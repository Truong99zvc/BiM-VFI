import torch
import logging
import os
import os.path as osp
import time
import datetime
import random
import wandb
import yaml
import json
import numpy as np

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, DataLoader

import utils
from utils.misc import print_cuda_statistics, is_main_process, get_rank, get_world_size
import datasets
import modules.models as models


class Trainer(object):
    """
    Wrapper for training, more related to engineering than research code
    """

    def __init__(self, cfgs):
        self.rank = get_rank()
        self.cfgs = cfgs
        self.is_master = (self.rank == 0)
        self.is_train = False

        env = cfgs['env']
        self.tot_gpus = get_world_size()
        self.distributed = (get_world_size() > 1)

        # Setup log, tensorboard, wandb
        if self.is_master:
            logger = utils.misc.set_save_dir(cfgs['log_dir'], cfgs["run_description"], replace=False)
            with open(osp.join(cfgs['cfg_dir'], f'cfg_{cfgs["run_description"]}.yaml'), 'w') as f:
                yaml.dump(cfgs, f, sort_keys=False)

            self.log = logger.info

            # ADD: Create epoch log file
            self.epoch_log_file = osp.join(cfgs['log_dir'], f'epoch_log_{cfgs["run_description"]}.txt')
            with open(self.epoch_log_file, 'w') as f:
                f.write("Epoch\tTrain_Loss\tVal_PSNR\tVal_SSIM\tLR\tTime\n")

            self.enable_tb = True


        else:
            self.log = lambda *args, **kwargs: None
            self.enable_tb = False
            self.enable_wandb = False

        self.make_datasets()
        self.model = models.make(cfgs)
        self.start_epoch = 0
        self.end_epoch = self.cfgs['max_epoch']

        # ADD: Early stopping parameters
        self.early_stopping_patience = cfgs.get('early_stopping_patience', 10)
        self.early_stopping_counter = 0
        self.early_stopping_enabled = cfgs.get('enable_early_stopping', False)
        
        if 'resume' in self.cfgs:
            run_id = self.model.load_checkpoint(self.cfgs['resume'])
            self.start_epoch = self.model.current_epoch
        else:
            run_id = wandb.util.generate_id()
        if self.is_master and env['wandb_upload']:
            self.enable_wandb = True
            self.cfgs['enable_wandb'] = True
            with open('wandb.yaml', 'r') as f:
                wandb_cfg = yaml.load(f, Loader=yaml.FullLoader)
            os.environ['WANDB_DIR'] = env['save_dir']
            os.environ['WANDB_NAME'] = env['exp_name']
            os.environ['WANDB_API_KEY'] = wandb_cfg['api_key']
            wandb.init(project=wandb_cfg['project'], entity=wandb_cfg['entity'], config=cfgs, id=run_id, name=env['exp_name'],
                       resume='allow')
        else:
            self.enable_wandb = False
            self.cfgs['enable_wandb'] = False

    def make_datasets(self):
            """
                By default, train dataset performs shuffle and drop_last.
                Distributed sampler will extend the dataset with a prefix to make the length divisible by tot_gpus, samplers should be stored in .dist_samplers.

                Cfg example:

                train/test_dataset:
                    name:
                    args:
                    loader: {batch_size: , num_workers: }
            """
            cfgs = self.cfgs
            self.dist_samplers = []

            def make_distributed_loader(dataset, batch_size, num_workers, shuffle=False, drop_last=False):
                sampler = DistributedSampler(dataset, shuffle=shuffle) if self.distributed else None
                loader = DataLoader(
                    dataset,
                    batch_size // self.tot_gpus,
                    drop_last=drop_last,
                    sampler=sampler,
                    shuffle=(shuffle and (sampler is None)),
                    num_workers=num_workers // self.tot_gpus,
                    pin_memory=True)
                return loader, sampler

            if cfgs.get('train_dataset') is not None:
                train_dataset = datasets.make(cfgs['train_dataset'])
                self.log(f'Train dataset: len={len(train_dataset)}')
                l = cfgs['train_dataset']['loader']
                self.train_loader, train_sampler = make_distributed_loader(
                    train_dataset, l['batch_size'], l['num_workers'], shuffle=True, drop_last=True)
                self.dist_samplers.append(train_sampler)
                
               # Chỉ thêm total_steps nếu dùng one_cycle_lr
                if cfgs['lr_scheduler']['name'] == 'one_cycle_lr':
                    total_steps = len(self.train_loader) * self.cfgs['max_epoch']
                    if 'args' not in cfgs['lr_scheduler']:
                        cfgs['lr_scheduler']['args'] = {}
                    cfgs['lr_scheduler']['args']['total_steps'] = total_steps
                    self.log(f'Total training steps: {total_steps}')

            if cfgs.get('test_dataset') is not None:
                test_dataset = datasets.make(cfgs['test_dataset'])
                self.log(f'Test dataset: len={len(test_dataset)}')
                l = cfgs['test_dataset']['loader']
                self.test_loader, test_sampler = make_distributed_loader(
                    test_dataset, l['batch_size'], l['num_workers'], shuffle=False, drop_last=False)
                self.dist_samplers.append(test_sampler)
            if cfgs.get('demo_dataset') is not None:
                self.demo_root = self.cfgs['demo_dataset']['args']['root_path']

    def train(self):
        print("Start training")
        start_time = time.time()
        self.is_train = True
        self.model.init_training_logger()
        self.best_performance = 0
        # torch.backends.cudnn.benchmark = True
        for epoch in range(self.start_epoch, self.end_epoch):
            epoch_start_time = time.time()
            
            if self.cfgs['distributed']:
                self.train_loader.batch_sampler.sampler.set_epoch(epoch)

            random.seed(self.cfgs['seed'] + epoch)
            np.random.seed(self.cfgs['seed'] + epoch)
            torch.random.manual_seed(self.cfgs['seed'] + epoch)
            torch.manual_seed(self.cfgs['seed'] + epoch)
            torch.cuda.manual_seed_all(self.cfgs['seed'] + epoch)

            self.model.train_one_epoch(self.train_loader, epoch)

            # Get current learning rate and training loss
            if is_main_process():
                current_lr = self.model.optimizer.param_groups[0]["lr"]
                avg_train_loss = self.model.metric_logger.loss.global_avg if hasattr(self.model.metric_logger, 'loss') else 0.0
                
                # Check if we should validate this epoch
                should_validate = ((epoch + 1) % self.cfgs['validate_every']) == 0
                
                if should_validate:
                    # Run validation
                    current_performance = self.validate()
                    
                    # Get validation metrics
                    val_psnr = self.model.metric_logger.psnr.global_avg
                    val_ssim = self.model.metric_logger.ssim.global_avg
                    
                    # Calculate epoch time
                    epoch_time = time.time() - epoch_start_time
                    epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
                    
                    # Log to file with validation metrics
                    with open(self.epoch_log_file, 'a') as f:
                        f.write(f"{epoch+1}\t{avg_train_loss:.6f}\t{val_psnr:.4f}\t{val_ssim:.4f}\t{current_lr:.8f}\t{epoch_time_str}\n")
                    
                    self.log(f"Epoch {epoch+1}/{self.end_epoch} - Loss: {avg_train_loss:.6f}, "
                            f"Val PSNR: {val_psnr:.4f}, Val SSIM: {val_ssim:.4f}, "
                            f"LR: {current_lr:.8f}, Time: {epoch_time_str}")
                    
                    # Check if best performance
                    if current_performance > self.best_performance:
                        self.best_performance = current_performance
                        self.model.save_checkpoint('model_{}.pth'.format(epoch + 1), is_best=1)
                        self.log(f"Best performance achieved at epoch {epoch+1} with PSNR of {self.best_performance:.4f}")
                        
                        # Reset early stopping counter
                        self.early_stopping_counter = 0
                    else:
                        # Increment early stopping counter
                        self.early_stopping_counter += 1
                        
                    # Check early stopping
                    if self.early_stopping_enabled and self.early_stopping_counter >= self.early_stopping_patience:
                        self.log(f"Early stopping triggered after {epoch+1} epochs. "
                                f"No improvement for {self.early_stopping_patience} validation epochs.")
                        self.log(f"Best PSNR: {self.best_performance:.4f}")
                        break
                else:
                    # No validation this epoch - just log training metrics
                    epoch_time = time.time() - epoch_start_time
                    epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
                    
                    # Log to file without validation metrics (use N/A or previous values)
                    with open(self.epoch_log_file, 'a') as f:
                        f.write(f"{epoch+1}\t{avg_train_loss:.6f}\tN/A\tN/A\t{current_lr:.8f}\t{epoch_time_str}\n")
                    
                    self.log(f"Epoch {epoch+1}/{self.end_epoch} - Loss: {avg_train_loss:.6f}, "
                            f"LR: {current_lr:.8f}, Time: {epoch_time_str}")

            if ((epoch + 1) % self.cfgs['save_every']) == 0 and is_main_process():
                self.model.save_checkpoint('model_{}.pth'.format(epoch + 1))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        if is_main_process():
            self.finalize_training()

            """ if ((epoch + 1) % self.cfgs['validate_every']) == 0:
                if is_main_process():
                    performance = self.validate()
                    if performance > self.best_performance:
                        self.best_performance = performance
                        self.model.save_checkpoint('model_{}.pth'.format(epoch + 1), is_best=1)
                        self.log(
                            "best performance achieved at epoch {} with performance of {}".format(epoch,
                                                                                                  self.best_performance))

            if ((epoch + 1) % self.cfgs['save_every']) == 0 and is_main_process():
                self.model.save_checkpoint('model_{}.pth'.format(epoch + 1))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        if is_main_process():
            self.finalize_training() """

    def validate(self):
        # return performance to save the best model, if there is no performance measure e.g. GAN then just return 0
        if not self.is_train:  # if mode == validation only
            self.model.init_validation_logger()
        return self.model.validate(self.test_loader)

    def benchmark(self):
        self.model.init_testing_logger()
        self.model.benchmark()


    def demo(self):
        self.model.init_demo_logger()
        self.model.demo(self.demo_root)

    def finalize_training(self):
        self.model.finalize_training()
