<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: 5;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<!-- Title -->
<h1 align="center"><b>CS420.Q12 - SELECTED TOPICS IN COMPUTER VISION</b></h1>

## TABLE OF CONTENTS
* [Course Introduction](#course-introduction)
* [Instructor](#instructor)
* [Students](#students)
* [Project](#project)
* [Environment Setting](#environment-setting)
* [Important Configuration Notes](#important-configuration-notes)
* [Dataset](#dataset)
* [Pretrained Model](#pretrained-model)
* [Evaluation](#evaluation)
* [Training](#training)
* [Demo](#demo)
  - [Command Line Demo](#command-line-demo)
  - [Web Demo](#web-demo)
* [Kaggle Notebook](#kaggle-notebook)
* [License](#license)

## COURSE INTRODUCTION
<a name="course-introduction"></a>
* **Course Name**: Selected Topics in Computer Vision
* **Course Code**: CS420
* **Class Code**: CS420.Q12
* **Academic Year**: 2025 - 2026
* **Start Date**: September 8, 2025
* **End Date**: December 27, 2025

## INSTRUCTOR
<a name="instructor"></a>
* **TS. Mai Tiến Dũng** - *dungmt@uit.edu.vn*

## STUDENTS
<a name="students"></a>
| Student ID | Name                | Github                                               | Email                   |
|:----------:|:-------------------:|:----------------------------------------------------:|:-----------------------:|
| 22521587   | Trương Phúc Trường  | [Truong99zvc](https://github.com/Truong99zvc/)      | 22521587@gm.uit.edu.vn  |
| 22521571   | Võ Đình Trung       | [votrung654](https://github.com/votrung654/)         | 22521571@gm.uit.edu.vn  |

## PROJECT
<a name="project"></a>
**Project Name**: BiM-VFI: BIDIRECTIONAL MOTION FIELD-GUIDED FRAME INTERPOLATION FOR VIDEO

This repository contains the implementation of BiM-VFI, a bidirectional motion field-guided frame interpolation method for video with non-uniform motions. The project is based on the CVPR 2025 paper by Wonyong Seo, Jihyong Oh, and Munchurl Kim.

## ENVIRONMENT SETTING
<a name="environment-setting"></a>

### Prerequisites
Before setting up the environment, make sure you have **Conda** installed on your system. You can download and install Conda from:
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (Recommended - lightweight)
- [Anaconda](https://www.anaconda.com/products/distribution) (Full distribution)

### Environment Setup

> **⚠️ Important Note**: The library versions in this repository differ from the original BiM-VFI repository. Since our training was conducted on **GTX 1650**, we use the latest PyTorch version with CUDA 13.0 support (`torch torchvision --index-url https://download.pytorch.org/whl/cu130`) for optimal compatibility.

```bash
conda create -n bimvfi python=3.11
conda activate bimvfi
pip install basicsr-fixed Ipython torchsummary moviepy pyyaml imageio packaging tqdm opencv-python tensorboardx ptflops pyiqa lpips stlpips_pytorch dists_pytorch torch torchvision --index-url https://download.pytorch.org/whl/cu130
conda install cupy -c conda-forge
```

### Additional Libraries for Web Demo
To run the web demo, you need to install the following additional libraries:

```bash
pip install flask werkzeug pillow scikit-image
```

**Note**: `opencv-python` and `torch` are already included in the main environment setup above.

## IMPORTANT CONFIGURATION NOTES
<a name="important-configuration-notes"></a>

### KMP_DUPLICATE_LIB_OK Environment Variable
In `main.py`, we added the following line that is not present in the original repository:
```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```
This environment variable resolves the "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized" error that can occur when multiple copies of the OpenMP runtime are linked into the program. This is a common issue on Windows systems when using libraries like NumPy, PyTorch, and OpenCV together.

### Absolute Path Configuration (Critical)
**⚠️ This is crucial for running the code successfully!**

After cloning the repository, you **must** modify the dataset and model paths in the configuration files located in the `cfgs/` directory. The default relative paths will not work and need to be changed to **absolute paths** corresponding to your local system.

#### For Training (`cfgs/bim_vfi_train_new.yaml`):
Change the dataset path from relative to absolute:
```yaml
# Before (will NOT work)
root_path: ./data/vimeo_triplet

# After (example - adjust to your actual path)
root_path: C:/Users/YourUsername/BiM-VFI/data/vimeo_triplet
```

#### For Evaluation (`cfgs/bim_vfi_benchmark.yaml`):
1. **Model path** - Change `resume` to absolute path:
```yaml
# Before
resume: ./save/train_new__400_epochs_NEW/checkpoints/model_best.pth

# After (example)
resume: C:/Users/YourUsername/BiM-VFI/save/train_new__400_epochs_NEW/checkpoints/model_best.pth
```

2. **Dataset path** - Change `root_path` to absolute path:
```yaml
# Before
root_path: ./data/vimeo_triplet

# After (example)
root_path: C:/Users/YourUsername/BiM-VFI/data/vimeo_triplet
```

## DATASET
<a name="dataset"></a>
### Download
You can download the Vimeo90K dataset used for training and testing from the following link:
> - [Vimeo90K](https://cove.thecvf.com/datasets/875)

### Preparation
After downloading the dataset, organize it according to the project structure. The dataset should be placed in the `data` directory.

## PRETRAINED MODEL
<a name="pretrained-model"></a>

This repository includes two models:

### 1. Original Pretrained Model (from Paper)
- **Path**: `pretrained/bim_vfi.pth`
- **Description**: This is the original pretrained model from the BiM-VFI paper. It is already included in this repository.

### 2. Our Retrained Model
- **Path**: `save/train_new__400_epochs_NEW/checkpoints/model_best.pth`
- **Description**: This model was retrained by our team from scratch on the Vimeo Triplet dataset. The training was configured for 400 epochs but **early stopped at epoch 330** due to convergence.

### Directory Structure
```
BiM-VFI/
├── pretrained/
│   └── bim_vfi.pth                    # Original paper's pretrained model
└── save/
    ├── eval_pretrained_model/         # Evaluation results of pretrained model
    │   └── logs/
    │       └── log_benchmark_['vimeo']_[['test']].txt
    ├── eval_train_330_epochs/         # Evaluation results of our retrained model
    │   └── logs/
    │       └── log_benchmark_['vimeo']_[['test']].txt
    └── train_new__400_epochs_NEW/     # Our retrained model
        └── checkpoints/
            └── model_best.pth         # Best model (early stopped at epoch 330)
```

## EVALUATION
<a name="evaluation"></a>

### Evaluation Results
The `save/` directory contains evaluation results for both models:

#### 1. Pretrained Model Evaluation
- **Location**: `save/eval_pretrained_model/logs/log_benchmark_['vimeo']_[['test']].txt`
- **Description**: Contains benchmark results (PSNR, SSIM, LPIPS, STLPIPS, NIQE) of the **original pretrained model** from the paper, evaluated on the Vimeo Triplet test set.

#### 2. Retrained Model Evaluation
- **Location**: `save/eval_train_330_epochs/logs/log_benchmark_['vimeo']_[['test']].txt`
- **Description**: Contains benchmark results (PSNR, SSIM, LPIPS, STLPIPS, NIQE) of **our retrained model** (trained for 330 epochs), evaluated on the Vimeo Triplet test set.

### Running Evaluation
Desired evaluation can be done by replacing `benchmark_dataset` section in `cfgs/bim_vfi_benchmark.yaml`.
* `name`: Name of benchmark datasets. The datasets that can be benchmarked are [_vimeo_, _vimeo\_septuplet_, _snu\_film_, _snu\_film\_arb_, _xtest_].
* `args`:
  * `root_path`: Path of each dataset. **Must be absolute path!**
  * `split`: Desired splits to evaluate. [_test_, _val_] for _vimeo_ and _vimeo\_septuplet_, [(_easy_), _medium_, _hard_, _extreme_] for _snu\_film_ and _snu\_film\_arb_, and [_single_, _multiple_] for _xtest_.
  * `pyr_lvl`: 3 for vimeo, 5 for snu_film, and 7 for xtest.
* `save_imgs`: `True` if you want to save interpolation results, else `False`. It takes much more time to save images.

Then, run below:
```bash
python main.py --cfg cfgs/bim_vfi_benchmark.yaml
```

## TRAINING
<a name="training"></a>

To train the model:
```bash
python main.py --cfg cfgs/bim_vfi_train_new.yaml
```

**Note**: Make sure to configure the absolute paths in `cfgs/bim_vfi_train_new.yaml` before running (see [Important Configuration Notes](#important-configuration-notes)).

## DEMO
<a name="demo"></a>
### Command Line Demo
<a name="command-line-demo"></a>
Custom videos in multiple images or video format can be interpolated as follows.

First, set demo root directory as follows:
  - video1.mp4 
  - video2.mp4
  - video3
    - img0.png
    - img1.png
    - ...
  - ...

Then, replace `root_path` in `cfgs/bim_vfi_demo.yaml` to desired data root, and run:
```bash
python main.py --cfg cfgs/bim_vfi_demo.yaml
```

### Web Demo
<a name="web-demo"></a>
The project includes a web-based demo interface for easy video frame interpolation. To run the web demo:

1. **Navigate to the web_demo directory**:
   ```bash
   cd web_demo
   ```

2. **Run the Flask application**:
   ```bash
   python app.py
   ```

3. **Access the web interface**:
   Open your web browser and navigate to `http://localhost:5000` (or `http://127.0.0.1:5000`)

The web demo provides the following features:
- **Image Pair Interpolation**: Upload two images and generate interpolated frames between them
- **Video Interpolation**: Upload a video file and interpolate frames between consecutive frames
- **Frame Sequence Interpolation**: Upload multiple frames and interpolate between them
- **Model Selection**: Choose between different pre-trained models (pretrained, trained_330)
- **Customizable Parameters**: Adjust the number of interpolated frames and output FPS

**Note**: Make sure you have installed all the required libraries mentioned in the [Environment Setting](#environment-setting) section, including the additional libraries for web demo (Flask, werkzeug, Pillow, scikit-image).

## KAGGLE NOTEBOOK
<a name="kaggle-notebook"></a>

For users **without a GPU** or those who want to train/evaluate the model on cloud resources, we provide a Kaggle notebook:

> 🔗 **Kaggle Notebook**: [https://www.kaggle.com/code/truong9/bim-vfi?scriptVersionId=289083440](https://www.kaggle.com/code/truong9/bim-vfi?scriptVersionId=289083440)

**Note**: When using the Kaggle notebook, you will also need to adjust some configurations in the YAML files to match the Kaggle environment paths (e.g., `/kaggle/input/` for datasets).

## LICENSE
<a name="license"></a>
The source codes including the checkpoint can be freely used for research and education only. Any commercial use should get formal permission from the principal investigator (Prof. Munchurl Kim, mkimee@kaist.ac.kr).

