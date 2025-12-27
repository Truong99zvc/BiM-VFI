import torch
import torch.nn.functional as F
import torch.nn as nn

from .backwarp import backwarp
from .resnet_encoder import ResNetPyramid
from .caun import CAUN
from .bimfn_hybrid import BiMFN_Hybrid
from .sn import SynthesisNetwork

from ..components import register

from utils.padder import InputPadder


def rotate_vector(vec, angle_rad):
    """
    Xoay vector 2D một góc angle_rad.
    vec: [B, 2, H, W] (u, v)
    angle_rad: [B, 1, H, W] (radians)
    """
    u = vec[:, 0:1, :, :]
    v = vec[:, 1:2, :, :]
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    u_new = u * cos_a - v * sin_a
    v_new = u * sin_a + v * cos_a
    return torch.cat([u_new, v_new], dim=1)


@register('bim_ifnet')
class BiM_IFNet(nn.Module):
    def __init__(self, pyr_level=3, feat_channels=32, **kwargs):
        super(BiM_IFNet, self).__init__()
        self.pyr_level = pyr_level
        self.mfe = ResNetPyramid(feat_channels)
        self.cfe = ResNetPyramid(feat_channels)
        self.bimfn = BiMFN_Hybrid(feat_channels)
        self.sn = SynthesisNetwork(feat_channels)
        self.feat_channels = feat_channels
        self.caun = CAUN(feat_channels)

    def forward_one_lvl(self, img0, img1, last_flow, last_occ, teacher_input_dict=None, time_period=0.5):
        # last_flow chứa 5 kênh: [base_flow(2), r(1), phi_vec(2)]

        # --- Upsample thông tin từ level trước ---
        last_flow_up = F.interpolate(
            input=last_flow.detach().clone(), scale_factor=2.0,
            mode="bilinear", align_corners=False
        )
        last_flow_up[:, :2] *= 2.0  # Scale flow
        last_occ_up = F.interpolate(
            input=last_occ.detach().clone(), scale_factor=2.0,
            mode="bilinear", align_corners=False
        )

        # --- Trích xuất đặc trưng ---
        feat0 = self.mfe(img0)[0]
        feat1 = self.mfe(img1)[0]

        # --- Dự đoán Residual (Dùng BiMFN mới nhẹ hơn) ---
        delta_base_flow, delta_r, delta_phi_vec, mask_res = self.bimfn(
            feat0, feat1, last_flow_up, last_occ_up
        )

        # --- Cập nhật tham số BiM tích lũy ---
        current_base_flow = last_flow_up[:, :2] + delta_base_flow
        current_r_logit = last_flow_up[:, 2:3] + delta_r
        current_r = torch.sigmoid(current_r_logit)
        current_phi_vec = last_flow_up[:, 3:5] + delta_phi_vec
        current_phi_vec = F.normalize(current_phi_vec, dim=1)
        current_occ = last_occ_up + mask_res

        # Params để lưu cho vòng lặp sau
        current_params_to_save = torch.cat(
            [current_base_flow, current_r_logit, current_phi_vec], dim=1
        )

        # --- BiM Equation Converter ---
        # Tính độ lớn
        base_mag = torch.norm(current_base_flow, dim=1, keepdim=True)
        mag_t0 = base_mag * current_r
        mag_t1 = base_mag * (1.0 - current_r)

        # Tính góc xoay
        raw_phi_angle = torch.atan2(current_phi_vec[:, 1:2], current_phi_vec[:, 0:1])
        rot_angle_t0 = -raw_phi_angle * (1 - time_period)
        rot_angle_t1 = raw_phi_angle * time_period

        # Tính vector hướng
        base_dir = current_base_flow / (base_mag + 1e-6)
        dir_t0 = rotate_vector(base_dir, rot_angle_t0)
        dir_t1 = rotate_vector(base_dir, rot_angle_t1)

        # Ra Flow cuối cùng
        flow_t0 = dir_t0 * mag_t0 * (-1.0)
        flow_t1 = dir_t1 * mag_t1

        # --- Synthesis (Sinh ảnh) ---
        flow_for_synth = torch.cat([flow_t0, flow_t1], dim=1)
        cfeat0_pyr = self.cfe(img0)
        cfeat1_pyr = self.cfe(img1)

        # Gọi mạng sinh ảnh
        # --- Synthesis (Sinh ảnh) ---
        flow_for_synth = torch.cat([flow_t0, flow_t1], dim=1)
        
        # [FIX ERROR] Tạo Flow Pyramid 3 cấp độ để khớp với SynthesisNetwork (SN)
        # SN yêu cầu flow ở các mức: gốc, giảm 2 lần, giảm 4 lần
        flow_pyramid = [flow_for_synth]
        
        # Level 1: Downsample 1/2
        f1 = F.interpolate(flow_for_synth, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        flow_pyramid.append(f1)
        
        # Level 2: Downsample 1/4 (từ f1 giảm tiếp 1/2)
        f2 = F.interpolate(f1, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        flow_pyramid.append(f2)

        cfeat0_pyr = self.cfe(img0) # Context Feature (thường trả về 3 level)
        cfeat1_pyr = self.cfe(img1)
        
        # Truyền flow_pyramid (3 phần tử) thay vì list 1 phần tử
        interp_img, out_occ, extra_dict = self.sn(
            img0, img1, cfeat0_pyr, cfeat1_pyr, flow_pyramid, current_occ
        )

        extra_dict['flow_res'] = flow_for_synth
        teacher_dict = {}

        return current_params_to_save, current_occ, interp_img, extra_dict, teacher_dict

    def forward(self, img0, img1, time_step,
                pyr_level=None, imgt=None, run_with_gt=False, **kwargs):
        if pyr_level is None: pyr_level = self.pyr_level
        N, _, H, W = img0.shape
        flowt0_pred_list = []
        flowt0_res_list = []
        flowt1_pred_list = []
        flowt1_res_list = []
        flow0t_tea_list = []
        flowt1_tea_list = []
        flowt0_pred_tea_list = []
        flowt0_res_tea_list = []
        flowt1_pred_tea_list = []
        flowt1_res_tea_list = []
        refine_mask_tea_list = []
        interp_imgs = []
        interp_imgs_tea = []

        padder = InputPadder(img0.shape, divisor=int(2 ** (pyr_level + 1)))

        ### Normalize input images
        with torch.set_grad_enabled(False):
            tenStats = [img0, img1]
            if self.training or run_with_gt:
                tenStats.append(imgt)
            tenMean_ = sum([tenIn.mean([1, 2, 3], True) for tenIn in tenStats]) / len(tenStats)
            tenStd_ = (sum([tenIn.std([1, 2, 3], False, True).square() + (
                    tenMean_ - tenIn.mean([1, 2, 3], True)).square() for tenIn in tenStats]) / len(tenStats)).sqrt()

            img0 = (img0 - tenMean_) / (tenStd_ + 0.0000001)
            img1 = (img1 - tenMean_) / (tenStd_ + 0.0000001)
            if self.training or run_with_gt:
                imgt = (imgt - tenMean_) / (tenStd_ + 0.0000001)

        ### Pad images for downsampling
        img0, img1 = padder.pad(img0, img1)
        if self.training or run_with_gt:
            imgt = padder.pad(imgt)

        N, _, H, W = img0.shape
        teacher_input_dict = dict()

        for level in list(range(pyr_level))[::-1]:
            ### Downsample images if needed
            if level != 0:
                scale_factor = 1 / 2 ** level
                img0_this_lvl = F.interpolate(
                    input=img0, scale_factor=scale_factor,
                    mode="bilinear", align_corners=False, antialias=True)
                img1_this_lvl = F.interpolate(
                    input=img1, scale_factor=scale_factor,
                    mode="bilinear", align_corners=False, antialias=True)
                if self.training or run_with_gt:
                    imgt_this_lvl = F.interpolate(
                        input=imgt, scale_factor=scale_factor,
                        mode="bilinear", align_corners=False, antialias=True)
                    teacher_input_dict['imgt_this_lvl'] = imgt_this_lvl
            else:
                img0_this_lvl = img0
                img1_this_lvl = img1
                if self.training or run_with_gt:
                    imgt_this_lvl = imgt
                    teacher_input_dict['imgt_this_lvl'] = imgt_this_lvl

            ### Initialize zero flows for lowest pyramid level
            if level == pyr_level - 1:
                # 5 kênh: 2 Flow + 1 R + 2 Phi cho kiến trúc lai
                last_flow = torch.zeros(
                    (N, 5, H // (2 ** (level + 1)), W // (2 ** (level + 1))), device=img0.device
                )
                # Mặc định Phi vector là (1, 0) tức cos=1, sin=0 (góc 0 độ)
                last_flow[:, 3, :, :] = 1.0 
                last_occ = torch.zeros(N, 1, H // (2 ** (level + 1)), W // (2 ** (level + 1)), device=img0.device)
            else:
                last_flow = flow  # Params tích lũy từ vòng trước
                last_occ = occ

            ### Single pyramid level run
            flow, occ, interp_img, extra_dict, teacher_dict = self.forward_one_lvl(
                img0_this_lvl, img1_this_lvl, last_flow, last_occ, teacher_input_dict, time_step)

            # Lấy Flow thực tế (4 kênh) từ extra_dict
            real_flow = extra_dict['flow_res']

            flowt0_pred_list.append((real_flow[:, :2]))
            flowt1_pred_list.append((real_flow[:, 2:]))
            flowt0_res_list.append(extra_dict['flow_res'][:, :2])
            flowt1_res_list.append(extra_dict['flow_res'][:, 2:])
            interp_imgs.append((interp_img) * (tenStd_ + 0.0000001) + tenMean_)
            
            if self.training or run_with_gt:
                if 'flow_t0_tea' in teacher_dict:
                    flowt0_pred_tea_list.append((teacher_dict['flow_t0_tea']))
                    flowt1_pred_tea_list.append((teacher_dict['flow_t1_tea']))
                    flowt0_res_tea_list.append(teacher_dict['flow_t0_res_tea'])
                    flowt1_res_tea_list.append(teacher_dict['flow_t1_res_tea'])
                    interp_imgs_tea.append((teacher_dict['interp_img_tea']) * (tenStd_ + 0.0000001) + tenMean_)
                    flow0t_tea_list.append(teacher_dict['flow_0t_res'][:, 2:])
                    flowt1_tea_list.append(teacher_dict['flow_t1_res'][:, :2])

        result_dict = {
            "imgt_preds": interp_imgs, "flowt0_pred_list": flowt0_pred_list[::-1],
            "flowt1_pred_list": flowt1_pred_list[::-1],
            'imgt_pred': padder.unpad(interp_imgs[-1].contiguous()),
            'flowt0_pred_tea_list': flowt0_pred_tea_list[::-1], 'flowt1_pred_tea_list': flowt1_pred_tea_list[::-1],
            'interp_imgs_tea': interp_imgs_tea, 'refine_mask_tea': refine_mask_tea_list,
            'flowt0_res_list': flowt0_res_list[::-1], 'flowt1_res_list': flowt1_res_list[::-1],
            'flowt0_res_tea_list': flowt0_res_tea_list[::-1], 'flowt1_res_tea_list': flowt1_res_tea_list[::-1],
            'flow0t_tea_list': flow0t_tea_list[::-1], 'flowt1_tea_list': flowt1_tea_list[::-1],
        }

        return result_dict
