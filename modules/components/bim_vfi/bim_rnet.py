import torch
import torch.nn as nn
import torch.nn.functional as F
from .backwarp import backwarp
from .sn import SynthesisNetwork
from .bimfn_hybrid import conv
from ..components import register
from utils.padder import InputPadder


class ContextNet(nn.Module):
    """
    Lightweight encoder similar to RIFE's ContextNet.
    Outputs features at 3 scales: 1/8, 1/4, 1/2 of input resolution.
    """
    def __init__(self, feat_channels=32):
        super(ContextNet, self).__init__()
        self.feat_channels = feat_channels
        
        # Initial conv
        self.conv0 = nn.Sequential(
            conv(3, feat_channels, 3, 1, 1),
            conv(feat_channels, feat_channels, 3, 1, 1),
        )
        
        # Level 2: 1/8 resolution
        self.conv1 = nn.Sequential(
            conv(feat_channels, feat_channels, 3, 2, 1),  # 1/2
            conv(feat_channels, feat_channels, 3, 1, 1),
        )
        self.conv2 = nn.Sequential(
            conv(feat_channels, feat_channels, 3, 2, 1),  # 1/4
            conv(feat_channels, feat_channels, 3, 1, 1),
        )
        self.conv3 = nn.Sequential(
            conv(feat_channels, feat_channels, 3, 2, 1),  # 1/8
            conv(feat_channels, feat_channels, 3, 1, 1),
        )
        
        # Level 1: 1/4 resolution (from 1/8 upsampled)
        self.conv4 = nn.Sequential(
            conv(feat_channels, feat_channels, 3, 1, 1),
        )
        
        # Level 0: 1/2 resolution (from 1/4 upsampled)
        self.conv5 = nn.Sequential(
            conv(feat_channels, feat_channels, 3, 1, 1),
        )
    
    def forward(self, img):
        """
        Returns features at 3 scales:
        - feat_l2: 1/8 resolution (coarsest)
        - feat_l1: 1/4 resolution
        - feat_l0: 1/2 resolution
        """
        feat0 = self.conv0(img)  # Full resolution
        
        # Level 2 (1/8)
        feat_l2 = self.conv3(self.conv2(self.conv1(feat0)))
        
        # Level 1 (1/4) - upsample from L2
        feat_l1 = F.interpolate(feat_l2, scale_factor=2, mode='bilinear', align_corners=False)
        feat_l1 = self.conv4(feat_l1)
        
        # Level 0 (1/2) - upsample from L1
        feat_l0 = F.interpolate(feat_l1, scale_factor=2, mode='bilinear', align_corners=False)
        feat_l0 = self.conv5(feat_l0)
        
        return feat_l2, feat_l1, feat_l0


class BiMIFBlock(nn.Module):
    """
    BiM-IFBlock: Similar to RIFE's IFBlock but predicts BiM parameters.
    Input: Features from img0, img1, and upsampled flow/motion from previous level.
    Output: 6 channels representing BiM parameters:
        - delta_flow (2): residual base flow
        - delta_r (1): residual ratio parameter
        - delta_phi (2): residual angle (cos, sin)
        - mask (1): occlusion mask residual
    """
    def __init__(self, feat_channels=32, c=64):
        super(BiMIFBlock, self).__init__()
        # Input: feat0 (feat_channels) + feat1 (feat_channels) + 
        #        last_flow (2) + last_r (1) + last_phi (2) + last_occ (1)
        #        = feat_channels * 2 + 6
        input_dim = feat_channels * 2 + 6
        
        self.conv0 = nn.Sequential(
            conv(input_dim, c, 3, 2, 1),
            conv(c, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),  # 6 layers like RIFE's IFBlock
        )
        # Output: delta_flow(2) + delta_r(1) + delta_phi(2) + mask(1) = 6 channels
        self.lastconv = nn.ConvTranspose2d(c, 6, 4, 4, 0)
    
    def forward(self, feat0, feat1, last_flow, last_r, last_phi, last_occ):
        """
        Args:
            feat0: Features from img0 [B, C, H, W]
            feat1: Features from img1 [B, C, H, W]
            last_flow: Base flow from previous level [B, 2, H, W]
            last_r: Ratio parameter from previous level [B, 1, H, W]
            last_phi: Angle parameter from previous level [B, 2, H, W] (cos, sin)
            last_occ: Occlusion mask from previous level [B, 1, H, W]
        Returns:
            delta_flow: Residual base flow [B, 2, H, W]
            delta_r: Residual ratio [B, 1, H, W]
            delta_phi: Residual angle [B, 2, H, W]
            mask_res: Residual occlusion mask [B, 1, H, W]
        """
        x = torch.cat([feat0, feat1, last_flow, last_r, last_phi, last_occ], dim=1)
        x = self.conv0(x)
        x = self.convblock(x) + x  # Residual connection
        tmp = self.lastconv(x)
        
        delta_flow = tmp[:, 0:2]
        delta_r = tmp[:, 2:3]
        delta_phi = tmp[:, 3:5]
        mask_res = tmp[:, 5:6]
        
        return delta_flow, delta_r, delta_phi, mask_res


@register('bim_rnet')
class BiMRNet(nn.Module):
    """
    BiM-RNet: Hybrid model combining BiM theory with RIFE's efficient cascaded architecture.
    Uses cascaded coarse-to-fine processing (Level 2 -> Level 1 -> Level 0) instead of iterative refinement.
    """
    def __init__(self, feat_channels=32, **kwargs):
        super(BiMRNet, self).__init__()
        self.feat_channels = feat_channels
        
        # Lightweight encoder (replaces heavy ResNetPyramid)
        self.encoder = ContextNet(feat_channels)
        
        # Context encoder for synthesis network
        from .resnet_encoder import ResNetPyramid
        self.cfe = ResNetPyramid(feat_channels)
        
        # BiM-IFBlocks for each pyramid level
        self.bim_block_l2 = BiMIFBlock(feat_channels, c=64)  # Level 2 (coarsest)
        self.bim_block_l1 = BiMIFBlock(feat_channels, c=64)  # Level 1
        self.bim_block_l0 = BiMIFBlock(feat_channels, c=64)  # Level 0 (finest)
        
        # Synthesis network for final image refinement
        self.sn = SynthesisNetwork(feat_channels)
    
    def get_flow_from_bim(self, base_flow, r, phi, time_step=0.5):
        """
        Convert BiM parameters to forward/backward flows (V_t->0, V_t->1).
        
        BiM Theory:
        - base_flow: Base motion vector
        - r: Ratio parameter (0 to 1), determines magnitude split
        - phi: Angle parameter (cos, sin), determines rotation
        - time_step: Interpolation time (typically 0.5 for middle frame)
        
        Formulas (based on bim_ifnet.py):
        - Compute base magnitude and direction
        - Rotate base direction by time-dependent angles derived from phi
        - Apply magnitude scaling based on r
        
        Args:
            base_flow: Base flow vector [B, 2, H, W]
            r: Ratio parameter [B, 1, H, W] (should be in [0, 1])
            phi: Angle parameter [B, 2, H, W] (cos, sin)
            time_step: Interpolation time [scalar]
        
        Returns:
            flow_t0: Flow from t to 0 [B, 2, H, W]
            flow_t1: Flow from t to 1 [B, 2, H, W]
        """
        # Normalize phi to unit vector
        phi_norm = F.normalize(phi, dim=1, eps=1e-6)
        phi_cos = phi_norm[:, 0:1, :, :]
        phi_sin = phi_norm[:, 1:2, :, :]
        
        # Compute base flow magnitude and direction
        base_mag = torch.norm(base_flow, dim=1, keepdim=True) + 1e-6
        base_dir = base_flow / base_mag
        
        # Compute angle from phi (in radians)
        raw_phi_angle = torch.atan2(phi_sin, phi_cos)
        
        # Compute time-dependent rotation angles
        rot_angle_t0 = -raw_phi_angle * (1.0 - time_step)
        rot_angle_t1 = raw_phi_angle * time_step
        
        # Rotate base direction by rot_angle_t0 and rot_angle_t1
        base_dir_u = base_dir[:, 0:1, :, :]
        base_dir_v = base_dir[:, 1:2, :, :]
        
        # Rotation for t->0: [cos(rot_angle_t0), -sin(rot_angle_t0); sin(rot_angle_t0), cos(rot_angle_t0)]
        cos_t0 = torch.cos(rot_angle_t0)
        sin_t0 = torch.sin(rot_angle_t0)
        dir_t0_u = base_dir_u * cos_t0 - base_dir_v * sin_t0
        dir_t0_v = base_dir_u * sin_t0 + base_dir_v * cos_t0
        dir_t0 = torch.cat([dir_t0_u, dir_t0_v], dim=1)
        
        # Rotation for t->1: [cos(rot_angle_t1), -sin(rot_angle_t1); sin(rot_angle_t1), cos(rot_angle_t1)]
        cos_t1 = torch.cos(rot_angle_t1)
        sin_t1 = torch.sin(rot_angle_t1)
        dir_t1_u = base_dir_u * cos_t1 - base_dir_v * sin_t1
        dir_t1_v = base_dir_u * sin_t1 + base_dir_v * cos_t1
        dir_t1 = torch.cat([dir_t1_u, dir_t1_v], dim=1)
        
        # Compute magnitudes for t->0 and t->1
        mag_t0 = base_mag * r
        mag_t1 = base_mag * (1.0 - r)
        
        # Compute final flows (matching bim_ifnet.py logic)
        flow_t0 = dir_t0 * mag_t0 * (-1.0)  # Negative for backward direction
        flow_t1 = dir_t1 * mag_t1
        
        return flow_t0, flow_t1
    
    def forward(self, img0, img1, time_step=0.5, **kwargs):
        """
        Cascaded forward pass through 3 pyramid levels.
        
        Args:
            img0: First input image [B, 3, H, W]
            img1: Second input image [B, 3, H, W]
            time_step: Interpolation time (default 0.5 for middle frame)
        
        Returns:
            Dictionary with:
                - imgt_pred: Predicted intermediate frame [B, 3, H, W]
                - flow: List of flows at each level
                - Other outputs compatible with training loop
        """
        B, _, H, W = img0.shape
        
        # Normalize input images (same as BiMVFI)
        with torch.set_grad_enabled(False):
            tenMean_ = (img0.mean([1, 2, 3], True) + img1.mean([1, 2, 3], True)) / 2.0
            tenStd_ = ((img0.std([1, 2, 3], False, True).square() + 
                       (tenMean_ - img0.mean([1, 2, 3], True)).square() +
                       img1.std([1, 2, 3], False, True).square() + 
                       (tenMean_ - img1.mean([1, 2, 3], True)).square()) / 2.0).sqrt()
            
            img0 = (img0 - tenMean_) / (tenStd_ + 0.0000001)
            img1 = (img1 - tenMean_) / (tenStd_ + 0.0000001)
        
        # Pad images for downsampling
        padder = InputPadder(img0.shape, divisor=16)  # 2^(3+1) = 16 for 3 levels
        img0, img1 = padder.pad(img0, img1)
        B, _, H, W = img0.shape
        
        # Extract features at 3 scales
        feat0_l2, feat0_l1, feat0_l0 = self.encoder(img0)
        feat1_l2, feat1_l1, feat1_l0 = self.encoder(img1)
        
        # Get context features for synthesis network
        cfeat0_pyr = self.cfe(img0)
        cfeat1_pyr = self.cfe(img1)
        
        # Initialize BiM parameters for Level 2 (coarsest)
        # Resolution at L2: H/8, W/8
        H_l2, W_l2 = H // 8, W // 8
        base_flow_l2 = torch.zeros(B, 2, H_l2, W_l2, device=img0.device)
        r_l2 = torch.ones(B, 1, H_l2, W_l2, device=img0.device) * time_step
        phi_l2 = torch.zeros(B, 2, H_l2, W_l2, device=img0.device)
        phi_l2[:, 0, :, :] = 1.0  # cos = 1, sin = 0 (angle = 0)
        occ_l2 = torch.zeros(B, 1, H_l2, W_l2, device=img0.device)
        
        # ========== Level 2 Processing (Coarsest: 1/8 resolution) ==========
        delta_flow_l2, delta_r_l2, delta_phi_l2, mask_res_l2 = self.bim_block_l2(
            feat0_l2, feat1_l2, base_flow_l2, r_l2, phi_l2, occ_l2
        )
        
        # Update BiM parameters
        base_flow_l2 = base_flow_l2 + delta_flow_l2
        r_l2 = torch.sigmoid(r_l2 + delta_r_l2)  # Ensure r in [0, 1]
        phi_l2 = F.normalize(phi_l2 + delta_phi_l2, dim=1, eps=1e-6)
        occ_l2 = occ_l2 + mask_res_l2
        
        # Convert to flows
        flow_t0_l2, flow_t1_l2 = self.get_flow_from_bim(base_flow_l2, r_l2, phi_l2, time_step)
        flow_l2 = torch.cat([flow_t0_l2, flow_t1_l2], dim=1)  # [B, 4, H/8, W/8]
        
        # ========== Level 1 Processing (1/4 resolution) ==========
        # Upsample flow and parameters from L2 to L1
        H_l1, W_l1 = H // 4, W // 4
        base_flow_l1 = F.interpolate(base_flow_l2, size=(H_l1, W_l1), mode='bilinear', align_corners=False) * 2.0
        r_l1 = F.interpolate(r_l2, size=(H_l1, W_l1), mode='bilinear', align_corners=False)
        phi_l1 = F.interpolate(phi_l2, size=(H_l1, W_l1), mode='bilinear', align_corners=False)
        phi_l1 = F.normalize(phi_l1, dim=1, eps=1e-6)  # Renormalize after interpolation
        occ_l1 = F.interpolate(occ_l2, size=(H_l1, W_l1), mode='bilinear', align_corners=False)
        
        # Warp features using current flow estimate
        flow_t0_l1_prev, flow_t1_l1_prev = self.get_flow_from_bim(base_flow_l1, r_l1, phi_l1, time_step)
        feat0_l1_warp = backwarp(feat0_l1, flow_t0_l1_prev)
        feat1_l1_warp = backwarp(feat1_l1, flow_t1_l1_prev)
        
        # Predict residual
        delta_flow_l1, delta_r_l1, delta_phi_l1, mask_res_l1 = self.bim_block_l1(
            feat0_l1_warp, feat1_l1_warp, base_flow_l1, r_l1, phi_l1, occ_l1
        )
        
        # Update BiM parameters
        base_flow_l1 = base_flow_l1 + delta_flow_l1
        r_l1 = torch.sigmoid(r_l1 + delta_r_l1)
        phi_l1 = F.normalize(phi_l1 + delta_phi_l1, dim=1, eps=1e-6)
        occ_l1 = occ_l1 + mask_res_l1
        
        # Convert to flows
        flow_t0_l1, flow_t1_l1 = self.get_flow_from_bim(base_flow_l1, r_l1, phi_l1, time_step)
        flow_l1 = torch.cat([flow_t0_l1, flow_t1_l1], dim=1)  # [B, 4, H/4, W/4]
        
        # ========== Level 0 Processing (Finest: 1/2 resolution) ==========
        # Upsample flow and parameters from L1 to L0
        H_l0, W_l0 = H // 2, W // 2
        base_flow_l0 = F.interpolate(base_flow_l1, size=(H_l0, W_l0), mode='bilinear', align_corners=False) * 2.0
        r_l0 = F.interpolate(r_l1, size=(H_l0, W_l0), mode='bilinear', align_corners=False)
        phi_l0 = F.interpolate(phi_l1, size=(H_l0, W_l0), mode='bilinear', align_corners=False)
        phi_l0 = F.normalize(phi_l0, dim=1, eps=1e-6)
        occ_l0 = F.interpolate(occ_l1, size=(H_l0, W_l0), mode='bilinear', align_corners=False)
        
        # Warp features using current flow estimate
        flow_t0_l0_prev, flow_t1_l0_prev = self.get_flow_from_bim(base_flow_l0, r_l0, phi_l0, time_step)
        feat0_l0_warp = backwarp(feat0_l0, flow_t0_l0_prev)
        feat1_l0_warp = backwarp(feat1_l0, flow_t1_l0_prev)
        
        # Predict residual
        delta_flow_l0, delta_r_l0, delta_phi_l0, mask_res_l0 = self.bim_block_l0(
            feat0_l0_warp, feat1_l0_warp, base_flow_l0, r_l0, phi_l0, occ_l0
        )
        
        # Update BiM parameters
        base_flow_l0 = base_flow_l0 + delta_flow_l0
        r_l0 = torch.sigmoid(r_l0 + delta_r_l0)
        phi_l0 = F.normalize(phi_l0 + delta_phi_l0, dim=1, eps=1e-6)
        occ_l0 = occ_l0 + mask_res_l0
        
        # Convert to final flows
        flow_t0_l0, flow_t1_l0 = self.get_flow_from_bim(base_flow_l0, r_l0, phi_l0, time_step)
        flow_l0 = torch.cat([flow_t0_l0, flow_t1_l0], dim=1)  # [B, 4, H/2, W/2]
        
        # ========== Upsample to Full Resolution and Synthesis ==========
        # Upsample flow to full resolution for synthesis
        flow_full = F.interpolate(flow_l0, size=(H, W), mode='bilinear', align_corners=False) * 2.0
        occ_full = F.interpolate(occ_l0, size=(H, W), mode='bilinear', align_corners=False)
        
        # Create flow pyramid for synthesis network (requires 3 levels)
        flow_pyramid = [
            flow_full,  # Level 0: full resolution
            flow_l0,    # Level 1: 1/2 resolution
            flow_l1,    # Level 2: 1/4 resolution
        ]
        
        # Use synthesis network to generate final image
        interp_img, occ_out, extra_dict = self.sn(
            img0, img1, cfeat0_pyr, cfeat1_pyr, flow_pyramid, occ_full
        )
        
        # Denormalize output
        interp_img = interp_img * (tenStd_ + 0.0000001) + tenMean_
        
        # Prepare output dictionary compatible with training loop
        # Note: BiMRNet doesn't use teacher-student training, so teacher-related keys use dummy zero tensors
        # This allows loss functions to run without errors (they'll compute regularization terms)
        # Lists are built from coarsest to finest (L2, L1, L0), then reversed to match expected format
        flowt0_pred_list = [flow_t0_l2, flow_t0_l1, flow_t0_l0]
        flowt1_pred_list = [flow_t1_l2, flow_t1_l1, flow_t1_l0]
        
        # Create dummy zero tensors for teacher flows
        # FlowSmoothnessTeacher1Loss expects [full_res, H/2, H/4] and does NOT reverse the list
        # FlowTeacherLoss expects [H/2, H/4, H/8] after reversal (finest to coarsest)
        # We need to create flows at the right resolutions for each loss
        # For FlowSmoothnessTeacher1Loss (no reversal): [full, H/2, H/4]
        # Get unpadded flow_full to match imgt resolution in loss function
        flow_full_unpadded_t0 = padder.unpad(flow_full[:, :2])
        flow_full_unpadded_t1 = padder.unpad(flow_full[:, 2:])
        flowt0_pred_tea_list = [
            torch.zeros_like(flow_full_unpadded_t0),  # Full resolution (unpadded) for i=0
            torch.zeros_like(flow_t0_l0),             # H/2 resolution for i=1
            torch.zeros_like(flow_t0_l1),            # H/4 resolution for i=2
        ]
        flowt1_pred_tea_list = [
            torch.zeros_like(flow_full_unpadded_t1),  # Full resolution (unpadded) for i=0
            torch.zeros_like(flow_t1_l0),              # H/2 resolution for i=1
            torch.zeros_like(flow_t1_l1),             # H/4 resolution for i=2
        ]
        # For other losses that expect [H/2, H/4, H/8] in finest-to-coarsest order (after reversal)
        # flowt0_res_list is [flow_t0_l0, flow_t0_l1, flow_t0_l2] (finest to coarsest) after reversal
        # So flowt0_res_tea_list should also be [flow_t0_l0, flow_t0_l1, flow_t0_l2] before reversal
        # Then after reversal it becomes [flow_t0_l2, flow_t0_l1, flow_t0_l0] which is wrong!
        # Actually, we need to create it in coarsest-to-finest order so after reversal it matches
        # Wait, let me check: flowt0_res_list = flowt0_pred_list[::-1] = [flow_t0_l2, flow_t0_l1, flow_t0_l0][::-1] = [flow_t0_l0, flow_t0_l1, flow_t0_l2]
        # So flowt0_res_list[i=0] = flow_t0_l0 (finest), flowt0_res_list[i=2] = flow_t0_l2 (coarsest)
        # flowt0_res_tea_list should match this order, so it should be [flow_t0_l0, flow_t0_l1, flow_t0_l2] (finest to coarsest)
        # But we're reversing it, so we need to create it as [flow_t0_l2, flow_t0_l1, flow_t0_l0] (coarsest to finest)
        # Then after reversal: [flow_t0_l0, flow_t0_l1, flow_t0_l2] which matches!
        flowt0_res_tea_list = [
            torch.zeros_like(flow_t0_l2),        # H/8 (coarsest, will be reversed to be at index 2)
            torch.zeros_like(flow_t0_l1),        # H/4 (will be reversed to be at index 1)
            torch.zeros_like(flow_t0_l0),        # H/2 (finest, will be reversed to be at index 0)
        ]
        flowt1_res_tea_list = [
            torch.zeros_like(flow_t1_l2),        # H/8 (coarsest)
            torch.zeros_like(flow_t1_l1),        # H/4
            torch.zeros_like(flow_t1_l0),        # H/2 (finest)
        ]
        flow0t_tea_list = [
            torch.zeros_like(flow_t0_l2),
            torch.zeros_like(flow_t0_l1),
            torch.zeros_like(flow_t0_l0)
        ]
        flowt1_tea_list = [
            torch.zeros_like(flow_t1_l2),
            torch.zeros_like(flow_t1_l1),
            torch.zeros_like(flow_t1_l0)
        ]
        
        # Create dummy teacher images for PhotoTeacherLoss (same shape as student prediction)
        # PhotoTeacherLoss expects a list, and will iterate over it
        # We provide dummy zero tensors so the loss can compute (will be regularization)
        imgt_pred_unpadded = padder.unpad(interp_img.contiguous())
        interp_imgs_tea = [
            torch.zeros_like(imgt_pred_unpadded),  # Level 0 (finest)
            torch.zeros_like(F.interpolate(imgt_pred_unpadded, scale_factor=0.5, mode='bilinear', align_corners=False)),  # Level 1
            torch.zeros_like(F.interpolate(imgt_pred_unpadded, scale_factor=0.25, mode='bilinear', align_corners=False)),  # Level 2
        ]
        # Also provide dummy warped images for the other PhotoTeacherLoss variant
        warped_img0_tea_list = interp_imgs_tea.copy()
        warped_img1_tea_list = interp_imgs_tea.copy()
        
        # Prepare flow lists
        # flowt0_res_list: used by loss functions, should be at [H/2, H/4, H/8] (finest to coarsest)
        # These need to match flowt0_res_tea_list resolutions exactly
        flowt0_res_list = [flow_t0_l0, flow_t0_l1, flow_t0_l2]  # [H/2, H/4, H/8] (finest to coarsest)
        flowt1_res_list = [flow_t1_l0, flow_t1_l1, flow_t1_l2]  # [H/2, H/4, H/8]
        
        # flowt0_pred_list: used by visualization, first element should be at full resolution
        # Upsample the finest flow (index 0) to full resolution for visualization
        flowt0_full = padder.unpad(F.interpolate(
            flow_t0_l0, 
            size=(imgt_pred_unpadded.shape[2], imgt_pred_unpadded.shape[3]),
            mode='bilinear', align_corners=False
        ) * 2.0)
        flowt1_full = padder.unpad(F.interpolate(
            flow_t1_l0,
            size=(imgt_pred_unpadded.shape[2], imgt_pred_unpadded.shape[3]),
            mode='bilinear', align_corners=False
        ) * 2.0)
        
        # Create separate list for visualization with upsampled first element
        # This avoids modifying flowt0_res_list which is used by loss functions
        flowt0_pred_list_reversed = [flowt0_full, flow_t0_l1, flow_t0_l2]  # [full_res, H/2, H/4]
        flowt1_pred_list_reversed = [flowt1_full, flow_t1_l1, flow_t1_l2]  # [full_res, H/2, H/4]
        
        result_dict = {
            'imgt_pred': imgt_pred_unpadded,
            'imgt_preds': [imgt_pred_unpadded],  # Single prediction
            'flowt0_pred_list': flowt0_pred_list_reversed,  # [full_res, H/2, H/4] for visualization
            'flowt1_pred_list': flowt1_pred_list_reversed,
            'flowt0_res_list': flowt0_res_list,  # [H/2, H/4, H/8] for loss functions
            'flowt1_res_list': flowt1_res_list,
            'flow': flow_pyramid,
            # Teacher-related keys (dummy zero tensors for compatibility)
            # Note: flowt0_pred_tea_list is NOT reversed because FlowSmoothnessTeacher1Loss doesn't reverse it
            'flowt0_pred_tea_list': flowt0_pred_tea_list,
            'flowt1_pred_tea_list': flowt1_pred_tea_list,
            'interp_imgs_tea': interp_imgs_tea[::-1],  # Dummy zero tensors with proper shapes
            'refine_mask_tea': [],
            'flowt0_res_tea_list': flowt0_res_tea_list[::-1],
            'flowt1_res_tea_list': flowt1_res_tea_list[::-1],
            'flow0t_tea_list': flow0t_tea_list[::-1],
            'flowt1_tea_list': flowt1_tea_list[::-1],
        }
        
        # Add extra_dict contents (may include warped_img0, warped_img1, etc.)
        result_dict.update(extra_dict)
        
        # Add warped images for teacher losses (dummy tensors with proper shapes)
        if 'warped_img0_tea_list' not in result_dict:
            result_dict['warped_img0_tea_list'] = warped_img0_tea_list[::-1]
        if 'warped_img1_tea_list' not in result_dict:
            result_dict['warped_img1_tea_list'] = warped_img1_tea_list[::-1]
        
        return result_dict

