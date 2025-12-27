import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class BiMFN_Hybrid(nn.Module):
    """
    Lightweight BiMFN - lấy cấu trúc từ IFBlock của RIFE
    Input: feat0, feat1, last_flow_up (5 kênh), last_occ_up (1 kênh)
    Output: delta_base_flow(2), delta_r(1), delta_phi_vec(2), mask_res(1)
    """
    def __init__(self, feat_channels=32):
        super(BiMFN_Hybrid, self).__init__()
        # Input channels:
        # feat0 (32) + feat1 (32) + last_flow_up (5: flow+r+phi) + last_occ (1) = 70
        input_dim = feat_channels * 2 + 6
        c = 64  # Số kênh trung gian (giống RIFE dùng 64-90)

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
            conv(c, c)  # 6 lớp conv giống IFBlock
        )
        # Output: delta_flow(2) + delta_r(1) + delta_phi_vec(2) + mask(1) = 6 kênh
        self.lastconv = nn.ConvTranspose2d(c, 6, 4, 4, 0)

    def forward(self, feat0, feat1, last_flow_up, last_occ_up):
        x = torch.cat((feat0, feat1, last_flow_up, last_occ_up), dim=1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)

        delta_base_flow = tmp[:, 0:2]
        delta_r = tmp[:, 2:3]
        delta_phi_vec = tmp[:, 3:5]
        mask_res = tmp[:, 5:6]

        return delta_base_flow, delta_r, delta_phi_vec, mask_res
