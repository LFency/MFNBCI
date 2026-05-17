import torch
import torch.nn as nn
import torch.nn.functional as F
from LOCAL import Local
from GLOBAL import GlobalBock

class ConvCrossAttention3D(nn.Module):
    def __init__(self, in_channels, inter_channels=None, kernel_size=3):
        super(ConvCrossAttention3D, self).__init__()
        if inter_channels is None:
            inter_channels = in_channels // 2

        padding = (kernel_size - 1) // 2
        self.query_conv = nn.Conv3d(in_channels, inter_channels, kernel_size=kernel_size, padding=padding)
        self.key_conv   = nn.Conv3d(in_channels, inter_channels, kernel_size=kernel_size, padding=padding)
        self.value_conv = nn.Conv3d(in_channels, in_channels,   kernel_size=kernel_size, padding=padding)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习缩放因子

    def forward(self, x1, x2):
        # x1, x2: [B, C, D, H, W]
        query = self.query_conv(x1)   # [B, C', D, H, W]
        key   = self.key_conv(x2)     # [B, C', D, H, W]
        value = self.value_conv(x2)   # [B, C , D, H, W]

        B, Cq, D, H, W = query.shape
        query = query.view(B, Cq, -1).permute(0, 2, 1)   # [B, N, C']
        key   = key.view(B, Cq, -1)                      # [B, C', N]
        attention = torch.bmm(query, key)                # [B, N, N]
        attention = F.softmax(attention, dim=-1)

        value = value.view(B, -1, D * H * W)             # [B, C, N]
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, N]
        out = out.view(B, -1, D, H, W)                   # [B, C, D, H, W]

        out = self.gamma * out + x1                      # 残差
        return out                                       # 别 squeeze(0)

class G_L_Cross(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.G = GlobalBock(in_channels)
        self.L = Local(in_channels)
        self.cross_G = ConvCrossAttention3D(in_channels)
        self.cross_L = ConvCrossAttention3D(in_channels)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.FC1 = nn.Linear(in_channels, in_channels//2)
        self.FC2 = nn.Linear(in_channels//2, in_channels//4)

        # 输入残差（向量级）+ 门控
        self.res_gap = nn.AdaptiveAvgPool3d(1)
        self.res_fc1 = nn.Linear(in_channels, in_channels//2)
        self.res_fc2 = nn.Linear(in_channels//2, in_channels//4)
        self.alpha_gl = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x: [B, C, D, H, W]
        x_global = self.G(x)                   # [B, C, D, H, W]
        x_local  = self.L(x)                   # [B, C, D, H, W]

        x_G = self.cross_G(x_global, x_local)  # [B, C, D, H, W]
        x_L = self.cross_L(x_local,  x_global) # [B, C, D, H, W]

        feat = self.gap(x_G + x_L).view(x.size(0), -1)   # [B, C]
        feat = self.FC2(self.FC1(feat))                  # [B, C/4]

        # 输入残差（来自原始 x）
        res = self.res_gap(x).view(x.size(0), -1)        # [B, C]
        res = self.res_fc2(self.res_fc1(res))            # [B, C/4]

        return feat + self.alpha_gl * res
