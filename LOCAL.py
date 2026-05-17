import torch
import torch.nn as nn
import torch.nn.functional as F

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)
class LayerNorm3D(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x
class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # GAP to [B, C, 1, 1, 1]
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, d, h, w = x.size()
        y = self.avg_pool(x).view(b, c)              # → [B, C]
        y = self.fc(y).view(b, c, 1, 1, 1)           # → [B, C, 1, 1, 1]
        return x * y.expand_as(x)
class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)          # [B, 1, D, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)        # [B, 1, D, H, W]
        x_cat = torch.cat([avg_out, max_out], dim=1)          # [B, 2, D, H, W]
        attn = self.sigmoid(self.conv(x_cat))                 # [B, 1, D, H, W]
        return x * attn
class EsNet(nn.Module):
    def __init__(self,in_channels,out_channel,kernel_size,padding=1):
        super().__init__()
        self.dwConv=nn.Sequential(
            nn.Conv3d(in_channels, out_channel, kernel_size=kernel_size, padding=padding,groups=in_channels),
            LayerNorm3D(out_channel),
            nn.Conv3d(out_channel, out_channel*2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channel*2, out_channel, kernel_size=1),
        )
        self.channel_attn=ChannelAttention3D(in_channels)
        self.spatial_attn=SpatialAttention3D()
    def forward(self,x):
        CA=self.channel_attn(x)
        SA=self.spatial_attn(x)
        y=CA+SA
        out=self.dwConv(x) + y
        return out


class Local(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        def conv_ln_Gelu(in_channels,out_channels, k):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=k),
                LayerNorm3D(out_channels),
                GELU()
            )

        self.branch1 = conv_ln_Gelu(in_channels, in_channels, 1)

        self.branch2 = nn.Sequential(
            conv_ln_Gelu(in_channels, in_channels, 1),
            EsNet(in_channels,in_channels,3),
        )

        self.branch3 = nn.Sequential(
            conv_ln_Gelu(in_channels, in_channels, 1),
            EsNet(in_channels, in_channels, 5,padding=2),
        )

        self.branch4 = nn.Sequential(
            conv_ln_Gelu(in_channels, in_channels, 1),
            EsNet(in_channels, in_channels, 5, padding=2),
            EsNet(in_channels, in_channels, 3),
        )
        self.branch5 = nn.Sequential(
            conv_ln_Gelu(in_channels, in_channels, 1),
            EsNet(in_channels, in_channels, 5, padding=2),
            EsNet(in_channels, in_channels, 3),
            EsNet(in_channels, in_channels, 3),
        )


    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)

        out4 = self.branch4(x)
        out5 = self.branch5(x)

        out=out1+ out2+ out3+ out4+ out5
        return out+x
