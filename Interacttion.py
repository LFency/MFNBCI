import torch
import torch.nn as nn
import torch.nn.functional as F
from swin2 import Swin3DClassifier

class Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.1
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x

class InteractionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mid_c=[]
    def forward(self, x):
        # x: [B, 2, D]  (D = 160*192*160)
        B, C, D = x.size()
        gram = torch.bmm(x, x.transpose(1, 2))  # [B, 2, 2]
        attn = F.softmax(gram, dim=-1)
        out = torch.bmm(attn, x)                # [B, 2, D]
        return out

class SupervisionModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.MRI_PET1 = nn.Linear(num_features, 2)
        self.MRI_PET2 = nn.Linear(2, 1)
        self.PET_MRI1 = nn.Linear(num_features, 2)
        self.PET_MRI2 = nn.Linear(2, 1)
        self.mid_c=[]

    def forward(self, c):
        c_MRI, c_PET = torch.unbind(c, dim=1)       # [B, D], [B, D]
        MRI_weight = self.MRI_PET2(self.MRI_PET1(c_MRI))  # [B, 1]
        PET_weight = self.PET_MRI2(self.PET_MRI1(c_PET))  # [B, 1]

        c_PET = torch.mul(c_PET, MRI_weight)        # [B, D]
        c_MRI = torch.mul(c_MRI, PET_weight)        # [B, D]

        supervision_Matrix = torch.stack([c_MRI, c_PET], dim=1)  # [B, 2, D]
        end = torch.cat([c, supervision_Matrix], dim=1)          # [B, 4, D]
        return end

class InteractionNet(nn.Module):
    def __init__(self, size, num_features, norm_layer=nn.LayerNorm):
        super().__init__()
        self.flatten = True
        self.size = size  # [160,192,160]
        dim = num_features

        self.supervisionModel = SupervisionModel(num_features)
        self.interactionModel = InteractionModel()
        self.model = Swin3DClassifier(
            in_channels=4,
            num_classes=2,
            embed_dim=96,
            patch_size=(8,8,8),
            input_shape=(160,192,160)
        )

        # === 旁路分支：从原始 ding_1+PET 直接到 logits，强力避免塌缩 ===
        self.bypass_pool = nn.AdaptiveAvgPool3d(1)
        self.bypass_head = nn.Sequential(
            nn.Flatten(),          # [B, 2,1,1,1] -> [B, 2]
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2)
        )
        self.beta_bypass = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x: [B, 2, D, H, W]
        x_raw = x  # 保存用于旁路

        # 旁路 logits（不会参与梯度截断，和主干一起训练）
        bypass_logits = self.bypass_head(self.bypass_pool(x_raw))  # [B, 2]

        if self.flatten:
            x = x.flatten(2)       # [B, 2, D*H*W]

        c = self.interactionModel(x)      # [B, 2, D*H*W]
        c = self.supervisionModel(c)      # [B, 4, D*H*W]
        c = c.reshape(c.shape[0], c.shape[1], self.size[0], self.size[1], self.size[2])  # [B,4,160,192,160]
        main_logits = self.model(c)       # [B, 2]

        return main_logits + self.beta_bypass * bypass_logits
