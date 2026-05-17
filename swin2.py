import torch
import torch.nn as nn
from einops import rearrange
from G_L import G_L_Cross
import torch.nn.functional as F

# ---------------- Patch Embedding (with residual from raw input) ----------------
class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels, embed_dim=96, patch_size=(8,8,8)):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 残差适配：等比例下采样 + 1x1x1 映射
        self.pool = nn.AvgPool3d(kernel_size=patch_size, stride=patch_size)
        self.res_conv = nn.Conv3d(in_channels, embed_dim, kernel_size=1)
        self.alpha_pe = nn.Parameter(torch.tensor(1.0))  # 可学习门控

    def forward(self, x):
        y = self.proj(x)                    # [B, C, D', H', W']
        r = self.res_conv(self.pool(x))     # [B, C, D', H', W']
        return y + self.alpha_pe * r        # 输入级残差

# ---------------- Window Attention ----------------
class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size=(4,4,4), heads=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.window_size = window_size
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)

# ---------------- Swin Block (fixed: add attention residual) ----------------
class SwinBlock3D(nn.Module):
    def __init__(self, dim, input_resolution, window_size=(4,4,4), heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.input_resolution = input_resolution

    def forward(self, x):
        # x: [B, D, H, W, C]
        B, D, H, W, C = x.shape
        wd, wh, ww = self.window_size

        pad_d = (wd - D % wd) % wd
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww
        x_pad = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))  # [B, Dp, Hp, Wp, C]
        Dp, Hp, Wp = x_pad.shape[1], x_pad.shape[2], x_pad.shape[3]

        tokens = x_pad.view(B, Dp // wd, wd, Hp // wh, wh, Wp // ww, ww, C)
        tokens = rearrange(tokens, 'b d1 wd h1 wh w1 ww c -> (b d1 h1 w1) (wd wh ww) c')

        # Attention 残差
        tokens_res = tokens
        attn_out = self.attn(self.norm1(tokens))
        tokens = tokens_res + attn_out

        # 还原
        x_attn = rearrange(
            tokens,
            '(b d1 h1 w1) (wd wh ww) c -> b (d1 wd) (h1 wh) (w1 ww) c',
            b=B, d1=Dp // wd, h1=Hp // wh, w1=Wp // ww, wd=wd, wh=wh, ww=ww
        )
        x_attn = x_attn[:, :D, :H, :W, :]

        # MLP 残差
        x = x_attn + self.mlp(self.norm2(x_attn))
        return x

# ---------------- Patch Merging ----------------
class PatchMerging3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Conv3d(dim, dim * 2, kernel_size=2, stride=2)
    def forward(self, x):
        return self.reduction(x)  # [B, C*2, D/2, H/2, W/2]

# ---------------- Swin3D Classifier with Residual Enhancements ----------------
class Swin3DClassifier(nn.Module):
    def __init__(
        self,
        in_channels=4,
        num_classes=2,
        embed_dim=96,
        patch_size=(8,8,8),
        window_size=(4,4,4),
        input_shape=(160,192,160),
        dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding3D(in_channels, embed_dim, patch_size)

        d, h, w = [s // p for s, p in zip(input_shape, patch_size)]

        # Stage 1
        self.stage1 = nn.Sequential(
            SwinBlock3D(embed_dim, (d, h, w), window_size),
            SwinBlock3D(embed_dim, (d, h, w), window_size)
        )
        self.G_L_C_1 = G_L_Cross(embed_dim)
        self.merge1 = PatchMerging3D(embed_dim)

        # Stage 2
        self.stage2 = nn.Sequential(
            SwinBlock3D(embed_dim*2, (d//2, h//2, w//2), window_size),
            SwinBlock3D(embed_dim*2, (d//2, h//2, w//2), window_size)
        )
        self.G_L_C_2 = G_L_Cross(embed_dim*2)
        self.merge2 = PatchMerging3D(embed_dim*2)

        # Stage 3
        self.stage3 = nn.Sequential(
            SwinBlock3D(embed_dim*4, (d//4, h//4, w//4), window_size),
            SwinBlock3D(embed_dim*4, (d//4, h//4, w//4), window_size)
        )
        self.G_L_C_3 = G_L_Cross(embed_dim*4)
        self.merge3 = PatchMerging3D(embed_dim*4)

        # Stage 4
        self.stage4 = nn.Sequential(
            SwinBlock3D(embed_dim*8, (d//8, h//8, w//8), window_size),
            SwinBlock3D(embed_dim*8, (d//8, h//8, w//8), window_size)
        )
        self.G_L_C_4 = G_L_Cross(embed_dim*8)

        self.norm = nn.LayerNorm(embed_dim*8)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(360, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        # === Stage 残差适配器 + 门控（注册成层，避免 forward 临时创建） ===
        self.adapt1 = nn.Conv3d(embed_dim,   embed_dim*2, kernel_size=1)
        self.adapt2 = nn.Conv3d(embed_dim,   embed_dim*4, kernel_size=1)
        self.adapt3 = nn.Conv3d(embed_dim,   embed_dim*8, kernel_size=1)
        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))
        self.alpha3 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # 初始 patch embedding
        x_patch = self.patch_embed(x)              # [B, C, D', H', W']
        x = x_patch.permute(0, 2, 3, 4, 1)         # [B, D', H', W', C]

        # -------- Stage 1 --------
        x = self.stage1(x)
        x1_perm = x.permute(0, 4, 1, 2, 3)         # [B, C, D', H', W']
        x_1 = self.G_L_C_1(x1_perm)

        y1 = self.merge1(x1_perm)                  # [B, 2C, D/2, H/2, W/2]
        r1 = F.interpolate(x_patch, size=y1.shape[2:], mode='trilinear', align_corners=False)
        r1 = self.adapt1(r1)
        x = y1 + self.alpha1 * r1

        # -------- Stage 2 --------
        x = x.permute(0, 2, 3, 4, 1)
        x = self.stage2(x)
        x2_perm = x.permute(0, 4, 1, 2, 3)
        x_2 = self.G_L_C_2(x2_perm)

        y2 = self.merge2(x2_perm)                  # [B, 4C, D/4, H/4, W/4]
        r2 = F.interpolate(x_patch, size=y2.shape[2:], mode='trilinear', align_corners=False)
        r2 = self.adapt2(r2)
        x = y2 + self.alpha2 * r2

        # -------- Stage 3 --------
        x = x.permute(0, 2, 3, 4, 1)
        x = self.stage3(x)
        x3_perm = x.permute(0, 4, 1, 2, 3)
        x_3 = self.G_L_C_3(x3_perm)

        y3 = self.merge3(x3_perm)                  # [B, 8C, D/8, H/8, W/8]
        r3 = F.interpolate(x_patch, size=y3.shape[2:], mode='trilinear', align_corners=False)
        r3 = self.adapt3(r3)
        x = y3 + self.alpha3 * r3

        # -------- Stage 4 --------
        x = x.permute(0, 2, 3, 4, 1)
        x = self.stage4(x)
        x = self.norm(x)
        x4_perm = x.permute(0, 4, 1, 2, 3)
        x_4 = self.G_L_C_4(x4_perm)

        # 最终融合
        out = torch.cat([x_1, x_2, x_3, x_4], dim=1)  # [B, 360]
        out = self.dropout(out)
        return self.head(out)

