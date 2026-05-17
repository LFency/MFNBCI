import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import option
from ptflops import get_model_complexity_info


class DilatedConv3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=2):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=1, padding=padding, dilation=dilation, bias=False,groups=in_ch)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))




def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# class PatchEmbed(nn.Module):
#     def __init__(self, input_shape=[160 , 192 , 160], patch_size=32, in_chans=1, num_features=4096, norm_layer=None,
#                  flatten=True):
#         super().__init__()
#         self.num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size) * (input_shape[2] // patch_size)
#         self.flatten = flatten
#
#         self.proj = nn.Conv3d(in_chans, num_features, kernel_size=[patch_size,patch_size,patch_size], stride=patch_size)
#         self.norm = norm_layer(num_features) if norm_layer else nn.Identity()
#
#     def forward(self, x):
#         # x=x.reshape(x.shape[0],4,160,192,160)
#         x = self.proj(x)
#
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#         x = self.norm(x)
#
#         return x

class PatchEmbed(nn.Module):
    def __init__(self, input_shape=[160, 192, 160], patch_size=32, in_chans=1,
                 num_features=4096, norm_layer=None, flatten=True):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        # 计算每个维度的 patch 数
        self.num_patches = (
            (input_shape[0] + patch_size - 1) // patch_size *
            (input_shape[1] + patch_size - 1) // patch_size *
            (input_shape[2] + patch_size - 1) // patch_size
        )

        # 使用 stride=patch_size 的 3D 卷积实现 patch embedding
        self.proj = nn.Conv3d(
            in_chans,
            num_features,
            kernel_size=[patch_size, patch_size, patch_size],
            stride=patch_size
        )
        self.norm = norm_layer(num_features) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, D, H, W = x.shape
        p = self.patch_size

        # 计算每个维度需要的 padding
        pad_d = (p - D % p) % p
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            # F.pad 的 padding 顺序是从最后一维开始： (W_left, W_right, H_left, H_right, D_left, D_right)
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        # 卷积提取 patch
        x = self.proj(x)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # [B, C, D', H', W'] -> [B, N, C]
        x = self.norm(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DenseAtten(nn.Module):
    def __init__(self,channel,dim,num_heads, qkv_bias=False, drop=0., attn_drop=0.,norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn_0 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.Dilaed_1=DilatedConv3DBlock(channel,channel)
        self.attn_1 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.Dilaed_2 = DilatedConv3DBlock(channel, channel)
        self.attn_2 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.Dilaed_3 = DilatedConv3DBlock(channel, channel)
        self.attn_3 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.Dilaed_4 = DilatedConv3DBlock(channel, channel)
        self.norm1 = norm_layer(dim)
    def forward(self,x):
        x_0 = self.attn_0(self.norm1(x))
        x_0=x_0.reshape(x_0.shape[0], x_0.shape[1], 16, 16, 16)
        x_0=self.Dilaed_1(x_0)
        x_0=x_0.reshape(x_0.shape[0], x_0.shape[1],4096)

        x_1 = self.attn_1(self.norm1(x_0))
        x_1 = x_1.reshape(x_1.shape[0], x_1.shape[1], 16, 16, 16)
        x_1 = self.Dilaed_2(x_1)
        x_1 = x_1.reshape(x_1.shape[0], x_1.shape[1], 4096)

        x_2 = x_1+x_0
        x_2 = self.attn_2(self.norm1(x_2))
        x_2 = x_2.reshape(x_2.shape[0], x_2.shape[1], 16, 16, 16)
        x_2 = self.Dilaed_3(x_2)
        x_2 = x_2.reshape(x_2.shape[0], x_2.shape[1], 4096)

        x_3 = x_2 + x_1 + x_0
        x_3 = self.attn_3(self.norm1(x_3))
        x_3 = x_3.reshape(x_3.shape[0], x_3.shape[1], 16, 16, 16)
        x_3 = self.Dilaed_4(x_3)
        x_3 = x_3.reshape(x_3.shape[0], x_3.shape[1], 4096)

        x_3= x_3 + x_2 + x_1 + x_0
        return x_3


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(self, channel,dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dense=DenseAtten(channel,dim,num_heads)

    def forward(self, x):
        # x = x + self.drop_path(self.attn(self.norm1(x)))

        x=x + self.drop_path(self.dense(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(
            self, input_shape=[160,192, 160], patch_size=16, in_chans=1, num_classes=2, num_features=4096,
            depth=1, num_heads=8, mlp_ratio=1.2, qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU
    ):
        super().__init__()
        # -----------------------------------------------#
        #   224, 224, 3 -> 196, 768
        # -----------------------------------------------#
        self.patch_embed = PatchEmbed(input_shape=input_shape, patch_size=patch_size, in_chans=in_chans,
                                      num_features=num_features)
        self.num_patches =(
            int((160-1) // patch_size+1) *
            int((192-1) // patch_size+1) *
            int((160-1) // patch_size+1)
        )
        self.num_features = num_features
        self.new_feature_shape = [int((160-1) // patch_size+1), int((192-1) // patch_size+1), int((160-1) // patch_size+1)]
        self.old_feature_shape = [int((160-1) // patch_size+1), int((192-1) // patch_size+1), int((160-1) // patch_size+1)]

        # --------------------------------------------------------------------------------------------------------------------#
        #   classtoken部分是transformer的分类特征。用于堆叠到序列化后的图片特征中，作为一个单位的序列特征进行特征提取。
        #
        #   在利用步长为16x16的卷积将输入图片划分成14x14的部分后，将14x14部分的特征平铺，一幅图片会存在序列长度为196的特征。
        #   此时生成一个classtoken，将classtoken堆叠到序列长度为196的特征上，获得一个序列长度为197的特征。
        #   在特征提取的过程中，classtoken会与图片特征进行特征的交互。最终分类时，我们取出classtoken的特征，利用全连接分类。
        # --------------------------------------------------------------------------------------------------------------------#
        #   196, 768 -> 197, 768
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_features))
        # --------------------------------------------------------------------------------------------------------------------#
        #   为网络提取到的特征添加上位置信息。
        #   以输入图片为224, 224, 3为例，我们获得的序列化后的图片特征为196, 768。加上classtoken后就是197, 768
        #   此时生成的pos_Embedding的shape也为197, 768，代表每一个特征的位置信息。
        # --------------------------------------------------------------------------------------------------------------------#
        #   197, 768 -> 197, 768
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, num_features))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.channel=self.old_feature_shape[0]*self.old_feature_shape[1]*self.old_feature_shape[2]+1
        # -----------------------------------------------#
        #   197, 768 -> 197, 768  12次
        # -----------------------------------------------#
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    channel=self.channel,
                    dim=num_features,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer
                ) for i in range(depth)
            ]
        )
        self.norm = norm_layer(num_features)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):

        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        cls_token_pe = self.pos_embed[:, 0:1, :]
        img_token_pe = self.pos_embed[:, 1:, :]

        print(img_token_pe.shape)
        img_token_pe = img_token_pe.view(1, *self.old_feature_shape, -1)
        print(img_token_pe.shape)
        img_token_pe = img_token_pe.permute(0, 4 , 1 , 2 , 3)
        # img_token_pe=img_token_pe.unsqueeze(0)
        img_token_pe = F.interpolate(img_token_pe, size=self.new_feature_shape, mode='trilinear', align_corners=False)
        img_token_pe = img_token_pe.permute(0, 2, 3, 4, 1).flatten(1, 3)

        pos_embed = torch.cat([cls_token_pe, img_token_pe], dim=1)

        x = self.pos_drop(x + pos_embed)
        # print(x.shape)
        x = self.blocks(x)
        x = self.norm(x)
        # print(x.shape)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        # print(x.shape)
        x = self.head(x)
        # print(x.shape)
        return x

    def freeze_backbone(self):
        backbone = [self.patch_embed, self.cls_token, self.pos_embed, self.pos_drop, self.blocks[:8]]
        for module in backbone:
            try:
                for param in module.parameters():
                    param.requires_grad = False
            except:
                module.requires_grad = False

    def Unfreeze_backbone(self):
        backbone = [self.patch_embed, self.cls_token, self.pos_embed, self.pos_drop, self.blocks[:8]]
        for module in backbone:
            try:
                for param in module.parameters():
                    param.requires_grad = True
            except:
                module.requires_grad = True


