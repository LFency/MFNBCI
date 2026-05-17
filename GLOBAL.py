import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self,C, num_heads=2, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads

        self.scale = (C // self.num_heads) ** -0.5
        self.qkv = nn.Linear(C, C * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, C, 3, self.num_heads, N // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x

class GlobalBock(nn.Module):
    def __init__(self,C):
        super().__init__()
        self.attn=Attention(C)

    def forward(self,x):
        x=x.permute(0,2,3,4,1)
        _,C,H,W,_=x.shape
        x = x.contiguous() .view(x.size(0), -1, x.size(-1))
        x=self.attn(x).view(x.size(0),C,H,W, x.size(-1)).permute(0,4,1,2,3)

        return  x
