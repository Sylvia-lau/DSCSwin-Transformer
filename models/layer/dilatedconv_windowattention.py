import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class DilatedconvReduction(nn.Module):
    '''


    '''
    def __init__(self, stage , dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        li = [[1,6,12,18],[1,4,8,12],[1,2,4,6],[1,2,3,4]]
        self.d1 = nn.Conv2d(dim, dim , kernel_size=3, stride=2, padding=0, bias=True, dilation=1,groups=dim)
        self.d2 = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=0, bias=True, dilation=2,groups=dim)
        self.d3 = nn.Conv2d(dim, dim , kernel_size=3, stride=2, padding=1, bias=True, dilation=3,groups=dim)
        self.d4 = nn.Conv2d(dim, dim , kernel_size=3, stride=2, padding=2, bias=True, dilation=4,groups=dim)
        # if stage==3:
        #     self.d1 = nn.Conv2d(dim, dim , kernel_size=3, stride=2, padding=0, bias=True, dilation=li[stage][0],groups=dim)
        #     self.d2 = nn.Conv2d(dim, dim , kernel_size=3, stride=2, padding=0, bias=True, dilation=li[stage][1],groups=dim)
        #     self.d3 = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=True, dilation=li[stage][2],groups=dim)
        #     self.d4 = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=2, bias=True, dilation=li[stage][3],groups=dim)
        self.p1 = nn.Conv2d(dim, dim*2, kernel_size=1, stride=1, padding=0)
        self.p2 = nn.Conv2d(dim, dim*2, kernel_size=1, stride=1, padding=0)
        self.p3 = nn.Conv2d(dim, dim*2, kernel_size=1, stride=1, padding=0)
        self.p4 = nn.Conv2d(dim, dim*2, kernel_size=1, stride=1, padding=0)

    def forward(self, x) :
        B,N,C = x.shape
        H,W = int(N ** 0.5),int(N ** 0.5)
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        d1 = self.d1(x_)
        p1 = self.p1(d1).reshape(B,C,-1).permute(0, 2, 1)
        p1 = p1.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        d2 = self.d2(x_)
        p2 = self.p2(d2).reshape(B, C, -1).permute(0, 2, 1)
        p2 = p2.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        d3 = self.d3(x_)
        p3 = self.p3(d3).reshape(B, C, -1).permute(0, 2, 1)
        p3 = p3.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        d4 = self.d4(x_)
        p4 = self.p4(d4).reshape(B, C, -1).permute(0, 2, 1)
        p4 = p4.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = torch.cat((p1[0],p2[0],p3[0],p4[0]),2)
        v = torch.cat((p1[1],p2[1],p3[1],p4[1]),2)
        return k,v
    #
    def flops(self, H, W):
        flops = 0
        for i in range(4):
            flops += (2*self.dim*3*3-1)*H*W*self.dim
        return flops

class Dilatedconv_WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, stage, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = DilatedconvReduction(stage, dim, num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        q = self.q(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k,v = self.kv(x)
         # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn #+ relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N ,input_resolution):
        # calculate flops for 1 window with token length of N
        # q = self.q(x)
        flops = N * self.dim
        # k,v = self.kv(x)
        flops += N*self.kv.flops(input_resolution[0],input_resolution[1])
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops