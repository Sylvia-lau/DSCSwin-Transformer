# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
# from models.layer.deform_conv import DeformConv2d


class DWT(nn.Module) :
    def __init__(self) :
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def dwt_init(self,x,mode='even') :
        device = torch.device('cpu')
        if mode == 'odd':
            p = torch.zeros((x.shape[0],x.shape[1],x.shape[3])).to(device)
            p = p.unsqueeze(2)
            x = torch.cat((x, p), 2)
            p = torch.zeros((x.shape[0],x.shape[1],x.shape[2])).to(device)
            p = p.unsqueeze(3)
            x = torch.cat((x, p), 3)
        x01 = x[:, :, 0 : :2, :] / 2
        x02 = x[:, :, 1 : :2, :] / 2
        # print(x.size(),x01.size(),x02.size())
        x1 = x01[:, :, :, 0 : :2]
        x2 = x02[:, :, :, 0 : :2]
        x3 = x01[:, :, :, 1 : :2]
        x4 = x02[:, :, :, 1 : :2]
        # print(x1.size(), x2.size(), x3.size(),x4.size())
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

    def forward(self, x,mode) :
        return self.dwt_init(x,mode)

class IWT(nn.Module) :
        def __init__(self) :
            super(IWT, self).__init__()
            self.requires_grad = False

        def iwt_init(self,x,mode) :
            r = 2
            in_batch, in_channel, in_height, in_width = x.size()
            # print([in_batch, in_channel, in_height, in_width])
            out_batch, out_channel, out_height, out_width = in_batch, int(
                in_channel / (r ** 2)), r * in_height, r * in_width
            x1 = x[:, 0 :out_channel, :, :] / 2
            x2 = x[:, out_channel :out_channel * 2, :, :] / 2
            x3 = x[:, out_channel * 2 :out_channel * 3, :, :] / 2
            x4 = x[:, out_channel * 3 :out_channel * 4, :, :] / 2
            h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cpu()
            h[:, :, 0 : :2, 0 : :2] = x1 - x2 - x3 + x4
            h[:, :, 1 : :2, 0 : :2] = x1 - x2 + x3 - x4
            h[:, :, 0 : :2, 1 : :2] = x1 + x2 - x3 - x4
            h[:, :, 1 : :2, 1 : :2] = x1 + x2 + x3 + x4
            if mode == 'odd' :
                h = h[:,:,:-1,:-1]
            return h

        def forward(self, x,mode='even') :
            return self.iwt_init(x,mode)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

class Block(nn.Module) :
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, kernelsize, dim, drop_path=0., layer_scale_init_value=1e-6) :
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.dwt = DWT()
        self.pwconv1 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(dim)
        self.act1 = nn.LeakyReLU()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.act2 = nn.LeakyReLU()
        self.pwconv2 = nn.Conv2d(dim, 4*dim,kernel_size=1)
        self.iwt = IWT()
        self.norm3 = nn.BatchNorm2d(dim)
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
        #                           requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x) :
        if x.size(-1)%2!=0:
            mode = 'odd'
        else:
            mode= 'even'
        input = x
        x = self.norm1(x)
        x = self.dwt(x, mode)
        x = self.pwconv1(x)
        x = self.norm2(x)
        x = self.act1(x)
        x = self.dwconv(x)
        # x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.act2(x)
        x = self.pwconv2(x)
        x = self.iwt(x,mode)
        x = self.norm3(x)
        x = input + self.drop_path(x)

        return x


class DwtConvNet(nn.Module) :
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ) :
        super().__init__()
        kernelsizes = [15,11,7,3]
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3) :
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4) :
            stage = nn.Sequential(
                *[Block(kernelsize=kernelsizes[i], dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m) :
        # if isinstance(m, (nn.Conv2d, nn.Linear)) :
        #     trunc_normal_(m.weight, std=.02)
        #     nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear) :
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None :
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)) :
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d) :
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def forward_features(self, x) :
        for i in range(4) :
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x) :
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module) :
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last") :
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"] :
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x) :
        if self.data_format == "channels_last" :
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first" :
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k" : "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k" : "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k" : "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k" : "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_base_22k" : "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k" : "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k" : "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def dwtconvnet_tt(pretrained=False, **kwargs) :
    model = DwtConvNet(depths=[1, 1, 1, 1], dims=[18,36,72,144],**kwargs)#96 828 192*3   576
    # if pretrained :
    #     url = model_urls['convnext_tiny_1k']
    #     checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
    #     model.load_state_dict(checkpoint["model"])
    return model


@register_model
def dwtconvnet_tiny(pretrained=False, **kwargs) :
    model = DwtConvNet(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained :
        url = model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def dwtconvnet_small(pretrained=False, **kwargs) :
    model = DwtConvNet(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained :
        url = model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def dwtconvnet_base(pretrained=False, in_22k=False, **kwargs) :
    model = DwtConvNet(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained :
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

