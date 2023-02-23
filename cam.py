# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:53

@author: mick.yi

入口类

"""
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import re
from pylab import *
import cv2
import numpy as np
import torch
from skimage import io
from torch import nn
from torchvision import models
from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from interpretability.guided_back_propagation import GuidedBackPropagation
from models.swin_transformer import SwinTransformer
import os
from PIL import Image
from collections import OrderedDict
from config import get_config
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model

def get_net(net_name, weight_path=None):
    """
    根据网络名称获取模型
    :param net_name: 网络名称
    :param weight_path: 与训练权重路径
    :return:
    """
    # 加载指定路径的权重参数
    # if weight_path is not None and net_name.startswith('densenet'):
    #     pattern = re.compile(
    #         r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    #     state_dict = torch.load(weight_path)
    #     for key in list(state_dict.keys()):
    #         res = pattern.match(key)
    #         if res:
    #             new_key = res.group(1) + res.group(2)
    #             state_dict[new_key] = state_dict[key]
    #             del state_dict[key]
    #     net.load_state_dict(state_dict)
    #     #net=torch.loadload(state_dict)
    # elif weight_path is not None:
    #     #net.load_state_dict(torch.load(weight_path))
    #     net=torch.load(weight_path,map_location='cpu')
    if net_name in ['swin']:
        config = get_config(r"./configs/swin_base_patch4_window7_224.yaml")
        model = build_model(config)
        # model_dict = model.state_dict()
        # pth = r"./weight/swin_base_patch4_window7_224_22k.pth"
        # params = torch.load(pth)
        # new_state_dict = OrderedDict()
        # # 修改 key
        # for k, v in params['model'].items() :
        #     if 'head' not in k :
        #         new_state_dict[k] = v
        # model_dict.update(new_state_dict)
        # model.load_state_dict(model_dict)
    return model


def get_last_conv_name(net):
    """
    获取网络的最后层的名字
    :param net:
    :return:
    """
    layer_name = net.layers[-1].blocks[-2].norm1
    # for name, m in net.named_modules() :
    #     print(name)
    # for name, m in net.named_modules():
    #     if isinstance(m, nn.Conv2d):
    #         layer_name = name
    # print(layer_name)
    return layer_name


def prepare_input(image):
    image = image.copy()

    # 归一化
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])

    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = image[np.newaxis, ...]  # 增加batch维

    return torch.tensor(image, requires_grad=True)


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam)  #, (heatmap * 255).astype(np.uint8)


def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dict, input_image_name, network, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    print(torch.from_numpy(image_dict['img']).shape,image_dict['cam'].shape)
    img_save = torch.cat((torch.from_numpy(image_dict['img']),torch.from_numpy(image_dict['cam']).float(),torch.from_numpy(image_dict['cam++']).float()),1)
    io.imsave(os.path.join(output_dir, '{}-{}.jpg'.format(prefix, network)), img_save)


def main(args):
    # 输入
    img = io.imread(args.image_path)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    image_dict = {'img' : img*255}
    inputs = prepare_input(img)
    # print(inputs)
    # 输出图像
    # 网络
    net = get_net(args.network, args.weight_path)
    # print(net)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # inputs=inputs.to(device)
    # net = net.cpu()
    # Grad-CAM
    layer_name = get_last_conv_name(net) if args.layer_name is None else args.layer_name
    grad_cam = GradCAM(net, layer_name)
    # inputs1=inputs.type(torch.cuda.FloatTensor)##################################
    mask = grad_cam(inputs, args.class_id,args.correct)  # cam mask#########
    if any(mask):
        image_dict['cam']= gen_cam(img, mask)
        grad_cam.remove_handlers()
        # Grad-CAM++
        grad_cam_plus_plus = GradCamPlusPlus(net, layer_name)
        mask_plus_plus = grad_cam_plus_plus(inputs, args.class_id)  # cam mask#######
        image_dict['cam++'] =  gen_cam(img, mask_plus_plus)
        grad_cam_plus_plus.remove_handlers()

        # GuidedBackPropagation
        gbp = GuidedBackPropagation(net)
        print(type(inputs))
        inputs.grad.zero_()  # 梯度置零#################################################
        # inputs.grad.zero_()  # 梯度置零#################################################
        # inputs2=inputs.type(torch.cuda.FloatTensor)##################################
        grad = gbp(inputs)  #######

        gb = gen_gb(grad)
        image_dict['gb'] = norm_image(gb)
        # 生成Guided Grad-CAM
        cam_gb = gb * mask[..., np.newaxis]
        image_dict['cam_gb'] = norm_image(cam_gb)

        save_image(image_dict, os.path.basename(args.image_path), args.network, args.output_dir)

if __name__ == '__main__':

    # for root, _, files in os.walk(r'D:\data\neck_region_pre\2'):
    #     for file in files:
    #         img = Image.open(os.path.join(root, file))
    #         # img.save(os.path.join(root, file))
    #         parser = argparse.ArgumentParser()
    #         parser.add_argument('--image-path', type=str,
    #                             default=os.path.join(root, file),
    #                             help='input image path')
    #         parser.add_argument('--network', type=str, default='swin',
    #                             help='ImageNet classification network')
    #         parser.add_argument('--weight-path', type=str, default=r'D:\project\neck region transformer\result\2021.6.2\swin_removemarkcross_randomcrop_batchsize8_lr0.001_SGD_imgsize224_epoch200_5fold.pth',
    #                             help='weight path of the model')
    #         parser.add_argument('--layer-name', type=str, default=None,
    #                             help='last convolutional layer name')
    #         parser.add_argument('--class-id', type=int, default=0,
    #                             help='class id')
    #         parser.add_argument('--output-dir', type=str, default=r'D:\project\neck region transformer\result\2021.6.2\cam_swin_removemarkcross_randomcrop\2',
    #                             help='output directory to save results')
    #         parser.add_argument('--correct', type=bool,
    #                             default=True,
    #                             help='getTrueorFalse')
    #         arguments = parser.parse_args()
    #         main(arguments)
    from PIL import Image
    path = r'D:\data\淋巴结数据\2021.8.3(23)\分区\5\21094843_查鸿坤\查鸿坤_40522244.jpg'
    src = Image.open(path)
    src = np.asarray(src)
    print(src.shape)
    import pydicom
    path = r'D:\data\淋巴结数据\2021.8.3(23)\分区\5\21094843_查鸿坤\查鸿坤_40522242.dcm'
    dcm = pydicom.dcmread(path)
    src = dcm.pixel_array
    print(src.shape)