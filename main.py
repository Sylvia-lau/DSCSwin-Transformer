from config import get_config
import numpy as np
from models.swin_transformer import SwinTransformer
from models.cswin import CSWin_96_24322_base_224
from load_data_vis_adamw_mixup import run
from timm.data import Mixup
# from thop import  profile
import torch
from models.cait import cait_s36_384,cait_s24_224,cait_s24_384
from models.mlpmixer import resmlp_36_224
from models.vision_transformer import vit_base_patch16_224
from models.visformer import visformer_small
from models.refiner import Refiner_ViT_L,Refiner_ViT_M,Refiner_ViT_S
from models.pvt_v2 import pvt_v2_b5
from models.densenet_pytorch import densenet201
from models.vgg import vgg16
from collections import OrderedDict
from models.newmodel import DeformSwinT
from models.DSPyconvSwin import DSPyconvSwin
from models.pyconvresnet import PyConvBlock,pyconvresnet50
from models.resnet import resnet152
from models.efficientnet import EfficientNet
from models.deformswinblockformer import Deformswinblockformer
################################################
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
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type=='deformswin':
        model = DeformSwinT(
                            block=PyConvBlock,
                            img_size=config.DATA.IMG_SIZE,
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
                            patch_norm=config.MODEL.SWIN.PATCH_NORM,
                            use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type=='dspyconvswin':
        model = DSPyconvSwin(
                            block=PyConvBlock,
                            layers=[3, 4, 6, 3],
                            img_size=config.DATA.IMG_SIZE,
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
                            patch_norm=config.MODEL.SWIN.PATCH_NORM,
                            use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type=='deformswinblockformer':
        model = Deformswinblockformer(
                            img_size=config.DATA.IMG_SIZE,
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
                            patch_norm=config.MODEL.SWIN.PATCH_NORM,
                            use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    else:
        print(model_type)
    return model

if __name__ == '__main__':
    test_li =[[],[],[],[],[],[],[],[],[],[]]
    ########swinB224#######################
    # log_dir = './swin_removemarkcross_randomcrop_cos_sam_batchsize8_lr0.001_SGD_imgsize224_epoch200_'
    # config = get_config(r"./configs/swin_base_patch4_window7_224.yaml")
    # main(config)
    # model = build_model(config)
    # path = r"./weight/swin_base_patch4_window7_224_22k.pth"

    ##########vision_transformer################
    # model_args = dict(num_classes=6)
    # model = vit_base_patch16_224(True,**model_args)

    ########visformer####################
    # model = visformer_small()
    # path = "./weight/visformer_small_patch16_224.pth"

    ##########CaiT######################
    # model_args = dict(num_classes=4)
    # model = cait_S36(True,**model_args)
    #
    # log_dir = './caits36384_removemarkcross_randomcrop_cos_SGD_batchsize8_lr0.001_imgsize384_epoch200_'

    ##########resmlp####################
    # model = resmlp_24_224(True)
    # log_dir = './resmlp24224_removemarkcross_randomcrop_cos_SGD_batchsize8_lr0.001_epoch200_'

    ##########refiner###################
    # model_args = dict(img_size=384, num_classes=4)
    # model = Refiner_ViT_L(True, **model_args)

    ########pvt2b5######################
    # model_args = dict(num_classes=4)
    # model = pvt_v2_b5(True, **model_args)

    #########densenet201################
    # model = densenet201(num_classes=6, drop_rate=0.5)
    # path = "./densenet201-c1103571.pth"
    # for k, v in params.items() :
    #     # k = k.replace('module.', '')
    #     if 'classifier' not in k :
    #     #     if 'layer1' in k:
    #     #     k = k.replace('layer1', 'pyconvlayer1')
    #         if 'denselayer' in k :
    #             klist = k.split('.')
    #             klist[-3] = klist[-3] + klist[-2]
    #             klist[-2] = klist[-1]
    #             klist.pop()
    #             k = '.'.join(klist)
    #         new_state_dict[k] = v

    #########resnet152##################
    # model_args = dict(num_classes=6)
    # model = resnet152(False, **model_args)
    # path = r'./weight/resnet152-b121ed2d.pth'
    # params = torch.load(path)
    # model_dict = model.state_dict()
    # new_state_dict = OrderedDict()
    # # 修改 key
    # for k, v in params.items() :  # ['state_dict_ema']
    #     if 'fc' not in k :
    #         new_state_dict[k] = v
    ##########efficientnet b7###############
    # model = EfficientNet.from_pretrained("efficientnet-b7", weights_path='./weight/adv-efficientnet-b7-4652b6dd.pth',num_classes=6, advprop=True)

    ##########vgg16#########################
    # model_args = dict(num_classes=6)
    # model = vgg16(False,**model_args)

    #########cswinB224####################
    # model_args = dict(num_classes=6)
    # model = CSWin_96_24322_base_224(True,**model_args)
    # path = './weight/cswin_base_224.pth'

    # f = int(1)
    config = get_config(r"./configs/swin_base_patch4_window7_224.yaml")
    f = 5
    for i in range(f) :
        model = build_model(config)
        path = r"./weight/swin_base_patch4_window7_224_22k.pth"
        model_dict = model.state_dict()
        params = torch.load(path)
        # model_dict = model.state_dict()
        new_state_dict = OrderedDict()
        # # 修改 key
        for k, v in params['model'].items() :
            if 'head' not in k :
                new_state_dict[k.replace('module.', '')] = v
        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of parameters:{}'.format(parameters / 1e6))
        # print('model flops:{}'.format(model.flops() / 1e9))
        # input = torch.randn(1, 3, 224, 224)  # 模型输入的形状,batch_size=1
        # flops, _ = profile(model, inputs=(input,))
        # print('flops:{}'.format(flops/1e9))
        mixup_fn = None
        mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
        if mixup_active :
            mixup_fn = Mixup(
                mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
                prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
                label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
        log_dir = './6-38moreroi_cos100_focalcbcenter_800'
        run(i, './tensorboard/6-38/6-38moreroi_cos100_focalcbcenter_800/', model, 800, 6, log_dir=log_dir, test_li=test_li)

    print(
        'avg: test loss %.4f , test acc max %.4f, test pre micro %.4f, test rec micro %.4f , test f1 micro %.4f ,test ppv %.4f,test pnv %.4f,test tpr %.4f,test tnr %.4f,test fpr %.4f ' % (
            sum(test_li[0]) / f, sum(test_li[1]) / f, sum(test_li[2]) / f, sum(test_li[3]) / f, sum(test_li[4]) / f,
            sum(test_li[5]) / f, sum(test_li[6]) / f, sum(test_li[7]) / f, sum(test_li[8]) / f,sum(test_li[9]) / f))
    print('test loss:', test_li[0])
    print('test acc:', test_li[1])
    print('test pre micro:', test_li[2])
    print('test rec micro:', test_li[3])
    print('test f1 micro:', test_li[4])
    print('test ppv:', test_li[5])
    print('test pnv:', test_li[6])
    print('test tpr:', test_li[7])
    print('test tnr:', test_li[8])
    print('test fpr:', test_li[9])




