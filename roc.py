from collections import OrderedDict

import torch
import torch.nn as nn
import os
import numpy as np
from torchvision.datasets import ImageFolder
from config import get_config
# from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
import torchvision.transforms as transforms
from sklearn.metrics import cohen_kappa_score
from models.swin_transformer_deformconv import SwinTransformer
from models.vision_transformer import vit_base_patch16_224,vit_base_patch16_224_in21k,VisionTransformer,vit_small_patch16_224
from models.visformer import visformer_small
from models.resnet import resnet50
# from models.cswin import CSWin_96_24322_base_224
from models.densenet_pytorch import densenet201
# from models.efficientnet import EfficientNet
from models.deformswinblockformer import Deformswinblockformer
# from models.mlp_mixer import resmlp_36_224,mixer_b16_224


def roc(model, test_path) :
    # 加载测试集和预训练模型参数
    # test_dir = os.path.join(data_root, 'test_images')
    # class_list = list(os.listdir(test_dir))
    # class_list.sort()
    transform_test = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        # Normalize a tensor image with mean and standard variance
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_dataset = ImageFolder(test_path, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True, num_workers=1)

    score_list = []  # 存储预测得分
    label_list = []  # 存储真实标签
    pred_list = []
    for i, (inputs, labels) in enumerate(test_loader) :

        outputs = model(inputs)
        # prob_tmp = torch.nn.Softmax(dim=1)(outputs) # (batchsize, nclass)
        # _,score_tmp = outputs  # (batchsize, nclass)
        _, predicted = outputs.max(1)

        score_list.extend(outputs[:,1].detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())
        pred_list.extend(predicted.cpu().numpy())

    score_array = np.array(score_list)
    label_array = np.array(label_list)
    pred_array = np.array(pred_list)
    # # 将label转换成onehot形式
    # label_tensor = torch.tensor(label_list)
    # label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    # label_onehot = torch.zeros(label_tensor.shape[0], 1)
    # label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    # label_onehot = np.array(label_onehot)

    print("score_array:", score_array.shape)  # (batchsize, classnum)
    print("label_onehot:", label_array.shape)  # torch.Size([batchsize, classnum])

    # 调用sklearn库，计算每个类别对应的fpr和tpr

    fpr_dict, tpr_dict, _ = roc_curve(label_array, score_array)
    roc_auc_dict = auc(fpr_dict, tpr_dict)
    kappa = cohen_kappa_score(pred_array, label_array)
    print('auc:',roc_auc_dict)
    print('kappa:',kappa)
    # # micro
    # fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    # roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])
    #
    # # macro
    # # First aggregate all false positive rates
    # all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
    # # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(num_class) :
    #     mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # # Finally average it and compute AUC
    #
    # fpr_dict["macro"] = all_fpr
    # tpr_dict["macro"] = mean_tpr
    # roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

    return fpr_dict,tpr_dict,roc_auc_dict

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

def  roc_models(fprlist,tprlist,auclist):
    # 绘制所有类别平均的roc曲线
    plt.figure()
    lw = 2

    #########resnet152###################
    plt.plot(fprlist[0], tprlist[0],
             label='Resnet50 = {0:0.2f}'
                   ''.format(auclist[0]),
             color='aqua')
    ##########densnenet201#############
    plt.plot(fprlist[1], tprlist[1],
             label='Densenet201 = {0:0.2f}'
                   ''.format(auclist[1]),
             color='cornflowerblue')
    ########Inceptionv3####################
    plt.plot(fprlist[2], tprlist[2],
             label='Inceptionv3 = {0:0.2f}'
                   ''.format(auclist[2]),
             color='deeppink')
    ##########seresnet50#################
    plt.plot(fprlist[3], tprlist[3],
             label=' SEResnet50 = {0:0.2f}'
                   ''.format(auclist[3]),
             color='darkorange')
    #########convnext_tiny###################
    plt.plot(fprlist[4], tprlist[4],
             label='Convnext_Tiny = {0:0.2f}'
                   ''.format(auclist[4]),
             color='coral')

    ##########visformer_small#################
    plt.plot(fprlist[5], tprlist[5],
             label='ViT_Small = {0:0.2f}'
                   ''.format(auclist[5]),
             color='beige')

    #########swinT#############
    plt.plot(fprlist[6], tprlist[6],
             label='Swin_Tiny = {0:0.2f}'
                   ''.format(auclist[6]),
             color='forestgreen')
    ########dwtconvnetT####################
    plt.plot(fprlist[7], tprlist[7],
             label='Dwtconvnet = {0:0.2f}'
                   ''.format(auclist[7]),
             color='darkred')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    name = 'roc2.jpg'
    plt.savefig(name)
    plt.show()

if __name__ == '__main__' :
    data_root = r'D:\project\yolov5\res\two stage lymph\val'  # 测试集路径
    # test_weights_path = r"C:\Users\admin\Desktop\fsdownload\epoch_0278_top1_70.565_'checkpoint.pth.tar'"  # 预训练模型参数
    num_class = 2  # 类别数量
##############resnet152######################################
    model_args = dict(num_classes=2)
    model = resnet50(False, **model_args)
    path = r'D:\project\yolov5\res\trueroi lymph\resnet50_cutmix_focalcb_trueroiresize224_500_1.pth'
    model_dict = model.state_dict()
    params = torch.load(path, map_location='cpu')
    new_state_dict = OrderedDict()
    # 修改 key
    for k, v in params['model'].items() :
        k = k.replace('module.', '')
        new_state_dict[k] = v
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    model.eval()
    resnet50_fpr,resnet50_tpr,resnet50_roc_auc_dict = roc(model,data_root)

#####################densenet201###############################
    model = densenet201(num_classes=2)
    path = r'D:\project\yolov5\res\trueroi lymph\densenet201_cutmix_focalcb_trueroiresize224_500_1.pth'
    model_dict = model.state_dict()
    params = torch.load(path, map_location='cpu')
    new_state_dict = OrderedDict()
    # 修改 key
    for k, v in params['model'].items() :
        k = k.replace('module.', '')
        new_state_dict[k] = v
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    model.eval()
    densenet201_fpr, densenet201_tpr, densenet201_roc_auc_dict = roc(model, data_root)


#####################inceptionv3###############################
    from models.inceptionv3 import inception_v3
    model_args = dict(num_classes=2)
    model = inception_v3(False, **model_args)
    path = r'D:\project\yolov5\res\trueroi lymph\inceptionv3_cutmix_focalcb_trueroiresize224_500_1.pth'
    model_dict = model.state_dict()
    params = torch.load(path, map_location='cpu')
    new_state_dict = OrderedDict()
    # 修改 key
    for k, v in params['model'].items() :
        k = k.replace('module.', '')
        new_state_dict[k] = v
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    model.eval()
    inceptionv3_fpr, inceptionv3_tpr, inceptionv3_roc_auc_dict = roc(model, data_root)

#####################mSEresnet50###############################
    from models.resnet import seresnet50
    model_args = dict(num_classes=2)
    model = seresnet50(False, **model_args)
    path = r'D:\project\yolov5\res\trueroi lymph\seresnet50_cutmix_focalcb_trueroiresize224_500_1.pth'
    model_dict = model.state_dict()
    params = torch.load(path, map_location='cpu')
    new_state_dict = OrderedDict()
    # 修改 key
    for k, v in params['model'].items() :
        k = k.replace('module.', '')
        new_state_dict[k] = v
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    model.eval()
    seresnet50_fpr, seresnet50_tpr, seresnet50_roc_auc_dict = roc(model, data_root)

#####################convnext_tiny###############################
    from models.convnext import convnext_tiny
    model_args = dict(num_classes=2)
    model = convnext_tiny(False, **model_args)
    path = r'D:\project\lymphnode_bm\result\convnextT_focalcb_trueroiresize224_500_pre6_1.pth'
    model_dict = model.state_dict()
    params = torch.load(path, map_location='cpu')
    new_state_dict = OrderedDict()
    # 修改 key
    for k, v in params['model'].items() :
        k = k.replace('module.', '')
        new_state_dict[k] = v
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    model.eval()
    convnextT_fpr, convnextT_tpr, convnextT_roc_auc_dict = roc(model, data_root)


# ##############vitB###############################################
#     model_args = dict(num_classes=6)
#     model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **model_args)
#     model = VisionTransformer(patch_size=model_kwargs['patch_size'],embed_dim=model_kwargs['embed_dim'],depth=model_kwargs['depth'],num_heads=model_kwargs['num_heads'],num_classes=model_kwargs['num_classes'])
#     path = r'D:\project\yolov5\res\trueroi lymph\'
#     model_dict = model.state_dict()
#     params = torch.load(path,map_location='cpu')
#     new_state_dict = OrderedDict()
#     for k, v in params['model'].items() :
#         k = k.replace('module.', '')
#         new_state_dict[k] = v
#     model_dict.update(new_state_dict)
#     model.load_state_dict(model_dict)
#     model.eval()
#     model.eval()
#     vitB_fpr,vitB_tpr,vitB_roc_auc_dict = roc(model,path)


##########visformer_small#######################################
    model_args = dict(num_classes=2)
    model = vit_small_patch16_224(False,**model_args)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of parameters:{}'.format(parameters / 1e6))
    path = r"D:\project\yolov5\res\trueroi lymph\vitS16_cutmix_focalcb_trueroiresize224_500_1.pth"
    params = torch.load(path,map_location='cpu')
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    # 修改 key
    for k, v in params['model'].items() :  # ['state_dict_ema']
        k = k.replace('module.', '')
        new_state_dict[k] = v
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    model.eval()
    visformerS_fpr,visformerS_tpr,visformerS_roc_auc_dict = roc(model,data_root)

###############swinB##############################################
    config = get_config(r"./configs/swin_tiny_patch4_window7_224.yaml")
    model = build_model(config)
    path = r"D:\project\yolov5\res\trueroi lymph\swinT_cutmix_focalcb_trueroiresize224_500_1.pth"
    model_dict = model.state_dict()
    params = torch.load(path,map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in params['model'].items() :
        k = k.replace('module.', '')
        new_state_dict[k] = v
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    model.eval()
    swinT_fpr,swinT_tpr,swinT_roc_auc_dict = roc(model,data_root)


#############dwtconvnetT############################################
    from models.dwtconvnet import dwtconvnet_tiny
    model_args = dict(num_classes=2)
    model = dwtconvnet_tiny(False, **model_args)
    path = r'D:\project\yolov5\res\trueroi lymph\convnextT_mixup_trueroiresize224_dwtconviwt_2000_78.99_re.pth'
    model_dict = model.state_dict()
    params = torch.load(path, map_location='cpu')
    new_state_dict = OrderedDict()
    # 修改 key
    for k, v in params['model'].items() :
        k = k.replace('module.', '')
        new_state_dict[k] = v
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    model.eval()


    dwt_fpr,dwt_tpr,dwt_roc_auc_dict = roc(model,data_root)




#########roc_models######################################
    fprlist = [resnet50_fpr,densenet201_fpr,inceptionv3_fpr,seresnet50_fpr,convnextT_fpr,visformerS_fpr,swinT_fpr,dwt_fpr]
    tprlist = [resnet50_tpr,densenet201_tpr,inceptionv3_tpr,seresnet50_tpr,convnextT_tpr,visformerS_tpr,swinT_tpr,dwt_tpr]
    auclist = [resnet50_roc_auc_dict,densenet201_roc_auc_dict,inceptionv3_roc_auc_dict,seresnet50_roc_auc_dict,convnextT_roc_auc_dict,visformerS_roc_auc_dict,swinT_roc_auc_dict,dwt_roc_auc_dict]
    roc_models(fprlist,tprlist,auclist)


