from config import get_config
from models.swin_transformer import SwinTransformer
import torch
from main import build_model
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from collections import OrderedDict
config = get_config(r"./configs/swin_base_patch4_window7_224_2.yaml")
model = build_model(config)
pth = r'D:\project\neck region transformer\result\2021.8.17\2c(156 234)swinB224_removemarkcross_randomcrop_cos_SGD_400\2c(156 234)swinB224_removemarkcross_randomcrop_cos_SGD_bs8_lr0.001_epoch400_4fold.pth'
params = torch.load(pth, map_location=torch.device('cpu'))
model_dict = model.state_dict()
new_state_dict = OrderedDict()
    # 修改 key
for k, v in params['model'].items() : #['state_dict_ema']
        param = k.split(".")
        k = ".".join(param[1:])
        new_state_dict[k] = v
model_dict.update(new_state_dict)
model.load_state_dict(model_dict)
model.eval()  # enter test mode

config2 = get_config(r"./configs/swin_base_patch4_window7_224_3.yaml")
model2 = build_model(config2)
pth2 = r'D:\project\neck region transformer\result\2021.8.17\3c156swinB224_removemarkcross_randomcrop_cos_SGD_500\3c156swinB224_removemarkcross_randomcrop_focalCB_cos_SGD_bs8_lr0.001_epoch500_4fold.pth'
params2 = torch.load(pth2, map_location=torch.device('cpu'))
model_dict2 = model2.state_dict()
new_state_dict2 = OrderedDict()
    # 修改 key
for k, v in params2['model'].items() : #['state_dict_ema']
        param = k.split(".")
        k = ".".join(param[1:])
        new_state_dict2[k] = v
model_dict2.update(new_state_dict2)
model2.load_state_dict(model_dict2)
model2.eval()  # enter test mode

config3 = get_config(r"./configs/swin_base_patch4_window7_224_4.yaml")
model3 = build_model(config3)
pth3 = r'D:\project\neck region transformer\result\2021.8.20\3c234swinB224_removemarkcross_randomcrop_focalCB_cos_SGD_500\3c234swinB224_removemarkcross_randomcrop_focalCB_cos_SGD_bs8_lr0.001_epoch500_4fold.pth'
params3 = torch.load(pth3, map_location=torch.device('cpu'))
model_dict3 = model3.state_dict()
new_state_dict3 = OrderedDict()
    # 修改 key
for k, v in params3['model'].items() : #['state_dict_ema']
        param = k.split(".")
        k = ".".join(param[1:])
        new_state_dict3[k] = v
model_dict3.update(new_state_dict3)
model3.load_state_dict(model_dict3)
model3.eval()  # enter test mode
train_path = r'D:\data\neck region\6c_1548_7-23\train\k4'
test_path = r'D:\data\neck region\6c_1548_7-23\test\k4'
# train_save156 = r'D:\data\neck region\4c_156_7-23\train\k2'
# test_save156 =  r'D:\data\neck region\4c_156_7-23\test\k2'
# train_save234 = r'D:\data\neck region\4c_234_7-23\train\k2'
# test_save234 = r'D:\data\neck region\4c_234_7-23\test\k2'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
transform_test = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        # Normalize a tensor image with mean and standard variance
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
# with torch.no_grad():
#     for root, dirs, _ in os.walk(train_path) :
#         for dir in dirs:
#             for root2, _, files2 in os.walk(os.path.join(root, dir)) :
#                     for file in files2 :
#
#                         pic = Image.open(os.path.join(root2,file))
#                         src = np.asarray(pic)
#                         input = transform_test(pic)
#                         c,h,w = input.size(2),input.size(0),input.size(1)
#                         input = input.view(1,h,w,c)
#                         input = input.to(device)
#                         output = model(input)
#                         _, predicted = output.max(1)
#                         c = predicted.numpy()[0]
#                         if c == 0:
#                             if dir in ['1','5','6']:
#                                 path = os.path.join(os.path.join(train_save156,dir),file)
#                             else:
#                                 path = os.path.join(os.path.join(train_save156, '0'),file)
#                         else:
#                             if dir in ['2','3','4']:
#                                 path = os.path.join(os.path.join(train_save234, dir),file)
#                             else:
#                                 path = os.path.join(os.path.join(train_save234,'0'),file)
#                         print(os.path.join(root2, file),dir,c)
#                         img = Image.fromarray(src)
#                         img.save(path)
correct,total = 0,0

with torch.no_grad():
    for root, dirs, _ in os.walk(test_path) :
        for dir in dirs:
            for root2, _, files2 in os.walk(os.path.join(root, dir)) :
                    total += len(files2)
                    t = len(files2)
                    cc = 0
                    for file in files2 :
                        pic = Image.open(os.path.join(root2, file))
                        src = np.asarray(pic)
                        input = transform_test(pic)
                        c, h, w = input.size(2), input.size(0), input.size(1)
                        input = input.view(1, h, w, c)
                        input = input.to(device)
                        output = model(input)
                        _, predicted = output.max(1)
                        c = predicted.numpy()[0]
                        print(dir,file,c)
                        if c == 0:
                            if dir in ['1','5','6']:
                                output2 = model2(input)
                                _, predicted2 = output2.max(1)
                                c2 = predicted2.numpy()[0]
                                print(c2)
                                if (dir == '1' and c2 == 0) or (dir == '5' and c2 == 1) or (dir == '6' and c2 == 2):
                                    cc+=1
                                    correct += 1
                        else:
                            if dir in ['2','3','4']:
                                output3 = model3(input)
                                _, predicted3 = output3.max(1)
                                c3 = predicted3.numpy()[0]
                                print(c3)
                                if (dir == '2' and c3 == 0) or (dir == '3' and c3 == 1) or (dir == '4' and c3 == 2) :
                                    cc+=1
                                    correct += 1
                    print(dir,"acc",cc/t*100)

print(correct/total*100)

