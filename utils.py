# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt




from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import cv2

imgPath = r'D:\data\2cbm7-25\train\b'
imgSavePath = r'D:\data\2cbm7-25\b'

k = 1
aug_data_number = 2

for root,dirs,files in os.walk(imgPath):
    for file in files:
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        p1 = 0.5
        p2 = 0.5
        # for i in item_split:
        im = Image.open(imgPath+'\\'+file)
        # label = Image.open(labelPath+'\\'+file)
        # print(im.mode)
        # plt.imshow(im)
        # plt.show()
        # name = os.path.basename(i)
#         im_aug1 = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.RandomHorizontalFlip(p1),
#             # transforms.RandomVerticalFlip(p2),
#             # transforms.RandomRotation(10, resample=False, expand=False, center=None),
#             # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
#             # transforms.RandomCrop(256),
#             transforms.Resize((256, 256))
#         ])

#         im_aug2 = transforms.Compose([
#             transforms.Resize((256, 256)),
#             # transforms.RandomHorizontalFlip(p1),
#             transforms.RandomVerticalFlip(p2),
#             # transforms.RandomRotation(10, resample=False, expand=False, center=None),
#             # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
#             # transforms.RandomCrop(256),
#             # transforms.Resize((256, 256))
#         ])
#         im_aug3 = transforms.Compose([
#             transforms.Resize((256, 256)),
#             # transforms.RandomHorizontalFlip(p1),
#             # transforms.RandomVerticalFlip(p2),
#             transforms.RandomRotation(20),
#             # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
#             # transforms.RandomCrop(256),
#             # transforms.Resize((256, 256))
#         ])
        im_aug4 = transforms.Compose([
            # transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # transforms.RandomHorizontalFlip(p1),
            transforms.RandomVerticalFlip(p2),
            # transforms.RandomRotation(20),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            # transforms.RandomCrop(256),
        ])
        im_aug5 = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p1),
            # transforms.RandomVerticalFlip(p2),
            # transforms.RandomRotation(20,  expand=False),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            # transforms.RandomCrop(256),
            # transforms.Resize((256, 256))
        ])
        im_aug6 = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p1),
            transforms.RandomVerticalFlip(p2),
            # transforms.RandomRotation(10, resample=False, expand=False, center=None),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            # transforms.RandomCrop(256),
            # transforms.Resize((256, 256))
        ])

        random.seed(seed)  # apply this seed to img tranfsorms

#         img = im_aug1(im)
#         data = np.asarray(img)
#         cv2.imwrite(os.path.join(imgSavePath, file.split('.')[0]+'a1.jpg'),data)
#         # label2 = im_aug1(label)
#         # data = np.asarray(label2)
#         # cv2.imwrite(os.path.join(labelSavePath, file.split('.')[0] + 'a1.jpg'), data)

#         img = im_aug2(im)
#         data = np.asarray(img)
#         cv2.imwrite(os.path.join(imgSavePath, file.split('.')[0]+'a2.jpg'),data)
        # label2 = im_aug2(label)
        # data = np.asarray(label2)
        # cv2.imwrite(os.path.join(labelSavePath, file.split('.')[0] + 'a2.jpg'), data)
        #
        # img = im_aug3(im)
        # data = np.asarray(img)
        # cv2.imwrite(os.path.join(imgSavePath, file.split('.')[0]+'a3.jpg'),data)
        # label2 = im_aug3(label)
        # data = np.asarray(label2)
        # cv2.imwrite(os.path.join(labelSavePath, file.split('.')[0] + 'a3.jpg'), data)

        img = im_aug4(im)
        data = np.asarray(img)
        src = Image.fromarray(data)
        src.save(os.path.join(imgSavePath, file.split('.')[0]+'a4.jpg'))
        # label2 = im_aug4(label)
        # data = np.asarray(label2)
        # cv2.imwrite(os.path.join(labelSavePath, file.split('.')[0] + 'a4.jpg'), data)

        img = im_aug5(im)
        data = np.asarray(img)
        src = Image.fromarray(data)
        src.save(os.path.join(imgSavePath, file.split('.')[0]+'a5.jpg'))
        # label2 = im_aug5(label)
        # data = np.asarray(label2)
        # cv2.imwrite(os.path.join(labelSavePath, file.split('.')[0] + 'a5.jpg'), data)

        img = im_aug6(im)
        data = np.asarray(img)
        src = Image.fromarray(data)
        src.save(os.path.join(imgSavePath, file.split('.')[0]+'a6.jpg'))
        # label2 = im_aug6(label)
        # data = np.asarray(label2)
        # cv2.imwrite(os.path.join(labelSavePath, file.split('.')[0] + 'a6.jpg'), data)


        # plt.title(name)
        # plt.show()


