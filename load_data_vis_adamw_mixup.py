from collections import OrderedDict
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from tensorboardX import SummaryWriter
from PIL import Image
import os
from optimizer.sam import SAM
from sklearn.metrics import recall_score,f1_score,accuracy_score,precision_score,confusion_matrix
import numpy as np
import warnings
from timm.scheduler.cosine_lr import CosineLRScheduler
from loss.softtarget import SoftTargetCrossEntropy,SoftTargetFocalCB
from loss.focalCB import CBLoss
from loss.center_loss import CenterLoss
warnings.filterwarnings('ignore')
#实现评价指标的计算
def cal_metrics(confusion_matrix):
    n_classes = confusion_matrix.shape[0]
    ALL,PPV,NPV,TPR,TNR,FPR = 0,0,0,0,0,0
    # res = [[],[],[],[],[]]
    # c = []
    for i in range(n_classes):
        # 逐步获取 真阳，假阳，真阴，假阴四个指标，并计算三个参数
        ALL = np.sum(confusion_matrix)
        # 对角线上是正确预测的
        tp = confusion_matrix[i, i]
        # 列加和减去正确预测是该类的假阳
        fp = np.sum(confusion_matrix[:, i]) - tp
        # 行加和减去正确预测是该类的假阴
        fn = np.sum(confusion_matrix[i, :]) - tp
        # 全部减去前面三个就是真阴
        tn = ALL - tp - fp - fn
        ppv = tp / (tp + fp)
        npv = tn / (fn + tn)
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        fpr = fp / (fp + tn)
        PPV += (tp+fn)*ppv/ALL
        NPV += (tp + fn) * npv / ALL
        TPR += (tp + fn) * tpr / ALL
        TNR += (tp + fn) * tnr / ALL
        FPR += (tp + fn) * fpr / ALL

    return PPV,NPV,TPR,TNR,FPR
def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)

def train(epoch, model, lossFunction, centerloss,optimizer, device, trainloader, mixup, scheduler, num_steps ):
    """train model using loss_fn and optimizer. When this function is called, model trains for one epoch.
    Args:
        train_loader: train data
        model: prediction model
        loss_fn: loss function to judge the distance between target and outputs
        optimizer: optimize the loss function
        get_grad: True, False
    output:
        total_loss: loss
        average_grad2: average grad for hidden 2 in this epoch
        average_grad3: average grad for hidden 3 in this epoch
    """
    print('\nEpoch: %d' % epoch)
    model.train()  # enter train mode
    train_loss = 0  # accumulate every batch loss in a epoch
    correct = 0  # count when model' prediction is correct i train set
    total = 0  # total number of prediction in train set
    l2_weight = 1e-7
    for batch_idx, (inputs, targets) in enumerate(trainloader) :
        if targets.shape[0] % 2 != 0 :
            inputs, targets = inputs[1 :], targets[1 :]
        inputs, targets = inputs.to(device), targets.to(device)  # load data to gpu device
        inputs, targets = Variable(inputs), Variable(targets)
        if mixup is not None :
            inputs, targets = mixup(inputs, targets)
        features,outputs = model(inputs)  # forward propagation return the value of softmax function
        loss = lossFunction(outputs, targets) +0.001* centerloss(features, targets, 'mixup')                  ###########+l2_weight * sum([torch.norm(p,2) for p in model.parameters()]) # compute loss
        optimizer.zero_grad()
        loss.backward()  # compute gradient of loss over parameters
        for param in centerloss.parameters() :
            param.grad.data *= (1. / 0.001)
        optimizer.step()  # update parameters with gradient descent

        train_loss += loss.item()  # accumulate every batch loss in a epoch
        _, predicted = outputs.max(1)  # make prediction according to the outputs
        # scheduler.step_update(epoch * num_steps + batch_idx)        # for label_idx in range(len(predicted)) :
        #     label = predicted[label_idx]
        #     if label != 0:
        #         if predicted[label_idx]==targets[label_idx]:
        #             correct+=1

        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()  # count how many predictions is correct
        # loss = smooth_crossentropy(outputs, targets)# use this loss for any training statistics
        # # first forward-backward pass
        #
        # loss.mean().backward(retain_graph=True)
        # optimizer.first_step(zero_grad=True)
        #
        # # second forward-backward pass
        # smooth_crossentropy(model(inputs), targets).mean().backward(retain_graph=True)  # make sure to do a full forward pass
        # optimizer.second_step(zero_grad=True)
        # with torch.no_grad():
        #     train_loss += loss.mean()# accumulate every batch loss in a epoch
        #     _, predicted = outputs.max(1)  # make prediction according to the outputs
        #     total += targets.size(0)
        #     correct += predicted.eq(targets).sum().item()  # count how many predictions is correct

    print('Train loss: %.3f | '
          % (train_loss / (batch_idx + 1)))
    return train_loss / (batch_idx + 1)

def test(model, lossFunction,centerloss, device, testloader):
    """
    test model's prediction performance on loader.
    When thid function is called, model is evaluated.
    Args:
        loader: data for evaluation
        model: prediction model
        loss_fn: loss function to judge the distance between target and outputs
    output:
        total_loss
        accuracy
    """
    global best_acc
    model.eval()  # enter test mode
    test_loss = 0  # accumulate every batch loss in a epoch
    correct = 0
    total = 0
    y_true,y_pred = [],[]
    l2_weight = 1e-7
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            features,outputs = model(inputs)
            # l2_regularization = torch.tensor(0.0)
            # l2_regularization = l2_regularization.to(device)
            # for param in model.parameters() :
            #     l2_regularization += torch.norm(param, 2)
            loss = lossFunction(outputs, targets) + 0.001*centerloss(features, targets,'norm')
            test_loss += loss.item()  # accumulate every batch loss in a epoch
            _, predicted = outputs.max(1)  # make prediction according to the outputs
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()  # count how many predictions is correct

            # print(targets.cpu().numpy(),predicted.cpu().numpy())
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        test_pre_micro = precision_score(y_true, y_pred, average='weighted')
        test_rec_micro = recall_score(y_true, y_pred, average='weighted')
        test_f1_micro = f1_score(y_true, y_pred, average='weighted')
        # 获取混淆矩阵
        sum_confusion_matrix = confusion_matrix(y_true, y_pred)
        PPV, NPV, TPR, TNR, FPR = cal_metrics(sum_confusion_matrix)
        # print loss and acc
        print(
            'Test Loss: %.3f  | Test Acc: %.3f (%d/%d) | Test Pre Micro: %.3f | Test Rec Micro: %.3f | Test F1 Micro: %.3f'
            % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total, 100. * test_pre_micro,
               100. * test_rec_micro, 100. * test_f1_micro))
        print(
            'Test PPV: %.3f  | Test NPV: %.3f | Test TPR: %.3f | Test TNR: %.3f | Test FPR: %.3f'
            % (100. * PPV, 100. * NPV, 100. * TPR, 100 * TNR, 100. * FPR))
    return test_loss / (
                batch_idx + 1), 100. * correct / total, 100. * test_pre_micro, 100. * test_rec_micro, 100. * test_f1_micro, 100. * PPV, 100. * NPV, 100. * TPR, 100. * TNR, 100. * FPR


class MyDataset(Dataset): # 继承Dataset类
    def __init__(self, root, transform=None, target_transform=None): # 定义txt_path参数
        imgs = []  # 定义imgs的列表
        c = 0
        for root1, dirs1, _ in os.walk(root) :
            for dir1 in dirs1 :
                for root2, _, files2 in os.walk(os.path.join(root1, dir1)) :
                    for file2 in files2 :
                        imgs.append((os.path.join(root2,file2), int(c))) # 存放进imgs列表中
                c+=1


        self.imgs = imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index] # fn代表图片的路径，label代表标签
        img = Image.open(fn).convert('RGBA')     # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1   参考：https://blog.csdn.net/icamera0/article/details/50843172

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.imgs)   # 返回图片的长度


def data_loader(k):
    # define method of preprocessing data for evaluating
    transform_train = transforms.Compose([
        # transforms.Resize([224, 224]),
        transforms.RandomResizedCrop(224),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomRotation(30),
        # transforms.ColorJitter(0.5,0.5,0.5,0.5),
        transforms.ToTensor(),
        # Normalize a tensor image with mean and standard variance
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        # Normalize a tensor image with mean and standard variance
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # prepare dataset by ImageFolder, data should be classified by directory
    trainset = torchvision.datasets.ImageFolder(root=r'/data/lyt/dataset/6c_6-38_moreroi/train/k' + str(k + 1),
                                                transform=transform_train)  # /data/lyt/dataset/neck_region_pre/6c_1382_new/train/k,D:\data\neck_region_pre(src)\train\k
    testset = torchvision.datasets.ImageFolder(root=r'/data/lyt/dataset/6c_6-38_moreroi/valid/k' + str(k + 1),
                                               transform=transform_test)  # /data/lyt/dataset/neck_region_pre/6c_1382_new/test/k,D:\data\neck_region_pre(src)\test\k
    # Data loader.
    # trainset = torchvision.datasets.ImageFolder(root=r'D:/data/7-25/train/k' + str(k + 1),
    #                                            transform=transform_train)  # /data/lyt/dataset/neck_region_pre/6c_1382_new/train/k,D:\data\neck_region_pre(src)\train\k
    # testset = torchvision.datasets.ImageFolder(root=r'D:/data/7-25/valid/k' + str(k + 1),
    #                                            transform=transform_test)  # /data/lyt/dataset/neck_region_pre/6c_1382_new/test/k,D:\data\neck_region_pre(src)\test\k
    # trainset = MyDataset(root=r'/data/lyt/dataset/neck_region_pre/6c_1382_heatmap/train/k'+str(k+1), transform=transform_train)
    # testset = MyDataset(root=r'/data/lyt/dataset/neck_region_pre/6c_1382_heatmap/test/k' + str(k + 1),transform=transform_test)

    # Combines a dataset and a sampler,

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False)
    return trainloader, testloader


def run(i,boardname,model, num_epochs,numclasses,log_dir = '',test_li=[],mixup=None):
    # 定义Summary_Writer
    log_path  = log_dir+'_'+str(i+1)+'.pth'
    writer = SummaryWriter(boardname)
    # load model into GPU device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    print(device)
    if 'cuda' in device:
        torch.cuda.empty_cache()
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True
    print(torch.cuda.device_count())
    # define the loss function and optimizer
    trainloader, testloader = data_loader(i)
    #optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9,0.999),lr=5e-4, weight_decay=0.05)
    name = str(i + 1) + 'fold_'
    labels = []
    for item in trainloader :
        labels.extend(item[1].tolist())
    lossFunction = SoftTargetFocalCB(numclasses, labels=labels, loss_type="focal", beta=0.99, gamma=0.5)
    lossFunction2 = CBLoss(numclasses, labels=labels, loss_type="focal", beta=0.99, gamma=0.5)
    center_loss = CenterLoss(num_classes=numclasses, feat_dim=1024, use_gpu=True)
    params = list(model.parameters()) + list(center_loss.parameters())
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=5e-4)
    n_iter_per_epoch = len(trainloader)
    num_steps = int(800 * n_iter_per_epoch)
    # warmup_steps = int(40 * n_iter_per_epoch)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)
    # scheduler = CosineLRScheduler(
    #     optimizer,
    #     t_initial=num_steps,
    #     t_mul=1.,
    #     lr_min=5e-6,
    #     warmup_lr_init=5e-7,
    #     warmup_t=warmup_steps,
    #     cycle_limit=1,
    #     t_in_epochs=False,
    # )
    # lossFunction = nn.CrossEntropyLoss()
    test_acc_li, test_pre_micro_li, test_rec_micro_li, test_f1_micro_li, test_loss_li, test_PPV_li, test_NPV_li, test_TPR_li, test_TNR_li, test_FPR_li = [], [], [], [], [], [], [], [], [], []
    maxacc = 0.0
    for epoch in range(num_epochs):
        train_loss = train(epoch, model, lossFunction, center_loss, optimizer, device, trainloader,mixup, scheduler, num_steps)
        test_loss, test_acc, test_pre_micro, test_rec_micro, test_f1_micro, PPV, NPV, TPR, TNR, FPR = test(model,lossFunction2,center_loss,device,testloader)
        scheduler.step()
        test_loss_li.append(test_loss)
        test_acc_li.append(test_acc)
        test_pre_micro_li.append(test_pre_micro)
        test_rec_micro_li.append(test_rec_micro)
        test_f1_micro_li.append(test_f1_micro)
        test_PPV_li.append(PPV)
        test_NPV_li.append(NPV)
        test_TPR_li.append(TPR)
        test_TNR_li.append(TNR)
        test_FPR_li.append(FPR)

        writer.add_scalars(name+'train_test_loss',{'train_loss': (train_loss), 'test_loss': (test_loss)}, epoch)
        writer.add_scalars(name+'test_cri', {'test_acc' : (test_acc), 'test_pre' : (test_pre_micro),'test_rec' : (test_rec_micro),'test_f1' : (test_f1_micro)}, epoch)
        if maxacc < test_acc:
            print('current_acc:',test_acc,',save_pth')
            maxacc = test_acc
            state = {'model' : model.state_dict(), 'optimizer' : optimizer.state_dict(), 'epoch' : epoch,
                     'result' : [test_loss, test_acc, test_pre_micro, test_rec_micro, test_f1_micro, PPV, NPV, TPR, TNR,
                                 FPR]}
            torch.save(state, log_path)

        # if (epoch + 1) % 40 == 0:
        #     lr = lr / 2
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
    index = test_acc_li.index(max(test_acc_li))
    test_li[0].append(test_loss_li[index])
    test_li[1].append(test_acc_li[index])
    test_li[2].append(test_pre_micro_li[index])
    test_li[3].append(test_rec_micro_li[index])
    test_li[4].append(test_f1_micro_li[index])
    test_li[5].append(test_PPV_li[index])
    test_li[6].append(test_NPV_li[index])
    test_li[7].append(test_TPR_li[index])
    test_li[8].append(test_TNR_li[index])
    test_li[9].append(test_FPR_li[index])

    print(
        'fold %d,第%d个epoch, test loss %.4f, test acc max %.4f, test pre micro %.4f,   test rec micro %.4f, test f1 micro %.4f, test PPV %.4f, test NPV %.4f, test TPR %.4f, test TNR %.4f, test FPR %.4f' %
        (i + 1, index, test_loss_li[index], test_acc_li[index], test_pre_micro_li[index], test_rec_micro_li[index],
         test_f1_micro_li[index], test_PPV_li[index], test_NPV_li[index], test_TPR_li[index], test_TNR_li[index],
         test_FPR_li[index]))
    writer.close()

