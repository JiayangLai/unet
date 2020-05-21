import os
import random
import time

import torch
import torchvision
from torch import nn, optim
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

import CONFIG
from unet import Unet_res
import pandas as pd
import numpy as np
import torch

from utils.dataload.lanedateset import LaneDataSet
from utils.loss.dice_loss import DiceLoss
from utils.metrics.mIOU import ComputeIoU


def train_model(model, criterion,metric, optimizer, scheduler, dataloaders, pre_loss,
                pre_epoch, models_path, Batch_size, num_epochs=2, labelshape=CONFIG.OUTPUT_SHAPE):
    since = time.time()
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = pre_loss

    count = 0
    for epoch in range(pre_epoch + 1, pre_epoch + num_epochs):
        print('-' * 20)

        print('Epoch {}/{}'.format(epoch, pre_epoch + num_epochs))

        # 每个epoch的训练和验证阶段
        for phase in ['train', 'val']:

            # phase = 'train'
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()  # 验证模式

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):  #
                count += 1
                inputs = inputs.type(torch.float32).transpose(1, 3)
                labels = labels.type(torch.float32).transpose(1, 2)

                inputs = inputs.to(device)
                # print(inputs.shape)
                # inputs = F.interpolate(inputs, size=[labelshape[0],labelshape[1]])

                labels = labels.to(device)
                # labels = torch.unsqueeze(labels, dim=0)
                # print(labels.shape)
                # labels = F.interpolate(labels, size=labelshape,mode='nearest')
                # labels = labels.squeeze(0)

                # 训练阶段开启梯度跟踪
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    loss = criterion(outputs.squeeze(1), labels)

                    # 仅在训练阶段进行后向+优化
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                if CONFIG.IS_PLOT and count % 20 == 0:
                    # print('loss:',loss.cpu().detach().numpy())
                    ypr = outputs.squeeze(1).cpu().detach().numpy()
                    ygt = labels.cpu().detach().numpy()
                    inp = inputs[0, :, :, :].cpu().detach().numpy()
                    # print(ypr)
                    # print(ygt)
                    # print('acc:',(abs(ypr-ygt)<0.1).sum()/(21.0*32))
                    ax1 = fig.add_subplot(131)
                    ax2 = fig.add_subplot(132)
                    ax3 = fig.add_subplot(133)
                    # ax4 = fig.add_subplot(234)
                    # ax5 = fig.add_subplot(235)
                    # im = Image.fromarray(np.uint8(cm.gist_earth(myarray) * 255))
                    ax1.imshow(inp[:3].astype(int).T)
                    # ax2.imshow(inp[1].T, cmap=plt.cm.gray)
                    # ax3.imshow(inp[2].T, cmap=plt.cm.gray)
                    ax2.imshow(ypr[0].T, cmap=plt.cm.gray)
                    ax3.imshow(ygt[0].T, cmap=plt.cm.gray)

                    plt.pause(0.1)
                    plt.clf()
                # 统计
                running_loss += loss.item() * inputs.size(0)
                metric(outputs.squeeze(1), labels)
                running_corrects += metric.get_miou()


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]#.double()
            # a = epoch_acc.cpu().detach().numpy()

            print('{} Loss: {:.9f} Acc: {:.9f}'.format(phase, epoch_loss, epoch_acc))
            # 记录最好的状态
            if phase == 'val' and epoch_loss <= best_loss:
                best_loss = epoch_loss

                # 保存整个模型
                torch.save(model, models_path + '/model_loss{:.6f}_acc{:.6f}_shape_{}_{}_epoch{}.pkl'.format(epoch_loss,
                                                                                                             epoch_acc,
                                                                                                             labelshape[
                                                                                                                 0],
                                                                                                             labelshape[
                                                                                                                 1],
                                                                                                             epoch))
                print('Better saved!')

    print('-' * 20)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))


def get_dataloaders():
    path = 'baiduLaneSegDataRead/data_320X320_npy_paths.csv'
    imgpathstable = pd.read_csv(path).to_numpy()
    np.random.seed(seed=2333)
    np.random.shuffle(imgpathstable)
    fifth = int(CONFIG.NUM_TOT_IMGS / 5)

    train_dst = LaneDataSet(imgpathstable[fifth:], testmode=CONFIG.IS_TEST_MODE)
    trainloader = torch.utils.data.DataLoader(train_dst, batch_size=CONFIG.TRAIN_BATCH_SIZE, shuffle=True,
                                              num_workers=3,
                                              pin_memory=False)

    val_dst = LaneDataSet(imgpathstable[:fifth], testmode=CONFIG.IS_TEST_MODE)
    valloader = torch.utils.data.DataLoader(val_dst, batch_size=CONFIG.TRAIN_BATCH_SIZE, shuffle=True, num_workers=3,
                                            pin_memory=False)
    return trainloader, valloader


if __name__ == '__main__':
    device = torch.device('cuda:0')
    # unet = Unet_res()
    models_path = 'saved_model_baseline'
    if CONFIG.IS_RETRAIN:
        model_name = 'model_loss0.541701_acc0.014362_shape_320_320_epoch12.pkl'
        pre_loss = float(model_name.split('loss')[1].split('_')[0])
        pre_epoch = int(model_name.split('epoch')[1].split('.')[0])
        unet = torch.load(models_path+"/"+model_name)
        print(f'load model: --{model_name}--')
    else:
        pre_loss = 1000.0
        pre_epoch = -1
        unet = Unet_res()


    unet = unet.to(device)
    metric = ComputeIoU()
    criterion = DiceLoss()
    # only train the upsampling networks
    # optimizer = optim.Adam(unet.parameters(), lr=3e-4)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, unet.parameters()), lr=3e-4)

    trainloader, valloader = get_dataloaders()
    dataloader = {'train': trainloader, 'val': valloader}
    dataset_sizes = {'train': CONFIG.NUM_TOT_IMGS - int(CONFIG.NUM_TOT_IMGS / 5), 'val': int(CONFIG.NUM_TOT_IMGS / 5)}
    if CONFIG.IS_TEST_MODE:
        dataset_sizes = {'train': int((CONFIG.NUM_TOT_IMGS - int(CONFIG.NUM_TOT_IMGS / 5))/100),
                         'val': int(CONFIG.NUM_TOT_IMGS / 5/100)}

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    num_epoch = 30
    # labelshape = (20, 20)
    if CONFIG.IS_PLOT:
        plt.ion()
        fig = plt.figure()
    train_model(unet, criterion,metric, optimizer, lr_scheduler, dataloader, pre_loss, pre_epoch,
                models_path, CONFIG.TRAIN_BATCH_SIZE, num_epochs=num_epoch, labelshape=CONFIG.OUTPUT_SHAPE)
