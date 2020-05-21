import os
import numpy as np
# import random
import matplotlib.pyplot as plt
# import collections
import torch
import torchvision
# import cv2
from PIL import Image


# import torchvision.transforms as transforms
from utils.dataload.baiduLabelConvrt import lbcnvrt


class LaneDataSet(torch.utils.data.Dataset):
    def __init__(self, Xs, testmode=False):
        super(LaneDataSet, self).__init__()
        if testmode:
            lth = Xs.shape[0]
            Xs = Xs[:int(lth/100)]

        self.files = []
        for row in Xs:
            img_file = row[0]
            label_file = row[1]
            self.files.append({
                "img": img_file,
                "label": label_file,
                # "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        '''load the datas'''
        # name = datafiles["name"]
        image = Image.open(datafiles["img"]).convert('RGB')#np.load(datafiles["img"])#
        image = np.array(image)
        # image.swapaxes([1,3])
        label = np.load(datafiles["label"])
        # label = Image.open(datafiles["label"]).convert('L')#np.load(datafiles["label"])#
        # label = np.array(label)
        # label = lbcnvrt(label)
        # size_origin = image[0].shape  # W * H

        I = np.asarray(image, np.float32)

        # I = I.transpose((2, 0, 1))  # transpose the  H*W*C to C*H*W
        L = np.asarray(np.array(label), np.float32)
        # print(I.shape,L.shape)
        return I.copy(), L.copy()#, np.array(size_origin)#, name


if __name__ == '__main__':
    DATA_DIRECTORY = './data'
    DATA_LIST_PATH = ''
    Batch_size = 8
    # MEAN = (104.008, 116.669, 122.675)
    dst = LaneDataSet(DATA_DIRECTORY, DATA_LIST_PATH,'')
    # just for test,  so the mean is (0,0,0) to show the original images.
    # But when we are training a model, the mean should have another value
    trainloader = torch.utils.data.DataLoader(dst, batch_size=Batch_size)
    # plt.ion()
    for i, data in enumerate(trainloader):
        print(i)
        imgs, labels, _, _ = data
        # print(imgs.shape)
        # print(labels.shape)
        # if i % 1 == 0:
        #     img = torchvision.utils.make_grid(imgs).numpy()
        #     img = img.astype(np.uint8)  # change the dtype from float32 to uint8,
        #     # because the plt.imshow() need the uint8
        #     img = np.transpose(img, (1, 2, 0))  # transpose the C*H*W to H*W*C
        #     # img = img[:, :, ::-1]
        #     plt.imshow(img)
        #     plt.show()
        #     plt.pause(0.5)
        #
        #     #            label = torchvision.utils.make_grid(labels).numpy()
        #     labels = labels.astype(np.uint8)  # change the dtype from float32 to uint8,
        #     #                                       # because the plt.imshow() need the uint8
        #     for i in range(labels.shape[0]):
        #         plt.imshow(labels[i], cmap='gray')
        #         plt.show()
        #         plt.pause(0.5)
