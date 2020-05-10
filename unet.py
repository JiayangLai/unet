import torch
from torch import nn

from bilinear_interpolation import Bilinear_upsample
from resnet_bottleneck import Resnet_bottleneck


class Unet_res(nn.Module):


    def __init__(self):
        super(Unet_res, self).__init__()

        res = Resnet_bottleneck(50)

        # padding 1 for alignment, if inputs is 224^2 that will be fine without it
        # self.pad1 = nn.ReflectionPad2d(1)

        # Using the res_152 from last works

        # Conv 1
        self.conv1 = res.conv1

        # Conv 2
        self.conv2 = res.conv2

        # Conv 3
        self.conv3 = res.conv3

        # Conv 4
        self.conv4 = res.conv4

        # Conv 5
        self.conv5 = res.conv5


        # Conv, BN and ReLU
        self.cbr5 = nn.Sequential(nn.Conv2d(2048,1024,3,1,padding=1),
                                 nn.BatchNorm2d(1024),
                                 nn.ReLU())
        self.cbr4 = nn.Sequential(nn.Conv2d(2048, 1024, 3, 1, padding=1),
                                  nn.BatchNorm2d(1024),
                                  nn.ReLU())
        self.cbr3 = nn.Sequential(nn.Conv2d(1536, 768, 3, 1, padding=1),
                                  nn.BatchNorm2d(768),
                                  nn.ReLU())
        self.cbr2 = nn.Sequential(nn.Conv2d(1024, 512, 3, 1, padding=1),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU())
        self.last = nn.Sequential(nn.Conv2d(512, 1, 3, 1, padding=1),
                                  nn.BatchNorm2d(1),
                                  nn.ReLU())


        # Upsample 1
        #self.upsamp1 = Bilinear_upsample() # Manually worse one, too slow with 2 for-loops
        self.upsamp = nn.Upsample(scale_factor=2) # use an official one to speed up



    def forward(self, x):
        # x = self.pad1(x)
        x = self.conv1(x)
        print(x.shape)
        x_l2 = self.conv2(x)
        x_l3 = self.conv3(x_l2)
        x_l4 = self.conv4(x_l3)
        x_l5 = self.conv5(x_l4)

        x_l5 = self.cbr5(x_l5)
        x_up4 = self.upsamp(x_l5)

        x_cat4 = torch.cat([x_up4,x_l4],1)
        x_cat4 = self.cbr4(x_cat4)
        x_up3 = self.upsamp(x_cat4)

        x_cat3 = torch.cat([x_up3, x_l3], 1)
        x_cat3 = self.cbr3(x_cat3)
        x_up2 = self.upsamp(x_cat3)

        x_cat2 = torch.cat([x_up2, x_l2], 1)
        x_cat2 = self.cbr2(x_cat2)
        x_up = self.upsamp(x_cat2)

        out = self.last(x_up)

        return out

if __name__ == '__main__':
    A = torch.rand([10, 3, 224, 224])#572
    unet101 = Unet_res()
    B = unet101.forward(A)
    print(B.shape)
