import torch
import torchvision
from torch import nn

from bilinear_interpolation import Bilinear_upsample
from resnet_bottleneck import Resnet_bottleneck


class Unet_res(nn.Module):


    def __init__(self):
        super(Unet_res, self).__init__()
        self.resnet1 = torchvision.models.resnet152(pretrained=True)

        # Conv, BN and ReLU
        self.cbr5 = nn.Sequential(nn.Conv2d(2048,1024,3,1,padding=1),
                                 nn.BatchNorm2d(1024),
                                 nn.ReLU())
        self.cbr4 = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, padding=1),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU())
        self.cbr3 = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU())
        self.cbr2 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU())
        self.last = nn.Sequential(nn.Conv2d(128, 1, 3, 1, padding=1),
                                  nn.BatchNorm2d(1),
                                  nn.ReLU())


        # Upsample 1
        #self.upsamp1 = Bilinear_upsample() # Manually worse one, too slow with 2 for-loops
        self.upsamp = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False) # use an official one to speed up



    def forward(self, x):
        with torch.no_grad():
            out1000, x_l0, x_l1, x_l2, x_l3, x_l4 = self.resnet1(x)

        x_l4 = self.cbr5(x_l4)
        x_up3 = self.upsamp(x_l4)

        x_cat3 = torch.cat([x_up3,x_l3],1)
        x_cat3 = self.cbr4(x_cat3)
        x_up2 = self.upsamp(x_cat3)

        x_cat2 = torch.cat([x_up2, x_l2], 1)
        x_cat2 = self.cbr3(x_cat2)
        x_up1 = self.upsamp(x_cat2)

        x_cat1 = torch.cat([x_up1, x_l1], 1)
        x_cat1 = self.cbr2(x_cat1)
        x_up = self.upsamp(x_cat1)

        out = self.last(x_up)

        return out

if __name__ == '__main__':
    device = torch.device('cuda:0')
    A = torch.rand([8, 3, 224, 224])#572
    A = A.to(device)
    unet101 = Unet_res()
    unet101 = unet101.to(device)
    B = unet101.forward(A)
    print(B.shape)
