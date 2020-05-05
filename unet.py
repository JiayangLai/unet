import torch
from torch import nn

from bilinear_interpolation import Bilinear_upsample
from resnet_bottleneck import Resnet_bottleneck


class Unet_res152(nn.Module):


    def __init__(self):
        super(Unet_res152, self).__init__()

        res101 = Resnet_bottleneck(101)

        # padding 1 for alignment
        self.pad1 = nn.ReflectionPad2d(1)

        # Using the res_152 of last works

        # Conv 1
        self.conv1 = res101.conv1

        # Conv 2
        self.conv2 = res101.conv2

        # Conv 3
        self.conv3 = res101.conv3

        # Conv 4
        self.conv4 = res101.conv4

        # Conv 5
        self.conv5 = res101.conv5

        # Upsample 1
        self.upsamp1 = Bilinear_upsample()



    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

if __name__ == '__main__':
    A = torch.rand([10, 3, 572, 572])#572
    unet152 = Unet_res152()
    B = unet152.forward(A)
    print(B.shape)
