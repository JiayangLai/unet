from torch import nn
import torch

class res_single_layer(nn.Module):

    def __init__(self,in_ch, out_ch, kernal_size, is_last_layer_in_block=False, is_stride_2=False):
        super(res_single_layer, self).__init__()
        padding_size = 0
        if kernal_size == 3:
            padding_size = 1
        if is_stride_2:
            self.conv = nn.Conv2d(in_ch, out_ch, kernal_size, stride=2, padding=padding_size)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, kernal_size, padding=padding_size)

        self.is_last_layer_in_block = is_last_layer_in_block
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.is_last_layer_in_block:
            return x
        else:
            return self.relu(x)


class inc_n_ch(nn.Module):
    def __init__(self, in_ch, out_ch,is_stride_2=False):
        super(inc_n_ch, self).__init__()
        if is_stride_2:
            self.identity = nn.Conv2d(in_ch, out_ch, 2,stride=2, padding=0)
        else:
            self.identity = nn.Conv2d(in_ch, out_ch, 1, padding=0)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.identity(x)
        x = self.bn(x)
        return x


class Res_block(torch.nn.Module):
    def __init__(self,in_ch,out_ch1,out_ch2, is_stride_2=False):
        super(Res_block, self).__init__()
        self.get_identity = inc_n_ch(in_ch, out_ch2,is_stride_2)

        self.blk_lyr1 = res_single_layer(in_ch, out_ch1, 1, is_stride_2=is_stride_2)
        self.blk_lyr2 = res_single_layer(out_ch1, out_ch1, 3)
        self.blk_lyr3 = res_single_layer(out_ch1, out_ch2, 1, is_last_layer_in_block=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.get_identity(x)
        x = self.blk_lyr1(x)
        x = self.blk_lyr2(x)
        x = self.blk_lyr3(x)

        x += identity

        x = self.relu(x)
        return x


class Resnet_bottleneck(nn.Module):
    def __init__(self,n=152):
        super(Resnet_bottleneck, self).__init__()
        # print(n)
        if n==152:
            ns_layer = [3, 8, 36, 3]
        elif n==101:
            ns_layer = [3, 4, 23, 3]
        elif n == 50:
            ns_layer = [3, 4, 6, 3]


        # Conv 1
        self.conv1 = nn.Conv2d(3, 64, (7, 7), stride=2,padding=3)

        # Conv 2
        layers_conv2 = [nn.MaxPool2d((3, 3), stride=2, padding=1)]
        layers_conv2.append(Res_block(64, 64, 256))
        for _ in range(ns_layer[0]-1):
            out_ch1 = 64
            out_ch2 = 256
            layers_conv2.append(Res_block(256,out_ch1,out_ch2))
        self.conv2 = nn.Sequential(*layers_conv2)

        # Conv 3
        out_ch1 = 128
        out_ch2 = 512
        layers_conv3 = [Res_block(256, out_ch1, out_ch2, is_stride_2=True)]
        for _ in range(ns_layer[1]-1):#7
            layers_conv3.append(Res_block(512, out_ch1, out_ch2))
        self.conv3 = nn.Sequential(*layers_conv3)

        # Conv 4
        out_ch1 = 256
        out_ch2 = 1024
        layers_conv4 =[Res_block(512, out_ch1, out_ch2, is_stride_2=True)]
        for _ in range(ns_layer[2]-1):
            layers_conv4.append(Res_block(1024, out_ch1, out_ch2))
        self.conv4 = nn.Sequential(*layers_conv4)

        # Conv 5
        out_ch1 = 512
        out_ch2 = 2048

        layers_conv5 = [Res_block(1024, out_ch1, out_ch2, is_stride_2=True)]
        for _ in range(ns_layer[3]-1):
            layers_conv5.append(Res_block(2048, out_ch1, out_ch2))
        self.conv5 = nn.Sequential(*layers_conv5)

        self.GAP = nn.AdaptiveAvgPool2d((1,1))

        # Full Connect
        self.fc = nn.Sequential(*[nn.Linear(2048,1000),nn.ReLU()])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.GAP(x)
        x = torch.squeeze(x)
        x = self.fc(x)
        return x




if __name__ == '__main__':
    A = torch.rand([10, 3, 224, 224])

    resnn = Resnet_bottleneck(101)
    B = resnn.forward(A)
    print(B.shape)

    from torchvision.models.resnet import resnet152
    resnn_official = resnet152()
    C = resnn_official.forward(A)
    print(C.shape)