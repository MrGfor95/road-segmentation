import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
#
# class conv_1to32(nn.modules):
#     def __init__(self,inplanes):
#         super(conv_1to32, self).__init__()
#         self.conv1 = conv3x3(inplanes, 8)
#         self.bn1 = nn.BatchNorm2d(8)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(8,32)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         return out

class noob(nn.Module):
    def __init__(self,numclass=1):
        super(noob,self).__init__()
    #     self.conv=conv_1to32
        resnet = models.resnet34(pretrained=True)
        # self.firstconv = resnet.conv1
        # self.firstbn = resnet.bn1
        # self.firstrelu = resnet.relu
        # self.firstmaxpool = resnet.maxpool
        # self.encoder1 = resnet.layer1
        # self.encoder2 = resnet.layer2
        # self.encoder3 = resnet.layer3
        # self.encoder4 = resnet.layer4

    # def forward(self, x):
    #     x=self.conv(x)
        # xout = self.firstconv(x)
        # xout = self.firstbn(xout)
        # xout = self.firstrelu(xout)
        # xout = self.firstmaxpool(xout)
        # e1 = self.encoder1(xout)
        # e2 = F.upsample(self.encoder2(e1),x.size()[2:])
        # e3 = self.encoder3(e2)
        # e4 = self.encoder4(e3)
        self.firstconv =nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encoder1 = resnet.layer1
        self.conv1=nn.Conv2d(1,512,kernel_size=1)

    def forward(self, x):
        xout = self.firstconv(x)
        xout = self.bn1(xout)
        xout = self.relu(xout)
        xout = self.maxpool(xout)
        xout= self.encoder1(xout)
        down1=F.upsample(x,(32,32),mode='bilinear', align_corners=True)
        conv1=self.conv1(down1)
        return  xout


if __name__ == '__main__':
    device=torch.device("cuda:0")
    input = torch.rand(1, 1, 1024,1024)
    input = input.to(device)
    net = noob()
    net.to(device)
    segout = net(input)
    print(segout.size())
    pass