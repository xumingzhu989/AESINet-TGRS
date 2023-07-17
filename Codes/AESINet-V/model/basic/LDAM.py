import torch
import torch.nn as nn
import cv2
import math
import torch.nn.functional as F
from model.basic.Attention import *
from model.basic.vig.gcn_lib.torch_vertex import *
from model.basic.DSIM import GraphProcess
eps = math.exp(-10)



class AP_MP(nn.Module):
    def __init__(self,stride=2):
        super(AP_MP,self).__init__()
        self.sz=stride
        self.gapLayer=nn.AvgPool2d(kernel_size=self.sz,stride=self.sz)
        self.gmpLayer=nn.MaxPool2d(kernel_size=self.sz,stride=self.sz)

    #关于数值大小的问题，一个改进思路：可以用softmax平衡权值，然后归一化
    def forward(self,x1,x2):
        B,C,H,W=x1.size()
        apimg=self.gapLayer(x1)
        mpimg=self.gmpLayer(x2)
        byimg=torch.norm(abs(apimg-mpimg),p=2,dim=1,keepdim=True)
        return byimg


class LDAModule(nn.Module):
    def __init__(self,channel):
        super(LDAModule,self).__init__()
        self.channel=channel

        self.conv1=BasicConv2d(channel,channel,3,padding=1)
        self.conv2=BasicConv2d(channel,channel,3,padding=1)

        self.CA1=ChannelAttention(self.channel)
        self.CA2=ChannelAttention(self.channel)
        self.SA1=SpatialAttention()
        self.SA2=SpatialAttention()

        self.glbamp=AP_MP()

        self.conv=BasicConv2d(channel*2+1,channel,kernel_size=1,stride=1)
        
        self.upsample2 = nn.Upsample(scale_factor=2 , mode='bilinear', align_corners=True)
        self.upSA=SpatialAttention()

    def forward(self,x,up):
        x1=self.conv1(x)
        x2=self.conv2(x)
        if(torch.is_tensor(up)):
            x2=x2*self.upSA(self.upsample2(up))+x2
            
        x1=x1+x1*self.CA1(x1)
        x2=x2+x2*self.CA2(x2)

        nx1=x1+x1*self.SA2(x2)
        nx2=x2+x2*self.SA1(x1)

        gamp=self.upsample2(self.glbamp(nx1,nx2))

        res=self.conv(torch.cat([nx1,gamp,nx2],dim=1))

        return res+x
        #可以把GAPGMP生成的图加到后面去，而不是链接
        #注意深层信息传下来的方式。放在两路径的一个分支上？放在最后？
