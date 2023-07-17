from re import X
import torch
from torch import nn, tensor
import torch.nn.functional as F
from model.basic.Attention import *
from model.basic.DSIM import GraphProcess


class MFEModule(nn.Module):
    def __init__(self,channel):
        super(MFEModule, self).__init__()
        self.upSA=SpatialAttention()
        self.upsample2 = nn.Upsample(scale_factor=2 , mode='bilinear', align_corners=True)
        self.downsample2=nn.MaxPool2d(2,stride=2)

        self.conv_mid1=BasicConv2d(channel, channel//8 , 3, padding=1, dilation=1)
        self.conv_mid2=BasicConv2d(channel, channel//8 , 3, padding=2, dilation=2)
        self.conv_mid3=BasicConv2d(channel, channel//8 , 3, padding=3, dilation=3)
        self.conv_mid4=BasicConv2d(channel, channel//8 , 3, padding=4, dilation=4)

        self.conv_up1=BasicConv2d(channel, channel//8 , 3, padding=1, dilation=1)
        self.conv_up2=BasicConv2d(channel, channel//8 , 3, padding=3, dilation=3)

        self.conv_down1=BasicConv2d(channel, channel//8 , 3, padding=1, dilation=1)
        self.conv_down2=BasicConv2d(channel, channel//8 , 3, padding=3, dilation=3)
        
        self.SA=SpatialAttention()
        self.CA=ChannelAttention(channel)

    def forward(self,x,up):
        x_mid1=self.conv_mid1(x)
        x_mid2=self.conv_mid2(x)
        x_mid3=self.conv_mid3(x)
        x_mid4=self.conv_mid4(x)

        x_up1=self.downsample2(self.conv_up1(self.upsample2(x)))
        x_up2=self.downsample2(self.conv_up2(self.upsample2(x)))

        x_down1=self.upsample2(self.conv_down1(self.downsample2(x)))
        x_down2=self.upsample2(self.conv_down2(self.downsample2(x)))

        x_mix=torch.cat((x_mid1,x_mid2,x_mid3,x_mid4,x_up1,x_up2,x_down1,x_down2),dim=1)

        x_mix=x_mix*self.CA(x_mix)+x_mix
        x_mix=x_mix*self.SA(x_mix)+x_mix

        if(torch.is_tensor(up)):
            x_mix=x_mix*self.upSA(self.upsample2(up))+x_mix

        x_mix=x_mix+x

        return x_mix





