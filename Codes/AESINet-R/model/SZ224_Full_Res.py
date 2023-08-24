import torch
from torch import nn
from model.basic.Attention import *
from model.basic.vgg import VGG
from model.basic.LDAM import *
from model.basic.MFEM import *
from model.basic.DSIM import *
from model.basic.resnet import resnet50_XMZ


class encoder(nn.Module):   
    def __init__(self):
        super(encoder, self).__init__()
        self.vgg = VGG('rgb')

    def forward(self,Xin):
        x1 = self.vgg.conv1(Xin) 
        x2 = self.vgg.conv2(x1)  
        x3 = self.vgg.conv3(x2)  
        x4 = self.vgg.conv4(x3)
        x5 = self.vgg.conv5(x4)
        return x1,x2,x3,x4,x5


class encoderRes(nn.Module):   
    def __init__(self):
        super(encoderRes, self).__init__()
        self.res = resnet50_XMZ(pretrained=1)
        self.conv2=BasicConv2d(256,128,kernel_size=1,stride=1)
        self.conv3=BasicConv2d(512,256,kernel_size=1,stride=1)
        self.conv4=BasicConv2d(1024,512,kernel_size=1,stride=1)
        self.conv5=BasicConv2d(2048,512,kernel_size=1,stride=1)

    def forward(self,Xin):
        x1,x2,x3,x4,x5=self.res(Xin) #128x
        x2=self.conv2(x2)   #56x
        x3=self.conv3(x3)   #28x
        x4=self.conv4(x4)   #14x
        x5=self.conv5(x5)   #7x
        return x1,x2,x3,x4,x5


class enhancer(nn.Module):
    def __init__(self):
        super(enhancer, self).__init__()
        self.o1=LDAModule(64)
        self.o2=LDAModule(128)
        self.o3=LDAModule(256)
        self.e4=MFEModule1(512)
        self.e5=MFEModule2(512)
        self.gcnlayer=DSIModule(1280,512)

    def forward(self,x1,x2,x3,x4,x5):
        xg=self.gcnlayer(x1,x2,x3,x4,x5)
        x5=self.e5(x5,xg)
        x4=self.e4(x4,x5)
        x3=self.o3(x3,x4)
        x2=self.o2(x2,x3)
        x1=self.o1(x1,x2)
        return x1,x2,x3,x4,x5


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

        self.upsample16 =nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8 , mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4 , mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2 , mode='bilinear', align_corners=True)

        self.decoder5 = nn.Sequential(
            BasicConv2d(512, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1),
            nn.Dropout(0.5),
        )
        self.decoder4 = nn.Sequential(
            BasicConv2d(1024, 512, 3, padding=1),
            BasicConv2d(512, 256, 3, padding=1),
            nn.Dropout(0.5),
        )
        self.decoder3 = nn.Sequential(
            BasicConv2d(512, 256, 3, padding=1),
            BasicConv2d(256, 128, 3, padding=1),
            nn.Dropout(0.5),
        )
        self.decoder2 = nn.Sequential(
            BasicConv2d(256, 128, 3, padding=1),
            BasicConv2d(128, 64, 3, padding=1),
            nn.Dropout(0.5),
        )
        self.decoder1 = nn.Sequential(
            BasicConv2d(128, 64, 3, padding=1),
            BasicConv2d(64, 32, 3, padding=1),
            nn.Dropout(0.5),
        )

        self.S5 = nn.Conv2d(512, 1, 3, stride=1, padding=1)
        self.S4 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        self.S3 = nn.Conv2d(128, 1, 3, stride=1, padding=1)
        self.S2 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        self.S1 = nn.Conv2d(32, 1, 3, stride=1, padding=1)

        self.sig = nn.Sigmoid()

    def forward(self, x1, x2, x3, x4, x5):
        x5 = self.decoder5(x5)  # 1/8
        x4 = self.decoder4(torch.cat((x4, self.upsample2(x5) ), 1))  # 1/8
        x3 = self.decoder3(torch.cat((x3, self.upsample2(x4) ), 1))  # 1/4
        x2 = self.decoder2(torch.cat((x2, self.upsample2(x3) ), 1))  # 1/2 
        x1 = self.decoder1(torch.cat((x1, self.upsample2(x2) ), 1))  # 1/1

        s5 = self.S5 (self.upsample2(x5))
        s4 = self.S4 (self.upsample2(x4))
        s3 = self.S3 (self.upsample2(x3))
        s2 = self.S2 (self.upsample2(x2))
        s1 = self.S1 (self.upsample2(x1))

        s1 = self.sig(s1.squeeze(dim=1))
        s2 = self.sig(self.upsample2 (s2).squeeze(dim=1))
        s3 = self.sig(self.upsample4 (s3).squeeze(dim=1))
        s4 = self.sig(self.upsample8 (s4).squeeze(dim=1))
        s5 = self.sig(self.upsample16(s5).squeeze(dim=1))

        return s1,s2,s3,s4,s5


class AESINet(nn.Module):
    def __init__(self,bnum,bnod,dim,loop,bias,res_pretrained):
        super(AESINet,self).__init__()
        self.encoder = encoderRes()
        self.enhancer= enhancer()
        self.decoder = decoder()
        
    def forward(self, Xin):
        x1,x2,x3,x4,x5=self.encoder(Xin)
        x1,x2,x3,x4,x5=self.enhancer(x1,x2,x3,x4,x5)
        s1,s2,s3,s4,s5=self.decoder(x1,x2,x3,x4,x5)
        return s1,s2,s3,s4,s5
