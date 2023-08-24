import torch
from torch import nn

# 0~1 normalization.
def MaxMinNormalization(x):
    Max = torch.max(x)
    Min = torch.min(x)
    x = torch.div(torch.sub(x, Min), 0.0001 + torch.sub(Max, Min))
    return x


def euc_dist(x):  # x is BNC
    sigma = 1
    t1 = x.unsqueeze(2)  # BN1C
    t2 = x.unsqueeze(1)  # B1NC
    # d = torch.sqrt((t1 - t2).pow(2).sum(3))
    d = (t1 - t2).pow(2).sum(3)
    d = torch.exp(-d/(2*sigma**2))
    return d  # BNN



class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)  # input is BCHW, output is BC11

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # input is BCHW, output is BC11
        out = max_out
        return self.sigmoid(out)


class LayAtt(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LayAtt, self).__init__()
        # self.conv0 = nn.Conv2d(1, 1, 7, padding=3, bias=True)
        self.conv1 = nn.Sequential(BasicConv2d(in_channel, in_channel//2, 3, padding=1),
                                   BasicConv2d(in_channel//2, in_channel, 3, padding=1))
        self.channel = out_channel

    def forward(self, x, y):
        # a = torch.sigmoid(self.conv0(y))  # y is B 1 H W
        a = torch.sigmoid(y)  # y is B 1 H W
        x_att = self.conv1(a.expand(-1, self.channel, -1, -1).mul(x))  # -1 means not changing the size of that dimension
        x = x #+ x_att
        return x

    def initialize(self):
        weight_init(self)

class RA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RA, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
        self.convs = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, 1, 3, padding=1),
        )
        self.channel = out_channel

    def forward(self, x, y):
        a = torch.sigmoid(-y)  # y is B 1 H W
        x = self.relu(self.bn(self.convert(x)))
        x = a.expand(-1, self.channel, -1, -1).mul(x)  # -1 means not changing the size of that dimension
        y = y + self.convs(x)
        return y

    def initialize(self):
        weight_init(self)


class NodeAtt(nn.Module):
    def __init__(self, in_channels):
        super(NodeAtt, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(1 * in_channels, in_channels),
                                 nn.ReLU(),
                                 nn.Linear(in_channels, 1))
        self.lin = nn.Linear(1 * in_channels, in_channels)

    def forward(self, x):  # x has shape [N, 1*in_channels]
        nodeatt = torch.sigmoid(self.mlp(x))  # has shape [N, 1]
        x_out = self.lin(x * nodeatt) + x   # [N, in_channels]
        return x_out