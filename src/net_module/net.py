import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_

try:
    from submodules import compact_conv_layer as conv
except:
    from net_module.submodules import compact_conv_layer as conv

import matplotlib.pyplot as plt

class ReExp_Layer(nn.Module):
    '''
    Description:
        A modified exponential layer.
        Only the negative part of the exponential retains.
        The positive part is linear: y=x+1.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        l = nn.ELU() # ELU: max(0,x)+min(0,α∗(exp(x)−1))
        return torch.add(l(x), 1) # assure no negative sigma produces!!!

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, with_batch_norm=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = conv(with_batch_norm, in_channels,  mid_channels, padding=1)
        self.conv2 = conv(with_batch_norm, mid_channels, out_channels, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, with_batch_norm=True):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, with_batch_norm=with_batch_norm)
        )

    def forward(self, x):
        return self.down_conv(x)

class UpBlock(nn.Module): # with skip connection
    def __init__(self, in_channels, out_channels, with_batch_norm=True, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels//2, with_batch_norm=with_batch_norm)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, with_batch_norm=with_batch_norm)

    def forward(self, x1, x2):
        # x1 is the front feature map, x2 is the skip-connection feature map
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class GridEncoder(nn.Module):
    # batch x channel x height x width
    def __init__(self, in_channels, num_classes=1, with_batch_norm=True, axes=None):
        super(GridEncoder,self).__init__()

        self.inc = DoubleConv(in_channels, 64, with_batch_norm=with_batch_norm)
        self.down1 = DownBlock(64, 128, with_batch_norm=with_batch_norm)
        self.down2 = DownBlock(128, 256, with_batch_norm=with_batch_norm)
        self.down3 = DownBlock(256, 512, with_batch_norm=with_batch_norm)
        # self.down4 = DownBlock(512, 512, with_batch_norm=with_batch_norm)
        self.outc = nn.Conv2d(512, num_classes, kernel_size=1)

        self.axes = axes

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        grid = self.outc(x)
        return grid

class GridCodec(nn.Module):
    # batch x channel x height x width
    def __init__(self, in_channels, num_classes=1, with_batch_norm=True, bilinear=True, axes=None):
        super(GridCodec,self).__init__()

        # self.inc = DoubleConv(in_channels, 64, with_batch_norm=with_batch_norm)
        # self.down1 = DownBlock(64, 128, with_batch_norm=with_batch_norm)
        # self.down2 = DownBlock(128, 256, with_batch_norm=with_batch_norm)
        # self.down3 = DownBlock(256, 512, with_batch_norm=with_batch_norm)
        # factor = 2 if bilinear else 1
        # self.down4 = DownBlock(512, 1024 // factor, with_batch_norm=with_batch_norm)
        # self.up1 = UpBlock(1024, 512 // factor, bilinear=bilinear, with_batch_norm=with_batch_norm)
        # self.outc = nn.Conv2d(256, num_classes, kernel_size=1)

        self.inc = DoubleConv(in_channels, 8, with_batch_norm=with_batch_norm)
        self.down1 = DownBlock(8, 16, with_batch_norm=with_batch_norm)
        self.down2 = DownBlock(16, 32, with_batch_norm=with_batch_norm)
        self.down3 = DownBlock(32, 32, with_batch_norm=with_batch_norm)
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(32, 64 // factor, with_batch_norm=with_batch_norm)
        self.up1 = UpBlock(64, 32 // factor, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.outc = nn.Conv2d(16, num_classes, kernel_size=1)

        # self.outl = ReExp_Layer()

        self.axes = axes

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        logits = self.outc(x6)
        # out = self.outl(logits)

        if self.axes is not None:
            for i, ax in enumerate(self.axes.ravel()):
                ax.cla()
                if i == 0:
                    ax.imshow(x[0,-1,:,:].cpu().detach().numpy())
                elif i == 1:
                    ax.imshow(x[0,-2,:,:].cpu().detach().numpy())
                else:
                    ax.imshow(x6[0,i,:,:].cpu().detach().numpy())
            # plt.pause(0.1)

        return logits

class UNet(nn.Module):
    # batch x channel x height x width
    def __init__(self, in_channels, num_classes=1, with_batch_norm=True, bilinear=True, axes=None):
        super(UNet,self).__init__()

        self.inc = DoubleConv(in_channels, 64, with_batch_norm=with_batch_norm)
        self.down1 = DownBlock(64, 128, with_batch_norm=with_batch_norm)
        self.down2 = DownBlock(128, 256, with_batch_norm=with_batch_norm)
        self.down3 = DownBlock(256, 512, with_batch_norm=with_batch_norm)
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(512, 1024 // factor, with_batch_norm=with_batch_norm)
        self.up1 = UpBlock(1024, 512 // factor, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.up2 = UpBlock(512, 256 // factor, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.up3 = UpBlock(256, 128 // factor, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.up4 = UpBlock(128, 64, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

        self.axes = axes

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':

    sample = torch.randn((1,3,200,200))
    net1 = UNet(3)
    net2 = GridEncoder(3)
    out1 = net1(sample)
    out2 = net2(sample)

    print(out1.shape)
    print(out2.shape)

