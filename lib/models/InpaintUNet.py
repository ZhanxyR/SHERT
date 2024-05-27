""" Full assembly of the parts to form the complete network """
import logging
import torch

from .unet_parts import *


class InpaintUNet(nn.Module):
    def __init__(self, cfg):
        super(InpaintUNet, self).__init__()

        self.uv_channels = cfg.uv_channels
        self.bilinear = cfg.bilinear
        self.inplanes = 64

        self.inc_uv = DoubleConv(self.uv_channels, self.inplanes)


        self.down1_uv = self._make_layer(Bottleneck, 64 + self.uv_channels, 128, 1, stride=2)
        self.down2_uv = self._make_layer(Bottleneck, 128, 256, 1, stride=2)
        self.down3_uv = self._make_layer(Bottleneck, 256, 512, 1, stride=2)

        factor = 2 if self.bilinear else 1
        self.down4_uv = self._make_layer(Bottleneck, 512, 1024 // factor, 1, stride=2)


        self.up1 = (Up(self.inplanes * 16, self.inplanes * 8 // factor, self.bilinear))
        self.up2 = (Up(self.inplanes * 8, self.inplanes * 4 // factor, self.bilinear))
        self.up3 = (Up(self.inplanes * 4, self.inplanes * 2 // factor, self.bilinear))
        self.up4 = (Up(self.inplanes * 2 + self.uv_channels, self.inplanes, self.bilinear))

        self.outc = nn.Sequential(
            DoubleConv(64, 32),
            DoubleConv(32, 16),
            nn.Conv2d(16, 3, kernel_size=1)
        )

        self.channel_attention_1 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                               nn.Conv2d(64 + self.uv_channels, 64 + self.uv_channels, kernel_size=1),
                                               nn.Sigmoid())

        self.channel_attention_2 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                               nn.Conv2d(64, 64, kernel_size=1),
                                               nn.Sigmoid())

    def _make_layer(self, block, inplanes, outplanes, block_num, stride=1):
        downsample = None
        if stride != 1 or outplanes != inplanes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )

        layers = []
        layers.append(block(inplanes, outplanes, stride, downsample))
        for i in range(1, block_num):
            layers.append(block(outplanes, outplanes))

        return nn.Sequential(*layers)

    def forward(self, features):

        # features torch.Size([2, 15, 1024, 1024]) 

        # torch.Size([2, 64, 1024, 1024])
        uv_fea = self.inc_uv(features)

        # torch.Size([2, 79, 1024, 1024])
        cat_fea = torch.cat([uv_fea, features], dim=1)

        # torch.Size([2, 79, 1, 1])
        attention1 = self.channel_attention_1(cat_fea)

        # torch.Size([2, 79, 1024, 1024])
        cat_fea = cat_fea.mul(attention1)

        # torch.Size([2, 128, 512, 512])
        uv_1 = self.down1_uv(cat_fea) 

        # torch.Size([2, 256, 256, 256])
        uv_2 = self.down2_uv(uv_1)

        # torch.Size([2, 512, 128, 128])
        uv_3 = self.down3_uv(uv_2)

        # torch.Size([2, 1024, 64, 64])
        uv_4 = self.down4_uv(uv_3)

        # torch.Size([2, 256, 128, 128])
        x_up = self.up1(uv_4, uv_3)

        # torch.Size([2, 128, 256, 256])
        x_up = self.up2(x_up, uv_2)

        # torch.Size([2, 64, 512, 512])
        x_up = self.up3(x_up, uv_1)

        # torch.Size([2, 64, 1024, 1024])
        x_up = self.up4(x_up, cat_fea)

        # torch.Size([2, 64, 1, 1])
        attention2 = self.channel_attention_2(x_up)

        # torch.Size([2, 64, 1024, 1024])
        x = x_up.mul(attention2)

        # torch.Size([2, 3, 1024, 1024])
        out = self.outc(x)

        return out
