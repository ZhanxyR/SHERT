import torch
from .unet_parts import *
from lib.tools.feature_projection import index
class RefineUNet(nn.Module):
    def __init__(self, cfg):

        super(RefineUNet, self).__init__()

        self.uv_channels = cfg.uv_channels
        self.image_channels = cfg.image_channels
        self.normal_channels = cfg.normal_channels
        self.out_channels = cfg.out_channels

        # self.bilinear = False
        self.inplanes = 64

        self.down1_uv = self._make_layer(Bottleneck, 38, 64, 1, stride=2)
        self.down2_uv = self._make_layer(Bottleneck, 64, 128, 1, stride=2)
        self.down3_uv = self._make_layer(Bottleneck, 128, 256, 1, stride=2)
        self.down4_uv = self._make_layer(Bottleneck, 256, 512, 1, stride=2)

        self.down1_img = self._make_layer(Bottleneck, self.image_channels + self.normal_channels, 32, 1, stride=2)
        self.down2_img = self._make_layer(Bottleneck, 32, 64, 1, stride=2)
        self.down3_img = self._make_layer(Bottleneck, 64, 128, 1, stride=2)
        # factor = 2 if self.bilinear else 1
        self.down4_img = self._make_layer(Bottleneck, 128, 256, 1, stride=2)

        self.up1_img = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                       nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                       nn.BatchNorm2d(128)
                    )
        self.up2_img = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                       nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
                       nn.BatchNorm2d(64)
                    )
        self.up3_img = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                       nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                       nn.BatchNorm2d(32)
                    )
        self.up4_img = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                       nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
                       nn.BatchNorm2d(16)
                    )

        self.up1_uv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                       nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
                       nn.BatchNorm2d(256)
                    )
        self.up2_uv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                       nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                       nn.BatchNorm2d(128)
                    )
        self.up3_uv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                       nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
                       nn.BatchNorm2d(64)
                    )
        self.up4_uv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                       nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                       nn.BatchNorm2d(32)
                    )

        self.outc = nn.Sequential(
            nn.Conv2d(70, 64, kernel_size=1),
            DoubleConv(64, 32),
            DoubleConv(32, 16),
            nn.Conv2d(16, self.out_channels, kernel_size=1)
        )

        self.channel_attention_1_img = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                               nn.Conv2d(64, 64, kernel_size=1),
                                               nn.Sigmoid())


        self.spatialAttention = SpatialAttentionModule()

        self.channel_attention_1_uv = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(38, 38, kernel_size=1), nn.Sigmoid())

        self.channel_attention_2_uv = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(70, 70, kernel_size=1), nn.Sigmoid())

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

    def forward(self, uvs, images, normals, xy, z):
        bs, _, h, w = images.shape

        # torch.Size([2, 32, 512, 512])
        img_1 = self.down1_img(torch.cat([images, normals], dim=1))
        # torch.Size([2, 64, 256, 256])
        img_2 = self.down2_img(img_1)

        # # torch.Size([2, 32, 512, 512])
        # img_1 = self.down1_img(images)
        # # torch.Size([2, 64, 256, 256])
        # img_2 = self.down2_img(torch.cat([img_1, normals], dim=1))

        img_attention = self.channel_attention_1_img(img_2)
        img_2 = img_2.mul(img_attention)

        # torch.Size([2, 128, 128, 128])
        img_3 = self.down3_img(img_2)

        # torch.Size([2, 256, 64, 64])
        img_4 = self.down4_img(img_3)


        # torch.Size([2, 128, 128, 128])
        img_up_1 = self.up1_img(img_4) + img_3

        # torch.Size([2, 64, 256, 256])
        img_up_2 = self.up2_img(img_up_1) + img_2

        # torch.Size([2, 32, 512, 512])
        img_up_3 = self.up3_img(img_up_2) + img_1

        # torch.Size([2, 16, 1024, 1024])
        img_up_4 = self.up4_img(img_up_3)

        # torch.Size([2, 25, 1024, 1024])
        img_feature = torch.cat([img_up_4, images, normals], axis=1)

        # torch.Size([2, 26, 1024, 1024])
        img_up_to_uv = index(img_feature, xy).reshape(bs, -1, h, w)
        img_up_to_uv = torch.cat([img_up_to_uv, z.reshape(bs, -1, h, w) * 2], dim=1)

        # # torch.Size([2, 38, 1024, 1024])
        cat_fea_uv = torch.cat([img_up_to_uv, uvs], dim=1)

        # # torch.Size([2, 41, 1, 1])
        attention1 = self.channel_attention_1_uv(cat_fea_uv)

        # torch.Size([2, 41, 1024, 1024])
        cat_fea_uv = cat_fea_uv.mul(attention1)

        # torch.Size([2, 64, 512, 512])
        uv_1 = self.down1_uv(cat_fea_uv)

        # torch.Size([2, 128, 256, 256])
        uv_2 = self.down2_uv(uv_1)

        # torch.Size([2, 256, 128, 128])
        uv_3 = self.down3_uv(uv_2)

        # torch.Size([2, 512, 64, 64])
        uv_4 = self.down4_uv(uv_3)

        # # torch.Size([2, 256, 128, 128])
        x_up = self.up1_uv(uv_4) + uv_3

        # # torch.Size([2, 128, 256, 256])
        x_up = self.up2_uv(x_up) + uv_2

        # # torch.Size([2, 64, 512, 512])
        x_up = self.up3_uv(x_up) + uv_1

        # # torch.Size([2, 32, 1024, 1024])
        x_up = self.up4_uv(x_up)

        # # torch.Size([2, 70, 1024, 1024])
        uv_img_cat = torch.cat([x_up, cat_fea_uv], dim=1)

        # torch.Size([2, 70, 1, 1])
        attention2 = self.channel_attention_2_uv(uv_img_cat)

        # # torch.Size([2, 70, 1024, 1024])
        x = uv_img_cat.mul(attention2)

        spatial_attention = self.spatialAttention(x)

        x = x.mul(spatial_attention)

        # # torch.Size([2, 3, 1024, 1024])
        out = self.outc(x)

        return out
