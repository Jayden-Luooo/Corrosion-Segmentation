import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        if bilinear: 
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(
            x,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        x = torch.cat([skip, x], dim=1) # skip connection
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=64, bilinear=False):
        super().__init__()
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # bottleneck 前面的 encoder 负责压缩图片，bottleneck 负责理解全局信息，后面的 decoder 负责恢复 mask 尺寸

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)





from torchvision.models import resnet34, ResNet34_Weights


class ResNetDecoderBlock(nn.Module):
    def __init__(self, decoder_channels, skip_channels, out_channels, dropout=0.3):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            decoder_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = nn.Sequential(
            DoubleConv(
                in_channels=out_channels + skip_channels,
                out_channels=out_channels,
            ),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = F.pad(
            x,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class FinalDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
            ),
            DoubleConv(out_channels, out_channels),
            nn.Dropout2d(p=0.2),
        )

    def forward(self, x):
        return self.block(x)


class ResNet34UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base_channels=64,
        bilinear=False,
        pretrained=True,
    ):
        super().__init__()

        if pretrained:
            weights = ResNet34_Weights.IMAGENET1K_V1
        else:
            weights = None

        backbone = resnet34(weights=weights)

        # -------------------------
        # ResNet34 encoder
        # -------------------------
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )

        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1 # residual block 1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4 # residual block 4
        # -------------------------
        # U-Net decoder
        # ResNet34 feature channels:
        # stem:   64
        # layer1: 64
        # layer2: 128
        # layer3: 256
        # layer4: 512
        # -------------------------
        self.decoder4 = ResNetDecoderBlock(
            decoder_channels=512,
            skip_channels=256,
            out_channels=256,
        )

        self.decoder3 = ResNetDecoderBlock(
            decoder_channels=256,
            skip_channels=128,
            out_channels=128,
        )

        self.decoder2 = ResNetDecoderBlock(
            decoder_channels=128,
            skip_channels=64,
            out_channels=64,
        )

        self.decoder1 = ResNetDecoderBlock(
            decoder_channels=64,
            skip_channels=64,
            out_channels=64,
        )

        self.final_decoder = FinalDecoderBlock(
            in_channels=64,
            out_channels=32,
        )

        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # input: [B, 3, 512, 512]

        x0 = self.stem(x)          # [B, 64, 256, 256]

        x1 = self.maxpool(x0)      # [B, 64, 128, 128]
        x1 = self.layer1(x1)       # [B, 64, 128, 128]

        x2 = self.layer2(x1)       # [B, 128, 64, 64]
        x3 = self.layer3(x2)       # [B, 256, 32, 32]
        x4 = self.layer4(x3)       # [B, 512, 16, 16]

        x = self.decoder4(x4, x3)  # [B, 256, 32, 32]
        x = self.decoder3(x, x2)   # [B, 128, 64, 64]
        x = self.decoder2(x, x1)   # [B, 64, 128, 128]
        x = self.decoder1(x, x0)   # [B, 64, 256, 256]

        x = self.final_decoder(x)  # [B, 32, 512, 512]

        return self.outc(x)        # [B, 1, 512, 512]


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, masks):
        bce_loss = self.bce(logits, masks)

        probs = torch.sigmoid(logits)
        probs = probs.flatten(start_dim=1)
        masks = masks.flatten(start_dim=1)

        intersection = (probs * masks).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + masks.sum(dim=1) + self.smooth
        )

        dice_loss = 1.0 - dice.mean()
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
