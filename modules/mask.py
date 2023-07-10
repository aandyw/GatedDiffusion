""" Learnable Mask with UNet """

import torch
import torch.nn as nn


class Mask(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act_fn: str = "sigmoid"):
        super().__init__()

        self.unet = UNet(in_channels, out_channels)
        self.conv_act = self.get_activation(act_fn)

    def get_activation(self, act_fn: str) -> nn.Module:
        if act_fn == "sigmoid":
            return nn.Sigmoid()
        elif act_fn == "relu":
            return nn.ReLU()
        elif act_fn == "leakyrelu":
            return nn.LeakyReLU()
        elif act_fn in ["swish", "silu"]:
            return nn.SiLU()
        elif act_fn == "mish":
            return nn.Mish()
        elif act_fn == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {act_fn}")

    def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        sample = self.unet(sample)
        out = self.conv_act(sample)
        return out


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Downsampling path
        self.down_conv1 = self.conv_block(in_channels, 64)
        self.down_conv2 = self.conv_block(64, 128)
        self.down_conv3 = self.conv_block(128, 256)
        self.down_conv4 = self.conv_block(256, 512)

        # Upsampling path
        self.up_transpose_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = self.conv_block(512, 256)
        self.up_transpose_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = self.conv_block(256, 128)
        self.up_transpose_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = self.conv_block(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        # Downsampling
        down1 = self.down_conv1(sample)
        down_pool1 = nn.functional.max_pool2d(down1, kernel_size=2, stride=2)
        down2 = self.down_conv2(down_pool1)
        down_pool2 = nn.functional.max_pool2d(down2, kernel_size=2, stride=2)
        down3 = self.down_conv3(down_pool2)
        down_pool3 = nn.functional.max_pool2d(down3, kernel_size=2, stride=2)
        down4 = self.down_conv4(down_pool3)

        # Upsampling
        up1 = self.up_transpose_conv1(down4)
        concat1 = torch.cat([up1, down3], dim=1)
        up_conv1 = self.up_conv1(concat1)
        up2 = self.up_transpose_conv2(up_conv1)
        concat2 = torch.cat([up2, down2], dim=1)
        up_conv2 = self.up_conv2(concat2)
        up3 = self.up_transpose_conv3(up_conv2)
        concat3 = torch.cat([up3, down1], dim=1)
        up_conv3 = self.up_conv3(concat3)

        # Output
        out = self.out_conv(up_conv3)
        return out
