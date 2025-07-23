import torch
import torch.nn as nn
import torch.nn.functional as F
from UnetLayers import *


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        normobj: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()
        factor = 2 if bilinear else 1

        self.inl = Conv(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
            pooling=None,
        )
        self.inr = Conv(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode="zeros",
            normalization=nn.GroupNorm(num_groups=64 // 4, num_channels=64),
            pooling=None,
        )
        self.down0l = Conv(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
            normalization=nn.GroupNorm(num_groups=128 // 4, num_channels=128),
            pooling=None,
        )  # 774
        self.down0r = Conv(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode="zeros",
            normalization=nn.GroupNorm(num_groups=256 // 4, num_channels=256),
            pooling=None,
        )
        self.down1l = Conv(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
            normalization=nn.GroupNorm(num_groups=256 // 4, num_channels=256),
            pooling=None,
        )
        self.down1r = Conv(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode="zeros",
            normalization=nn.GroupNorm(num_groups=512 // 4, num_channels=512),
            pooling=None,
        )
        self.down2l = Conv(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
            normalization=nn.GroupNorm(num_groups=512 // 4, num_channels=512),
            pooling=None,
        )
        self.down2r = Conv(
            in_channels=1024,
            out_channels=1024,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode="zeros",
            normalization=nn.GroupNorm(num_groups=1024 // 4, num_channels=1024),
            pooling=None,
        )
        self.down3l = Conv(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
            normalization=nn.GroupNorm(num_groups=512 // 4, num_channels=512),
            pooling=None,
        )
        self.down3r = Conv(
            in_channels=1024,
            out_channels=1024,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode="zeros",
            normalization=nn.GroupNorm(num_groups=1024 // 4, num_channels=1024),
            pooling=None,
        )

        upconv_options = {"upconv": "bilinear", "scale_factor": 2}

        self.up0l = UpConv(**upconv_options)
        self.up0m = Conv(
            in_channels=1024,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
            normalization=nn.GroupNorm(num_groups=1024 // 4, num_channels=1024),
            pooling=None,
        )
        self.up0r = Conv(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
            normalization=nn.GroupNorm(num_groups=512 // 4, num_channels=512),
            pooling=None,
        )
        self.up1l = UpConv(**upconv_options)
        self.up1m = Conv(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
            normalization=nn.GroupNorm(num_groups=512 // 4, num_channels=512),
            pooling=None,
        )
        self.up1r = Conv(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
            normalization=nn.GroupNorm(num_groups=256 // 4, num_channels=256),
            pooling=None,
        )
        self.up2l = UpConv(**upconv_options)
        self.up2m = Conv(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
            normalization=nn.GroupNorm(num_groups=256 // 4, num_channels=256),
            pooling=None,
        )
        self.up2r = Conv(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
            normalization=nn.GroupNorm(num_groups=128 // 4, num_channels=128),
            pooling=None,
        )
        self.up3l = UpConv(**upconv_options)
        self.up3m = Conv(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
            normalization=nn.GroupNorm(num_groups=128 // 4, num_channels=128),
            pooling=None,
        )
        self.up3r = Conv(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
            normalization=nn.GroupNorm(num_groups=64 // 4, num_channels=64),
            pooling=None,
        )
        self.up4l = UpConv(**upconv_options)
        self.up4m = Conv(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
            normalization=nn.GroupNorm(num_groups=64 // 4, num_channels=64),
            pooling=None,
        )
        self.up4r = Conv(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
            normalization=nn.GroupNorm(num_groups=1, num_channels=3),
            pooling=None,
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Height and Width must satisfy to ensure no padding
        # needs to be added during concatenation:
        # Height (in pixels) % 2^n = 0, with n levels of downsampling
        # Width (in pixels) % 2^n = 0, with n levels of downsampling
        # n = 4 here, so 2^n = 16
        h, w = img.shape[-2:]
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom))

        in0 = self.inr(self.inl(img))
        x0 = self.down0r(self.down0l(in0))
        x1 = self.down1r(self.down1l(x0))
        x2 = self.down2r(self.down2l(x1))
        x3 = self.down3r(self.down3l(x2))
        xup = self.up0r(self.up0m(self.up0l(x2, x3)))
        xup = self.up1r(self.up1m(self.up1l(x1, xup)))
        xup = self.up2r(self.up2m(self.up2l(x0, xup)))
        xup = self.up3r(self.up3m(self.up3l(in0, xup)))
        logits = self.outc(xup)

        if any(self.pad):
            left, right, top, bottom = self.pad
            logits = logits[
                ..., top : logits.shape[-2] - bottom, left : logits.shape[-1] - right
            ]

        return logits
