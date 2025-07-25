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
    ):
        super().__init__()

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

        self.down = {}
        in_channels = 64
        next_channels = in_channels * 2

        for num_down_layers in range(4):
            self.down[f"down{num_down_layers}"] = DoubleConv(
                normalization0=nn.GroupNorm(
                    num_groups=in_channels // 4, num_channels=in_channels
                ),
                normalization1=nn.GroupNorm(
                    num_groups=next_channels // 4, num_channels=next_channels
                ),
                ascending=False,
                **{
                    "in_channels": in_channels,
                    "out_channels": next_channels,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "padding_mode": "zeros",
                    "pooling": None,
                },
            )
            in_channels *= 2
            next_channels *= 2

        upconv_options = {"upconv": "bilinear", "scale_factor": 2}

        self.up = {}
        in_channels = 1024
        next_channels = in_channels / 2
        for num_up_layers in range(4):
            self.up[f"up_sample{num_up_layers}"] = UpConv(**upconv_options)
            self.up[f"up_conv{num_up_layers}"] = self.DoubleConv(
                normalization0=nn.GroupNorm(
                    num_groups=in_channels // 4, num_channels=in_channels
                ),
                normalization1=nn.GroupNorm(
                    num_groups=next_channels // 4, num_channels=next_channels
                ),
                ascending=True,
                **{
                    "in_channels": in_channels,
                    "out_channels": next_channels,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "padding_mode": "zeros",
                    "pooling": None,
                },
            )
            in_channels = next_channels
            next_channels /= 2

        self.outc = OutConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            padding_mode="zeros",
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
        pad = (pad_left, pad_right, pad_top, pad_bottom)
        img = F.pad(img, pad)

        in0 = self.inr(self.inl(img))
        x0 = self.down["down0"](in0)
        x1 = self.down["down1"](x0)
        x2 = self.down["down2"](x1)
        x3 = self.down["down3"](x2)
        xup = self.up["up_conv0"](self.up["up_sample0"](x2, x3))
        xup = self.up["up_conv1"](self.up["up_sample1"](x1, xup))
        xup = self.up["up_conv2"](self.up["up_sample2"](x0, xup))
        xup = self.up["up_conv3"](self.up["up_sample3"](in0, xup))
        logits = self.outc(xup)

        # This will crop out any of the padded sections.
        # to improve model updates by not considering unuseful areas (padded areas)
        if any(pad):
            left, right, top, bottom = pad
            logits = logits[
                ..., top : logits.shape[-2] - bottom, left : logits.shape[-1] - right
            ]

        return logits
