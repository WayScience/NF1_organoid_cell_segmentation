import torch
import torch.nn as nn
import torch.nn.functional as F

from .UNetLayers import Conv, DoubleConv, OutConv, UpConv


class UNet(nn.Module):
    """
    Relies on the following properties to be satisfied:
    - Image height mod 16 = 0
    - Image width mod 16 = 0
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.inl = Conv(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
            normalization=nn.BatchNorm2d,
            pooling=None,
        )
        self.inr = Conv(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
            normalization=nn.BatchNorm2d,
            pooling=None,
        )

        self.down = nn.ModuleDict()
        in_ch = 64
        out_ch = in_ch * 2
        encoder_channels = []

        for i in range(4):
            self.down[f"down{i}"] = DoubleConv(
                normalization=nn.BatchNorm2d,
                ascending=False,
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros",
                pooling=None,
            )
            in_ch = out_ch
            encoder_channels.append(out_ch)
            out_ch *= 2

        upconv_options = {"upconv": "bilinear", "scale_factor": 2}

        encoder_channels.sort(reverse=True)
        self.up = nn.ModuleDict()
        self.up_convs = nn.ModuleDict()

        for i in range(4):
            self.up[f"up_sample{i}"] = UpConv(**upconv_options)

        for i, enc_channel in enumerate(encoder_channels):
            in_ch = enc_channel + (enc_channel // 2)
            out_ch = enc_channel // 2
            self.up_convs[f"up_conv{i}"] = DoubleConv(
                normalization=nn.BatchNorm2d,
                ascending=True,
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros",
                pooling=None,
            )

        self.outc = OutConv(
            in_channels=out_ch,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            padding_mode="zeros",
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:

        in0 = self.inr(self.inl(img))  # [B, 64, H/2, W/2]
        x0 = self.down["down0"](in0)  # [B, 128, H/2, W/2]
        x1 = self.down["down1"](x0)  # [B, 256, H/4, W/4]
        x2 = self.down["down2"](x1)  # [B, 512, H/8, W/8]
        x3 = self.down["down3"](x2)  # [B, 1024, H/16, W/16]

        xup = self.up["up_sample0"](x3, x2)
        xup = self.up_convs["up_conv0"](xup)

        xup = self.up["up_sample1"](xup, x1)
        xup = self.up_convs["up_conv1"](xup)

        xup = self.up["up_sample2"](xup, x0)
        xup = self.up_convs["up_conv2"](xup)

        xup = self.up["up_sample3"](xup, in0)
        xup = self.up_convs["up_conv3"](xup)

        logits = self.outc(xup)
        return logits.unsqueeze(2)  # Adds a z-dimension
