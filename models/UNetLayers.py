import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        normalization: nn.Module,
        padding_mode: str = "zeros",
        pooling: nn.Module = None,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            bias=False,
        )

        self.norm = normalization(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pooling = pooling

    def forward(self, imgmap: torch.Tensor) -> torch.Tensor:
        x = self.conv(imgmap)
        x = self.norm(x)
        x = self.relu(x)
        if self.pooling:
            x = self.pooling(x)
        return x


class DoubleConv(nn.Module):
    """
    Can be used in both the encoder and decoder of a unet
    """
    def __init__(
        self,
        normalization: nn.Module,
        ascending: bool,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        padding_mode: str = "zeros",
        pooling: nn.Module = None,
    ):
        super().__init__()

        self.xl = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            normalization=normalization,
            pooling=pooling,
        )

        self.xr = Conv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1 if ascending else 2,
            padding=padding,
            padding_mode=padding_mode,
            normalization=normalization,
            pooling=None,
        )

    def forward(self, tenmap: torch.Tensor) -> torch.Tensor:
        return self.xr(self.xl(tenmap))


class UpConv(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        upconv = kwargs.pop("upconv", None)

        if upconv is None:
            raise ValueError("Missing required argument: upconv")

        if upconv == "bilinear":
            self.up = nn.Upsample(**kwargs)
        else:
            self.up = nn.ConvTranspose2d(**kwargs)

    def pad_match(self, encmap: torch.Tensor, decmap: torch.Tensor) -> torch.Tensor:
        encmap = self.up(encmap)
        diffY = decmap.size()[2] - encmap.size()[2]
        diffX = decmap.size()[3] - encmap.size()[3]

        return F.pad(
            encmap, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )

    def forward(self, encmap: torch.Tensor, decmap: torch.Tensor) -> torch.Tensor:
        encmap = self.pad_match(encmap, decmap)
        encmap = torch.cat([decmap, encmap], dim=1)
        return encmap  # Note: return concatenated map; conv is done in DoubleConv


class OutConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )

    def forward(self, convmap: torch.Tensor) -> torch.Tensor:
        return self.conv(convmap)
