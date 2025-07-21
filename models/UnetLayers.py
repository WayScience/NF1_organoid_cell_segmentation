import torch
import torch.nn as nn
import torch.nn.functional as F


class DownConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        padding_mode: str = "zeros",
        normobj: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            bias=False,
        )

        if normobj in [nn.BatchNorm2d, nn.InstanceNorm2d]:
            self.norm = normobj(out_channels)
        elif normobj == nn.GroupNorm:
            self.norm = normobj(num_groups=1, num_channels=out_channels)
        else:
            raise ValueError("Unsupported normalization layer.")

        self.relu = nn.ReLU(inplace=True)

    def forward(self, imgmap: torch.Tensor) -> torch.Tensor:
        x = self.conv(imgmap)
        x = self.norm(x)
        x = self.relu(x)
        return x


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

    def pad_match(self, encmap, decmap):
        encmap = self.up(encmap)
        diffY = decmap.size()[2] - encmap.size()[2]
        diffX = decmap.size()[3] - encmap.size()[3]

        return F.pad(
            encmap, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )

    def forward(self, encmap, decmap):
        encmap = self.pad_match(encmap, decmap)
        encmap = torch.cat([decmap, encmap], dim=1)
        return self.up(encmap)


class OutConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        padding_mode: str = "zeros",
    ):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )

    def forward(self, convmap):
        return torch.sigmoid(self.conv(convmap))
