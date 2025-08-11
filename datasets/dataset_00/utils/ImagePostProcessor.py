import torch


class ImagePostProcessor:
    def __init__(self, padding: tuple[int, int, int, int]):
        self.pad_left, self.pad_right, self.pad_top, self.pad_bottom = padding

    def __call__(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Remove padding from the image.
        Supports input shapes:
          - (C, H, W)
          - (B, C, H, W)
        """
        if img.ndim == 3:
            # (C, H, W)
            _, h, w = img.shape
            top = self.pad_top
            bottom = h - self.pad_bottom
            left = self.pad_left
            right = w - self.pad_right

            return img[:, top:bottom, left:right]

        elif img.ndim == 4:
            # (B, C, H, W)
            _, _, h, w = img.shape
            top = self.pad_top
            bottom = h - self.pad_bottom
            left = self.pad_left
            right = w - self.pad_right

            return img[:, :, top:bottom, left:right]

        else:
            raise ValueError(
                f"Unsupported input tensor shape {img.shape}. Expected (C,H,W) or (B,C,H,W)."
            )
