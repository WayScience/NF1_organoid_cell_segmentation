from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class ImagePreProcessor:
    def __init__(
        self,
        image_selector: Any,
        pad_to_multiple: int,
        input_transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
    ):
        self.input_max_pixel_value = image_selector.input_max_pixel_value
        self.input_ndim = image_selector.input_ndim
        self.target_ndim = image_selector.target_ndim
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.pad_to_multiple = pad_to_multiple
        self.device = image_selector.device

    def pad_to(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """
        Pad tensor (C,H,W) so H and W are divisible by pad_to_multiple.
        """

        _, h, w = img.shape
        pad_h = (self.pad_to_multiple - h % self.pad_to_multiple) % self.pad_to_multiple
        pad_w = (self.pad_to_multiple - w % self.pad_to_multiple) % self.pad_to_multiple

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padding = (
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
        )
        img = F.pad(img, padding, mode="constant", value=0)

        return img, padding

    def format_img(self, img: np.ndarray, img_dims: int) -> torch.Tensor:
        """
        Formats an image base on the number of image dimensions.
        """

        if img_dims == 2:
            img = torch.from_numpy(img).unsqueeze(0)

        elif img_dims == 3:
            img = torch.from_numpy(img)

        else:
            raise ValueError(
                f"The number of dimensions in your image should be 2 or 3. It is currently {img_dims}"
            )

        return img.to(dtype=torch.float32, device=self.device)

    def __call__(
        self, input_img: np.ndarray, target_img: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int, int, int, int]]:

        if self.input_transform is not None:
            input_img = self.input_transform(image=input_img)["image"]

        if self.target_transform is not None:
            target_img = self.target_transform(image=target_img)["image"]

        input_img = input_img / self.input_max_pixel_value

        target_img = (target_img != 0).astype(np.float32)

        input_img = self.format_img(input_img, self.input_ndim)
        target_img = self.format_img(target_img, self.target_ndim)

        input_img, input_padding = self.pad_to(input_img)

        processed_data = {
            "input_image": input_img,
            "input_padding": input_padding,
        }

        if target_img is not None:
            processed_data["target_image"] = target_img

        return processed_data
