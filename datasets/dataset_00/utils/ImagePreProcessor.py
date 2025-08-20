from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class ImagePreProcessor:
    """
    Processes the image prior to model training and inferencing
    """

    def __init__(
        self,
        pad_to_multiple: int,
        device: str = "cuda",
        input_transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
    ):
        self.pad_to_multiple = pad_to_multiple
        self.device = device
        self.input_transform = input_transform
        self.target_transform = target_transform

    def set_image_specs(
        self, input_max_pixel_value: int, input_ndim: int, target_ndim: int, **kwargs
    ) -> None:
        self.input_max_pixel_value = input_max_pixel_value
        self.input_ndim = input_ndim
        self.target_ndim = target_ndim

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
        self, input_img: np.ndarray, target_img: np.ndarray = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int, int, int, int]]:

        if self.input_transform is not None:
            input_img = self.input_transform(image=input_img)["image"]

        if self.target_transform is not None:
            target_img = self.target_transform(image=target_img)["image"]

        input_img = input_img / self.input_max_pixel_value

        target_img = (target_img != 0).astype(np.float32)

        input_img = self.format_img(input_img, self.input_ndim)
        target_img = self.format_img(target_img, self.target_ndim)

        processed_data = {
            "input_image": input_img,
            "target_image": target_img,
        }

        return processed_data
