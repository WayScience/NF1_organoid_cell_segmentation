from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class AllSlicesImagePreProcessor:
    """
    Processes the image prior to model training and inferencing
    """

    def __init__(
        self,
        image_specs: dict[str, Any],
        crop_slices: int,
        crop_height: int,
        crop_width: int,
        device: str = "cuda",
        input_transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        pad_type: str = "reflection",
    ):
        self.device = device
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.pad_type = pad_type

    def format_img(self, img: np.ndarray, img_dims: int) -> torch.Tensor:
        """
        Formats an image base on the number of image dimensions.
        img can be either of the following dimensions:
        (Z, H, W)
        (H, W)
        where:
        Z: Number of Z-slices
        H: Height of the image (in pixels)
        W: Width of the image (in pixels)
        """

        if img_dims != 3:
            raise ValueError(
                f"The number of dimensions in your image should be 2 or 3. It is currently {img_dims}"
            )
        img = torch.from_numpy(img)

        return img.to(dtype=torch.float32, device=self.device)

    def __call__(
        self, input_img: np.ndarray, target_img: np.ndarray = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int, int, int, int]]:
        """
        Both images can be either of the following dimensions:
        (Z, H, W)
        (H, W)
        where:
        Z: Number of Z-slices
        H: Height of the image (in pixels)
        W: Width of the image (in pixels)
        """

        if self.input_transform is not None:
            input_img = self.input_transform(image=input_img)["image"]

        if self.target_transform is not None:
            target_img = self.target_transform(image=target_img)["image"]

        input_img = input_img / self.image_specs["input_max_pixel_value"]

        target_img = (target_img != 0).astype(np.float32)

        input_img = self.format_img(input_img, self.image_specs["input_ndim"])
        target_img = self.format_img(target_img, self.image_specs["target_ndim"])

        processed_data = {
            "input_image": input_img,
            "target_image": target_img,
        }

        return processed_data
