from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from instance_to_semantic_segmentation import instance_to_semantic_segmentation


class ImagePreProcessor:
    """
    Processes the image prior to model training and inferencing
    """

    def __init__(
        self,
        image_specs: dict[str, Any],
        device: str = "cuda",
        cell_boundary_thickness: Optional[int] = None,
        input_transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
    ):
        self.image_specs = image_specs
        self.device = device
        self.cell_boundary_thickness = cell_boundary_thickness
        self.input_transform = input_transform
        self.target_transform = target_transform

    def set_image_specs(
        self, input_max_pixel_value: int, input_ndim: int, target_ndim: int, **kwargs
    ) -> None:
        self.input_max_pixel_value = input_max_pixel_value
        self.input_ndim = input_ndim
        self.target_ndim = target_ndim

    def format_img(self, img: np.ndarray) -> torch.Tensor:
        """
        Formats an image, with dimensions:
        (C, Z, H, W)

        where:
        C: Number of semantic segmentation masks
        Z: Number of Z-slices
        H: Height of the image (in pixels)
        W: Width of the image (in pixels)
        """

        return torch.from_numpy(img).to(dtype=torch.float32, device=self.device)

    def crop_img(self, img: np.ndarray):
        """
        Images can be the following dimensions:
        (C, Z, H, W)
        where:
        C: Number of semantic segmentation masks
        Z: Number of Z-slices
        H: Height of the image (in pixels)
        W: Width of the image (in pixels)
        """

        if self.cell_boundary_thickness is None:
            return img

        return img[
            :,
            :,
            self.cell_boundary_thickness : -self.cell_boundary_thickness,
            self.cell_boundary_thickness : -self.cell_boundary_thickness,
        ]

    def __call__(
        self, input_img: np.ndarray, target_img: np.ndarray
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int, int, int, int]]:
        """
        Images can be the following dimensions:
        (Z, H, W)
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

        target_img = instance_to_semantic_segmentation(instance_mask=target_img)
        target_img = self.crop_img(img=target_img)

        input_img = self.format_img(input_img)
        target_img = self.format_img(target_img)

        processed_data = {
            "input_image": input_img,
            "target_image": target_img,
        }

        return processed_data
