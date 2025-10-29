import pathlib
from collections import defaultdict
from typing import Any, Optional, Union

import numpy as np
import tifffile
import torch


class ImageSelector:
    """
    Selects the z-slices and crops to be passed as input to the segmentation model.
    """

    def __init__(
        self,
        input_crop_shape: tuple[int],
        target_crop_shape: tuple[int],
        image_specs: Optional[int] = None,
        slice_stride: int = 1,
        crop_stride: int = 256,
        device: Union[str, torch.device] = "cuda",
    ):
        """
        input_crop_shape:
            Expects the crop shape to be: (Z, H, W)
            Z -> Number of Z slices
            H -> Image Height
            W -> Image Width

        target_crop_shape:
            Expects image of shape: (Z, H, W)
            Z -> Number of Z slices
            H -> Image Height
            W -> Image Width

        slice_stride: int
            Number of slices to move by when selecting images.

        crop_stride: int
            Size of the the stride in both height and width to move by when selecting images.
        """

        if len(input_crop_shape) != 3 or len(target_crop_shape) != 3:
            raise ValueError(
                "The input and target must both have three dimensions of (z-slices, height, width)"
            )

        for dimension in range(1, 3):
            if input_crop_shape[dimension] != target_crop_shape[dimension]:
                raise ValueError(
                    "The height and width of both the input and target crops must be equal."
                )

        self.input_crop_shape = input_crop_shape
        self.target_crop_shape = target_crop_shape

        self.crop_margin = 0 if image_specs is None else image_specs["crop_margin"]

        self.slice_stride = slice_stride
        self.crop_stride = crop_stride

        self.input_neighbors_per_side = self.input_crop_shape[0] // 2
        self.target_neighbors_per_side = self.target_crop_shape[0] // 2
        self.device = device

        if self.input_crop_shape[0] % 2 == 0:
            raise ValueError("The model only accepts an odd number of slices")

    def set_image_specs(
        self,
        input_max_pixel_value: int,
        image_shape: tuple[int],
        **kwargs,
    ) -> None:
        self.image_shape = image_shape
        self.input_max_pixel_value = input_max_pixel_value

    """
    image_shape:
        Expects image of shape: (Z, H, W)
        Z -> Number of Z slices
        H -> Image Height
        W -> Image Width
    """

    def select_zslices(
        self, img: np.ndarray, img_path: pathlib.Path, neighbors_per_side: int
    ) -> dict[pathlib.Path, list[dict[str, Any]]]:
        """
        Expects image of shape: (Z, H, W)
        Z -> Number of Z slices
        H -> Image Height
        W -> Image Width
        """

        zslice_groups = defaultdict(list)

        for z_index in range(
            neighbors_per_side,
            img.shape[0] - neighbors_per_side,
            self.slice_stride,
        ):

            selected_zslices = []

            for z_index_neigh in range(
                z_index - neighbors_per_side,
                z_index + neighbors_per_side + 1,
            ):
                selected_zslices.append(z_index_neigh)

            zslice_groups[img_path.parent].append(
                {"file_path": img_path, "z_slices": selected_zslices}
            )

        return zslice_groups

    def filter_symmetric_centered_overlapping_slices(
        self,
        data_locations: list[dict[str, dict[pathlib.Path, dict[str, Any]]]],
    ) -> list[dict[str, Any]]:
        """
        Filters by overlap and symmetry of z-slices from input and target images.
        """

        overlapping_data = []

        for locations in data_locations:
            for location in locations["target"]:
                target_info = locations["target"][location]
                input_info = locations["input"][location]

                for target_slice_data in target_info:
                    for input_slice_data in input_info:
                        if self._has_centered_symmetric_overlap(
                            target_slice_data["z_slices"], input_slice_data["z_slices"]
                        ):
                            overlapping_data.append(
                                {
                                    "input_slices": input_slice_data["z_slices"],
                                    "input_path": input_slice_data["file_path"],
                                    "target_slices": target_slice_data["z_slices"],
                                    "target_path": target_slice_data["file_path"],
                                }
                            )
                            break
        return overlapping_data

    @staticmethod
    def _has_centered_symmetric_overlap(
        target_slices: list[int], input_slices: list[int]
    ) -> bool:
        """
        Determines if the target and input slice are symmetric, and
        if the target slices are centered with the input slices, where
        number of target slices <= number of input slices.

        We may care about inferencing on slices present in the target image
        that are not present in the input image because a cell may span
        multiple slices. Therefore, the model may leverage this this information
        to potentially improve segmentation performance.
        """

        def is_symmetric(slices: list[int]) -> bool:
            if len(slices) % 2 == 0:
                return False
            mid = len(slices) // 2
            center = slices[mid]
            left = slices[:mid]
            right = slices[mid + 1 :]
            expected_left = [center - i for i in range(1, mid + 1)][::-1]
            expected_right = [center + i for i in range(1, mid + 1)]
            return left == expected_left and right == expected_right

        return (
            is_symmetric(target_slices)
            and is_symmetric(input_slices)
            and target_slices[len(target_slices) // 2]
            == input_slices[len(input_slices) // 2]
            and len(target_slices) <= len(input_slices)
        )

    def generate_crop_coords(self):
        """
        Naive crop selection
        """

        crop_coords = []
        y_starts = list(
            range(
                0,
                self.image_shape[1]
                - (2 * self.crop_margin)  # Both sides of the images were removed
                - self.crop_stride
                + 1,
                self.crop_stride,
            )
        )
        x_starts = list(
            range(
                0,
                self.image_shape[2] - (2 * self.crop_margin) - self.crop_stride + 1,
                self.crop_stride,
            )
        )
        for y in y_starts:
            for x in x_starts:
                crop_coords.append(
                    {
                        "height_start": y,
                        "height_end": y + self.input_crop_shape[1],
                        "width_start": x,
                        "width_end": x + self.input_crop_shape[2],
                    }
                )
        return crop_coords

    def set_crop_coords(self, data_slices: list[dict[str, Any]]):
        data_crops = []
        for sample_idx in range(len(data_slices)):
            base_data = data_slices[sample_idx]
            for crop in self.crop_coords:
                data_crop = base_data.copy()
                data_crop["crop_coords"] = crop
                data_crops.append(data_crop)

        return data_crops

    def __call__(
        self, img_paths: list[dict[str, pathlib.Path]]
    ) -> list[dict[str, Any]]:
        """
        Orchestrates image, z-slice, and crop selection
        """

        self.crop_coords = self.generate_crop_coords()
        data_locations = []

        for img_path in img_paths:
            z_slices_input = self.select_zslices(
                tifffile.imread(img_path["input_path"]).astype(np.float32)
                / self.input_max_pixel_value,
                img_path["input_path"],
                neighbors_per_side=self.input_neighbors_per_side,
            )

            target_img = tifffile.imread(img_path["target_path"])
            target_img = (target_img != 0).astype(np.float32)

            z_slices_target = self.select_zslices(
                target_img,
                img_path["target_path"],
                neighbors_per_side=self.target_neighbors_per_side,
            )

            data_locations.append({"input": z_slices_input, "target": z_slices_target})

        return self.set_crop_coords(
            self.filter_symmetric_centered_overlapping_slices(data_locations)
        )
