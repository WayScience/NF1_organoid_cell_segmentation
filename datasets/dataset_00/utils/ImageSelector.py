import pathlib
from collections import defaultdict
from typing import Any, Union

import numpy as np
import tifffile
import torch


class ImageSelector:
    """
    Selects the z-slices to be passed as input to the segmentation model.
    """

    def __init__(
        self,
        number_of_slices: int = 1,
        slice_stride: int = 1,
        crop_stride: int = 256,
        crop_height: int = 256,
        crop_width: int = 256,
        device: Union[str, torch.device] = "cuda",
    ):
        self.number_of_slices = number_of_slices
        self.slice_stride = slice_stride
        self.crop_stride = crop_stride
        self.crop_height = crop_height
        self.crop_width = crop_width

        self.neighbors_per_side = self.number_of_slices // 2
        self.device = device

        if self.number_of_slices % 2 == 0:
            raise ValueError("The model only accepts an odd number of slices")

    def set_image_specs(
        self,
        input_max_pixel_value: int,
        image_height,
        image_width,
        **kwargs,
    ) -> None:
        self.input_max_pixel_value = input_max_pixel_value
        self.image_height = image_height
        self.image_width = image_width

    def select_zslices(
        self, img: np.ndarray, img_path: pathlib.Path
    ) -> dict[pathlib.Path, list[dict[str, Any]]]:

        zslice_groups = defaultdict(list)

        for z_index in range(
            self.neighbors_per_side,
            img.shape[0] - self.neighbors_per_side,
            self.slice_stride,
        ):

            selected_zslices = []

            for z_index_neigh in range(
                z_index - self.neighbors_per_side,
                z_index + self.neighbors_per_side + 1,
            ):
                selected_zslices.append(z_index_neigh)

            zslice_groups[img_path.parent].append(
                {"file_path": img_path, "z_slices": selected_zslices}
            )

        return zslice_groups

    def select_symmetric_centered_overlapping_slices(
        self,
        data_locations: list[dict[str, dict[pathlib.Path, dict[str, Any]]]],
    ) -> list[dict[str, Any]]:
        """Selects overlapping z-slices from input and target images."""
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
        target_slice: list[int], input_slice: list[int]
    ) -> bool:

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
            is_symmetric(target_slice)
            and is_symmetric(input_slice)
            and target_slice[len(target_slice) // 2]
            == input_slice[len(input_slice) // 2]
            and len(target_slice) <= len(input_slice)
        )

    def generate_crop_coords(self):
        crop_coords = []
        y_starts = list(
            range(0, self.image_height - self.crop_stride + 1, self.crop_stride)
        )
        x_starts = list(
            range(0, self.image_width - self.crop_stride + 1, self.crop_stride)
        )
        for y in y_starts:
            for x in x_starts:
                crop_coords.append(
                    {
                        "height_start": y,
                        "height_end": y + self.crop_height,
                        "width_start": x,
                        "width_end": x + self.crop_width,
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

        self.crop_coords = self.generate_crop_coords()
        data_locations = []

        for img_path in img_paths:
            z_slices_input = self.select_zslices(
                tifffile.imread(img_path["input_path"]).astype(np.float32)
                / self.input_max_pixel_value,
                img_path["input_path"],
            )

            target_img = tifffile.imread(img_path["target_path"])
            target_img = (target_img != 0).astype(np.float32)

            z_slices_target = self.select_zslices(
                target_img,
                img_path["target_path"],
            )

            data_locations.append({"input": z_slices_input, "target": z_slices_target})

        return self.set_crop_coords(
            self.select_symmetric_centered_overlapping_slices(data_locations)
        )
