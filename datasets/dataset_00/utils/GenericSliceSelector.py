import pathlib
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import tifffile


class GenericSliceSelector:
    """
    Selects the z-slices to be passed as input to the segmentation model.
    """

    def __init__(
        self,
        number_of_slices: int,
        filter_black_slices: bool = False,
        stride: int = 1,
        black_threshold: int = 1 / 16,
    ):
        self.number_of_slices = number_of_slices
        self.filter_black_slices = filter_black_slices
        self.black_threshold = black_threshold
        self.stride = stride

        self.neighbors_per_side = self.number_of_slices // 2

        if self.number_of_slices % 2 == 0:
            raise ValueError("The model only accepts an odd number of slices")

    def get_image_specs(self, img_paths: list[dict[str, pathlib.Path]]) -> None:
        """
        Get the max possible pixel value of the input images.
        """
        self.input_max_pixel_value = np.iinfo(
            tifffile.imread(img_paths[0]["input_path"]).astype(np.float32).dtype
        ).max

        self.target_max_pixel_value = np.iinfo(
            tifffile.imread(img_paths[0]["target_path"]).astype(np.float32).dtype
        ).max

    def is_black(self, slice: np.ndarray) -> bool:
        return np.mean(slice) < self.black_threshold

    def select_all_nonblack(
        self, img: np.ndarray, img_path: pathlib.Path
    ) -> dict[pathlib.Path, list[dict[str, Any]]]:
        """
        Select all non-black z-slice samples efficiently by minimizing computation of slice averages.
        """
        black_zslices = set()
        zslice_groups = defaultdict(list)

        for z_index in range(
            self.neighbors_per_side, img.shape[0] - self.neighbors_per_side, self.stride
        ):

            reject_slice = True
            selected_zslices = []

            for z_index_neigh in range(
                z_index - self.neighbors_per_side,
                z_index + self.neighbors_per_side + 1,
            ):
                if (
                    z_index_neigh in black_zslices or self.is_black(img[z_index_neigh])
                ) and self.filter_black_slices:
                    reject_slice = False
                    black_zslices.add(z_index_neigh)
                    break

                selected_zslices.append(z_index_neigh)

            if not reject_slice:
                continue

            zslice_groups[img_path.parent].append(
                {"file_path": img_path, "z_slices": selected_zslices}
            )

        return zslice_groups

    def select_overlapping_slices(
        self,
        data_locations: list[dict[str, dict[pathlib.Path, dict[str, Any]]]],
    ) -> list[dict[str, Any]]:
        """Selects overlapping z-slices from input and target images."""
        overlapping_data = []

        for locations in data_locations:
            for location in locations["target"]:
                target_info = locations["target"][location]
                input_info = locations["input"][location]

                for target_slice in target_info["z_slices"]:
                    for input_slice in input_info["z_slices"]:
                        if self._has_overlap(target_slice, input_slice):
                            overlapping_data.append(
                                {
                                    "input_slices": input_slice,
                                    "input_path": input_info["file_path"],
                                    "target_slices": target_slice,
                                    "target_path": target_info["file_path"],
                                }
                            )
                            break
        return overlapping_data

    @staticmethod
    def _has_overlap(target_slice: list[int], input_slices: list[int]) -> bool:
        t_start, t_end = target_slice
        i_start, i_end = input_slices
        return t_start >= i_start and t_end <= i_end

    def __call__(
        self, img_paths: list[dict[str, pathlib.Path]]
    ) -> list[dict[str, Any]]:
        """Select slices using the chosen mode."""

        self.get_image_specs(img_paths=img_paths)
        data_locations = []

        for img_path in img_paths:
            z_slices_input = self.select_all_nonblack(
                tifffile.imread(img_path["input"]).astype(np.float32)
                / self.input_max_pixel_value,
                img_path["input"],
            )

            z_slices_target = self.select_all_nonblack(
                tifffile.imread(img_path["target"]).astype(np.float32)
                / self.target_max_pixel_value,
                img_path["target"],
            )

            data_locations.append({"input": z_slices_input, "target": z_slices_target})

        return self.select_overlapping_slices(data_locations)
