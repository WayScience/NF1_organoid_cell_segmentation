import pathlib
from collections import defaultdict
from typing import Any

import numpy as np
import tifffile


class NonBlackSliceSelector:
    """
    Selects the z-slices to be passed as input to the segmentation model from a selection of modes.
    """

    def __init__(
        self,
        data_slices: int,
        number_of_slices: int,
        strategy: str = "all_nonblack",
        black_threshold: int = 1 / 16,
        stride: int = 1,
    ):
        self.data_slices = data_slices
        self.number_of_slices = number_of_slices
        self.black_threshold = black_threshold
        self.stride = stride

        self.input_max_pixel_value = np.iinfo(
            tifffile.imread(data_slices[0]["input"]).astype(np.float32).dtype
        ).max

        self.target_max_pixel_value = np.iinfo(
            tifffile.imread(data_slices[0]["target"]).astype(np.float32).dtype
        ).max

        self.neighbors_per_side = self.number_of_slices // 2

        if self.number_of_slices % 2 == 0:
            raise ValueError("The model only accepts an odd number of slices")

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

            select_slice = True
            selected_zslices = []

            for z_index_neigh in range(
                z_index - self.neighbors_per_side,
                z_index + self.neighbors_per_side + 1,
            ):
                if z_index_neigh in black_zslices or self.is_black(img[z_index_neigh]):
                    select_slice = False
                    black_zslices.add(z_index_neigh)
                    break

                selected_zslices.append(z_index_neigh)

            if not select_slice:
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

    def __call__(self) -> list[dict[str, Any]]:
        """Select slices using the chosen mode."""

        data_locations = []
        for img_path in self.data_slices:
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
