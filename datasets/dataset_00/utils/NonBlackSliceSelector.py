import pathlib
from collections import defaultdict

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

        if self.mode == "all_nonblack":
            self.selector_func = self.select_all_nonblack

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def is_black(self, slice: np.ndarray):
        return np.mean(slice) < self.black_threshold

    def select_all_nonblack(self, img: np.ndarray, img_path: pathlib.Path):
        """
        Select all nonblack z-slice samples efficiently by minimizing computation of slice averages.
        """
        black_zslices = []
        zslice_groups = defaultdict(list)

        for z_index in range(self.neighbors_per_side, img.shape[0], self.stride):

            if z_index + self.neighbors_per_side > img.shape[0] - 1:
                break

            select_slice = True

            for z_index_neigh in range(
                z_index - self.neighbors_per_side, z_index + self.neighbors_per_side, 1
            ):

                selected_zslices = []

                if z_index_neigh in black_zslices or self.is_black(img[z_index_neigh]):
                    select_slice = False
                    black_zslices.append(z_index_neigh)
                    break

                if len(selected_zslices) < 2:
                    selected_zslices.append(z_index_neigh)

            if not select_slice:
                continue

            zslice_groups[img_path.parent].append(
                {"file_path": img_path, "z_slices": selected_zslices}
            )

        return zslice_groups

    def select_overlapping_slices(
        self,
        data_locations: list[
            dict[str, dict[pathlib.Path, list[list[dict[str, int]]]]]
        ],  # Probably need to change this type hint
    ):

        overlapping_data = []

        for locations in data_locations:
            for location in locations["target"].keys():
                for target_slices in locations["target"][location]["z_slices"]:
                    target_overlaps = False

                    for input_slices in locations["input"][location]["z_slices"]:
                        if (
                            target_slices[0] >= input_slices[0]
                            and target_slices[0] <= input_slices[1]
                            and target_slices[1] >= input_slices[0]
                            and target_slices[1] <= input_slices[1]
                        ):
                            target_overlaps = True
                            break

                    if target_overlaps:
                        overlapping_data.append(
                            {
                                "input_slices": input_slices[0],
                                "input_path": locations["input"][location]["file_path"],
                                "target_slices": target_slices[0],
                                "target_path": locations["target"][location][
                                    "file_path"
                                ],
                            }
                        )
        return overlapping_data

    def __call__(self):
        """Select slices using the chosen mode."""

        data_locations = []
        for img_path in self.data_slices:
            z_slices_input = self.selector_func(
                tifffile.imread(img_path["input"]).astype(np.float32)
                / self.input_max_pixel_value,
                img_path["input"],
            )

            z_slices_target = self.selector_func(
                tifffile.imread(img_path["target"]).astype(np.float32)
                / self.target_max_pixel_value,
                img_path["target"],
            )

            data_locations.append({"input": z_slices_input, "target": z_slices_target})

        return self.select_overlapping_slices(data_locations)
