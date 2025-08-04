import pathlib
from collections import defaultdict
from typing import Any

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

        input_example = tifffile.imread(img_paths[0]["input_path"])
        target_example = tifffile.imread(img_paths[0]["target_path"])

        self.input_max_pixel_value = np.iinfo(input_example.dtype).max

        self.input_ndim = input_example.ndim
        self.target_ndim = target_example.ndim

    def is_black(self, slice: np.ndarray) -> bool:
        if self.filter_black_slices:
            return np.mean(slice) < self.black_threshold

        return False

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
                if z_index_neigh in black_zslices or self.is_black(img[z_index_neigh]):
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

    def __call__(
        self, img_paths: list[dict[str, pathlib.Path]]
    ) -> list[dict[str, Any]]:

        self.get_image_specs(img_paths=img_paths)
        data_locations = []

        for img_path in img_paths:
            z_slices_input = self.select_all_nonblack(
                tifffile.imread(img_path["input_path"]).astype(np.float32)
                / self.input_max_pixel_value,
                img_path["input_path"],
            )

            target_img = tifffile.imread(img_path["target_path"])
            target_img = (target_img != 0).astype(np.float32)

            z_slices_target = self.select_all_nonblack(
                target_img,
                img_path["target_path"],
            )

            data_locations.append({"input": z_slices_input, "target": z_slices_target})

        return self.select_symmetric_centered_overlapping_slices(data_locations)
