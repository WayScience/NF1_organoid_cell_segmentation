import numpy as np
import tifffile


class ZSliceSelector:
    """
    Selects the z-slices to be passed as input to the segmentation model from a selection of modes.
    """
    def __init__(
        self,
        data_slices: int,
        number_of_slices: int,
        mode: str = "nonblack",
        black_threshold: int = 1 / 16,
        stride: int = 1,
    ):
        self.data_slices = data_slices
        self.number_of_slices = number_of_slices
        self.mode = mode
        self.black_threshold = black_threshold
        self.stride = stride

        self.max_pixel_value = np.iinfo(
            tifffile.imread(data_slices[0]).astype(np.float32).dtype
        ).max
        self.neighbors_per_side = self.number_of_slices // 2

        if self.number_of_slices % 2 == 0:
            raise ValueError("The model only accepts an odd number of slices")

        if self.mode == "nonblack":
            self.selector_func = self.select_all_nonblack

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def is_black(self, slice: np.ndarray):
        return np.mean(slice) < self.black_threshold

    def select_all_nonblack(self, img: np.ndarray):
        """
        Select all nonblack z-slice samples efficiently by minimizing computation of slice averages.
        """
        black_zslices = []
        zslice_groups = []

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

            zslice_groups.append(selected_zslices)

        return zslice_groups

    def __call__(self):
        """
        Select slices using the choosen mode.
        """

        data_locations = []
        for img_path in self.data_slices:
            data_locations.append(
                {
                    "path": img_path,
                    "z-slices": self.selector_func(
                        tifffile.imread(img_path).astype(np.float32)
                        / self.max_pixel_value
                    ),
                }
            )
