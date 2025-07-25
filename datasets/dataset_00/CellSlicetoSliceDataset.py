import pathlib
from typing import Iterable, Optional, Union

import numpy as np
import tifffile
import torch
from albumentations import ImageOnlyTransform
from PIL import Image
from torch.utils.data import Dataset
from utils.ZSliceSelector import ZSliceSelector


class CellSlicetoSliceDataset(Dataset):
    """
    Iterable Slice Dataset for brightfield and masked Images,
    which supports applying transformations to the inputs and targets.
    """

    def __init__(
        self,
        root_data_path: pathlib.Path,
        patient_folders: list[str],
        input_transform: Optional[ImageOnlyTransform] = None,
        target_transform: Optional[ImageOnlyTransform] = None,
        device: Union[torch.device, str] = "cuda",
        **zslice_selector_configs,
    ):
        self.root_data_path = root_data_path
        self.patient_folders = patient_folders

        self.__input_transform = input_transform
        self.__target_transform = target_transform

        self.device = device
        self.data_paths = self.store_fov_patients()
        self.data_slices = ZSliceSelector(self.data_paths, **zslice_selector_configs)

        input_example = tifffile.imread(self.data_slices[0]["input_path"]).astype(
            np.float32
        )
        target_example = tifffile.imread(self.data_slices[0]["target_path"]).astype(
            np.float32
        )

        self.input_ndim = input_example.ndim
        self.target_ndim = target_example.ndim

        self.input_max_pixel_value = np.iinfo(input_example.dtype).max
        self.target_max_pixel_value = np.iinfo(target_example.dtype).max

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

    def store_fov_patients(self):
        """
        Store the FOV paths of the specified patients using by
        performing a recursive search from the specified root_data_path.
        """

        root_dir = pathlib.Path(self.root_data_path)

        image_mask_pairs = []

        for patient in self.patient_folders:
            sample_path = root_dir / patient

            brightfield_paths = sample_path.rglob(
                "profiling_input_images/**/*TRANS.tif"
            )

            for bright_path in brightfield_paths:
                mask_path = bright_path.with_name(
                    bright_path.with_name("cell_masks.tiff")
                )

                if mask_path.exists():
                    image_mask_pairs.append({"input": bright_path, "target": mask_path})

        return image_mask_pairs

    @property
    def input_transform(self):
        return self.__input_transform

    @property
    def target_transform(self):
        return self.__target_transform

    @property
    def metadata(self):
        if self.well is None or self.fov is None or self.patient is None:
            raise ValueError("The metadata are not defined.")
        return {
            "Metadata_Well": self.well,
            "Metadata_Fov": self.fov,
            "Metadata_Patient": self.patient,
            "Metadata_ID": self.id,
        }

    def __getitem__(self, _idx: int):
        """Returns input and target data pairs."""

        self.input_path = self.data_slices[_idx]["input_path"]
        self.target_path = self.data_slices[_idx]["target_path"]
        self.well, self.fov = self.data_slices[_idx]["input_path"].parent.split("-")
        self.input_slices = self.data_slices["input_slices"]
        self.target_slices = self.data_slices["target_slices"]

        self.patient = self.data_slices[_idx].parents[2]
        input_slice_str = "".join(map(str, sorted(self.input_slices, reverse=False)))
        target_slice_str = "".join(map(str, sorted(self.target_slices, reverse=False)))

        self.id = (
            f"{self.patient}{self.well}{self.fov}{input_slice_str}{target_slice_str}"
        )

        input_image = (
            tifffile.imread(self.input_path).astype(np.float32)
            / self.input_max_pixel_value
        )[self.input_slices]

        target_image = (
            tifffile.imread(self.target_path).astype(np.float32)
            / self.target_max_pixel_value
        )[self.target_slices]

        if self.__input_transform:
            input_image = self.__input_transform(image=input_image)["image"]

        if self.__target_transform:
            target_image = self.__target_transform(image=target_image)["image"]

        input_image = self.format_img(input_image, self.input_ndim)
        target_image = self.format_img(target_image, self.target_ndim)

        return {
            "input": input_image,
            "target": target_image,
            "metadata": self.metadata,
            "input_path": self.input_path,
            "target_path": self.target_path,
        }
