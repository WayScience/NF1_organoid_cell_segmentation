import pathlib
from typing import Iterable, Optional

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
        **zslice_selector_configs,
    ):
        self.root_data_path = root_data_path
        self.patient_folders = patient_folders

        self.__input_transform = input_transform
        self.__target_transform = target_transform

        self.data_paths = self.store_fov_patients()
        self.data_slices = ZSliceSelector(self.data_paths, **zslice_selector_configs)

        self.input_max_pixel_value = np.iinfo(
            tifffile.imread(self.data_slices[0]["input_path"]).astype(np.float32).dtype
        ).max

        self.target_max_pixel_value = np.iinfo(
            tifffile.imread(self.data_slices[0]["target_path"]).astype(np.float32).dtype
        ).max

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

        self.patient = self.data_slices[_idx].parents[2]
        self.id = f"{self.patient}{self.well}{self.fov}"

        input_image = (
            tifffile.imread(self.input_path).astype(np.float32)
            / self.input_max_pixel_value
        )

        target_image = (
            tifffile.imread(self.target_path).astype(np.float32)
            / self.target_max_pixel_value
        )

        if self.__input_transform:
            input_image = self.__input_transform(image=input_image)["image"]
            input_image = torch.from_numpy(input_image).unsqueeze(0).float()

        if self.__target_transform:
            target_image = self.__target_transform(image=target_image)["image"]
            target_image = torch.from_numpy(target_image).unsqueeze(0).float()

        return {
            "input": input_image,
            "target": target_image,
            "metadata": self.metadata,
            "input_path": self.input_path,
            "target_path": self.target_path,
        }
