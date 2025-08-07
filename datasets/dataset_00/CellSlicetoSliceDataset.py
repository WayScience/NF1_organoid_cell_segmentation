import pathlib
from typing import Any, Iterable, Optional, Union

import numpy as np
import tifffile
import torch
from albumentations import ImageOnlyTransform
from PIL import Image
from torch.utils.data import Dataset


class CellSlicetoSliceDataset(Dataset):
    """
    Iterable Slice Dataset for brightfield and masked Images,
    which supports applying transformations to the inputs and targets.
    """

    def __init__(
        self,
        root_data_path: pathlib.Path,
        patient_folders: list[str],
        image_selector: Any,
        image_preprocessor: Any,
        input_transform: Optional[ImageOnlyTransform] = None,
        target_transform: Optional[ImageOnlyTransform] = None,
    ):
        self.root_data_path = root_data_path
        self.patient_folders = patient_folders

        self.__input_transform = input_transform
        self.__target_transform = target_transform

        self.data_paths = self.get_image_paths()
        self.data_slices = image_selector(self.data_paths)
        self.image_preprocessor = image_preprocessor(image_selector, pad_to_multiple=16)

        self.input_max_pixel_value = image_selector.input_max_pixel_value
        self.input_ndim = image_selector.input_ndim
        self.target_ndim = image_selector.target_ndim

        self.split_data = False

    def get_image_paths(self):
        """
        Get the image path of patients.
        """

        root_dir = pathlib.Path(self.root_data_path)

        image_mask_pairs = []

        for patient in self.patient_folders:
            sample_path = root_dir / patient

            brightfield_paths = sample_path.rglob(
                "profiling_input_images/**/*TRANS.tif"
            )

            for bright_path in brightfield_paths:
                mask_path = bright_path.with_name("cell_masks.tiff")

                if mask_path.exists():
                    image_mask_pairs.append(
                        {"input_path": bright_path, "target_path": mask_path}
                    )

        return image_mask_pairs

    def __len__(self):
        return len(self.data_slices)

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
            "Metadata_Input_Slices": self.input_slices,
            "Metadata_Target_Slices": self.target_slices,
            "Metadata_ID": self.id,
        }

    def __getitem__(self, _idx: int):
        """Returns input and target data pairs."""

        self.input_path = self.data_slices[_idx]["input_path"]
        self.target_path = self.data_slices[_idx]["target_path"]
        self.well, self.fov = str(self.input_path.parent).split("/")[-1].split("-")
        self.input_slices = sorted(
            self.data_slices[_idx]["input_slices"], reverse=False
        )
        self.target_slices = sorted(
            self.data_slices[_idx]["target_slices"], reverse=False
        )

        self.patient = str(self.input_path.parents[2]).split("/")[-1]
        input_slice_str = "".join(map(str, self.input_slices))
        target_slice_str = "".join(map(str, self.target_slices))

        self.id = f"{self.patient}{self.well}{self.fov}"

        # Ensure only the data for splitting is returned rather than loading each image
        if self.split_data:
            return {
                "metadata": self.metadata,
                "input_path": self.input_path,
                "target_path": self.target_path,
            }

        input_image = (tifffile.imread(self.input_path).astype(np.float32))[
            self.input_slices
        ]

        target_image = (tifffile.imread(self.target_path).astype(np.float32))[
            self.target_slices
        ]

        self.processing_data = self.image_preprocessor(input_img=input_image)
        input_image = self.processing_data.pop("input_image")

        return {
            "input": input_image,
            "target": target_image,
            "metadata": self.metadata,
            "processing_metadata": self.processing_data,
            "input_path": self.input_path,
            "target_path": self.target_path,
        }
