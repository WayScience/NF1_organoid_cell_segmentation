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
    which support image selection and preprocessing.
    """

    def __init__(
        self,
        root_data_path: pathlib.Path,
        patient_folders: list[str],
        image_selector: Any,
        image_preprocessor: Any,
    ):
        self.root_data_path = root_data_path
        self.patient_folders = patient_folders

        self.data_paths = self.get_image_paths()
        image_specs = self.get_image_specs()
        image_selector.set_image_specs(**image_specs)
        self.data_crops = image_selector(self.data_paths)
        self.image_preprocessor = image_preprocessor
        self.image_preprocessor.set_image_specs(**image_specs)

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

    def get_image_specs(self) -> None:

        input_example = tifffile.imread(self.data_paths[0]["input_path"])
        target_example = tifffile.imread(self.data_paths[0]["target_path"])

        if target_example.ndim == 2:
            image_height, image_width = target_example.shape
        elif target_example.ndim == 3:
            image_height, image_width = (
                target_example.shape[1],
                target_example.shape[2],
            )
        else:
            raise ValueError(f"Unexpected target shape: {target_example.shape}")

        return {
            "input_max_pixel_value": np.iinfo(input_example.dtype).max,
            "input_ndim": input_example.ndim,
            "target_ndim": target_example.ndim,
            "image_height": image_height,
            "image_width": image_width,
        }

    def __len__(self):
        return len(self.data_crops)

    @property
    def metadata(self):
        if self.well is None or self.fov is None or self.patient is None:
            raise ValueError("The metadata are not defined.")
        return {
            "Metadata_Well": self.well,
            "Metadata_FOV": self.fov,
            "Metadata_Patient": self.patient,
            "Metadata_Input_Slices": self.input_slices,
            "Metadata_Target_Slices": self.target_slices,
            "Metadata_Crop_Coordinates": self.crop_coords,
            "Metadata_ID": self.id,
            "Metadata_Dataset_ID": self.dataset_id,
            "Metadata_Sample_ID": self.sample_id,
        }

    def __getitem__(self, _idx: int):
        """Returns input and target data pairs."""

        # For easier retrieval of image data once sampled
        self.dataset_id = _idx

        self.input_path = self.data_crops[_idx]["input_path"]
        self.target_path = self.data_crops[_idx]["target_path"]
        self.well, self.fov = str(self.input_path.parent).split("/")[-1].split("-")
        self.input_slices = sorted(self.data_crops[_idx]["input_slices"], reverse=False)
        self.target_slices = sorted(
            self.data_crops[_idx]["target_slices"], reverse=False
        )
        self.crop_coords = self.data_crops[_idx]["crop_coords"]

        self.patient = str(self.input_path.parents[2]).split("/")[-1]
        input_slice_str = "".join(map(str, self.input_slices))
        target_slice_str = "".join(map(str, self.target_slices))

        # For splitting image data
        # Ensures each fov belongs to a different split
        self.id = f"{self.patient}{self.well}{self.fov}"

        # For sampling images per epoch
        h0, h1 = self.crop_coords["height_start"], self.crop_coords["height_end"]
        w0, w1 = self.crop_coords["width_start"], self.crop_coords["width_end"]
        self.sample_id = f"{self.id}{input_slice_str}{target_slice_str}{h0}{h1}{w0}{w1}"

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

        self.processing_data = self.image_preprocessor(
            input_img=input_image, target_img=target_image
        )
        input_image = self.processing_data.pop("input_image")
        target_image = self.processing_data.pop("target_image")

        # Three dimensions are expected for both the input and target images
        input_image = input_image[
            :,
            self.crop_coords["height_start"] : self.crop_coords["height_end"],
            self.crop_coords["width_start"] : self.crop_coords["width_end"],
        ]

        target_image = target_image[
            :,
            self.crop_coords["height_start"] : self.crop_coords["height_end"],
            self.crop_coords["width_start"] : self.crop_coords["width_end"],
        ]

        return {
            "input": input_image,
            "target": target_image,
            "metadata": self.metadata,
            "processing_metadata": self.processing_data,
            "input_path": self.input_path,
            "target_path": self.target_path,
        }
