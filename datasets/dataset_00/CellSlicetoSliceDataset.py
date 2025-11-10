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
        image_paths: list[pathlib.Path],
        image_specs: dict[str, Any],
        image_selector: Any,
        image_preprocessor: Any,
        image_cache_path: Optional[pathlib.Path] = None,
        input_image_name: Optional[str] = None,
        target_image_name: Optional[str] = None,
    ):

        image_selector.set_image_specs(**image_specs)
        self.data_crops = image_selector(image_paths)
        self.image_preprocessor = image_preprocessor
        self.image_preprocessor.set_image_specs(**image_specs)

        self.image_cache_path = image_cache_path

        self.split_data = False

        self.input_image_name = (
            "input_image.tiff" if input_image_name is None else input_image_name
        )

        self.target_image_name = (
            "target_image.tiff" if target_image_name is None else target_image_name
        )

        self.device = self.image_preprocessor.device

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

    def process_load_images(self):
        """
        Processes and loads images.
        """

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
        return (
            input_image[
                :,
                self.crop_coords["height_start"] : self.crop_coords["height_end"],
                self.crop_coords["width_start"] : self.crop_coords["width_end"],
            ],
            # Target images also have semantic channels (C,Z,H,W)
            target_image[
                :,
                :,
                self.crop_coords["height_start"] : self.crop_coords["height_end"],
                self.crop_coords["width_start"] : self.crop_coords["width_end"],
            ],
        )

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

        cache_dir = None
        if self.image_cache_path is not None:
            cache_dir = self.image_cache_path / self.sample_id

        if cache_dir is not None and cache_dir.exists():
            # Load from cache
            input_image = torch.from_numpy(
                tifffile.imread(cache_dir / self.input_image_name)
            ).to(device=self.device)

            target_image = torch.from_numpy(
                tifffile.imread(cache_dir / self.target_image_name)
            ).to(device=self.device)

        else:
            # Load images
            input_image, target_image = self.process_load_images()
            if cache_dir is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(
                    cache_dir / self.input_image_name, input_image.cpu().numpy()
                )
                tifffile.imwrite(
                    cache_dir / self.target_image_name, target_image.cpu().numpy()
                )

        return {
            "input": input_image,
            "target": target_image,
            "metadata": self.metadata,
            "processing_metadata": self.processing_data,
            "input_path": self.input_path,
            "target_path": self.target_path,
        }
