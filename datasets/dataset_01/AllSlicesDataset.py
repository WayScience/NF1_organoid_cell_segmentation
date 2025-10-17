import pathlib
from typing import Any, Iterable, Optional, Union

import numpy as np
import tifffile
import torch
from albumentations import ImageOnlyTransform
from PIL import Image
from torch.utils.data import Dataset


class AllSlicesDataset(Dataset):
    """
    Iterable Dataset for all slices of brightfield and masked Images,
    which support image selection and preprocessing.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        image_specs: dict[str, Any],
        image_preprocessor: Any,
    ):

        self.dataset = dataset
        self.image_preprocessor = image_preprocessor
        self.image_preprocessor.set_image_specs(**image_specs)

        # Don't load images, just process the metadata
        self.dataset.split_data = True
        self.split_data = False

        self.useless_metadata = [
            "Metadata_Input_Slices",
            "Metadata_Target_Slices",
            "Metadata_Crop_Coordinates",
            "Metadata_Sample_ID",
        ]

    def __getitem__(self, _idx: int):
        """Returns input and target data pairs."""

        sample = self.dataset[_idx]

        for meta in self.useless_metadata:
            sample["metadata"].pop(meta, None)

        # Ensure only the data for splitting is returned rather than loading each image
        if self.split_data:
            return sample

        input_image = tifffile.imread(sample["input_path"]).astype(np.float32)

        target_image = tifffile.imread(sample["target_path"]).astype(np.float32)

        self.processing_data = self.image_preprocessor(
            input_img=input_image, target_img=target_image
        )
        input_image = self.processing_data.pop("input_image")
        target_image = self.processing_data.pop("target_image")

        return {
            "input": input_image,
            "target": target_image,
            "metadata": sample["metadata"],
            "processing_metadata": self.processing_data, # Should include stride and final crop coordinates
            "input_path": sample["input_path"],
            "target_path": sample["target_path"],
        }
