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
        image_cache_path: Optional[pathlib.Path] = None,
        input_image_name: Optional[str] = None,
        target_image_name: Optional[str] = None,
    ):

        self.dataset = dataset
        self.image_preprocessor = image_preprocessor
        self.image_preprocessor.set_image_specs(**image_specs)

        # Don't load images, just process the metadata
        self.dataset.split_data = True
        self.image_cache_path = image_cache_path

        self.split_data = False
        self.device = self.image_preprocessor.device

        self.input_image_name = (
            "whole_input_image.tiff" if input_image_name is None else input_image_name
        )

        self.target_image_name = (
            "whole_target_image.tiff"
            if target_image_name is None
            else target_image_name
        )

        self.useless_metadata = [
            "Metadata_Input_Slices",
            "Metadata_Target_Slices",
            "Metadata_Crop_Coordinates",
            "Metadata_Sample_ID",
        ]

    def process_load_images(
        self, sample: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Processes and loads images.
        """
        input_image = tifffile.imread(sample["input_path"]).astype(np.float32)

        target_image = tifffile.imread(sample["target_path"]).astype(np.float32)

        self.processing_data = self.image_preprocessor(
            input_img=input_image, target_img=target_image
        )
        return self.processing_data.pop("input_image"), self.processing_data.pop(
            "target_image"
        )

    def __getitem__(self, _idx: int):
        """Returns input and target data pairs."""

        sample = self.dataset[_idx]

        for meta in self.useless_metadata:
            sample["metadata"].pop(meta, None)

        # Ensure only the data for splitting is returned rather than loading each image
        if self.split_data:
            return sample

        cache_dir = None
        if self.image_cache_path is not None:
            cache_dir = self.image_cache_path / sample["metadata"]["Metadata_ID"]

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
            input_image, target_image = self.process_load_images(sample=sample)
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
            "metadata": sample["metadata"],
            "input_path": sample["input_path"],
            "target_path": sample["target_path"],
        }
