import pathlib
from typing import Any, Optional

import numpy as np
import pandas as pd
import tifffile
import torch

from .image_padding_specs import compute_patch_mapping
from .save_utils import save_image_locally, save_image_mlflow


class SaveWholeSlices:
    """
    Saves all 2D images and slices to a 3D tiff format either locally or in MLflow.
    """

    def __init__(
        self,
        image_dataset: torch.utils.data.Dataset,
        image_dataset_idxs: list[int],
        image_specs: dict[str, Any],
        stride,
        crop_shape,
        pad_mode="reflect",
        image_postprocessor: Any = lambda x: x,
        local_save_path: Optional[pathlib.Path] = None,
        **kwargs,
    ):

        self.image_dataset = image_dataset
        self.image_dataset_idxs = image_dataset_idxs
        self.image_specs = image_specs
        self.stride = stride
        self.crop_shape = crop_shape
        self.pad_mode = pad_mode
        self.image_postprocessor = image_postprocessor
        self.local_save_path = local_save_path

        self.unique_image_dataset_idxs = []
        self.reduce_dataset_idxs(image_dataset=image_dataset)

        self.pad_width, self.original_crop_coords = compute_patch_mapping(
            image_specs=image_specs,
            crop_shape=crop_shape,
            stride=stride,
            pad_slices=True,
        )

        self.epoch = None

    def reduce_dataset_idxs(self, image_dataset: torch.utils.data.Dataset):
        """
        For reducing the dataset to only unique indices.
        """
        self.unique_image_dataset_idxs = []

        for sample_idx in self.image_dataset_idxs:
            if (
                image_dataset[sample_idx]["metadata"]["Metadata_ID"]
                not in self.unique_image_dataset_idxs
            ):
                self.unique_image_dataset_idxs.append(sample_idx)

    def predict_target(
        self, padded_image: torch.Tensor, model: torch.nn.Module
    ) -> torch.Tensor:
        """
        padded_image:
            Expects image of shape: (Z, H, W)
            Z -> Number of Z slices
            H -> Image Height
            W -> Image Width
        """

        output = torch.zeros(
            *padded_image.shape,
            dtype=torch.float32,
            device=padded_image.device,
        )
        weight = torch.zeros_like(output)

        spatial_ranges = [
            range(0, s - c, st)
            for s, c, st in zip(padded_image.shape, self.crop_shape, self.stride)
        ]

        for idx in torch.cartesian_prod(
            *[torch.tensor(list(r)) for r in spatial_ranges]
        ):
            start = idx.tolist()
            end = [s + c for s, c in zip(start, self.crop_shape)]

            slices = tuple(slice(s, e) for s, e in zip(start, end))
            crop = padded_image[slices].unsqueeze(0)  # add batch dim

            with torch.no_grad():
                generated_prediction = self.image_postprocessor(
                    generated_prediction=model(crop)
                ).squeeze(0)

            # Accumulate prediction and weights
            output[slices] += generated_prediction
            weight[slices] += 1.0

        output /= weight

        return output[self.original_crop_coords]

    def pad_image(self, input_image: torch.Tensor) -> torch.Tensor:
        """
        input_image:
            Expects image of shape: (Z, H, W)
            Z -> Number of Z slices
            H -> Image Height
            W -> Image Width
        """

        padded_image = np.pad(
            input_image.detach().cpu().numpy(),
            pad_width=self.pad_width,
            mode=self.pad_mode,
        )

        padded_image = torch.from_numpy(padded_image).to(
            dtype=torch.float32, device=input_image.device
        )

        return padded_image

    def save_image(
        self,
        image_path: pathlib.Path,
        image_type: str,
        image: torch.Tensor,
    ) -> bool:
        """
        - Determines if the image is completely black or not.
        - Saves images in the correct format to the hardcoded path.
        """

        if not ((image > 0.0) & (image < 1.0)).any():
            if image_type == "input":
                raise ValueError("Pixels should be between 0 and 1 in the input image")

        if image_type == "target":
            image = (image != 0).float()

        image = (image * 255).byte().cpu().numpy()

        # Black images will not be saved
        if np.max(image) == 0:
            return False

        image_suffix = ".tiff" if ".tif" in image_path.suffix else image_path.suffix

        filename = f"3D_{image_type}_{image_path.stem}{image_suffix}"

        fov_well_name = image_path.parent.name
        patient_name = image_path.parents[2].name

        save_image_path_folder = f"{patient_name}/{fov_well_name}"
        save_image_path_folder = (
            f"whole_image/epoch_{self.epoch:02}/{save_image_path_folder}"
            if self.epoch is not None
            else save_image_path_folder
        )

        if self.local_save_path is None:
            print("Started_save")
            save_image_mlflow(
                image=image,
                save_image_path_folder=save_image_path_folder,
                image_filename=filename,
            )
        else:
            save_image_path_folder = self.local_save_path / save_image_path_folder
            save_image_locally(
                image=image,
                save_image_path_folder=save_image_path_folder,
                image_filename=filename,
            )

        return True

    def __call__(
        self,
        model: torch.nn.Module,
        epoch: Optional[int] = None,
    ) -> None:

        self.epoch = epoch
        for sample_idx in self.unique_image_dataset_idxs:

            sample_image = self.save_image(
                image_path=self.image_dataset[sample_idx]["target_path"],
                image_type="target",
                image=self.image_dataset[sample_idx]["target"],
            )

            # Only save these images if the segmentation mask isn't black
            if sample_image:
                padded_image = self.pad_image(
                    input_image=self.image_dataset[sample_idx]["input"]
                )

                generated_prediction = self.predict_target(
                    padded_image=padded_image, model=model
                )

                self.save_image(
                    image_path=self.image_dataset[sample_idx]["input_path"],
                    image_type="input",
                    image=self.image_dataset[sample_idx]["input"],
                )

                self.save_image(
                    image_path=self.image_dataset[sample_idx]["target_path"],
                    image_type="generated-prediction",
                    image=generated_prediction,
                )
