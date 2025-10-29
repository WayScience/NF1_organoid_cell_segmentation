import pathlib
import tempfile
from typing import Any, Optional

import mlflow
import numpy as np
import tifffile
import torch

from .save_utils import save_image_mlflow


class SaveEpochSlices:
    """
    Saves image crops containing multiple slices
    """

    def __init__(
        self,
        image_dataset: torch.utils.data.Dataset,
        image_postprocessor: Any = lambda x: x,
        image_dataset_idxs: Optional[list[int]] = None,
    ) -> None:

        self.image_dataset = image_dataset
        self.image_dataset_idxs = image_dataset_idxs
        self.crop_key_order = ["height_start", "height_end", "width_start", "width_end"]
        self.image_postprocessor = image_postprocessor

        self.epoch = None
        self.metadata = None

        self.image_dataset_idxs = (
            range(len(image_dataset))
            if image_dataset_idxs is None
            else image_dataset_idxs
        )

    def save_image(
        self,
        image_path: pathlib.Path,
        image_type: str,
        image: torch.Tensor,
    ) -> bool:
        if not ((image > 0.0) & (image < 1.0)).any():
            if image_type == "input":
                raise ValueError("Pixels should be between 0 and 1 in the input image")

        if image_type == "target":
            image = (image != 0).float()

        image = (image * 255).byte().cpu().numpy()

        # Black segmentation masks will not be saved
        if np.max(image) == 0:
            return False

        input_slices_name = "_".join(map(str, self.metadata["Metadata_Input_Slices"]))
        target_slices_name = "_".join(map(str, self.metadata["Metadata_Target_Slices"]))

        crop_name = "_".join(
            str(self.metadata["Metadata_Crop_Coordinates"][k])
            for k in self.crop_key_order
        )

        image_suffix = ".tiff" if ".tif" in image_path.suffix else image_path.suffix

        image_filename = (
            f"3D_{image_type}_{image_path.stem}__{crop_name}__{image_suffix}"
        )

        fov_well_name = image_path.parent.name
        patient_name = image_path.parents[2].name

        save_image_path_folder = f"cropped_images/epoch_{self.epoch:02}/{patient_name}/{fov_well_name}/{input_slices_name}__{target_slices_name}"

        save_image_mlflow(
            image=image,
            save_image_path_folder=save_image_path_folder,
            image_filename=image_filename,
        )

        return True

    def predict_target(
        self, image: torch.Tensor, model: torch.nn.Module
    ) -> torch.Tensor:
        return self.image_postprocessor(model(image.unsqueeze(0)).squeeze(0))

    def __call__(self, model: torch.nn.Module, epoch: int) -> None:
        self.epoch = epoch
        for sample_idx in self.image_dataset_idxs:
            sample = self.image_dataset[sample_idx]
            self.metadata = sample["metadata"]

            sample_image = self.save_image(
                image_path=sample["target_path"],
                image_type="target",
                image=sample["target"],
            )

            # Only save these images if the segmentation mask isn't black
            if sample_image:
                self.save_image(
                    image_path=sample["input_path"],
                    image_type="input",
                    image=sample["input"],
                )

                generated_prediction = self.predict_target(
                    image=sample["input"], model=model
                )

                self.save_image(
                    image_path=sample["target_path"],
                    image_type="generated-prediction",
                    image=generated_prediction,
                )
