import pathlib
import tempfile
from typing import Optional

import mlflow
import numpy as np
import tifffile
import torch


class SaveEpochSlices:

    def __init__(self, image_dataset_idxs: list[int], data_split: str) -> None:

        self.image_dataset_idxs = image_dataset_idxs
        self.data_split = data_split
        self.crop_key_order = ["height_start", "height_end", "width_start", "width_end"]

    def save_image_mlflow(
        self,
        image: torch.Tensor,
        save_image_path_folder: pathlib.Path,
        image_filename: str,
    ) -> None:

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = pathlib.Path(tmp_dir) / image_filename
            tifffile.imwrite(save_path, image.astype(np.uint8))

            mlflow.log_artifact(
                local_path=save_path, artifact_path=save_image_path_folder
            )

    def save_image(
        self,
        image_path: pathlib.Path,
        image_type: str,
        image: torch.Tensor,
    ) -> None:
        if not ((image > 0.0) & (image < 1.0)).any():
            if image_type == "input":
                raise ValueError("Pixels should be between 0 and 1 in the input image")
            image = (image != 0).float()

        image = (image * 255).byte().cpu().numpy()

        input_slices_name = "_".join(map(str, self.metadata["Metadata_Input_Slices"]))
        target_slices_name = "_".join(map(str, self.metadata["Metadata_Target_Slices"]))

        crop_name = "_".join(
            str(self.metadata["Metadata_Crop_Coordinates"][k])
            for k in self.crop_key_order
        )

        filename = (
            f"{image_path.name}_{image_type}"
            if image_type == "generated_prediction"
            else image_path.name
        )

        image_filename = f"{crop_name}__{filename}"

        fov_well_name = image_path.parent.name
        patient_name = image_path.parents[2].name

        save_image_path_folder = pathlib.Path(
            f"{patient_name}/{fov_well_name}/{input_slices_name}__{target_slices_name}"
        )

        self.save_image_mlflow(
            image=image,
            save_image_path_folder=save_image_path_folder,
            image_filename=image_filename,
        )

    def predict_target(self, image: torch.Tensor, model: torch.nn.Module):
        return model(image.unsqueeze(0)).squeeze(0)

    def __call__(
        self, dataset: torch.utils.data.Dataset, model: torch.nn.Module
    ) -> None:
        for sample_idx in self.image_dataset_idxs:
            sample = dataset[sample_idx]
            self.metadata = sample["metadata"]

            self.save_image(
                image_path=sample["input_path"],
                image_type="input",
                image=sample["input"],
            )

            self.save_image(
                image_path=sample["target_path"],
                image_type="target",
                image=sample["target"],
            )

            generated_prediction = self.predict_target(
                image=sample["input"], model=model
            )

            self.save_image(
                image_path=sample["target_path"],
                image_type="target",
                image=generated_prediction,
            )
