import pathlib
import tempfile

import mlflow
import numpy as np
import tifffile
import torch


def save_image_mlflow(
    image: torch.Tensor,
    save_image_path_folder: str,
    image_filename: str,
) -> None:

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = pathlib.Path(tmp_dir) / image_filename
        tifffile.imwrite(save_path, image.astype(np.uint8))

        mlflow.log_artifact(local_path=save_path, artifact_path=save_image_path_folder)


def save_image_locally(
    image: torch.Tensor,
    save_image_path_folder: str,
    image_filename: str,
) -> None:

    save_image_path_folder = pathlib.Path(save_image_path_folder)
    save_image_path_folder.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(save_image_path_folder / image_filename, image.astype(np.uint8))
