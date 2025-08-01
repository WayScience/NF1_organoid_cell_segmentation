import pathlib
import tempfile
from typing import Optional, Sequence, Union

import mlflow
import numpy as np
import tifffile
import torch
from utils.SampleImages import SampleImages


class SaveEpochSlices:

    def __init__(self, sample_images_obj: SampleImages, data_split: str) -> None:

        self.sampled_images = sample_images_obj()
        self.data_split = data_split

    def load_and_slice(
        self,
        image: np.ndarray,
        slices: Optional[Union[slice, Sequence[int]]] = None,
    ) -> np.ndarray:

        if image.ndim == 2:
            return image.copy()
        elif image.ndim == 3:
            if slices is None:
                return image.copy()
            return image[slices].copy()
        else:
            raise ValueError(f"Unsupported image dimensions: {image.shape}")

    def save_image_mlflow(self, image: np.ndarray, image_path: pathlib.Path) -> None:
        path_parts = list(image_path.relative_to(image_path.parents[3]).parts)
        path_parts.remove("profiling_input_images")
        mlflow_path = pathlib.Path(*path_parts)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = pathlib.Path(tmp_dir) / image_path.name
            tifffile.imwrite(save_path, image.astype(np.uint8))

            mlflow.log_artifact(local_path=save_path, artifact_path=mlflow_path)

    def save_image(
        self,
        image_path: pathlib.Path,
        image_type: str,
        slice_groups: Optional[Sequence[Union[int, slice]]] = None,
    ) -> None:
        if slice_groups is None:
            loop_iter = range(1)
        else:
            loop_iter = slice_groups

        image = tifffile.imread(image_path)
        for slices_idx, slices in enumerate(loop_iter):
            image_processed = self.load_and_slice(image=image, slices=slices)

            if not np.any((image_processed > 0.0) & (image_processed < 1.0)):
                if image_type == "input":
                    raise ValueError(
                        "Pixels should be between 0 and 1 in the input image"
                    )
                image_processed = image_processed != 0

            image_processed *= 255
            image_processed = image_processed.astype(np.uint8)

            slice_image_path = image_path.with_name(
                f"{slices_idx:02d}_{image_type}_{image_path.name}"
            )
            slice_image_path = (
                slice_image_path.parent / self.data_split / slice_image_path.name
            )
            self.save_image_mlflow(image=image_processed, image_path=slice_image_path)

    def __call__(self, model: torch.nn.Module) -> None:
        for metadata in self.sampled_images.values():
            self.save_image(
                image_path=metadata["input_path"],
                image_type="input",
                slice_groups=metadata["Metadata_Input_Slices"],
            )

            self.save_image(
                image_path=metadata["target_path"],
                image_type="target",
                slice_groups=metadata["Metadata_Target_Slices"],
            )
