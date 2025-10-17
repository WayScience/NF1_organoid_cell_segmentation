import pathlib
from typing import Any

import numpy as np
import tifffile


def get_image_paths(
    patient_folders: list[pathlib.Path],
) -> list[dict[str, pathlib.Path]]:
    """
    Get the image path of patients.
    """

    image_mask_pairs = []

    for patient in patient_folders:

        brightfield_paths = patient.rglob("profiling_input_images/**/*TRANS.tif")

        for bright_path in brightfield_paths:
            mask_path = bright_path.with_name("cell_masks.tiff")

            if mask_path.exists():
                image_mask_pairs.append(
                    {"input_path": bright_path, "target_path": mask_path}
                )

    return image_mask_pairs


def get_image_specs(image_paths: list[dict[str, pathlib.Path]]) -> dict[str, Any]:

    input_example = tifffile.imread(image_paths[0]["input_path"])
    target_example = tifffile.imread(image_paths[0]["target_path"])

    if target_example.ndim == 2:
        image_height, image_width = target_example.shape
        image_slices = None
    elif target_example.ndim == 3:
        image_slices, image_height, image_width = target_example.shape
    else:
        raise ValueError(f"Unexpected target shape: {target_example.shape}")

    return {
        "input_max_pixel_value": np.iinfo(input_example.dtype).max,
        "input_ndim": input_example.ndim,
        "target_ndim": target_example.ndim,
        "image_shape": [image_slices, image_height, image_width],
    }
