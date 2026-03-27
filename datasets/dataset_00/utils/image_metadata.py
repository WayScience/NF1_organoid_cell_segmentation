import pathlib
from typing import Any, Optional

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

        cell_mask_paths = patient.rglob("segmentation_masks/**/cell_mask.tiff")
        bright_path_prefix = patient / "zstack_images"

        for mask_path in cell_mask_paths:
            well_site_name = mask_path.parents[0].name

            bright_path = (
                bright_path_prefix / f"{well_site_name}/{well_site_name}_TRANS.tif"
            )

            if bright_path.exists():
                image_mask_pairs.append(
                    {"input_path": bright_path, "target_path": mask_path}
                )
            else:
                raise FileNotFoundError(
                    f"The brightfield path doesn't exist for patient {patient.name} and well-site {well_site_name}. Consider modifying image_metadata.py in the first dataset utils to correct the pathing"
                )

    return image_mask_pairs


def get_image_specs(
    image_paths: list[dict[str, pathlib.Path]], crop_margin: Optional[int] = None
) -> dict[str, Any]:

    input_example = tifffile.imread(image_paths[0]["input_path"])
    target_example = tifffile.imread(image_paths[0]["target_path"])

    if target_example.ndim == 2:
        image_height, image_width = target_example.shape
        image_slices = None
    elif target_example.ndim == 3:
        image_slices, image_height, image_width = target_example.shape
    else:
        raise ValueError(f"Unexpected target shape: {target_example.shape}")

    crop_margin = 0 if crop_margin is None else crop_margin

    return {
        "input_max_pixel_value": np.iinfo(input_example.dtype).max,
        "input_ndim": input_example.ndim,
        "target_ndim": target_example.ndim,
        "image_shape": [image_slices, image_height, image_width],
        "crop_margin": crop_margin,
    }
