#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pathlib
import sys
import tempfile

import numpy as np
import tifffile
from nviz.image import tiff_to_zarr
from nviz.view import view_zarr_with_napari


# # Inputs

# In[ ]:


original_img_dir_path = pathlib.Path("temp_example").resolve(strict=True)
img_dir_paths = {tif_file for tif_file in original_img_dir_path.rglob("*.tif*")}


# # Parameters
# Came from the microscope

# In[ ]:


scaling_values = [1.0, 0.1, 0.1]


# # Split, Convert, and View Images

# In[ ]:


with tempfile.TemporaryDirectory() as tmpdir:
    temp_img_dir_path = pathlib.Path(tmpdir) / "tiff_images"
    temp_img_dir_path.mkdir(parents=True, exist_ok=True)

    temp_zarr_dir_path = pathlib.Path(tmpdir) / "zarr_output"
    temp_zarr_dir_path.mkdir(parents=True, exist_ok=True)

    for img_path in img_dir_paths:

        image = tifffile.imread(img_path)
        image_dir = img_path.relative_to(original_img_dir_path)

        # Save 2D representations of 3D tiff images
        # nViz can't convert 3D tiff images to an OME-tiff or OME-zarr format yet
        for zslice_idx in range(image.shape[0]):
            save_path = temp_img_dir_path / image_dir
            pathlib.Path(save_path.parent).mkdir(parents=True, exist_ok=True)
            save_path = save_path.with_name(
                f"{save_path.stem}_{zslice_idx:02d}{save_path.suffix}"
            )
            tifffile.imwrite(save_path, image[zslice_idx])
            thresholded_segmentations_save_path = save_path.parent

            segmentation_image_path = (
                thresholded_segmentations_save_path
                / f"3D_generated-segmentation_{zslice_idx:02d}.tiff"
            )

            # Save 2D thresholded segmentations images
            if not segmentation_image_path.exists():
                tifffile.imwrite(
                    segmentation_image_path,
                    (image[zslice_idx] >= 55).astype(np.uint8) * 255,
                )

    tiff_to_zarr(
        image_dir=save_path.parent,
        output_path=temp_zarr_dir_path / "output.zarr",
        channel_map={
            "TRANS": "Brightfield",
            "prediction_background": "Predicted Background Probabilities",
            "prediction_cell": "Predicted Cell Boundary Probabilities",
            "prediction_inner": "Predicted Inner-Cell Probabilities",
            "target_background": "Target Background Segmentation",
            "target_cell": "Target Cell Boundary Segmentation",
            "target_inner": "Target Inner-Cell Segmentation",
        },
        scaling_values=scaling_values,
    )

    viewer = view_zarr_with_napari(
        zarr_dir=temp_zarr_dir_path / "output.zarr",
        scaling_values=scaling_values,
        headless=False,
    )

