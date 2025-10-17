#!/usr/bin/env python
# coding: utf-8

# # Convert Images
# Convert 3D tiff images to a format that can visualized by nViz.

# In[ ]:


import pathlib
import sys

import numpy as np
import pandas as pd
import tifffile


# # Inputs
# Path to the images

# In[ ]:


artifacts_path = pathlib.Path(sys.argv[1])
img_dir_paths = {tif_file for tif_file in artifacts_path.rglob("*.tif*")}


# # Outputs

# In[ ]:


modified_artifacts_path = pathlib.Path(sys.argv[2])


# In[ ]:


for img_path in img_dir_paths:

    image = tifffile.imread(img_path)
    image_dir = img_path.relative_to(artifacts_path)

    # Save 2D representations of 3D tiff images
    # nViz can't convert 3D tiff images to an OME-tiff or OME-zarr format yet
    for zslice_idx in range(image.shape[0]):
        save_path = modified_artifacts_path / image_dir
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

