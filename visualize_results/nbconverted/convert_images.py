#!/usr/bin/env python
# coding: utf-8

# # Convert Images
# Convert images to a format that can converted by the nViz tiff_to_zarr function.

# In[ ]:


import pathlib
import sys

import numpy as np
import pandas as pd
import tifffile


# # Inputs
# Path to the images

# In[ ]:


artifacts_path = sys.argv[1]
img_dir_paths = {
    tif_file.parent for tif_file in pathlib.Path(artifacts_path).rglob("*.tif*")
}


# # Outputs

# In[ ]:


modified_artifacts_path = sys.argv[2]


# In[ ]:


def get_base_name(img_path: str) -> str:
    """
    Get the image name.
    """

    if "generated_prediction" in img_path:
        return "placeholder_generated-segmentation.tiff"
    elif "TRANS" in img_path:
        return "placeholder_brightfield.tiff"
    else:
        return "placeholder_pipeline-segmentation.tiff"


# # Process the Data
# - Convert the 3D images to 2D.
# - Change the value after the first underscore to the type of image represented so it can be visualized as a layer.
# - Create a separate folder for the same fovs with different crop coordinates

# In[1]:


for img_dir_path in img_dir_paths:
    rel_dir = img_dir_path.relative_to(artifacts_path)
    target_dir_base = modified_artifacts_path / rel_dir

    for img_path in img_dir_path.rglob("*.tif*"):
        img = tifffile.imread(img_path)
        img_bit_depth = img.dtype.itemsize * 8

        if img.ndim != 3:
            raise ValueError("Script not needed, images should already be 2D")

        mod_img_dir = target_dir_base / img_path.stem.split("__")[0]
        mod_img_dir.mkdir(parents=True, exist_ok=True)

        base_name = get_base_name(str(img_path))

        for z_index in range(img.shape[0]):
            mod_img_path = (
                mod_img_dir
                / f"{pathlib.Path(base_name).stem}_{z_index:02d}{pathlib.Path(base_name).suffix}"
            )
            tifffile.imwrite(mod_img_path, img[z_index])

            # Generate the thresholded image as well
            if "generated-segmentation" in base_name:
                max_pixel_val = (2**img_bit_depth) - 1
                thresh_image = np.where(
                    img > int(max_pixel_val * 0.1), max_pixel_val, 0
                )[z_index]

                if img_bit_depth == 8:
                    thresh_image = thresh_image.astype(np.uint8)

                else:
                    thresh_image = thresh_image.astype(np.uint16)

                tifffile.imwrite(
                    mod_img_dir
                    / f"{pathlib.Path(base_name).stem}-thresholded_{z_index:02d}{pathlib.Path(base_name).suffix}",
                    thresh_image,
                )

