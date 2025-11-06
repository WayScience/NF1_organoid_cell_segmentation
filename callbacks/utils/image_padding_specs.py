import pathlib
from typing import Any, Optional

import numpy as np
import tifffile


def compute_pad(
    image_dim_size: int, crop_dim_size: int, stride: int
) -> tuple[int, int]:
    """Compute how much padding is needed for one dimension."""
    if image_dim_size <= crop_dim_size:
        raise ValueError(
            "The crop dimensions must be smaller than the image dimensions."
        )

    remainder = (image_dim_size - crop_dim_size) % stride
    total_pad = stride - remainder

    return total_pad // 2, total_pad - total_pad // 2


def compute_patch_mapping(
    image_specs: dict[str, Any],
    crop_shape: tuple[int],
    stride: tuple[int],
    pad_slices: bool = True,
) -> tuple[tuple[tuple[int, int], ...], tuple[slice, ...]]:
    """
    Compute the reflection padding required to evenly tile a 3D image
    with patches of the given crop size and stride. Also returns the
    bounding box coordinates to recover the original image region
    after padding.

    Parameters
    ----------
    image_specs : dict[str, Any]
        Dictionary containing image dimensions with the key:
        - "image_shape": tuple[int]
            Expects Image shape to be: (Z, H, W)
            Z -> Number of Z slices
            H -> Image Height
            W -> Image Width

    crop_shape : tuple[int]
        Expects the crop shape to be: (Z, H, W)
        Z -> Number of Z slices
        H -> Image Height
        W -> Image Width

    stride : tuple[int]
        Expects stride shapeto be: (Z, H, W)
        Z -> Number of Z slices
        H -> Stride Height
        W -> Stride Width

    pad_slices: bool
        Whether to pad the z-slices dimension or not.

    Returns
    -------
    tuple
        A tuple of two elements:
        - Padding Amounts: ((z_before, z_after), (h_before, h_after), (w_before, w_after))
            Amount of reflection padding applied to each dimension.
            Padding is applied before converting instance segmentations to semantic segmentation masks.
            Therefore, there is no need to specify zero padding for the semantic mask channel.
        - Original Crop Coordinates: (slice(None), slice_z, slice_h, slice_w)
            Slice objects that can be used to recover
            the original unpadded image from the padded one.
            There should be no slicing of semantic maps, so the first dimension will
            be discounted.
    """

    padding_slices = tuple([crop_shape[0] // 2] * 2) if pad_slices else (0, 0)

    padding_height = compute_pad(
        image_dim_size=image_specs["image_shape"][1],
        crop_dim_size=crop_shape[1],
        stride=stride[1],
    )

    padding_width = compute_pad(
        image_dim_size=image_specs["image_shape"][2],
        crop_dim_size=crop_shape[2],
        stride=stride[2],
    )

    return (padding_slices, padding_height, padding_width), (
        slice(None),
        slice(padding_slices[0], image_specs["image_shape"][0] + padding_slices[0]),
        slice(padding_height[0], image_specs["image_shape"][1] + padding_height[0]),
        slice(padding_width[0], image_specs["image_shape"][2] + padding_width[0]),
    )
