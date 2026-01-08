import numpy as np
from scipy.ndimage import distance_transform_edt


def instance_to_semantic_segmentation(
    instance_mask: np.ndarray, boundary_thickness: int = 2
):
    """
    Convert 3D instance mask to semantic mask with fixed boundary thickness per instance.
    Channel order: 0=background, 1=cell interior, 2=boundary

    instance_mask can be the following dimensions:
    (Z, H, W)

    where:
    Z: Number of Z-slices
    H: Height of the image (in pixels)
    W: Width of the image (in pixels)
    """
    Z, H, W = instance_mask.shape
    semantic_mask = np.zeros((3, Z, H, W), dtype=np.uint8)

    # Background
    semantic_mask[0] = (instance_mask == 0).astype(np.uint8)

    for z in range(Z):
        slice_mask = instance_mask[z]
        boundary_slice = np.zeros((H, W), dtype=bool)
        interior_slice = np.zeros((H, W), dtype=bool)

        # Process each instance separately
        for label in np.unique(slice_mask):
            if label == 0:
                continue  # skip background

            cell_mask = slice_mask == label
            if not np.any(cell_mask):
                continue

            # Distance transform inside this cell
            dist = distance_transform_edt(cell_mask)

            # Boundary: pixels within boundary_thickness of edge
            boundary_mask = (dist <= boundary_thickness) & cell_mask
            interior_mask = (dist > boundary_thickness) & cell_mask

            # Add to slice masks (combine all instances)
            boundary_slice |= boundary_mask
            interior_slice |= interior_mask

        semantic_mask[1, z] = interior_slice.astype(np.uint8)
        semantic_mask[2, z] = boundary_slice.astype(np.uint8)

    return semantic_mask
