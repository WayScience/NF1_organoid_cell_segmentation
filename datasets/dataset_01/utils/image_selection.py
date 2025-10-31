def select_unique_image_idxs(image_dataset_idxs, image_dataset):
    """
    Identify redundant indices in the original dataset.
    This dataset contains FOV crops as samples, whereas I want samples containing whole images
    from multiple z-slices from the other dataset.
    """

    unique_image_dataset_idxs = []

    for sample_idx in image_dataset_idxs:
        if (
            image_dataset[sample_idx]["metadata"]["Metadata_ID"]
            not in unique_image_dataset_idxs
        ):
            unique_image_dataset_idxs.append(sample_idx)

    return unique_image_dataset_idxs
