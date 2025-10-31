from torch.utils.data import dataset


def select_unique_image_idxs(image_dataset_idxs: list[int], image_dataset: dataset):
    """
    Identify redundant indices in the original dataset.
    This dataset contains FOV crops as samples, whereas I want samples containing whole images
    from multiple z-slices from the other dataset.
    """

    unique_image_dataset_idxs = []
    original_split_decision = image_dataset.split_data
    image_dataset.split_data = True

    for sample_idx in image_dataset_idxs:
        if (
            image_dataset[sample_idx]["metadata"]["Metadata_ID"]
            not in unique_image_dataset_idxs
        ):
            unique_image_dataset_idxs.append(sample_idx)

    image_dataset.split_data = original_split_decision

    return unique_image_dataset_idxs
