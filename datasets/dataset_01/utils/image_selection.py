from torch.utils.data import dataset


def select_unique_image_idxs(image_dataset_idxs: list[int], image_dataset: dataset):
    """
    Identify redundant indices in the original dataset.
    This dataset contains FOV crops as samples, whereas I want samples containing whole images
    from multiple z-slices from the other dataset.
    """

    unique_image_dataset_idxs = set()
    metadata_idxs = set()
    len_metadata_idx = 0
    original_split_decision = image_dataset.split_data
    image_dataset.split_data = True

    for sample_idx in image_dataset_idxs:
        metadata_idxs.add(image_dataset[sample_idx]["metadata"]["Metadata_ID"])

        next_len_metadata_idx = len(metadata_idxs)

        if next_len_metadata_idx > len_metadata_idx:
            unique_image_dataset_idxs.add(sample_idx)

        len_metadata_idx = next_len_metadata_idx

    image_dataset.split_data = original_split_decision

    return unique_image_dataset_idxs
