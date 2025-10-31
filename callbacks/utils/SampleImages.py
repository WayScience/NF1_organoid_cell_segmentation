import copy
from typing import Any, Dict, Optional, Union

from farmhash import Fingerprint64
from torch.utils.data import DataLoader, Dataset, Subset


class SampleImages:
    """
    Hash sampling of image data and wrangling of image metadata and paths.
    """

    def __init__(
        self,
        datastruct: Union[Dataset, DataLoader],
        image_fraction: float,
        dataset_idxs: Optional[list[int]] = None,
    ) -> None:

        self.is_dataset = dataset_idxs is not None
        self.datastruct = datastruct
        self.dataset_idxs = dataset_idxs
        self.divisor = 10**6

        if self.is_dataset:
            self.original_data_split = self.datastruct.split_data
            self.datastruct.split_data = True
        else:
            self.original_data_split = self.datastruct.dataset.dataset.split_data
            self.datastruct.dataset.dataset.split_data = True

        image_fraction = min(1.0, image_fraction)

        self.upper_thresh = int(self.divisor * image_fraction)

    def sample_images(self, metadata_dataset_id: int, metadata_sample_id: int) -> int:
        """
        Hash-based sampling using sample ids.
        """

        if metadata_sample_id is None:
            return metadata_sample_id

        # Samples in the batch may have different IDs
        image_num_id = Fingerprint64(metadata_sample_id) % self.divisor
        if not image_num_id < self.upper_thresh:
            return None

        return metadata_dataset_id

    def __call__(self) -> list[int]:
        image_dataset_idxs = []

        data_iterable = (
            (self.datastruct[idx] for idx in self.dataset_idxs)
            if self.is_dataset
            else self.datastruct
        )

        for data in data_iterable:
            metadata = data["metadata"]
            dataset_ids = metadata["Metadata_Dataset_ID"]
            sample_ids = metadata["Metadata_Sample_ID"]

            if self.is_dataset:
                dataset_sample_pairs = [(dataset_ids, sample_ids)]

            else:
                dataset_sample_pairs = zip(dataset_ids, sample_ids)

            for dataset_id, sample_id in dataset_sample_pairs:
                image_dataset_idx = self.sample_images(dataset_id, sample_id)

                if image_dataset_idx is not None:
                    image_dataset_idxs.append(image_dataset_idx)

        if not image_dataset_idxs:
            raise ValueError(
                "No images were sampled. Consider changing your thresholds and ensuring there is enough data in the dataloader."
            )

        if self.is_dataset:
            self.datastruct.split_data = self.original_data_split
        else:
            self.datastruct.dataset.dataset.split_data = self.original_data_split

        return image_dataset_idxs
