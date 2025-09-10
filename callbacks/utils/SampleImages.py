import copy
from typing import Any, Dict

from farmhash import Fingerprint64
from torch.utils.data import DataLoader, Subset


class SampleImages:
    """
    Hash sampling of image data and wrangling of image metadata and paths.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        number_of_images: int,
    ) -> None:
        self.dataloader = dataloader
        self.number_of_images = number_of_images
        self.divisor = 10**6
        self.dataloader.dataset.dataset.split_data = True

        image_fraction = self.number_of_images / len(self.dataloader.dataset)

        self.upper_thresh = int(self.divisor * image_fraction)

    def __call__(self) -> list[int]:
        image_dataset_idxs = []

        for batch_data in self.dataloader:
            metadata = batch_data["metadata"]
            input_paths = batch_data["input_path"]
            target_paths = batch_data["target_path"]

            for metadata_dataset_id, metadata_sample_id in zip(
                metadata["Metadata_Dataset_ID"], metadata["Metadata_Sample_ID"]
            ):
                if metadata_sample_id is None:
                    continue

                # Samples in the batch may have different IDs
                image_num_id = Fingerprint64(metadata_sample_id) % self.divisor
                if not image_num_id < self.upper_thresh:
                    continue

                image_dataset_idxs.append(metadata_dataset_id)

        if not image_dataset_idxs:
            raise ValueError(
                "No images were sampled. Consider changing your thresholds and ensuring there is enough data in the dataloader."
            )

        self.dataloader.dataset.dataset.split_data = False
        return image_dataset_idxs
