import copy
from typing import Any, Dict

from farmhash import Fingerprint64
from torch.utils.data import DataLoader


class SampleImages:
    """
    Hash sampling of image data and wrangling of image metadata and paths.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        splitter: Any,
        data_split: str,
        number_of_images: int,
    ) -> None:
        self.divisor = 10**6
        dataloader.dataset.split_data = True
        self.dataloader = dataloader

        image_fraction = number_of_images / len(self.dataloader.dataset)

        if data_split == "validation":
            self.lower_thresh = splitter.upper_train_thresh
            self.upper_thresh = (
                int(splitter.upper_val_thresh * image_fraction) + self.lower_thresh
            )

        else:
            raise ValueError("Please specify a valid argument for data_split")

    def __call__(self) -> Dict[str, Dict[str, Any]]:
        images_metadata: Dict[str, Dict[str, Any]] = {}

        for batch_data in self.dataloader:
            metadata = batch_data["metadata"]
            input_paths = batch_data["input_path"]
            target_paths = batch_data["target_path"]

            for idx, (input_path, target_path) in enumerate(
                zip(input_paths, target_paths)
            ):
                metadata_id = metadata["Metadata_ID"][idx]
                if metadata_id is None:
                    continue

                image_num_id = Fingerprint64(metadata_id) % self.divisor
                if not self.lower_thresh < image_num_id < self.upper_thresh:
                    continue

                if metadata_id not in images_metadata:
                    images_metadata[metadata_id] = {
                        key: value[idx]
                        for key, value in metadata.items()
                        if key != "Metadata_ID"
                    }
                    images_metadata[metadata_id]["Metadata_Input_Slices"] = []
                    images_metadata[metadata_id]["Metadata_Target_Slices"] = []
                    images_metadata[metadata_id]["input_path"] = input_path
                    images_metadata[metadata_id]["target_path"] = target_path

                images_metadata[metadata_id]["Metadata_Input_Slices"].append(
                    metadata["Metadata_Input_Slices"][idx]
                )
                images_metadata[metadata_id]["Metadata_Target_Slices"].append(
                    metadata["Metadata_Target_Slices"][idx]
                )

        for metadata in images_metadata.values():
            metadata["Metadata_Input_Slices"].sort(reverse=False)
            metadata["Metadata_Target_Slices"].sort(reverse=False)

        if not images_metadata:
            raise ValueError(
                "No images were sampled. Consider changing your thresholds and ensuring there is enough data in the dataloader."
            )

        return images_metadata
