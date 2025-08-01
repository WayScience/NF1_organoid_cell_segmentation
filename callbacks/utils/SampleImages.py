import copy
from typing import Any, Dict

from farmhash import Fingerprint64
from torch.utils.data import DataLoader


class SampleImages:
    """
    Hash sampling of image data and wrangling of image metadata and paths.
    """

    def __init__(
        self, lower_threshold: int, upper_threshold: int, dataloader: DataLoader
    ) -> None:
        self.divisor = 10**6
        self.lower_threshold = int(lower_threshold * self.divisor)
        self.upper_threshold = int(upper_threshold * self.divisor)
        self.dataloader = dataloader

    def __call__(self) -> Dict[str, Dict[str, Any]]:

        images_metadata: Dict[str, Dict[str, Any]] = {}

        for batch_data in self.dataloader:
            for input_path, target_path, image_metadata in zip(
                batch_data["input_path"],
                batch_data["target_path"],
                batch_data["metadata"],
            ):
                image_metadata_copy = image_metadata.copy()
                image_num_id = (
                    Fingerprint64(image_metadata_copy["Metadata_ID"]) % self.divisor
                )

                if self.lower_threshold < image_num_id < self.upper_threshold:
                    metadata_id = image_metadata_copy.pop("Metadata_ID", None)

                    if metadata_id is not None:
                        if metadata_id not in images_metadata:
                            images_metadata[metadata_id] = copy.deepcopy(
                                image_metadata_copy
                            )
                            images_metadata[metadata_id]["Metadata_Input_Slices"] = []
                            images_metadata[metadata_id]["Metadata_Target_Slices"] = []

                        for data_type in ["Input", "Target"]:

                            metadata_key = f"Metadata_{data_type}_Slices"

                            images_metadata[metadata_id][metadata_key].append(
                                image_metadata_copy[metadata_key]
                            )

                            images_metadata[metadata_id][metadata_key] = sorted(
                                images_metadata[metadata_id][metadata_key],
                                reverse=False,
                            )

                            if data_type == "Input":
                                images_metadata[metadata_id]["input_path"] = input_path

                            else:
                                images_metadata[metadata_id][
                                    "target_path"
                                ] = target_path

        for metadata in images_metadata.values():
            for key in ["Metadata_Input_Slices", "Metadata_Target_Slices"]:
                metadata[key] = sorted(metadata[key], reverse=False)

        if not images_metadata:
            raise ValueError(
                "No images were sampled. Consider changing your thresholds and ensuring there is enough data in the dataloader."
            )

        return images_metadata
