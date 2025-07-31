import copy

from farmhash import Fingerprint64
from torch.utils.data import DataLoader


class SampleImages:

    def __init__(
        self, lower_threshold: int, upper_threshold: int, dataloader: DataLoader
    ):
        self.divisor = 10**6
        self.lower_threshold = int(lower_threshold * self.divisor)
        self.upper_threshold = int(upper_threshold * self.divisor)
        self.dataloader = dataloader

    def __call__(self):

        images_metadata = {}

        for batch_data in self.dataloader:
            for image_metadata in batch_data["metadata"]:
                image_metadata_copy = image_metadata.copy()
                image_num_id = (
                    Fingerprint64(image_metadata_copy["Metadata_ID"]) % self.divisor
                )

                if (
                    image_num_id < self.upper_threshold
                    and image_num_id > self.lower_threshold
                ):
                    metadata_id = image_metadata_copy.pop("Metadata_ID", None)

                    if metadata_id is not None:
                        if metadata_id not in images_metadata:
                            images_metadata[metadata_id] = copy.deepcopy(
                                image_metadata_copy
                            )
                            images_metadata[metadata_id]["Metadata_Input_Slices"] = []
                            images_metadata[metadata_id]["Metadata_Target_Slices"] = []

                        images_metadata[metadata_id]["Metadata_Input_Slices"].append(
                            image_metadata_copy["Metadata_Input_Slices"]
                        )

                        images_metadata[metadata_id]["Metadata_Target_Slices"].append(
                            image_metadata_copy["Metadata_Target_Slices"]
                        )

        if not images_metadata:
            raise ValueError(
                "No images were sampled. Consider changing your thresholds and ensuring there is enough data in the dataloader."
            )

        return images_metadata
