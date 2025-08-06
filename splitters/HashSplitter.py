import pathlib
from collections import defaultdict
from typing import Tuple

from farmhash import Fingerprint64
from torch.utils.data import DataLoader, Dataset, Subset, default_collate


class HashSplitter:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 16,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.val_frac = val_frac

    @staticmethod
    def custom_collate_fn(batch):
        """
        Ensures that pathlib paths can be collated by the dataloader, and
        ensures that other metadata consist of lists instead of pytorch tensors.
        """
        collated = {}
        skip_collate_subkeys = {"Metadata_Input_Slices", "Metadata_Target_Slices"}

        for key in batch[0]:
            values = [d[key] for d in batch]

            if isinstance(values[0], pathlib.Path):
                collated[key] = values

            elif key == "metadata":
                metadata_collated = {}
                for meta_key in values[0]:
                    meta_values = [v[meta_key] for v in values]

                    if meta_key in skip_collate_subkeys:
                        # Convert tensors to lists here
                        metadata_collated[meta_key] = [
                            v.tolist() if hasattr(v, "tolist") else v
                            for v in meta_values
                        ]
                    else:
                        metadata_collated[meta_key] = default_collate(meta_values)

                collated["metadata"] = metadata_collated

            else:
                collated[key] = default_collate(values)

        return collated

    def __call__(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        divisor = 10**6
        self.upper_train_thresh = int(self.train_frac * divisor)
        self.upper_val_thresh = int(self.val_frac * divisor) + self.upper_train_thresh
        splits = defaultdict(list)

        for idx, sample in enumerate(self.dataset):
            sample_id = Fingerprint64(sample["metadata"]["Metadata_ID"]) % divisor
            if sample_id < self.upper_train_thresh:
                splits["train"].append(idx)
            elif sample_id < self.upper_val_thresh:
                splits["val"].append(idx)
            else:
                splits["test"].append(idx)

        def make_loader(indices, shuffle):
            return DataLoader(
                Subset(self.dataset, indices),
                batch_size=self.batch_size,
                shuffle=shuffle,
                collate_fn=self.custom_collate_fn,
            )

        return (
            make_loader(splits["train"], shuffle=True),
            make_loader(splits["val"], shuffle=False),
            make_loader(splits["test"], shuffle=False),
        )
