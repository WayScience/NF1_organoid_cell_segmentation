import pathlib
from collections import defaultdict
from typing import Tuple

from farmhash import Fingerprint64
from torch.utils.data import DataLoader, Dataset, Subset
from datasets.dataset_00.utils.Collator import collator


class HashSplitter:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 16,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
    ) -> None:
        dataset.split_data = True
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.val_frac = val_frac

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
                collate_fn=collator,
            )

        self.dataset.split_data = False
        return (
            make_loader(splits["train"], shuffle=True),
            make_loader(splits["val"], shuffle=False),
            make_loader(splits["test"], shuffle=False),
        )
