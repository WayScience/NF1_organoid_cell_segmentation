import pathlib
from collections import defaultdict

from farmhash import Fingerprint64
from torch.utils.data import DataLoader, Dataset, Subset


class HashSplitter:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 16,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
    ) -> None:
        self.dataset: Dataset = dataset
        self.batch_size: int = batch_size
        self.train_frac: float = train_frac
        self.val_frac: float = val_frac

    def __call__(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        divisor = 10**6
        upper_train_thresh = int(self.train_frac * divisor)
        upper_val_thresh = int(self.val_frac * divisor) + upper_train_thresh
        data_indices = defaultdict(list)

        for sample_idx, sample_pair in enumerate(self.dataset):
            sample_num_id = Fingerprint64(sample_pair["Metadata_ID"]) % divisor

            if sample_num_id < upper_train_thresh:
                data_indices["train_idxs"].append(sample_idx)
            elif sample_num_id >= upper_val_thresh:
                data_indices["test_idxs"].append(sample_idx)
            else:
                data_indices["val_idxs"].append(sample_idx)

        train_dataloader = DataLoader(
            Subset(self.dataset, data_indices["train_idxs"]),
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_dataloader = DataLoader(
            Subset(self.dataset, data_indices["val_idxs"]),
            batch_size=self.batch_size,
            shuffle=False,
        )

        test_dataloader = DataLoader(
            Subset(self.dataset, data_indices["test_idxs"]),
            batch_size=self.batch_size,
            shuffle=False,
        )

        return train_dataloader, val_dataloader, test_dataloader
