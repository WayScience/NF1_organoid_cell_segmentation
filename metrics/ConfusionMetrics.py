from typing import Optional, Union

import torch

from .AbstractMetric import AbstractMetric


class ConfusionMetrics(AbstractMetric):
    """
    Computes binary metrics such as precision and recall.
    Assumes generated_predictions are raw logits.
    """

    def __init__(
        self,
        mask_idx_mapping: dict[int, str],
        mask_weights: Optional[torch.Tensor] = None,
        prediction_threshold: float = 0.5,
        device: Union[str, torch.device] = "cuda",
        **kwargs,
    ):
        super().__init__()

        self.mask_idx_mapping = mask_idx_mapping

        self.mask_weights = (
            torch.ones(3, device=device) if mask_weights is None else mask_weights
        )

        if self.mask_weights.ndim != 1 or self.mask_weights.shape[0] != 3:
            raise ValueError(
                "The mask_weights torch tensor must have one dimension with three weights."
            )

        self.mask_weights_sum = self.mask_weights.sum()

        self.prediction_threshold = prediction_threshold
        self.device = torch.device(device)
        self.reset()

    def reset(self):
        self.true_positives = torch.zeros(3, device=self.device)
        self.false_positives = torch.zeros(3, device=self.device)
        self.false_negatives = torch.zeros(3, device=self.device)

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        data_split_logging: str,
        **kwargs,
    ) -> dict[str, torch.Tensor] | None:

        if generated_predictions.shape != targets.shape:
            raise ValueError(
                "The generated predictions and targets must be the same shape."
            )

        if generated_predictions.ndim == 4:
            generated_predictions = generated_predictions.unsqueeze(0)
            targets = targets.unsqueeze(0)

        preds = (generated_predictions > self.prediction_threshold).float()

        tp = (preds * targets).sum(dim=(0, 2, 3, 4))
        fp = (preds * (1 - targets)).sum(dim=(0, 2, 3, 4))
        fn = ((1 - preds) * targets).sum(dim=(0, 2, 3, 4))

        self.data_split_logging = data_split_logging

        self.true_positives += tp.detach()
        self.false_positives += fp.detach()
        self.false_negatives += fn.detach()

        return None

    def get_metric_data(self) -> dict[str, torch.Tensor]:

        metrics = {}

        precision = torch.where(
            self.true_positives + self.false_positives != 0,
            self.true_positives / (self.true_positives + self.false_positives),
            torch.tensor(0.0, device=self.device),
        )

        recall = torch.where(
            self.true_positives + self.false_negatives != 0,
            self.true_positives / (self.true_positives + self.false_negatives),
            torch.tensor(0.0, device=self.device),
        )

        tp_total = (self.mask_weights * self.true_positives).sum()
        fp_total = (self.mask_weights * self.false_positives).sum()
        fn_total = (self.mask_weights * self.false_negatives).sum()

        precision_total = torch.where(
            (tp_total + fp_total) != 0,
            tp_total / (tp_total + fp_total),
            torch.tensor(0.0, device=self.device),
        )

        recall_total = torch.where(
            (tp_total + fn_total) != 0,
            tp_total / (tp_total + fn_total),
            torch.tensor(0.0, device=self.device),
        )

        metrics[f"precision_total_{self.data_split_logging}"] = precision_total
        metrics[f"recall_total_{self.data_split_logging}"] = recall_total

        for mask_idx, mask_name in self.mask_idx_mapping.items():
            key = f"{mask_name}_component_{self.data_split_logging}"
            precision_key = f"precision_{key}"
            recall_key = f"recall_{key}"
            metrics[precision_key] = precision[mask_idx]
            metrics[recall_key] = recall[mask_idx]

        self.reset()
        return metrics
