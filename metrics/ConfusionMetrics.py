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
        prediction_threshold: float = 0.5,
        use_logits: bool = True,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()
        self.prediction_threshold = prediction_threshold
        self.use_logits = use_logits
        self.device = device
        self.reset()

    def reset(self):
        self.true_positives = torch.tensor(0.0, device=self.device)
        self.false_positives = torch.tensor(0.0, device=self.device)
        self.false_negatives = torch.tensor(0.0, device=self.device)

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

        probs = (
            torch.sigmoid(generated_predictions)
            if self.use_logits
            else generated_predictions
        )
        preds = (probs > self.prediction_threshold).float()

        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()

        self.data_split_logging = data_split_logging

        self.true_positives += tp.detach()
        self.false_positives += fp.detach()
        self.false_negatives += fn.detach()

        return None

    def get_metric_data(self) -> dict[str, torch.Tensor]:
        precision = (
            self.true_positives / (self.true_positives + self.false_positives)
            if (self.true_positives + self.false_positives) > 0
            else torch.tensor(0.0, device=self.device)
        )

        recall = (
            self.true_positives / (self.true_positives + self.false_negatives)
            if (self.true_positives + self.false_negatives) > 0
            else torch.tensor(0.0, device=self.device)
        )

        key_precision = f"precision__{self.data_split_logging}"
        key_recall = f"recall__{self.data_split_logging}"

        self.reset()
        return {key_precision: precision, key_recall: recall}
