from typing import Optional

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
        device: str = "cuda",
    ):
        super().__init__()
        self.prediction_threshold = prediction_threshold
        self.use_logits = use_logits
        self.device = device
        self.data_split_logging: Optional[str] = None
        self.reset()

    def reset(self):
        self.true_positives = torch.tensor(0.0, device=self.device)
        self.false_positives = torch.tensor(0.0, device=self.device)
        self.false_negatives = torch.tensor(0.0, device=self.device)

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        data_split_logging: Optional[str] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor] | None:

        if generated_predictions.shape != targets.shape:
            raise ValueError(
                "The generated predictions and targets must be the same shape."
            )

        probs = torch.sigmoid(generated_predictions) if self.use_logits else generated_predictions
        preds = (probs > self.prediction_threshold).float()

        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()

        if data_split_logging is None:
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            return {"precision": precision, "recall": recall}

        self.data_split_logging = data_split_logging

        self.true_positives += tp.detach()
        self.false_positives += fp.detach()
        self.false_negatives += fn.detach()

        return None

    def get_metric_data(self) -> dict[str, torch.Tensor]:
        precision = (
            self.true_positives / (self.true_positives + self.false_positives + 1e-6)
            if (self.true_positives + self.false_positives) > 0
            else torch.tensor(0.0, device=self.device)
        )

        recall = (
            self.true_positives / (self.true_positives + self.false_negatives + 1e-6)
            if (self.true_positives + self.false_negatives) > 0
            else torch.tensor(0.0, device=self.device)
        )

        key_precision = f"precision_{self.data_split_logging}"
        key_recall = f"recall_{self.data_split_logging}"

        self.reset()
        return {key_precision: precision, key_recall: recall}
