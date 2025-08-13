from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .AbstractMetric import AbstractMetric


class BCE(AbstractMetric):
    """
    Computes the Binary Cross Entropy from Logits.
    """

    def __init__(
        self,
        is_loss: bool = False,
        use_logits: bool = True,
        reduction: str = "mean",
        device: str = "cuda",
    ):
        super().__init__()
        self.is_loss = is_loss
        self.use_logits = use_logits
        self.device = device

        if self.is_loss:
            bce_fn = nn.BCEWithLogitsLoss if self.use_logits else nn.BCELoss
            self.bce_fn = bce_fn(reduction=reduction).to(self.device)

        self.reset()

    def reset(self):
        self.total_loss = torch.tensor(0.0, device=self.device)
        self.total_elements = 0

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        data_split_logging: Optional[str] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor] | torch.Tensor:

        if generated_predictions.shape != targets.shape:
            raise ValueError(
                "The generated predictions and targets must be the same shape."
            )

        if data_split_logging is None:
            return self.bce_fn(generated_predictions, targets)

        self.data_split_logging = data_split_logging

        if not self.use_logits:
            bce_per_pixel = F.binary_cross_entropy(
                generated_predictions, targets, reduction="none"
            )
        else:
            bce_per_pixel = F.binary_cross_entropy_with_logits(
                generated_predictions, targets, reduction="none"
            )

        self.total_loss += bce_per_pixel.sum().detach().to(self.device)
        self.total_elements += bce_per_pixel.numel()

        return None

    def get_metric_data(self) -> dict[str, torch.Tensor]:
        average_loss = (
            self.total_loss / self.total_elements
            if self.total_elements > 0
            else torch.tensor(0.0, device=self.device)
        )

        key = "bce_loss" if self.is_loss else "bce"
        key += f"_{self.data_split_logging}"

        self.reset()
        return {key: average_loss}
