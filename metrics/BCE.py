from typing import Optional, Union

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
        mask_idx_mapping: Optional[dict[int, str]] = None,
        mask_weights: Optional[torch.Tensor] = None,
        is_loss: bool = False,
        use_logits: bool = True,
        device: Union[str, torch.device] = "cuda",
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

        if mask_idx_mapping is None and not is_loss:
            raise ValueError(
                "The mask index mapping must be defined if BCE is used as a loss."
            )

        self.mask_weights_sum = self.mask_weights.sum()

        self.is_loss = is_loss
        self.use_logits = use_logits
        self.device = torch.device(device)
        self.bce_fn = None
        self.bce_pixel_func = None

        if self.is_loss:
            bce_fn = nn.BCEWithLogitsLoss if self.use_logits else nn.BCELoss
            self.bce_fn = bce_fn(reduction="none").to(self.device)

        # For accumulating the BCE across pixels
        self.bce_pixel_func = (
            F.binary_cross_entropy_with_logits  # This function is more efficient if using logits
            if self.use_logits
            else F.binary_cross_entropy
        )

        self.reset()

    def reset(self):
        self.total_loss = torch.zeros(3, device=self.device)
        self.total_elements = 0

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        data_split_logging: Optional[str] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        data_split_logging should not be defined if computing the loss for backpropagation
        """

        if generated_predictions.shape != targets.shape:
            raise ValueError(
                "The generated predictions and targets must be the same shape."
            )

        if data_split_logging is None and not self.is_loss:
            raise ValueError(
                "If the metric is not a loss, then it must be used for logging."
            )

        if generated_predictions.ndim == 4:
            generated_predictions = generated_predictions.unsqueeze(0)
            targets = targets.unsqueeze(0)

        if data_split_logging is None:
            bce_losses = self.bce_fn(generated_predictions, targets)
            per_mask_losses = bce_losses.mean(dim=(0, 2, 3, 4))
            return (self.mask_weights * per_mask_losses).sum() / self.mask_weights_sum

        bce_func = self.bce_fn if data_split_logging is None else self.bce_pixel_func

        per_mask_losses = bce_func(
            generated_predictions,
            targets,
            reduction="none",
        )

        self.data_split_logging = data_split_logging

        self.total_loss += per_mask_losses.sum(dim=(0, 2, 3, 4)).detach()
        self.total_elements += targets[:, 0, ::].numel()

        return None

    def get_metric_data(self) -> dict[str, torch.Tensor]:
        average_loss = (
            self.total_loss / self.total_elements
            if self.total_elements > 0
            else torch.zeros(3, device=self.device)
        )

        metrics = {}
        base_prefix = "_loss_" if self.is_loss else "_"
        component_prefix = f"{base_prefix}component_" if self.is_loss else "_"

        if self.total_elements > 0:
            bce_total = (self.mask_weights * self.total_loss).sum() / (
                3 * self.total_elements
            )
        else:
            bce_total = torch.zeros(3, device=self.device)

        metrics[f"bce_total{base_prefix}{self.data_split_logging}"] = bce_total

        for mask_idx, mask_name in self.mask_idx_mapping.items():
            if self.total_elements > 0:
                value = average_loss[mask_idx]
            else:
                value = torch.zeros(1, device=self.device)

            metrics[f"bce_{mask_name}{component_prefix}{self.data_split_logging}"] = (
                value
            )

        self.reset()
        return metrics
