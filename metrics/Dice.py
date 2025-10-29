from typing import Optional, Union

import torch

from .AbstractMetric import AbstractMetric


class Dice(AbstractMetric):
    """
    Computes the dice score and IOU.
    Assumes generated_predictions are raw logits.
    """

    def __init__(
        self,
        mask_idx_mapping: dict[int, str],
        mask_weights: Optional[torch.Tensor] = None,
        smooth: float = 1e-6,
        use_logits: bool = True,
        prediction_threshold=0.5,
        is_loss: bool = False,
        device: Union[str, torch.device] = "cuda",
    ):
        """
        mask_idx_mapping:
            Contains the mappings from mask indices to mask names.
        """
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

        self.smooth = smooth
        self.use_logits = use_logits
        self.prediction_threshold = prediction_threshold
        self.is_loss = is_loss
        self.device = device
        self.data_split_logging: Optional[str] = None
        self.reset()

    def reset(self):
        self.total_intersection = torch.zeros(3, device=self.device)
        self.total_denominator = torch.zeros(3, device=self.device)
        self.total_union = torch.zeros(3, device=self.device)

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        data_split_logging: Optional[str] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor] | None:
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

        batch_size, num_channels, z_slices = generated_predictions.shape[:3]

        probs = (
            torch.sigmoid(generated_predictions)
            if self.use_logits
            else generated_predictions
        )
        preds_flat = (
            (probs > self.prediction_threshold)
            .float()
            .view(batch_size, num_channels, z_slices, -1)
        )
        targets_flat = targets.view(batch_size, num_channels, z_slices, -1)

        intersection = (preds_flat * targets_flat).sum(dim=(0, 2, 3))
        denominator = preds_flat.sum(dim=(0, 2, 3)) + targets_flat.sum(dim=(0, 2, 3))
        union = denominator - intersection

        # Smoothing is used here, because this could be a loss so it should be defined.
        if data_split_logging is None:
            dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
            return (self.mask_weights * dice).sum() / self.mask_weights_sum

        self.data_split_logging = data_split_logging

        self.total_intersection += intersection.detach().to(self.device)
        self.total_denominator += denominator.detach().to(self.device)
        self.total_union += union.detach().to(self.device)

        return None

    def get_metric_data(self) -> dict[str, torch.Tensor]:
        total_dice = (2.0 * self.total_intersection) / self.total_denominator
        total_dice = torch.where(
            self.total_denominator != 0,
            (2.0 * self.total_intersection) / self.total_denominator,
            torch.tensor(0.0, device=self.device),
        )

        average_dice = (self.mask_weights * total_dice).sum() / self.mask_weights_sum

        total_iou = torch.where(
            self.total_union != 0,
            self.total_intersection / self.total_union,
            torch.tensor(0.0, device=self.device),
        )
        average_iou = (self.mask_weights * total_iou).sum() / self.mask_weights_sum

        prefix = "loss_" if self.is_loss else ""
        key_dice = f"dice_total_{prefix}{self.data_split_logging}"

        key_iou = f"iou_total_component_{self.data_split_logging}"

        metrics = {key_dice: average_dice, key_iou: average_iou}

        for mask_idx, mask_name in self.mask_idx_mapping.items():
            metrics[f"iou_{mask_name}_component_{self.data_split_logging}"] = total_iou[
                mask_idx
            ]
            metrics[f"dice_{mask_name}_loss_component_{self.data_split_logging}"] = (
                total_dice[mask_idx]
            )

        self.reset()
        return metrics
