from typing import Optional

import torch

from .AbstractMetric import AbstractMetric


class Dice(AbstractMetric):
    """
    Computes the dice score and IOU.
    Assumes generated_predictions are raw logits.
    """

    def __init__(
        self,
        smooth: float = 1e-6,
        use_logits: bool = True,
        prediction_threshold=0.5,
        is_loss: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        self.smooth = smooth
        self.use_logits = use_logits
        self.prediction_threshold = prediction_threshold
        self.is_loss = is_loss
        self.device = device
        self.data_split_logging: Optional[str] = None
        self.reset()

    def reset(self):
        self.total_intersection = torch.tensor(0.0, device=self.device)
        self.total_denominator = torch.tensor(0.0, device=self.device)
        self.total_union = torch.tensor(0.0, device=self.device)

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

        batch_size = generated_predictions.size(0)

        probs = (
            torch.sigmoid(generated_predictions)
            if self.use_logits
            else generated_predictions
        )
        preds_flat = (probs > self.prediction_threshold).float().view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)

        intersection = (preds_flat * targets_flat).sum()
        denominator = preds_flat.sum() + targets_flat.sum()
        union = denominator - intersection

        if data_split_logging is None:
            dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

            return dice

        self.data_split_logging = data_split_logging

        self.total_intersection += intersection.detach().to(self.device)
        self.total_denominator += denominator.detach().to(self.device)
        self.total_union += union.detach().to(self.device)

        return None

    def get_metric_data(self) -> dict[str, torch.Tensor]:
        if self.total_denominator == 0:
            average_dice = torch.tensor(0.0, device=self.device)
            average_iou = torch.tensor(0.0, device=self.device)
        else:
            average_dice = (2.0 * self.total_intersection) / self.total_denominator
            average_iou = self.total_intersection / self.total_union

        prefix = "loss" if self.is_loss else ""
        key_dice = f"dice_{prefix}_{self.data_split_logging}"

        other_prefix = f"{prefix}_component" if prefix == "loss" else ""
        key_iou = f"iou_{other_prefix}_{self.data_split_logging}"

        self.reset()
        return {key_dice: average_dice, key_iou: average_iou}
