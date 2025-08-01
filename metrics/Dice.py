import torch
from torch import nn


class Dice(nn.Module):
    """
    Computes the dice score, and also an intermediary loss (IOU).
    """

    def __init__(self, smooth: float = 1e-6, is_loss: bool = False):
        super().__init__()
        self.smooth = smooth
        self.is_loss = is_loss

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        data_split: str,
    ) -> dict[str, torch.Tensor]:

        if generated_predictions.shape != targets.shape:
            raise ValueError(
                "The generated predictions and targets must be the same shape."
            )

        batch_size = generated_predictions.size(0)
        preds_flat = generated_predictions.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)

        intersection = (preds_flat * targets_flat).sum(dim=1)
        union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection
        denominator = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

        iou = (intersection + self.smooth) / (union + self.smooth)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        prefix = "loss" if self.is_loss else ""

        return {
            f"dice_{prefix}_{data_split}": (
                (1.0 - dice).mean() if self.is_loss else dice.mean()
            ),
            f"iou_{prefix}_{data_split}": (
                (1.0 - iou).mean() if self.is_loss else iou.mean()
            ),
        }
