from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn


class AbstractMetric(nn.Module, ABC):
    """
    Abstract base class for all metrics.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self) -> torch.Tensor | None:
        pass

    @abstractmethod
    def get_metric_data(self) -> dict[str, torch.Tensor]:
        pass
