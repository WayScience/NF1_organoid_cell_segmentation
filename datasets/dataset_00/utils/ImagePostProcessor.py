from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast


class ImagePostProcessor:
    """
    Processes generated predictions computed from the model.
    """

    def __init__(self, device: Union[str, torch.device] = "cuda", use_amp: bool = True):
        self.device = device
        self.use_amp = use_amp

    def __call__(self, generated_prediction: torch.Tensor) -> torch.Tensor:

        with autocast(enabled=self.use_amp, device_type=self.device):
            return torch.sigmoid(generated_prediction)
