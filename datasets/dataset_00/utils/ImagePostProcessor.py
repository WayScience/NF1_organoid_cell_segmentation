from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class ImagePostProcessor:
    """
    Processes generated predictions computed from the model.
    """

    def __init__(self):
        pass

    def __call__(self, generated_prediction: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(generated_prediction)
