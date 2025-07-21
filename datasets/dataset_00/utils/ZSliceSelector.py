import pathlib
from collections import defaultdict

import numpy as np
import tifffile
from NonBlackSliceSelector import NonBlackSliceSelector


class ZSliceSelectorManager:
    """
    Decides how to select z-slices.
    """

    def __init__(self, **kwargs):

        mode = kwargs.pop("mode", None)

        if mode is None:
            raise ValueError("Missing required argument: mode")

        if kwargs["mode"] == "nonblack":
            self.slice_selector = NonBlackSliceSelector(**kwargs)

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def __call__(self):
        self.slice_selector()
