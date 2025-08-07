from typing import Any, Union

import torch
from torch.utils.data import DataLoader

from metrics.AbstractMetric import AbstractMetric


class UNetTrainer:
    """
    Orchestrates training and evaluation of segmentation modeling from brightfield images.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_optimizer: torch.optim.Optimizer,
        model_loss: AbstractMetric,
        train_dataloader: Union[torch.utils.data.Dataset, DataLoader],
        val_dataloader: Union[torch.utils.data.Dataset, DataLoader],
        image_postprocessor: Any,
        callbacks: Any,
        epochs: int = 10,
        device: str = "cuda",
    ) -> None:

        self.model = model
        self.model_optimizer = model_optimizer
        self.model_loss = model_loss
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.image_postprocessor = image_postprocessor
        self.callbacks = callbacks
        self.epochs = epochs
        self.device = device

    def train(self) -> None:
        train_data = {}
        train_data["continue_training"] = True

        for epoch in range(self.epochs):
            if not train_data["continue_training"]:
                break

            train_data["epoch"] = epoch
            train_data["callback_hook"] = "on_epoch_start"
            self.callbacks(**train_data)

            self.model = self.model.to(self.device)
            self.model.train()

            for batch, batch_data in enumerate(self.train_dataloader):
                train_data["callback_hook"] = "on_batch_start"
                train_data["batch"] = batch
                train_data["batch_data"] = batch_data
                self.callbacks(**train_data)

                train_data["generated_predictions"] = self.image_postprocessor(
                    img=self.model(batch_data["input"].to(self.device))
                )

                train_data["model_update_loss"] = self.model_loss(
                    _targets=batch_data["target"].to(self.device),
                    _generated_predictions=train_data["generated_predictions"],
                )

                # Update the Model
                self.model_optimizer.zero_grad()
                train_data["model_update_loss"].backward()
                self.model_optimizer.step()

                train_data["model"] = self.model
                train_data["callback_hook"] = "on_batch_end"

                self.callbacks(**train_data)

            train_data["callback_hook"] = "on_epoch_end"
            train_data["continue_training"] = self.callbacks(
                val_dataloader=self.val_dataloader, **train_data
            )

            if not train_data["continue_training"]:
                break

    def __call__(self) -> None:
        self.train()
