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
        callbacks: Any,
        image_postprocessor: Any = lambda x: x,
        epochs: int = 10,
        device: Union[str, torch.device] = "cuda",
        use_amp: bool = True,
    ) -> None:

        self.model = model
        self.model_optimizer = model_optimizer
        self.model_loss = model_loss
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.callbacks = callbacks
        self.image_postprocessor = image_postprocessor
        self.epochs = epochs
        self.device = device
        self.use_amp = use_amp  # Automatic Mixed Precision (AMP)

        if self.use_amp and self.device.startswith("cuda"):
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

    @property
    def best_loss_value(self):
        return self.callbacks.best_loss_value

    def train(self) -> None:
        train_data = {}
        train_data["continue_training"] = True
        train_data["use_amp"] = self.use_amp
        train_data["device"] = self.device

        self.model = self.model.to(self.device)

        for epoch in range(self.epochs):
            if not train_data["continue_training"]:
                break

            train_data["epoch"] = epoch
            train_data["callback_hook"] = "on_epoch_start"
            self.callbacks(**train_data)

            self.model.train()

            for batch, batch_data in enumerate(self.train_dataloader):
                train_data["callback_hook"] = "on_batch_start"
                train_data["batch"] = batch
                train_data["batch_data"] = batch_data
                self.callbacks(**train_data)

                inputs = batch_data["input"].to(self.device)
                targets = batch_data["target"].to(self.device)

                if self.use_amp and self.scaler is not None:
                    with torch.amp.autocast("cuda"):
                        generated_predictions = self.image_postprocessor(
                            self.model(inputs)
                        )
                        loss = self.model_loss(
                            targets=targets,
                            generated_predictions=generated_predictions,
                        )
                else:
                    generated_predictions = self.image_postprocessor(self.model(inputs))
                    loss = self.model_loss(
                        targets=targets,
                        generated_predictions=generated_predictions,
                    )

                train_data["generated_predictions"] = generated_predictions
                train_data["model_update_loss"] = loss

                self.model_optimizer.zero_grad()
                if self.use_amp and self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.model_optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.model_optimizer.step()

                train_data["model"] = self.model
                train_data["callback_hook"] = "on_batch_end"

                self.callbacks(**train_data)

            train_data["callback_hook"] = "on_epoch_end"
            train_data["continue_training"] = self.callbacks(
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                **train_data,
            )

            if not train_data["continue_training"]:
                break

    def __call__(self) -> None:
        self.train()
