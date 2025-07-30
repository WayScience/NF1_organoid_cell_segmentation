from typing import List, Optional

import mlflow
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader


class Callbacks:
    def __init__(
        self,
        metrics: List,
        loss: Module,
        num_samples_to_plot: Optional[int] = None,
    ):
        self.metrics = metrics
        self.loss = loss
        self.num_samples_to_plot = num_samples_to_plot
        self.best_loss_value = float("inf")
        self.early_stopping_counter = 0

    def _log_epoch_metrics(
        self,
        time_step: int,
        model: Module,
        dataloader: DataLoader,
        data_split: str,
        generated_predictions: Tensor,
        targets: Tensor,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:

        model.eval()

        with torch.no_grad():
            for inputs, targets in dataloader:
                generated_predictions = model(inputs)
                pixel_probs = torch.sigmoid(generated_predictions)

                for metric in self.metrics:
                    metric(
                        generated_predictions=generated_predictions,
                        generated_pixel_probs=pixel_probs,
                        targets=targets,
                        data_split_logging=data_split,
                        **kwargs,
                    )

                self.loss(
                    generated_predictions=generated_predictions,
                    generated_pixel_probs=pixel_probs,
                    targets=targets,
                    data_split=data_split,
                    **kwargs,
                )

        for name, loss_value in self.loss.metric_data().items():
            mlflow.log_metric(f"{data_split}_{name}", loss_value, step=time_step)

        for metric in self.metrics:
            for name, metric_value in metric.metric_data().items():
                mlflow.log_metric(f"{data_split}_{name}", metric_value, step=time_step)

    def _on_epoch_start(self, epoch: int, **kwargs) -> None:
        print(f"Starting epoch {epoch}")

    def _on_batch_start(self, batch: int, **kwargs) -> None:
        print(f"Starting batch {batch}")

    def _on_epoch_end(
        self,
        epoch: int,
        model: Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        **kwargs,
    ) -> None:

        for data_split, dataloader in [
            ("train", train_dataloader),
            ("validation", val_dataloader),
        ]:

            self._log_epoch_metrics(
                model=model,
                dataloader=dataloader,
                data_split=data_split,
                time_step=epoch,
                **kwargs,
            )

    def _on_batch_end(self, batch: int, **kwargs) -> None:
        pass

    def __call__(self, callback_hook: str, **kwargs) -> None:
        getattr(self, f"_on_{callback_hook}")(**kwargs)
