from typing import List, Optional, Union

import mlflow
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from utils.SaveEpochSlices import SaveEpochSlices


class Callbacks:
    def __init__(
        self,
        metrics: List,
        loss: Module,
        early_stopping_counter_threshold: int,
        image_savers: Optional[Union[SaveEpochSlices, List[SaveEpochSlices]]] = None,
    ):
        self.metrics = metrics
        self.loss = loss
        self.early_stopping_counter_threshold = early_stopping_counter_threshold
        self.image_savers = image_savers
        self.best_loss_value = float("inf")
        self.early_stopping_counter = 0
        self.loss_value = None

    def _log_metrics(self, time_step: int, data_split: str):
        self.loss_value = None
        for name, loss_value in self.loss.metric_data().items():
            if "loss" in name and "component" not in name:
                self.loss_value = loss_value

            mlflow.log_metric(f"{data_split}_{name}", loss_value, step=time_step)

        if self.loss_value is None:
            raise ValueError(
                "The loss name should contain the string 'loss' and shouldn't contain the string 'component'"
            )

        for metric in self.metrics:
            for name, metric_value in metric.metric_data().items():
                mlflow.log_metric(f"{data_split}_{name}", metric_value, step=time_step)

    def _log_epoch_metrics(
        self,
        time_step: int,
        model: Module,
        dataloader: DataLoader,
        data_split: str,
        **kwargs,
    ) -> None:

        model.eval()

        # Compute Metrics
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

        self._log_metrics(time_step=time_step, data_split=data_split)

    def _assess_early_stopping(self, epoch: int, **kwargs) -> bool:
        if self.best_loss_value > self.loss_value:
            self.best_loss_value = self.loss_value
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_counter_threshold:
                print(f"Early stopping triggered at epoch {epoch}")
                mlflow.log_param("early_stopping_epoch", epoch)
                return False
        return True

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

        if self.image_savers is not None and not isinstance(self.image_savers, list):
            self.image_savers()

        elif isinstance(self.image_savers, list):
            for image_saver in self.image_savers:
                image_saver()

        return self._assess_early_stopping(epoch=epoch)

    def _on_batch_end(self, batch: int, **kwargs) -> None:
        pass

    def __call__(self, callback_hook: str, **kwargs) -> None:
        getattr(self, f"_on_{callback_hook}")(**kwargs)
