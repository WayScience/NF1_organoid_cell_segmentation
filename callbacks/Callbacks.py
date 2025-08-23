from typing import Any, List, Optional, Union

import mlflow
import torch
from mlflow.models import ModelSignature
from mlflow.models.signature import infer_signature
from torch.amp import autocast
from torch.nn import Module
from torch.utils.data import DataLoader


class Callbacks:
    """
    Triggered at certain points during model training according to the callback hook.
    """

    def __init__(
        self,
        metrics: List,
        loss: Module,
        early_stopping_counter_threshold: int,
        image_savers: Optional[Union[Any, List[Any]]] = None,
        image_postprocessor: Any = lambda x: x,
    ):
        self.metrics = metrics
        self.loss = loss
        self.early_stopping_counter_threshold = early_stopping_counter_threshold
        self.image_savers = image_savers
        self.image_postprocessor = image_postprocessor
        self.best_loss_value = float("inf")  # We always want to minimize the loss
        self.early_stopping_counter = 0
        self.loss_value = None

    def _log_metrics(self, time_step: int):
        """
        Log metrics at an predefined time step (including components of metrics)
        """

        for name, loss_value in self.loss.get_metric_data().items():

            # Stores the loss value to assess if training should stop early
            if "loss" in name and "component" not in name:
                self.loss_value = loss_value

            mlflow.log_metric(name, loss_value, step=time_step)

        if self.loss_value is None:
            raise ValueError(
                "The loss name should contain the string 'loss' and shouldn't contain the string 'component'"
            )

        for metric in self.metrics:
            for name, metric_value in metric.get_metric_data().items():
                mlflow.log_metric(name, metric_value, step=time_step)

    def _log_epoch_metrics(
        self,
        time_step: int,
        model: Module,
        dataloader: DataLoader,
        data_split: str,
        device: Union[str, torch.device] = "cuda",
        **kwargs,
    ) -> None:

        model.eval()

        with torch.no_grad():
            for samples in dataloader:
                with autocast(enabled=kwargs["use_amp"], device_type=device):
                    generated_predictions = model(samples["input"])

                    self.loss(
                        generated_predictions=generated_predictions,
                        targets=samples["target"],
                        data_split_logging=data_split,
                    )

                generated_predictions = self.image_postprocessor(
                    model(samples["input"])
                )

                for metric in self.metrics:
                    metric(
                        generated_predictions=generated_predictions,
                        targets=samples["target"],
                        data_split_logging=data_split,
                    )

            self._log_metrics(time_step=time_step)

    def _assess_early_stopping(
        self, epoch: int, signature: ModelSignature, model: Module, **kwargs
    ) -> bool:
        if self.best_loss_value > self.loss_value:
            self.best_loss_value = self.loss_value
            self.early_stopping_counter = 0

            mlflow.pytorch.log_model(
                model,
                name="model",
                signature=signature,
                step=epoch,
            )
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_counter_threshold:
                print(f"Early stopping triggered at epoch {epoch}")
                mlflow.log_param("early_stopping_epoch", epoch)
                return False
        return True

    def _prepare_signature(
        self, input_example: torch.Tensor, model: Module
    ) -> ModelSignature:
        model.eval()
        with torch.no_grad():
            output_example = (
                self.image_postprocessor(model(input_example)).detach().cpu().numpy()
            )

        input_numpy = input_example.detach().cpu().numpy().astype("float32")

        return infer_signature(input_numpy, output_example)

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
        device: Union[str, torch.device] = "cuda",
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
                device=device,
                **kwargs,
            )

        # Images can be saved in different ways if desired in the future
        if self.image_savers is not None and not isinstance(self.image_savers, list):
            self.image_savers(
                dataset=val_dataloader.dataset.dataset, model=model, epoch=epoch
            )

        val_sample = next(iter(val_dataloader))
        val_sample = val_sample["input"]
        signature = self._prepare_signature(input_example=val_sample, model=model)

        return self._assess_early_stopping(
            epoch=epoch, signature=signature, model=model
        )

    def _on_batch_end(self, batch: int, **kwargs) -> None:
        pass

    def __call__(self, callback_hook: str, **kwargs) -> None:
        """
        Must return to possibly stop model training early
        """
        return getattr(self, f"_{callback_hook}")(**kwargs)
