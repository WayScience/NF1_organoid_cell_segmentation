from typing import List, Optional

import mlflow
import torch


class Callbacks:
    def __init__(
        self,
        metrics: List,
        loss: torch.nn.Module,
        num_samples_to_plot: Optional[int] = None,
    ):
        self.metrics = metrics
        self.loss = loss
        self.num_samples_to_plot = num_samples_to_plot

    def _log_loss(
        self, time_step, data_split, predictions, pixel_probs, targets, **kwargs
    ):
        loss = self.loss(
            generated_predictions=predictions,
            generated_pixel_probs=pixel_probs,
            targets=targets,
            data_split=data_split,
            **kwargs,
        )
        for name, value in loss.items():
            mlflow.log_metric(f"{data_split}_{name}", value, step=time_step)

    def _log_metrics(
        self, time_step, data_split, predictions, pixel_probs, targets, **kwargs
    ):
        for metric in self.metrics:
            metric_data = metric(
                generated_predictions=predictions,
                generated_pixel_probs=pixel_probs,
                targets=targets,
                data_split=data_split,
                **kwargs,
            )
            for name, value in metric_data.items():
                mlflow.log_metric(f"{data_split}_{name}", value, step=time_step)

    def inference(
        self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, **kwargs
    ):
        model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = model(inputs)
                all_preds.append(outputs)
                all_targets.append(targets)

        return torch.cat(all_preds), torch.cat(all_targets)

    def _on_epoch_start(self, epoch, **kwargs):
        print(f"Starting epoch {epoch}")

    def _on_batch_start(self, batch, **kwargs):
        print(f"Starting batch {batch}")

    def _on_epoch_end(
        self, epoch, generated_predictions, targets, val_dataloader, **kwargs
    ):
        train_probs = torch.sigmoid(generated_predictions)

        val_preds, val_targets = self.inference(
            kwargs["model"], val_dataloader, **kwargs
        )
        val_probs = torch.sigmoid(val_preds)

        for split, preds, probs, targs in [
            ("train", generated_predictions, train_probs, targets),
            ("validation", val_preds, val_probs, val_targets),
        ]:
            self._log_loss(epoch, split, preds, probs, targs, **kwargs)
            self._log_metrics(epoch, split, preds, probs, targs, **kwargs)

    def _on_batch_end(self, batch, **kwargs):
        pass

    def __call__(self, callback_hook, **kwargs):
        getattr(self, f"_on_{callback_hook}")(**kwargs)
