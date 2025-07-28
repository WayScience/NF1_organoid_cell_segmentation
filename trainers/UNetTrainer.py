import copy
import pathlib
from collections import defaultdict
from inspect import signature
from typing import Optional

import mlflow
import torch
import utils.visualize_model_performance as vm
from farmhash import Fingerprint64
from losses.AbstractLoss import AbstractLoss
from torch.utils.data import DataLoader, random_split


class UNetTrainer:
    """
    Orchestrates training and evaluation of paired stain-to-stain translational modeling.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_optimizer: torch.optim.Optimizer,
        model_loss: AbstractLoss,
        train_dataloader: torch.utils.data.Dataset,
        val_dataloader: torch.utils.data.Dataset,
        callbacks=False,
        epochs: int = 10,
        patience: int = 5,
    ):

        self.model = model
        self.model_optimizer = model_optimizer
        self.model_loss = model_loss
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs

        # Also known as an early stopping counter threshold
        self._patience = patience

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_model = None
        self.best_loss = float("inf")
        self.early_stop_counter = 0

    def _log_metrics(self, _metrics: defaultdict[str, float], _datasplit: str):

        for metric_name, metric_value in _metrics.items():
            mlflow.log_metric(
                f"{_datasplit}_batch_averaged_{metric_name}_per_epoch",
                metric_value,
                step=self._epoch,
            )

    @property
    def model(self):
        return self.best_model

    def train(self):
        #train_data = {"val_dataloader": self.val_dataloader}
        train_data["continue_training"] = True

        for epoch in range(self.epochs):
            if not train_data["continue_training"]:
                break

            train_data["callback_hook"] = "on_epoch_start"
            self.callbacks(**train_data)
            #print(f"Starting epoch {epoch}")

            self.model.train()

            for batch, batch_data in enumerate(self.train_dataloader):
                train_data["callback_hook"] = "on_batch_start"
                train_data["batch"] = batch
                train_data["batch_data"] = batch_data
                self.callbacks(**train_data)
                #print(f"Starting batch {batch_idx}")

                train_data["generated_predictions"] = self.model(batch_data["input"])
                train_data["model_loss"] = self.model_loss(
                    _targets=targets,
                    _generated_predictions=generated_predictions,
                )

                # Update the Model
                self.model_optimizer.zero_grad()
                train_data["model_loss"].backward()
                self.model_optimizer.step()

                train_data["model"] = self.model
                train_data["callback_hook"] = "on_batch_end"

                self.callbacks(**train_data)

            train_data["callback_hook"] = "on_epoch_end"
            self.callbacks(**train_data)
            """

                training_losses[self._generator_loss.metric_name] += generator_loss
                training_losses[
                    self._discriminator_loss.metric_name
                ] += discriminator_loss

                if self._auxiliary_metrics:

                    auxiliary_metric_parameters = {
                        "_generated_outputs": generated_outputs,
                        "_fake_classification_outputs": fake_classification_outputs,
                        "_real_classification_outputs": real_classification_outputs,
                        "_targets": targets,
                    }

                    for metric in self._auxiliary_metrics:
                        param_names = signature(metric.forward).parameters.keys()
                        parameters = {
                            k: v
                            for k, v in auxiliary_metric_parameters.items()
                            if k in list(param_names)
                        }
                        auxiliary_metric_values[metric.metric_name] += metric(
                            **parameters
                        )

            if self._auxiliary_metrics:
                auxiliary_metric_values = {
                    loss_name: loss / len(self.train_loader)
                    for loss_name, loss in auxiliary_metric_values.items()
                }
                training_losses = training_losses | auxiliary_metric_values

            self._log_metrics(training_losses, "train")

            validation_losses = self.evaluation_losses(_data_loader=self.val_loader)
            self._log_metrics(validation_losses, "validation")

            # Define the validation loss to use for early stopping
            if validation_losses[self._generator_loss.metric_name] < self._best_loss:
                self._best_loss = validation_losses[self._generator_loss.metric_name]
                self._early_stop_counter = 0
                self._best_generator_model = self._generator_model
                self._best_discriminator_model = self._discriminator_model

            else:
                self._early_stop_counter += 1

            mlflow.log_metric(
                "early_stopping_counter_per_epoch",
                self._early_stop_counter,
                step=self._epoch,
            )

            if self._early_stop_counter >= self._patience:
                break

        return (
            self._best_loss,
            self._best_generator_model,
            self._best_discriminator_model,
        )

    def visualize_per_epoch_examples(
        self,
        _inputs: torch.Tensor,
        _outputs: torch.Tensor,
        _targets: torch.Tensor,
        _data_loader: DataLoader,
        _input_name: str,
    ):
        """Use the image's unique name to consistently decide if the image should be sampled based on the predetermined target frequency."""

        if self._example_images_per_epoch > 0:

            num_cells = len(_data_loader.dataset)
            divisor = 10_000
            mod_cutoff = (self._example_images_per_epoch / num_cells) * divisor
            normalization_factor = self.val_dataset.dataset.input_transform[
                0
            ].normalization_factor

            if self._save_pretrained_generated_imgs:
                _pretrained_outputs = self._pretrained_model(_inputs).detach()

            pretrained_outputs = None

            for input_idx, input_name in enumerate(_input_name):

                # Remove channel component of name
                start_ch_idx = input_name.find("CH")
                input_name = (
                    input_name[: start_ch_idx - 1] + input_name[start_ch_idx + 3 :]
                )
                input_name = input_name.replace("_illumcorrect.tiff", "")

                if mod_cutoff > (Fingerprint64(input_name) % divisor):

                    imgs_path = pathlib.Path(
                        f"generated_image_epoch_montage/{input_name}"
                    )
                    imgs_path.mkdir(parents=True, exist_ok=True)
                    img_path = f"{imgs_path}/epoch_{self._epoch}_{input_name}"

                    input = vm.format_img(
                        _tensor_img=_inputs[input_idx].unsqueeze(1),
                        _normalization_factor=normalization_factor,
                    )

                    output = vm.format_img(
                        _tensor_img=_outputs[input_idx],
                        _normalization_factor=normalization_factor,
                    )

                    if self._save_pretrained_generated_imgs:
                        pretrained_outputs = vm.format_img(
                            _tensor_img=_pretrained_outputs[input_idx],
                            _normalization_factor=normalization_factor,
                        )

                    target = vm.format_img(
                        _tensor_img=_targets[input_idx],
                        _normalization_factor=normalization_factor,
                    )

                    vm.visualize_stains(
                        _input=input,
                        _output=output,
                        _target=target,
                        _image_path=img_path,
                        _title=input_name,
                        _pretrained_output=pretrained_outputs,
                    )

    def evaluation_losses(self, _data_loader: torch.utils.data.DataLoader):
        """Computes the loss for an evaluation datasplit, e.g. validation or testing."""

        self._generator_model.eval()
        self._discriminator_model.eval()
        losses = defaultdict(float)
        auxiliary_metric_values = defaultdict(float)

        for inputs, targets, metadata in _data_loader:
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            generated_outputs = self._generator_model(inputs).detach()
            fake_classification_outputs = self._discriminator_model(inputs).detach()
            generator_loss = self._generator_loss(
                _fake_classification_outputs=fake_classification_outputs,
                _generated_outputs=generated_outputs,
                _targets=targets,
                _epoch=0,
            )

            self.visualize_per_epoch_examples(
                _inputs=inputs,
                _outputs=generated_outputs,
                _targets=targets,
                _data_loader=_data_loader,
                _input_name=metadata["input_name"],
            )

            real_classification_outputs = self._discriminator_model(targets).detach()

            discriminator_loss = self._discriminator_loss(
                _gradients=torch.zeros_like(targets),
                _real_classification_outputs=real_classification_outputs,
                _fake_classification_outputs=fake_classification_outputs,
            )

            losses[self._generator_loss.metric_name] += generator_loss
            losses[self._discriminator_loss.metric_name] += discriminator_loss

            if self._auxiliary_metrics:

                auxiliary_metric_parameters = {
                    "_generated_outputs": generated_outputs,
                    "_fake_classification_outputs": fake_classification_outputs,
                    "_real_classification_outputs": real_classification_outputs,
                    "_targets": targets,
                }

                for metric in self._auxiliary_metrics:
                    param_names = signature(metric.forward).parameters.keys()
                    parameters = {
                        k: v
                        for k, v in auxiliary_metric_parameters.items()
                        if k in list(param_names)
                    }
                    auxiliary_metric_values[metric.metric_name] += metric(**parameters)

        losses = {
            loss_name: loss / len(_data_loader) for loss_name, loss in losses.items()
        }

        if auxiliary_metric_values:
            auxiliary_metric_values = {
                loss_name: loss / len(self.train_loader)
                for loss_name, loss in auxiliary_metric_values.items()
            }
            losses = losses | auxiliary_metric_values

        return losses
        """
