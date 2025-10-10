import pathlib
import random
from typing import Any

import joblib
import mlflow
import numpy as np
import optuna
import torch

from callbacks.Callbacks import Callbacks
from callbacks.utils.SampleImages import SampleImages
from callbacks.utils.SaveEpochSlices import SaveEpochSlices
from callbacks.utils.SaveWholeSlices import SaveWholeSlices
from datasets.dataset_00.CellSlicetoSliceDataset import CellSlicetoSliceDataset
from datasets.dataset_00.utils.image_metadata import (get_image_paths,
                                                      get_image_specs)
from datasets.dataset_00.utils.ImagePostProcessor import ImagePostProcessor
from datasets.dataset_00.utils.ImagePreProcessor import ImagePreProcessor
from datasets.dataset_00.utils.ImageSelector import ImageSelector
from datasets.dataset_01.AllSlicesDataset import AllSlicesDataset
from metrics.BCE import BCE
from metrics.ConfusionMetrics import ConfusionMetrics
from metrics.Dice import Dice
from models.UNet import UNet
from splitters.HashSplitter import HashSplitter
from trainers.UNetTrainer import UNetTrainer

# |%%--%%| <EZ6SOiX7Wc|Twmb6mPjf5>


class OptimizationManager:
    """
    Optuna objective function with MLflow logging.
    """

    def __init__(
        self,
        trainer: Any,
        hash_splitter: Any,
        dataset: Any,
        callbacks_args: dict[str, Any],
        **trainer_kwargs,
    ):
        self.trainer = trainer
        self.hash_splitter = hash_splitter
        self.dataset = dataset
        self.callbacks_args = callbacks_args
        self.trainer_kwargs = trainer_kwargs

    def __call__(self, trial):
        batch_size = trial.suggest_int("batch_size", 1, 18)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

        train_dataloader, val_dataloader, test_dataloader = self.hash_splitter(
            batch_size=batch_size
        )
        self.trainer_kwargs["train_dataloader"] = train_dataloader
        self.trainer_kwargs["val_dataloader"] = val_dataloader

        optimizer_params = {
            "params": self.trainer_kwargs["model"].parameters(),
            "lr": lr,
            "betas": (0.5, 0.999),
        }

        loss_trainer = BCE(
            is_loss=True, use_logits=True, reduction="mean", device=device
        )

        # We do not care about the gradient stability when evaluating performance
        loss_callbacks = BCE(
            is_loss=True, use_logits=True, reduction="mean", device=device
        )

        metrics = [
            Dice(
                use_logits=False, prediction_threshold=0.5, is_loss=False, device=device
            ),
            ConfusionMetrics(use_logits=False, prediction_threshold=0.5, device=device),
        ]

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            optimizer = torch.optim.Adam(**optimizer_params)
            self.trainer_kwargs["model_optimizer"] = optimizer

            opt_params = optimizer.param_groups[0].copy()
            del opt_params["params"]
            mlflow.log_params({f"optimizer_{k}": v for k, v in opt_params.items()})
            mlflow.log_param("batch_size", batch_size)
            mlflow.set_tag("optimizer_class", optimizer.__class__.__name__.lower())

            self.trainer_kwargs["callbacks"] = Callbacks(
                **self.callbacks_args | {"metrics": metrics, "loss": loss_callbacks}
            )

            trainer_obj = self.trainer(
                **self.trainer_kwargs | {"model_loss": loss_trainer}
            )
            trainer_obj.train()

            return trainer_obj.best_loss_value


# |%%--%%| <Twmb6mPjf5|ExjIoeHzuw>
r"""°°°
# Inputs
°°°"""
# |%%--%%| <ExjIoeHzuw|uHCF3KHHYz>

root_data_path = pathlib.Path("big_drive/NF1_organoid_processed_patients").resolve(
    strict=True
)
patient_folders = [p for p in root_data_path.iterdir() if p.is_dir()]

# |%%--%%| <uHCF3KHHYz|9NUAycuR83>

device = "cuda"
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

mlflow.log_param("random_seed", 0)

description = """
Optimization of the first segmentation model with the following:
- UNet Generator
- One-to-One slice segmentation mask prediction
- Does not perform any QC or filtering of image or filtering of slices
- Trained on all slices
- Each input slice is normalized
- Each input is padded to preserve dimensionality (the height and width dimensions will now be divisible by 16)
"""
mlflow.set_tag("mlflow.note.content", description)

# |%%--%%| <9NUAycuR83|RXyUovWJFX>

image_paths = get_image_paths(patient_folders=patient_folders)
image_specs = get_image_specs(image_paths=image_paths)

# |%%--%%| <RXyUovWJFX|muTDx2W917>

input_crop_shape = (3, 512, 512)

img_selector = ImageSelector(
    input_crop_shape=input_crop_shape,
    target_crop_shape=(1, 512, 512),
    slice_stride=1,
    crop_stride=512,
    device=device,
)

image_preprocessor = ImagePreProcessor(image_specs=image_specs, device=device)
image_postprocessor = ImagePostProcessor()

crop_image_dataset = CellSlicetoSliceDataset(
    image_paths=image_paths,
    image_specs=image_specs,
    image_selector=img_selector,
    image_preprocessor=image_preprocessor,
)

whole_image_dataset = AllSlicesDataset(
    dataset=crop_image_dataset,
    image_specs=image_specs,
    image_preprocessor=image_preprocessor,
)

# |%%--%%| <muTDx2W917|Ljn54YK9d8>

hash_splitter = HashSplitter(
    dataset=crop_image_dataset,
    train_frac=0.8,
    val_frac=0.1,
)

# Batch size is arbitrary here, it is just to sample images, which won't change with batch_size
_, val_dataloader, _ = hash_splitter(batch_size=10)

image_dataset_idxs = SampleImages(dataloader=val_dataloader, number_of_images=300)()

whole_image_saver = SaveWholeSlices(
    image_dataset=whole_image_dataset,
    image_dataset_idxs=image_dataset_idxs,
    image_specs=image_specs,
    stride=(1, 256, 256),
    crop_shape=input_crop_shape,
    pad_mode="reflect",
    image_postprocessor=image_postprocessor,
)

image_prediction_saver = SaveEpochSlices(
    image_dataset=val_dataloader.dataset.dataset,
    image_postprocessor=image_postprocessor,
    image_dataset_idxs=image_dataset_idxs,
)

# |%%--%%| <Ljn54YK9d8|sv6R19116h>

callbacks_args = {
    "early_stopping_counter_threshold": 5,
    "image_savers": [image_prediction_saver, image_prediction_saver],
    "image_postprocessor": image_postprocessor,
}

# |%%--%%| <sv6R19116h|1ibSiDMEcz>

unet = UNet(in_channels=3, out_channels=1)

# |%%--%%| <1ibSiDMEcz|yAnz5nSUyL>

optimization_manager = OptimizationManager(
    trainer=UNetTrainer,
    hash_splitter=hash_splitter,
    dataset=crop_image_dataset,
    callbacks_args=callbacks_args,
    model=unet,
    epochs=30,
    device=device,
)

study = optuna.create_study(study_name="model_training", direction="minimize")
study.optimize(optimization_manager, n_trials=6)

# |%%--%%| <yAnz5nSUyL|bVaGWMfHn6>

joblib.dump(study, "optuna_study.joblib")
mlflow.log_artifact("optuna_study.joblib")
