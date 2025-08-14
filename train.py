import pathlib
import random
from typing import Any

import mlflow
import numpy as np
import optuna
import torch

from callbacks.Callbacks import Callbacks
from callbacks.utils.SampleImages import SampleImages
from callbacks.utils.SaveEpochSlices import SaveEpochSlices
from datasets.dataset_00.CellSlicetoSliceDataset import CellSlicetoSliceDataset
from datasets.dataset_00.utils.ImagePreProcessor import ImagePreProcessor
from datasets.dataset_00.utils.ImageSelector import ImageSelector
from metrics.BCE import BCE
from metrics.ConfusionMetrics import ConfusionMetrics
from metrics.Dice import Dice
from models.UNet import UNet
from splitters.HashSplitter import HashSplitter
from trainers.UNetTrainer import UNetTrainer

# |%%--%%| <EZ6SOiX7Wc|Twmb6mPjf5>


class OptimizationManager:
    """
    For running optuna optimization studies.
    """

    def __init__(self, trainer: Any, **trainer_kwargs):

        self.trainer = trainer
        self.trainer_kwargs = trainer_kwargs

    def __call__(self, trial):
        optimizer_params = {
            "params": self.trainer_kwargs["model"].parameters(),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "betas": (0.5, 0.999),
        }

        with mlflow.start_run(nested=True, run_name=f"trial {trial.number}"):
            optimizer = torch.optim.Adam(**optimizer_params)
            self.trainer_kwargs["model_optimizer"] = optimizer
            opt_params = optimizer.param_groups[0].copy()

            del opt_params["params"]
            opt_params = {
                f"optimizer_{opt_param_name}": opt_param
                for opt_param_name, opt_param in opt_params.items()
            }

            mlflow.log_params(opt_params)
            mlflow.set_tag("optimizer_class", optimizer.__class__.__name__.lower())
            self.trainer(**self.trainer_kwargs)
            self.trainer.train()


# |%%--%%| <Twmb6mPjf5|sNNHe7R4s3>
r"""°°°
## Find the root of the git repo on the host system
°°°"""
# |%%--%%| <sNNHe7R4s3|5uHY3JoEKK>

# Get the current working directory
cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd

else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break

# Check if a Git root directory was found
if root_dir is None:
    raise FileNotFoundError("No Git root directory found.")

# |%%--%%| <5uHY3JoEKK|ExjIoeHzuw>
r"""°°°
# Inputs
°°°"""
# |%%--%%| <ExjIoeHzuw|uHCF3KHHYz>

root_data_path = root_dir / "big_drive/NF1_organoid_processed_patients"
patient_folders = [[p for p in root_data_path.iterdir() if root_data_path.is_dir()][0]]

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


# |%%--%%| <9NUAycuR83|cunSoruBAQ>

img_selector = ImageSelector(
    number_of_slices=1,
    slice_stride=1,
    crop_stride=256,
    crop_height=256,
    crop_width=256,
    device=device,
)
image_preprocessor = ImagePreProcessor(pad_to_multiple=16, device=device)

img_dataset = CellSlicetoSliceDataset(
    root_data_path=root_data_path,
    patient_folders=patient_folders,
    image_selector=img_selector,
    image_preprocessor=image_preprocessor,
)

# |%%--%%| <cunSoruBAQ|gUtaBucbVw>

hash_splitter = HashSplitter(
    dataset=img_dataset, batch_size=10, train_frac=0.8, val_frac=0.1
)

# |%%--%%| <gUtaBucbVw|H2ARqRccUc>

train_dataloader, val_dataloader, test_dataloader = hash_splitter()

# |%%--%%| <H2ARqRccUc|84VWoN8eUi>

image_dataset_idxs = SampleImages(
    dataloader=val_dataloader,
    number_of_images=200,  # 35,
)()

# |%%--%%| <84VWoN8eUi|TjsS5lPUGF>

image_saver = SaveEpochSlices(
    image_dataset_idxs=image_dataset_idxs, data_split="validation"
)

# |%%--%%| <TjsS5lPUGF|CNv30fImCD>

loss = BCE(is_loss=True, use_logits=True, reduction="mean", device=device)
metrics = [
    BCE(is_loss=False, use_logits=True, reduction="mean", device=device),
    Dice(use_logits=True, prediction_threshold=0.5, is_loss=False, device=device),
    ConfusionMetrics(use_logits=True, prediction_threshold=0.5, device=device),
]

callbacks = Callbacks(
    metrics=metrics,
    loss=loss,
    early_stopping_counter_threshold=13,
    image_savers=image_saver,
)

# |%%--%%| <CNv30fImCD|unwqdPi1Wn>

unet = UNet(in_channels=1, out_channels=1).to(device)

# |%%--%%| <unwqdPi1Wn|h3r11MNDxO>

optimizer_params = {
    "params": unet.parameters(),
    "lr": 1e-2,
    "betas": (0.5, 0.999),
}

optimizer = torch.optim.Adam(**optimizer_params)

# |%%--%%| <h3r11MNDxO|0Gkz2JJSt9>

torch.cuda.empty_cache()

# |%%--%%| <0Gkz2JJSt9|6ybuV3PiSm>

trainer = UNetTrainer(
    model=unet,
    model_optimizer=optimizer,
    model_loss=loss,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    callbacks=callbacks,
    epochs=2,
    device="cuda",
    use_amp=True,
)
trainer.train()

# |%%--%%| <6ybuV3PiSm|mWPQQ5jFxo>

"""
optimization_manager = OptimizationManager(
    trainer=UNetTrainer,
    trainer_kwargs={
        "model": unet,
        "model_loss": loss,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "image_postprocessor": image_postprocessor,
        "callbacks": callbacks,
        "epochs": 2,
        "device": "cuda",
    },
)

study = optuna.create_study(direction="minimize")
study.optimize(optimization_manager, n_trials=2)
"""
