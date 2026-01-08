# NF1_organoid_cell_segmentation

This project aims to segment single-cells within diseased NF1 organoids by developing a deep-learning segmentation model.

## Model Development Structure

Training results are logged in MLflow and are not currently available in the github.
Each model's code is tracked within git's commit comment history. A trained model's code can be identified by a singular commit. Model commits can be identified by the commit message, which indicate the model's Experiment ID, Run ID, and possibly the model's codename in MLflow. These model commits may be located in main or other branches.

## Training a Model

Each model is trained by executing `train.py` from the MLproject file so the results can be logged to MLflow. The paths inside `train.py` are needed to create the dataset for training the model.

## Model Descriptions
All models below were trained with Experiment ID: 185714796021641116

These are the models developed in reverse chronological order:

---

**Commit:** `07e0368d2d435e4ad04902e242cfee934adc50e0` <br>
**Codename:** fearless-eel-256

> **Note:** The binary cross entropy (BCE) was not computed after the first trial (trial 0) <br>

- **Architecture:** UNet Generator
- **Task:** One-to-One slice segmentation mask prediction
- **QC / Filtering:** Does not perform any QC or filtering of images or slices
- **Data:**
  - Trained and evaluated on all patient data except patient NF0014 (Holdout Patient)
  - The model was developed using multiple segmentation masks:
    - Background
    - Inner-cell
    - Cell-boundary
- **Preprocessing:**
  - Each input slice is normalized
  - Each input is padded to preserve dimensionality (height and width divisible by 16)
- **Loss:**
  - The BCE loss is modified to account for segmentation mask pixel frequencies
    - Computed using an exponential moving average during training by initially randomly sampling image patches
  - The logged BCE used a random alpha exponential moving average hyperparameter

---

