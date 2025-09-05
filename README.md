# NF1_organoid_cell_segmentation

This project aims to segment single-cells within diseased NF1 organoids by developing a deep-learning segmentation model.

## Model Development Structure

Training results are logged in MLflow and are not currently available in the github.
Each model's code is tracked within git's commit comment history. A trained model's code can be identified by a singular commit. Model commits can be identified by the commit message, which indicate the model's Experiment ID, Run ID, and possibly the model's codename in MLflow. These model commits may be located in main or other branches.

## Training a Model

Each model is trained by executing `train.py` from the MLproject file so the results can be logged to MLflow. The paths inside `train.py` are needed to create the dataset for training the model.

## Model Descriptions
All models will be trained with Experiment ID: 310992065458859481

These are the models developed in reverse chronological order:

---

**Commit:** `25f5b66e2b56dbf9c489645f6eb56610120c4ad6` <br>
**Codename:** luxuriant-shad-666

- **Architecture:** UNet Generator
- **Task:** One-to-One slice segmentation mask prediction
- **QC / Filtering:** Does not perform any QC or filtering of images or slices
- **Training Data:** Trained on all slices
- **Preprocessing:**
  - Each input slice is normalized
  - Each input is padded to preserve dimensionality (height and width divisible by 16)

---

**Commit:** `b9c5ab4af0ef5db82f3575988d3e9eaeb891bd98` <br>
**Codename:** casual-stork-830

> **Important!!!** The computer crashed during this run due to a device-side CUDA error <br>
> **Note:** The first two sub-runs completed without error

- **Architecture:** UNet Generator
- **Task:** One-to-One slice segmentation mask prediction
- **QC / Filtering:** Does not perform any QC or filtering of images or slices
- **Training Data:** Trained on all slices
- **Preprocessing:**
  - Each input slice is normalized
  - Each input is padded to preserve dimensionality (height and width divisible by 16)

---

**Commit:** `d993cd928bb181fd723d9be4bd8f28ff87ec14a0` <br>
**Codename:** bright-calf-792

> **Important!!!** The pyproject.toml file and the uv.lock file were not included in this commit <br>
> **Important!!!** The computer crashed during this run

- **Architecture:** UNet Generator
- **Task:** One-to-One slice segmentation mask prediction
- **QC / Filtering:** Does not perform any QC or filtering of images or slices
- **Training Data:** Trained on all slices
- **Preprocessing:**
  - Each input slice is normalized
  - Each input is padded to preserve dimensionality (height and width divisible by 16)

---

**Commit:** `1ccccd1595607ddbebac95f93aa6ecad6184c414` <br>
**Codename:** stylish-flea-695

> **Note:** Manually ended early because images weren't being saved <br>
> **Important!!!** Did not commit the collator, so this model will also need the collator committed later <br>
> **Important!!!** The pyproject.toml file and the uv.lock file were not included in this commit

- **Architecture:** UNet Generator
- **Task:** One-to-One slice segmentation mask prediction
- **QC / Filtering:** Does not perform any QC or filtering of images or slices
- **Training Data:** Trained on all slices
- **Preprocessing:**
  - Each input slice is normalized
  - Each input is padded to preserve dimensionality (height and width divisible by 16)

---
