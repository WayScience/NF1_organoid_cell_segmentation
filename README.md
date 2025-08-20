# NF1_organoid_cell_segmentation

This project aims to segment single-cells within diseased NF1 organoids by developing a deep-learning segmentation model.

## Model Development Structure

Training results are logged in MLflow and are not currently available in the github.
Each model's code is tracked within git's commit comment history. A trained model's code can be identified by a singular commit. Model commits can be identified by the commit message, which indicate the model's Experiment ID, Run ID, and possibly the model's codename in MLflow. These model commits may be located in main or other branches.

## Training a Model

Each model is trained by executing `train.py` from the MLproject file so the results can be logged to MLflow. The paths inside `train.py` are needed to create the dataset for training the model.
