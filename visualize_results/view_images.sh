#!/bin/bash
# Converts and views images

set -e

# Path to the folder containing the mlflow images
mlflow_artifacts_path=$1

temp_image_dir=$(mktemp -d)

python3 convert_images.py "$mlflow_artifacts_path" "$temp_image_dir"

while true; do
    temp_zarr_dir=$(mktemp -d)
    chosen_dir=$(find "$temp_image_dir" -type f -name "*.tiff" -exec dirname {} \; | sort -u | fzf -x) || exit
    echo "You choose the directory $chosen_dir"

    uvx nviz tiff_to_zarr \
        --image_dir "$chosen_dir" \
        --output_path "$temp_zarr_dir/output.zarr" \
        --channel_map '{"brightfield":"Brightfield","generated_prediction":"Generated Segmentation","TRANS":"Pipeline Segmentation", "thresholded": "Thresholded Generated Segmentation"}' \
        --scaling_values '(1.0,0.1,0.1)'  # Microscope values

    uvx nviz view_zarr \
        --zarr_dir "$temp_zarr_dir/output.zarr" \
        --scaling_values '(1.0,0.1,0.1)' \
        --headless "False"
done
