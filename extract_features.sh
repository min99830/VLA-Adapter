#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

python vla-scripts/extract_features.py \
    --config_file_path pretrained_models/LIBERO-Spatial-Pro \
    --vlm_path pretrained_models/LIBERO-Spatial-Pro \
    --resum_vla_path pretrained_models/LIBERO-Spatial-Pro \
    --dataset_name libero_spatial_no_noops \
    --data_root_dir data/libero \
    --batch_size 8 \
    --use_minivlm True \
    --use_proprio True