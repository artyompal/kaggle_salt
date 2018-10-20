#!/bin/bash

# handle errors
set -e

GPU=${1:-0}
BATCH_SIZE=60
DIRECTORY=${2:-"../model_weights"}

for f in $DIRECTORY/*.pth; do
    python3.6 ./make_preds.py --gpu=$GPU --weight_file $f --batch_size=$BATCH_SIZE --flip_tta
done

