#!/bin/bash

# Activating the conda environment
source activate openmmlab

FOLD=0

python -m torch.distributed.launch --nnodes=1 --nproc_per_node=1 train.py configs/radio/radio_linear_8xb2-80k_voc-512x512.py --launcher pytorch --cfg-options "val_dataloader.dataset.ann_file=ImageSets/val_${FOLD}.txt" --cfg-options "train_dataloader.dataset.ann_file=ImageSets/val_${FOLD}.txt" --cfg-options "test_dataloader.dataset.ann_file=ImageSets/test.txt"