#!/bin/bash

# Activating the conda environment
source activate openmmlab

FOLD=1
RANDOM_PORT=$((RANDOM % 64512 + 1024))
work_dir=segformer_work_dirs
python -m torch.distributed.launch --nnodes=1 --nproc_per_node=1 --master_port=$RANDOM_PORT train.py configs/segformer/segformer_mit-b5-b5_512x512_80k_calcium.py --launcher pytorch --cfg-options "val_dataloader.dataset.ann_file=ImageSets/val_${FOLD}.txt" --cfg-options "train_dataloader.dataset.ann_file=ImageSets/train_${FOLD}.txt" --cfg-options "test_dataloader.dataset.ann_file=ImageSets/test.txt" --work-dir $work_dir/segformer_mit-b5-b5_512x512_80k_calcium_fold${FOLD} --cfg-options "test_evaluator.output_dir=$work_dir/test_fold${FOLD}" --amp