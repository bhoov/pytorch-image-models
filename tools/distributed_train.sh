#!/bin/bash -l
# 
# Wrap a pytorch_lightning training script for SLURM training
# 
# - Number of nodes in the script is automatically set, number of devices per node needs to be passed in
# - Make sure the correct python environment is activated before calling the script

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12532 # Random number on master addr
export NODE_RANK=$SLURM_NODEID
export WORLD_SIZE=$SLURM_NTASKS # In the new AIMOS system, this is the number of nodes because specify tasks is now confusing
export GPUS_PER_NODE=$SLURM_GPUS_PER_NODE

echo MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE NODE_RANK=$NODE_RANK GPUS_PER_NODE=$GPUS_PER_NODE

$@
# python ~/Projects/timm/tools/lightning_train.py /gpfs/u/home/DAMT/DAMThvrb/scratch-shared/datasets/imagenet1k --model et_base_patch16_224 --num-classes 1000 --pin-mem --no-prefetcher --batch-size 32 --val-split val --aa rand --reprob 0.5 --mixup 0.8 --cutmix 1.0 --strategy ddp_find_unused_parameters_false --accelerator gpu --max_epochs -1 --lr 0.05 --devices $GPUS_PER_NODE --num_nodes $WORLD_SIZE