#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=4
#SBATCH --gpus-per-node=6
#SBATCH --time=360
#SBATCH -o slurm_%J-%a.out
#SBATCH -e slurm_%J-%a.err
#SBATCH --signal=SIGUSR1@90
#SBATCH --array=1-20%1

# Adapted from `lightning_sbatch.sbatch`

source activate timm

## With manually set env variables
export WORLD_SIZE=$SLURM_NNODES
export GPUS_PER_NODE=$SLURM_GPUS_PER_NODE

# Path dependent, run from the same folder containing distributed_train.sh
srun bash ~/Projects/timm/tools/distributed_train.sh "python lightning_train.py  /gpfs/u/home/DAMT/DAMThvrb/scratch-shared/datasets/imagenet1k --model et_base_patch16_224 --num-classes 1000 --pin-mem --no-prefetcher --batch-size 32 --val-split val --aa rand --reprob 0.5 --mixup 0.8 --cutmix 1.0 --strategy ddp_find_unused_parameters_false --accelerator gpu --max_epochs 1000 --epochs 1000 --lr 0.01 --exp_dir /gpfs/u/home/DAMT/DAMThvrb/scratch/new-lightning-logs/early_tests --exp_name et-base-long --gradient_clip_val=0.5 --progress_bar_refresh_rate 200 --devices $GPUS_PER_NODE --num_nodes $WORLD_SIZE" 