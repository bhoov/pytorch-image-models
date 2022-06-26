#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --time=60
#SBATCH -o slurm_%J.out
#SBATCH -e slurm_%J.err
#SBATCH --signal=SIGUSR1@90

source activate timm

## With manually set env variables

export WORLD_SIZE=$SLURM_NNODES
export GPUS_PER_NODE=$SLURM_GPUS_PER_NODE

# Path dependent
srun bash ~/Projects/timm/tools/distributed_train.sh "python lightning_train.py  /gpfs/u/home/DAMT/DAMThvrb/scratch-shared/datasets/imagenet1k --model et_base_patch16_224 --num-classes 1000 --pin-mem --no-prefetcher --batch-size 32 --val-split val --aa rand --reprob 0.5 --mixup 0.8 --cutmix 1.0 --strategy ddp_find_unused_parameters_false --accelerator gpu --max_epochs -1 --lr 0.05 --exp_dir /gpfs/u/home/DAMT/DAMThvrb/scratch/new-lightning-logs/early_tests --exp_name et-small-data2 --gradient_clip_val=0.5 --limit_train_batches=50 --limit_val_batches=10 --val_check_interval=50 --devices $GPUS_PER_NODE --num_nodes $WORLD_SIZE"
