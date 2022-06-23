#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=2
#SBATCH -o slurm_%J.out
#SBATCH -e slurm_%J.err
conda activate timm


srun python cluster_env.py
#srun echo "NTASKS=$SLURM_NTASKS"
#srun echo "NODELIST="${SLURM_JOB_NODELIST}; master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1); export MASTER_ADDR=$master_addr; export MASTER_PORT=12543; echo "MASTER_ADDR="$MASTER_ADDR; echo #$SLURM_NODEID; 
#wait
