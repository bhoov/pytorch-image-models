import argparse
import os
from pathlib import Path
from typing import *
import os
import platform
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument("job_name", type=str, help="Name of the job, must be provided")
parser.add_argument("cmd", type=str, help="Pytorch Lightning training script to run on cluster")

parser.add_argument("-l", "--log_dir", type=str, default=None, help="Directory for where to save the tensorboard and stdlogs files. If none provided, default to '~/scratch/tb_logs/{job_name}'")
parser.add_argument("-n", "--num_nodes", type=int, default=4, help="Number of nodes to request")
parser.add_argument("-g", "--gpus_per_node", type=int, default=6, help="Number of gpus per node to request")
parser.add_argument("-t", "--time", type=int, default=360, help="Minutes")
parser.add_argument("-a", "--n_resubmit", type=int, default=20, help="Number of times to resubmit")

def create_lightning_job(
    job_name: str, # Name of the job, must be provided
    cmd: str, # Command to be run
    logdir:Optional[str]=None,
    num_nodes:int=4,
    gpus_per_node:int=6,
    time:int=360, # Minutes
    n_resubmit:int=20, # Number of times to resubmit
):
    if logdir is None:
        logdir = Path.home() / f"scratch/tb_logs/{job_name}"
    else:
        logdir = Path(log_dir) / job_name

    stdlogs_dir = logdir / "stdlogs"
    stdlogs_dir.mkdir(exist_ok=True, parents=True)

    py_cmd = args.cmd + f" --devices $GPUS_PER_NODE --num_nodes $WORLD_SIZE --exp_dir {logdir} --exp_name {job_name}"

    sbatch_cmd = f"""\
#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes={num_nodes}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --time={time}
#SBATCH -o {stdlogs_dir / "slurm_%J-%a.out"}
#SBATCH -e {stdlogs_dir / "slurm_%J-%a.err"}
#SBATCH --signal=SIGUSR1@90
#SBATCH --array=1-{n_resubmit}%1
#SBATCH -J {job_name}


# Adapted from `lightning_sbatch.sbatch`

source activate timm

## With manually set env variables
export WORLD_SIZE=$SLURM_NNODES
export GPUS_PER_NODE=$SLURM_GPUS_PER_NODE

# Path dependent, run from the same folder containing distributed_train.sh
srun bash ~/Projects/timm/tools/distributed_train.sh "{py_cmd}"
    """
    return sbatch_cmd, logdir

def submit_sbatch_cmd(logdir, job_name, sbatch_cmd):
    script = logdir / f"{job_name}.tmp.sh"
    with open(script, 'w') as f:
        print(sbatch_cmd, file=f, flush=True)

    p = subprocess.Popen(['sbatch', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    stdout = stdout.decode("utf-8")
    job_id = stdout.split(" ")[-1].strip()
    
    print(f"Job {job_id} is submitted.")

    print("Generating submitted sbatch script at {}".format(script))

    newscript = logdir / f"{job_name}-{job_id}.sh"

    with open(newscript, 'w') as f:
        print(sbatch_cmd, file=f, flush=True)
    
    script.unlink()
    
    print("Done")
    

if __name__ == "__main__":
    args = parser.parse_args()
    sbatch_cmd, logdir = create_lightning_job(
        args.job_name,
        args.cmd,
        logdir=args.log_dir,
        num_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        time=args.time,
        n_resubmit=args.n_resubmit
    )
    
    submit_sbatch_cmd(logdir, args.job_name, sbatch_cmd)
#     print(sbatch_cmd)
#     script = logdir / f"{args.job_name}.tmp.sh"
#     with open(script, 'w') as f:
#         print(sbatch_cmd, file=f, flush=True)

#     p = subprocess.Popen(['sbatch', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     stdout, stderr = p.communicate()
#     stdout = stdout.decode("utf-8")
#     job_id = stdout.split(" ")[-1].strip()
    
#     print(f"Job {job_id} is submitted.")

#     print("Generate submitted sbatch script at {}".format(script))

#     newscript = logdir / f"{args.job_name}-{job_id}.sh"

#     with open(newscript, 'w') as f:
#         print(sbatch_cmd, file=f, flush=True)
    
#     script.unlink()
