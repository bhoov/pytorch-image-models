#!/usr/bin/env python3

"""
Much like submit_job_aimos.py, but with the following changes:
- Assumes that '--devices' and '--num_nodes' of the 'cmd' need to be set
- Adds a SLURM usr kill signal to notify the lightning process that the job is almost done and needs to save
- A way to reload automatically from the logging directory.
- Creates environment variables that are needed for the python script to launch correctly
    MASTER_ADDR
    MASTER_PORT
    WORLD_SIZE
    NODE_RANK 

Submit a lightning python script must be configured in the following way:
- The trainer's hyperparamters are exposed through argparse ( to add the correct number of devices and nodes)

"""

import argparse
import os
import platform
import subprocess

parser = argparse.ArgumentParser(description='SLURM job submission')
parser.add_argument('-n', '--num_nodes', default=1, type=int, help='number of nodes')
parser.add_argument('-g', '--gpus_per_node', default=None, type=int, help='number of gpus per node')
parser.add_argument('-c', '--cpus_per_gpu', default=None, type=int, help='number of cpus per node')
parser.add_argument('-d', '--num_deps', default=1, type=int, help='number of consecutive jobs')
parser.add_argument('-j', '--job_name', default='my_job', help='job name and log name prefix')
parser.add_argument('-t', '--time', default=360, type=int, help='minutes')
parser.add_argument('--wd', default=os.getcwd(), type=str, help='root of working directory')
parser.add_argument('cmd', help='whole command, quoted by ""', metavar='CMD')

args = parser.parse_args()


def ifnone(a,b):
    if a is None:
        return b
    return a

def main():

    home = os.path.expanduser("~")
    log_path = os.path.join(home, 'sbatch_scripts')
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    arch = platform.uname().processor

    if arch == 'x86_64':
        num_cpus=ifnone(args.cpus_per_gpu, 10)
        num_gpus=ifnone(args.gpus_per_node, 8)
        partition='npl'
    elif arch == 'ppc64le':
        num_cpus=ifnone(args.cpus_per_gpu, 24)
        num_gpus=ifnone(args.gpus_per_node, 6)
        partition='el8'

    cmd = args.cmd + f" --devices {num_gpus} --num_nodes {args.num_nodes}"

    for i in range(args.num_deps):
        dependency = "" if i == 0 else "#SBATCH --dependency=afterany:{}".format(job_id)

        #env_variables MASTER_ADDR=dcs107 MASTER_PORT=14502 WORLD_SIZE=2 NODE_RANK=0
        export_var = "MASTER_ADDR,MASTER_PORT,WORLD_SIZE,NODE_RANK"
        sbatch_job = f"""#!/bin/bash -x

#SBATCH -J {args.job_name}
#SBATCH -o {log_path}/my_job_%j.out
#SBATCH -e {log_path}/my_job_%j.err
#SBATCH --nodes={args.num_nodes}
#SBATCH --gpus-per-node={num_gpus}
#SBATCH --cpus-per-gpu={num_cpus}
#SBATCH --time={args.time}
#SBATCH -p {partition}
{dependency}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12532 # Random number on master addr
export NODE_RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "NODE_RANK=$NODE_RANK"

source activate timm
cd {args.wd}
srun --export={export_var} {cmd}
wait
"""
        print(sbatch_job)
        script = "{}/{}.sh".format(log_path, args.job_name)
        print("Generate sbatch script at {}".format(script))
        with open(script, 'w') as f:
            print(sbatch_job, file=f, flush=True)

        p = subprocess.Popen(['sbatch', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        stdout = stdout.decode("utf-8")
        job_id = stdout.split(" ")[-1]
        print(f"Job {job_id.strip()} is submitted.")


if __name__ == "__main__":
    main()
