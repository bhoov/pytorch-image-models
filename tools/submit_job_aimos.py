#!/usr/bin/env python3

import argparse
import os
import platform
import subprocess

parser = argparse.ArgumentParser(description='SLURM job submission')
parser.add_argument('-n', '--num_nodes', default=1, type=int, help='number of nodes')
parser.add_argument('-d', '--num_deps', default=1, type=int, help='number of consecutive jobs')
parser.add_argument('-j', '--job_name', default='my_job', help='job name and log name prefix')
parser.add_argument('-t', '--time', default=6, type=int, help='hours')
parser.add_argument('--wd', default=os.getcwd(), type=str, help='root of working directory')
parser.add_argument('cmd', help='whole command, quoted by ""', metavar='CMD')

args = parser.parse_args()

def main():

    home = os.path.expanduser("~")
    log_path = os.path.join(home, 'sbatch_scripts')
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    arch = platform.uname().processor

    if arch == 'x86_64':
        num_cpus=10
        num_gpus=8
        partition='npl'
    elif arch == 'ppc64le':
        num_cpus=24
        num_gpus=6
        partition='el8'

    for i in range(args.num_deps):
        dependency = "" if i == 0 else "#SBATCH --dependency=afterany:{}".format(job_id)

        sbatch_job = """#!/bin/bash -x

#SBATCH -J {job_name}
#SBATCH -o {log_path}/my_job_%j.out
#SBATCH -e {log_path}/my_job_%j.err
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --nodes={num_nodes}
#SBATCH --cpus-per-gpu={num_cpus}
#SBATCH --time=0{time}:00:00
#SBATCH -p {partition}
{dependency}

cd {wd}
srun {cmd}
wait
""".format(job_name=args.job_name, num_nodes=args.num_nodes, wd=args.wd, log_path=log_path, time=args.time, num_gpus=num_gpus, num_cpus=num_cpus, partition=partition, dependency=dependency, cmd=args.cmd)

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
