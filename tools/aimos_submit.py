#!/usr/bin/env python3

import argparse
import os
import platform
import subprocess

parser = argparse.ArgumentParser(description='SLURM job submission')
parser.add_argument('-n', '--nodes', default=1, type=int, help='number of nodes')
parser.add_argument('-d', '--num_deps', default=1, type=int, help='number of consecutive jobs')
parser.add_argument('-ng', '--ngpus', default=None, help='number of gpus per node')
parser.add_argument('-t', '--time', default=360, type=int, help='minutes')
parser.add_argument('--wd', default=os.getcwd(), type=str, help='root of working directory')
parser.add_argument('-nc', "--ncpus", default=10, type=int, help='Number cpus per gpu')

parser.add_argument('--init_dep', type=str, default='')
parser.add_argument('--model', default='', help='model name')
parser.add_argument('--suffix', default='', help='string suffix appended to job_dir')
parser.add_argument('--job_dir', default='', help='directory for saving slurm stdout, stderr, main log, and checkpoints')
parser.add_argument('--resume', default='', help='give the resume path for the first job, the chained jobs will figure out where is the checkpoint accordingly.')
parser.add_argument('cmd', help='whole command, quoted by ""', metavar='CMD')

args = parser.parse_args()


def main():

    log_folder = args.model if args.suffix == '' else args.model + "-" + args.suffix
    if args.job_dir == "":
        args.job_dir = "checkpoint/experiments/"
    args.job_dir = os.path.join(args.job_dir, log_folder)
    if not os.path.exists(args.job_dir):
        os.makedirs(args.job_dir)

    job_name = log_folder

    args.cmd = args.cmd + f" --output {args.job_dir} --experiment 00 --model {args.model}"

    arch = platform.uname().processor

    if arch == 'x86_64':
        num_gpus=8 if args.ngpus is None else args.ngpus
        partition='npl'
    elif arch == 'ppc64le':
        num_gpus=6 if args.ngpus is None else args.ngpus
        partition='el8'
    num_cpus=10 if args.ncpus is None else args.ncpus
    
    for i in range(args.num_deps):
        cmd = args.cmd
        if i == 0:
            if args.resume != '':
                cmd += f" --resume {args.resume}"
            if args.init_dep:
                dependency = f"#SBATCH --dependency=afterany:{args.init_dep}"
            else:
                dependency = ""
        else:
            dependency = "#SBATCH --dependency=afterany:{}".format(job_id)
            cmd = cmd + f" --resume {args.job_dir}/checkpoint.pth"
        
        sbatch_job = """#!/bin/bash -x

#SBATCH -J {job_name}
#SBATCH -o {log_folder}/%j.out
#SBATCH -e {log_folder}/%j.err
#SBATCH --cpus-per-gpu={num_cpus}
#SBATCH --gpus-per-node={num_gpus}
#SBATCH --nodes={num_nodes}
#SBATCH --time={time}
{dependency}

bash ~/.bashrc
export MASTER_HOSTNAME=`srun hostname -s | sort | head -n 1`
export OMP_NUM_THREADS=1
conda activate timm
cd {wd}
""".format(job_name=job_name, num_nodes=args.nodes, wd=args.wd, log_folder=args.job_dir, time=args.time,
           num_gpus=num_gpus, partition=partition, dependency=dependency, num_cpus=num_cpus)

        sbatch_job += "srun ./tools/train.sh main.py {cmd}\n".format(cmd=cmd)
        #exit(0)
        script = "{}/{}.sh".format(args.job_dir, job_name)
        print("Generate sbatch script at {}".format(script))
        with open(script, 'w') as f:
            print(sbatch_job, file=f, flush=True)

        print("SCRIPPTTTTT")
        print(script)

        p = subprocess.Popen(['sbatch', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        stdout = stdout.decode("utf-8")
        print("STDOUT: ", stdout)
        stderr = stderr.decode("utf-8")
        print("STDerr: ", stderr)
        job_id = stdout.split(" ")[-1]
        print(f"Job {job_id.strip()} is submitted.")

    #os.system('sbatch {}'.format(script))


if __name__ == "__main__":
    main()
