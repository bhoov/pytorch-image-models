#!/usr/bin/env python3

import argparse
import os
import platform
import subprocess

parser = argparse.ArgumentParser(description='SLURM job submission')
parser.add_argument('-n', '--num_nodes', default=1, type=int, help='number of nodes', nargs='+')
parser.add_argument('--num_jobs', default=1, type=int, help='number of sub_jobs')
parser.add_argument('-d', '--num_deps', default=1, type=int, help='number of consecutive jobs')
parser.add_argument('-j', '--job_name', default='my_job', help='job name and log name prefix')
parser.add_argument('-t', '--time', default=6, type=int, help='hours')
parser.add_argument('--bb', action='store_true', help='request burst buffer. Unsupported for my training')
parser.add_argument('--wd', default=os.getcwd(), type=str, help='root of working directory')
parser.add_argument('cmd', help='whole command, quoted by ""', metavar='CMD', nargs='+')

args = parser.parse_args()

if len(args.num_nodes) == 1 and args.num_jobs != 1:
    args.num_nodes = args.num_nodes * args.num_jobs
elif args.num_jobs == 1 and  len(args.num_nodes) > 1:
    args.num_jobs = len(args.num_nodes)
    

def parse_bb(cmd, nj):
    dataset = None
    modality = []
    elements = cmd.split(" ")
    use_pyav = False
    for i, x in enumerate(elements):
        if x == '--dataset':
            dataset = elements[i+1]
            break
            
    for i, x in enumerate(elements):
        if x == '--modality':
            for j in range(i+1, len(elements)):
                if elements[j][0] == '-':
                    break
                modality.append(elements[j])
            break

    for i, x in enumerate(elements):
        if x == '--use_pyav':
            use_pyav = True
            break

    if modality == []:
        modality = ['rgb']

    datadir = []
    if use_pyav:
        modality.remove('rgb')
        # find original datadir
        for i, x in enumerate(elements):
            if x == '--datadir':
                for j in range(i+1, len(elements)):
                    if elements[j][0] == '-':
                        break
                    datadir.append(elements[j])
                break

    out = ''
    if dataset is None or len(modality) == 0:
        print("can not find dataset and modality from command, skip copy data to bb.")
        return out, cmd
    else:
        bb_path = '/mnt/nvme/uid_{}'.format(os.getuid())
        out = f"mpirun -hostfile ~/tmp/hosts.$SLURM_JOBID.{nj} copy_to_bb.py {dataset} {bb_path} --modality {' '.join(modality)} & \n"
        # update command to use bb_path

    cmd += f" --datadir {' '.join(datadir)} "
    for m in modality:
        if m == 'sound':
            cmd += "{}/{}/{} ".format(bb_path, dataset, 'audio')
        elif m == 'rgb' or m == 'rgbdiff':
            cmd += "{}/{}/{} ".format(bb_path, dataset, 'rgb-lmdb')
        elif m == 'flow':
            cmd += "{}/{}/{} ".format(bb_path, dataset, 'flow-lmdb')
    return out, cmd


def main():

    home = os.path.expanduser("~")
    log_path = os.path.join(home, 'tmp', 'sbatch_scripts')
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    arch = platform.uname().processor

    if arch == 'x86_64':
        loaded_module = "module load gcc cuda openmpi"
        num_gpus=8
        partition='npl'
    elif arch == 'ppc64le':
        loaded_module = "module load spectrum-mpi"
        num_gpus=6
        partition='el8'

    total_num_nodes = sum(args.num_nodes)

    for i in range(args.num_deps):
        dependency = "" if i == 0 else "#SBATCH --dependency=afterany:{}".format(job_id)
    
        sbatch_job = """#!/bin/bash -x

#SBATCH -J {job_name}
#SBATCH -o {log_path}/my_job_%j.out
#SBATCH -e {log_path}/my_job_%j.err
#SBATCH --gres=gpu:{num_gpus}{with_bb}
#SBATCH --nodes={num_nodes}
#SBATCH --cpus-per-gpu=10
#SBATCH --time=0{time}:00:00
#SBATCH -p {partition}
{dependency}
echo NODELIST
echo $SLURM_JOB_NODELIST

# SLURM_NPROCS and SLURM_NTASK_PER_NODE env variables are set by the SBATCH directive nodes, ntasks-per-node above.
if [ \"x$SLURM_NPROCS\" = \"x\" ]
  then
  if [ \"x$SLURM_NTASKS_PER_NODE\" = \"x\" ]
    then
      SLURM_NTASKS_PER_NODE=1
  fi
  SLURM_NPROCS=`expr $SLURM_JOB_NUM_NODES \\* $SLURM_NTASKS_PER_NODE`
  else
    if [ \"x$SLURM_NTASKS_PER_NODE\" = \"x\" ]
    then
      SLURM_NTASKS_PER_NODE=`expr $SLURM_NPROCS / $SLURM_JOB_NUM_NODES`
    fi
fi
srun hostname -s | sort -u > ~/tmp/hosts.$SLURM_JOBID
#awk \"{{ print \$0 \\"-ib slots=$SLURM_NTASKS_PER_NODE\\"; }}\" ~/tmp/hosts.$SLURM_JOBID >~/tmp/tmp.$SLURM_JOBID
awk \"{{ print \$0 \\" slots=$SLURM_NTASKS_PER_NODE\\"; }}\" ~/tmp/hosts.$SLURM_JOBID >~/tmp/tmp.$SLURM_JOBID
mv ~/tmp/tmp.$SLURM_JOBID ~/tmp/hosts.$SLURM_JOBID
{loaded_module}

cd {wd}
""".format(job_name=args.job_name, num_nodes=total_num_nodes, with_bb=',nvme' if args.bb else '', wd=args.wd, log_path=log_path, time=args.time,
           loaded_module=loaded_module, num_gpus=num_gpus, partition=partition, dependency=dependency)


        # split to sub_jobs
        for nj in range(args.num_jobs):
            end_line = sum(args.num_nodes[:(nj+1)])
            sbatch_job += f"head -n+{end_line} ~/tmp/hosts.$SLURM_JOBID | tail -n {args.num_nodes[nj]} > ~/tmp/hosts.$SLURM_JOBID.{nj}\n"
            if args.bb:
                out, new_cmd = parse_bb(args.cmd[nj], nj)
                sbatch_job += out
                args.cmd[nj] = new_cmd
        if args.bb:
            sbatch_job += "wait \n"
        for nj in range(args.num_jobs):
            #sbatch_job += f"head -n+{args.num_nodes * (nj + 1)} ~/tmp/hosts.$SLURM_JOBID | tail -n {args.num_nodes} > ~/tmp/hosts.$SLURM_JOBID.{nj}\n"
            #sbatch_job += f"export NODELIST=`split_file_to_line.py ~/tmp/hosts.$SLURM_JOBID.{nj}`\n"
            sbatch_job += f"mpirun -hostfile ~/tmp/hosts.$SLURM_JOBID.{nj} {args.cmd[nj]} --hostfile ~/tmp/hosts.$SLURM_JOBID.{nj} & \n"
            #sbatch_job += f"mpirun -hostfile ~/tmp/hosts.$SLURM_JOBID.{nj} {args.cmd[nj]} & \n"
            #sbatch_job += f"srun --nodelist=$NODELIST {args.cmd[nj]} --hostfile ~/tmp/hosts.$SLURM_JOBID.{nj}"
            #sbatch_job += f"srun --nodelist=$NODELIST {args.cmd[nj]} & \n"
            #sbatch_job += f"srun --exclude=$NODELIST {args.cmd[nj]} & \n"
            
            #if nj != args.num_jobs - 1:
            #    sbatch_job += " &\n"
            #else:
            #    sbatch_job += "\n"
        sbatch_job += "wait\n"
        print(sbatch_job)
        #exit(0)
        script = "{}/{}.sh".format(log_path, args.job_name)
        print("Generate sbatch script at {}".format(script))
        with open(script, 'w') as f:
            print(sbatch_job, file=f, flush=True)

        p = subprocess.Popen(['sbatch', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        stdout = stdout.decode("utf-8")
        print("STDout: ", stdout)
        stderr = stderr.decode("utf-8")
        print("STDerr: ", stderr)

        job_id = stdout.split(" ")[-1]
        print(f"Job {job_id.strip()} is submitted.")

    #os.system('sbatch {}'.format(script))


if __name__ == "__main__":
    main()

