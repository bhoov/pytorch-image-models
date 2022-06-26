from timm.utils.sbatch_tools import create_lightning_job, submit_sbatch_cmd

n_resubmit = 8
base_log_dir = "/gpfs/u/home/DAMT/DAMThvrb/scratch/new-lightning-logs/early_tests"
lrs = [0.02, 0.05, 0.1, 0.2, 0.3]
job_names = [f"et_base_lr-{lr}" for lr in lrs]


def make_cmd(lr):
    return f"python lightning_train.py /gpfs/u/home/DAMT/DAMThvrb/scratch-shared/datasets/imagenet1k --model et_base_patch16_224 --num-classes 1000 --pin-mem --no-prefetcher --batch-size 32 --val-split val --aa rand --reprob 0.5 --mixup 0.8 --cutmix 1.0 --strategy ddp_find_unused_parameters_false --accelerator gpu --max_epochs 600 --epochs 600 --lr {lr} --exp_dir --gradient_clip_val=0.5 --progress_bar_refresh_rate 100"

cmds = [make_cmd(lr) for lr in lrs]
for lr in lrs:
    job_name = f"et_base_lr-{lr}"
    cmd = f"python lightning_train.py /gpfs/u/home/DAMT/DAMThvrb/scratch-shared/datasets/imagenet1k --model et_base_patch16_224 --num-classes 1000 --pin-mem --no-prefetcher --batch-size 32 --val-split val --aa rand --reprob 0.5 --mixup 0.8 --cutmix 1.0 --strategy ddp_find_unused_parameters_false --accelerator gpu --max_epochs 200 --epochs 200 --lr {lr} --gradient_clip_val=0.5 --progress_bar_refresh_rate 100"
    sbatch_cmd, logdir = create_lightning_job(
        job_name,
        cmd,
        logdir=base_log_dir,
        n_resubmit=n_resubmit,
    )
    print(sbatch_cmd)
    submit_sbatch_cmd(logdir, job_name, sbatch_cmd)