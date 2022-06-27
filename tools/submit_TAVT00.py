from timm.utils.sbatch_tools import create_lightning_job, submit_sbatch_cmd

n_resubmit = 8
base_log_dir = "/gpfs/u/home/DAMT/DAMThvrb/scratch/tb_logs/TAVT00"
model_names = [
    "avt_base_patch16_224",
    "tavt_nobias",
    "tavt_relu",
    "tavt_symmlp",
    "tavt_parallel",
    "tavt_elnorm",
    "tavt_newatt",
    "tavt_base0",
    "tavt_base0_nobias",
    "tavt_base0_alpha",
    "tavt_base0_alpha_nobias",
    "tavt_noproj",
]

for mname in model_names:
    job_name = mname
    cmd = f"python lightning_train.py /gpfs/u/home/DAMT/DAMThvrb/scratch-shared/datasets/imagenet1k --model {mname} --num-classes 1000 --pin-mem --no-prefetcher --batch-size 32 --val-split val --aa rand --reprob 0.5 --mixup 0.8 --cutmix 1.0 --strategy ddp_find_unused_parameters_false --accelerator gpu --max_epochs 600 --epochs 600 --lr 0.05 --gradient_clip_val=0.5 --progress_bar_refresh_rate 500"
    sbatch_cmd, logdir = create_lightning_job(
        job_name,
        cmd,
        logdir=base_log_dir,
        n_resubmit=n_resubmit,
    )
    print(sbatch_cmd)
    submit_sbatch_cmd(logdir, job_name, sbatch_cmd)