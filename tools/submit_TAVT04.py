# Where we test initializations of the weighted sum
from timm.utils.sbatch_tools import create_lightning_job, submit_sbatch_cmd

n_resubmit = 8
base_log_dir = "/gpfs/u/home/DAMT/DAMThvrb/scratch/tb_logs/TAVT04"

# ============================
# Basic NAME experiment lookup
# ============================
mnames = [
    "tavt04_base0_hmix",
    "tavt_base0_nobias",
    "tavt04_newatt_usewqk",
    "tavt04_newatt_usewqk_hmix"
    "tavt04_newatt_usek",
    "tavt04_newatt_usek_hmix",
    "tavt04_newatt_useq",
    "tavt04_newatt_useq_hmix",
    "tavt04_newatt_usewkq",
    "tavt04_newatt_usewkq_hmix",
    "tavt_kisv_noproj",
]

for mname in mnames:
    job_name = f"{mname}"
    cmd = f"python lightning_train.py /gpfs/u/home/DAMT/DAMThvrb/scratch-shared/datasets/imagenet1k --model {mname} --num-classes 1000 --pin-mem --no-prefetcher --batch-size 32 --val-split val --aa rand --reprob 0.5 --mixup 0.8 --cutmix 1.0 --strategy ddp_find_unused_parameters_false --accelerator gpu --max_epochs 600 --epochs 600 --lr 0.05 --gradient_clip_val=0.5 --progress_bar_refresh_rate 500"
        
    sbatch_cmd, logdir = create_lightning_job(
        job_name,
        cmd,
        logdir=base_log_dir,
        n_resubmit=n_resubmit,
    )
    print(sbatch_cmd)
    submit_sbatch_cmd(logdir, job_name, sbatch_cmd)
    
    
# ============================
# Old Attention
# ============================
old_att = [
    (768,1),
    (384,2),
    (256,3),
    (128,6),
    (64,12),
    (32,24)
]

for num_heads, head_dim in old_att:
    kwargs = {
        "num_heads": num_heads,
        "head_dim": head_dim
    }
    model_name = f"tavt04_zdim{head_dim}_{num_heads}heads"
    job_name = f"{model_name}"
    cmd = f"python lightning_train.py /gpfs/u/home/DAMT/DAMThvrb/scratch-shared/datasets/imagenet1k --model tavt04_zdimX_Yheads --num-classes 1000 --pin-mem --no-prefetcher --batch-size 32 --val-split val --aa rand --reprob 0.5 --mixup 0.8 --cutmix 1.0 --strategy ddp_find_unused_parameters_false --accelerator gpu --max_epochs 600 --epochs 600 --lr 0.05 --gradient_clip_val=0.5 --progress_bar_refresh_rate 500 --num_heads {num_heads} --head_dim {head_dim}"
        
    sbatch_cmd, logdir = create_lightning_job(
        job_name,
        cmd,
        logdir=base_log_dir,
        n_resubmit=n_resubmit,
    )
    print(sbatch_cmd)
    submit_sbatch_cmd(logdir, job_name, sbatch_cmd)
    
# ============================
# New Attention
# ============================
new_att = [
    (1, 64),
    (1,128),
    (1,256),
    (1,768),
    (1,1536),
    (3,64),
    (3,256),
    (3,768),
    (3,1536),
]

for num_heads, head_dim in new_att:
    kwargs = {
        "num_heads": num_heads,
        "head_dim": head_dim
    }
    model_name = f"tavt04_newatt_hmix_zdim{head_dim}_{num_heads}heads"
    job_name = f"{model_name}"
    cmd = f"python lightning_train.py /gpfs/u/home/DAMT/DAMThvrb/scratch-shared/datasets/imagenet1k --model tavt04_newatt_hmix_zdimX_Yheads --num-classes 1000 --pin-mem --no-prefetcher --batch-size 32 --val-split val --aa rand --reprob 0.5 --mixup 0.8 --cutmix 1.0 --strategy ddp_find_unused_parameters_false --accelerator gpu --max_epochs 600 --epochs 600 --lr 0.05 --gradient_clip_val=0.5 --progress_bar_refresh_rate 500 --num_heads {num_heads} --head_dim {head_dim}"
        
    sbatch_cmd, logdir = create_lightning_job(
        job_name,
        cmd,
        logdir=base_log_dir,
        n_resubmit=n_resubmit,
    )
    print(sbatch_cmd)
    submit_sbatch_cmd(logdir, job_name, sbatch_cmd)
    
# ============================
# Base0
# ============================
new_att = [
    (1, 64),
    (1,128),
    (1,256),
    (1,768),
    (1,1536),
    (3,64),
    (3,256),
    (3,768),
    (3,1536),
]

for num_heads, head_dim in new_att:
    kwargs = {
        "num_heads": num_heads,
        "head_dim": head_dim
    }
    model_name = f"tavt04_base0_hmix_zdim{head_dim}_{num_heads}heads"
    job_name = f"{model_name}"
    cmd = f"python lightning_train.py /gpfs/u/home/DAMT/DAMThvrb/scratch-shared/datasets/imagenet1k --model tavt04_base0_hmix_zdimX_Yheads --num-classes 1000 --pin-mem --no-prefetcher --batch-size 32 --val-split val --aa rand --reprob 0.5 --mixup 0.8 --cutmix 1.0 --strategy ddp_find_unused_parameters_false --accelerator gpu --max_epochs 600 --epochs 600 --lr 0.05 --gradient_clip_val=0.5 --progress_bar_refresh_rate 500 --num_heads {num_heads} --head_dim {head_dim}"
        
    sbatch_cmd, logdir = create_lightning_job(
        job_name,
        cmd,
        logdir=base_log_dir,
        n_resubmit=n_resubmit,
    )
    print(sbatch_cmd)
    submit_sbatch_cmd(logdir, job_name, sbatch_cmd)
    

    
## TEST BEFORE RUNNING
# python lightning_train.py /gpfs/u/home/DAMT/DAMThvrb/scratch-shared/datasets/imagenet1k --model tavt04_base0_hmix_zdimX_Yheads --num-classes 1000 --pin-mem --no-prefetcher --batch-size 32 --val-split val --aa rand --reprob 0.5 --mixup 0.8 --cutmix 1.0 --strategy ddp_find_unused_parameters_false --accelerator gpu --max_epochs 600 --epochs 600 --lr 0.05 --gradient_clip_val=0.5 --progress_bar_refresh_rate 500 --num_heads 1 --head_dim 64 --devices 1 --num_nodes 1 --exp_dir /gpfs/u/home/DAMT/DAMThvrb/scratch/tb_logs/TST --exp_name tavt04_base0_hmix_zdim64_1heads

# python lightning_train.py /gpfs/u/home/DAMT/DAMThvrb/scratch-shared/datasets/imagenet1k --model tavt04_base0_hmix_zdimX_Yheads --num-classes 1000 --pin-mem --no-prefetcher --batch-size 32 --val-split val --aa rand --reprob 0.5 --mixup 0.8 --cutmix 1.0 --strategy ddp_find_unused_parameters_false --accelerator gpu --max_epochs 600 --epochs 600 --lr 0.05 --gradient_clip_val=0.5 --progress_bar_refresh_rate 500 --num_heads 1 --devices 1 --num_nodes 1 --exp_dir /gpfs/u/home/DAMT/DAMThvrb/scratch/tb_logs/TST --exp_name tavt04_base0_hmix_zdim64_1heads

# python lightning_train.py /gpfs/u/home/DAMT/DAMThvrb/scratch-shared/datasets/imagenet1k --model tavt04_newatt_usewkq --num-classes 1000 --pin-mem --no-prefetcher --batch-size 32 --val-split val --aa rand --reprob 0.5 --mixup 0.8 --cutmix 1.0 --strategy ddp_find_unused_parameters_false --accelerator gpu --max_epochs 600 --epochs 600 --lr 0.05 --gradient_clip_val=0.5 --progress_bar_refresh_rate 500 --devices 1 --num_nodes 1 --exp_dir /gpfs/u/home/DAMT/DAMThvrb/scratch/tb_logs/TST --exp_name tavt04_newatt_usewkq