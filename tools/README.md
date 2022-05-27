# Method 1

> The `aimos_submit.py` script

Example provided by Rameswar

```
python3 tools/aimos_submit.py --nodes 8 --ngpus 6 -d 1 --wd /gpfs/u/home/SIFA/SIFApndr/scratch/action-vit --suffix 3x3_f8_aug --job_dir checkpoint/action_vit_conv/ssv2/big_resnetv2 --model action_conv_big_resnetv2_50x1_bitm_in21k_3 " --data_dir /gpfs/u/home/SIFA/SIFApndr/scratch-shared/datasets/action/ssv2/videos --use_pyav --dataset st2stv2 --opt adamw --lr 1e-4 --epochs 50 --sched cosine --duration 8 --no-amp --batch-size 16 --pretrained --super_img_rows 3  --warmup-epochs 5 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --disable_scaleup"
```

- Change `wd` working directory -> ??
- What is suffix? :: Appended to job directory
- Choose job directory
- How do I automatically resume?
- Why is model separate from the quoted command?
- Change data dir
- Remove custom flags
    - use_pyav
    - super_img_rows
- No pretrained model currently

```
python3 tools/aimos_submit.py --nodes 8 --ngpus 6 -d 1 --wd /gpfs/u/home/DAMT/DAMThvrb/Projects/timm --suffix 00 --job_dir checkpoint/TDCS --model et_base_patch16_224 " /gpfs/u/home/DAMT/DAMThvrb/scratch-shared/datasets/imagenet1k --dataset imagenet --opt adamw --lr 5e-4 --epochs 300 --sched cosine --batch-size 16 --warmup-epochs 5 --mixup 0.8 --cutmix 1.0 --drop-path 0.0"
```

By default, this command looks for a `main.py` script in the `wd`. This `main.py` takes a lot of arguments similarly to the `train.py` of timm

# Method 2

> Any script can be submitted to multiple nodes

```
submit_job_dev_split_jobs_v2.py -n 8 -d 1 --job_name ap --wd /gpfs/u/home/RBST/RBSTpndr/scratch/Concept-VLN/ "python train_mlm_ap.py --batch-size 576 --weight-mlm 0. --weight-ap 1. --epochs 10 --name PREVALENT-only-AP --wd 1e-4 --multiprocessing-distributed --auto_resume"
```

Changing for myself:


```
python tools/submit_job_dev_split_jobs_v2.py -n 2 -d 1 --job_name TDCS00 --wd /gpfs/u/home/DAMT/DAMThvrb/Projects/timm "python train.py ~/scratch-shared/datasets/imagenet1k --model vit_base_patch16_224 --pretrained --local_rank $device --output ~/scratch/timmlogs/ --experiment tmp1 --workers 55 --pin-mem --mixup 0.8 --cutmix 1. --reprob 0.25 --warmup-epochs 5 --sched cosine --opt adamw --aa rand --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225"
```
