#!/bin/bash

for device in {0..5} 
do
echo "$device"
python train.py ~/scratch-shared/datasets/imagenet1k --model vit_base_patch16_224 --pretrained --local_rank $device --output ~/scratch/timmlogs/ --experiment tmp1 --workers 55 --pin-mem --mixup 0.8 --cutmix 1. --reprob 0.25 --warmup-epochs 5 --sched cosine --opt adamw --aa rand --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 &
done
