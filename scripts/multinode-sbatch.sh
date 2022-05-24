#!/bin/bash

## I couldn't get torchscript to work
# bash distributed_train.sh 6 ~/scratch-shared/datasets/imagenet1k --model vit_base_patch16_224 --pretrained --output ~/scratch/timmlogs/ --experiment tmp3 --workers 55 --pin-mem --mixup 0.8 --cutmix 1. --reprob 0.25 --warmup-epochs 5 --sched cosine --opt adamw --aa rand --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 --workers 50 --lr 0.001 --torchscript

## I can train from scratch with a base vision transformer
bash distributed_train.sh 6 ~/scratch-shared/datasets/imagenet1k --model vit_base_patch16_224 --pretrained --output ~/scratch/timmlogs/ --experiment tmp3 --workers 55 --pin-mem --mixup 0.8 --cutmix 1. --reprob 0.25 --warmup-epochs 5 --sched cosine --opt adamw --aa rand --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 --workers 50 --lr 0.001

## I can train from scratch with my energy transformer?
bash distributed_train.sh 6 ~/scratch-shared/datasets/imagenet1k --model et_base_patch16_224 --pretrained --output ~/scratch/timmlogs/ --experiment tmp3 --workers 55 --pin-mem --mixup 0.8 --cutmix 1. --reprob 0.25 --warmup-epochs 5 --sched cosine --opt adamw --aa rand --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 --workers 50 --lr 0.001

# I need WORLD_SIZE and local rank and ??

## I don't know how to resume from a checkpoint automatically

## I don't know how to launch with WORLD_SIZE>1
