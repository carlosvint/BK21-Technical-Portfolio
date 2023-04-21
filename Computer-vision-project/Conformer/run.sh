#!/usr/bin/env bash

# Train
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
OUTPUT='./output/Conformer_small_patch16_CIFAR10pretrained1_lr5e4_100epochs'

python -m torch.distributed.launch --master_port 50132 --nproc_per_node=6 --use_env main.py \
                                   --model Conformer_tiny_patch16 \
                                   --data-set CIFAR10 \
                                   --batch-size 64 \
                                   --lr 0.0005 \
                                   --num_workers 4 \
                                   --data-path /nas/datahub/cifar10/ \
                                   --output_dir ${OUTPUT} \
                                   --epochs 100

# Inference
#CUDA_VISIBLE_DEVICES=0, python main.py  --model Conformer_tiny_patch16 --eval --batch-size 64 \
#                --input-size 224 \
#                --data-set IMNET \
#                --num_workers 4 \
#                --data-path /nas/datahub/imagenet/ \
#                --epochs 100 \
#                --resume /nas/home/carlos/Conformer/output/Conformer_small_patch16_batch_1024_lr1e-3_300epochs/checkpoint.pth


