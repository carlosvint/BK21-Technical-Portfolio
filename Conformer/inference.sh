CUDA_VISIBLE_DEVICES=0, python main.py  --model Conformer_small_patch16 --eval --batch-size 64 \
                --input-size 224 \
                --data-set IMNET \
                --num_workers 4 \
                --data-path /nas/datahub/imagenet/ \
                --epochs 100 \
                --resume /nas/home/carlos/Conformer/output/Conformer_small_patch16_batch_1024_lr1e-3_300epochs/checkpoint.pth