#!/bin/bash
# Based on solo-learn-1.0.3/bash_files/pretrain/custom/byol.sh
# Train without labels.
# To train with labels, simply remove --no_labels
# --val_dir is optional and will expect a directory with subfolder (classes)
# --dali flag is also supported
echo "Running on: "`/usr/bin/hostname`
export PYTHONPATH=.
# python3 -m debugpy --listen 0.0.0.0:5678 --wait-for-client self_supervised/pretrain_acdc.py \
python3 self_supervised/pretrain_acdc.py \
    --dataset custom \
    --backbone resnet50 \
    --data_dir ~/data/acdc \
    --train_dir training \
    --no_labels \
    --max_epochs 401 \
    --gpus 0 \
    --accelerator gpu \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 1.0 \
    --classifier_lr 0.1 \
    --weight_decay 1e-5 \
    --batch_size 128 \
    --num_workers 10 \
    --brightness 0.2 \
    --contrast 0.2 \
    --saturation 0.2 \
    --hue 0.1 \
    --color_jitter_prob 0.2 \
    --gray_scale_prob 0.2 \
    --horizontal_flip_prob 0.5 \
    --gaussian_prob 0.0 0.0 \
    --solarization_prob 0.0 0.0 \
    --mean 67.27297657740893\
    --std 84.6606962344396 \
    --num_crops_per_aug 1 1 \
    --name MoreCkpts_BYOL-Imagenet-Acdc_oldaugs \
    --project ACDC-Pretrain \
    --wandb \
    --save_checkpoint \
    --checkpoint_dir artifacts-acdc/pretrain \
    --method byol \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 8192 \
    --proj_output_dim 256 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --log_every_n_steps 1 \
    --pretrained_weights "resnet50_byol_imagenet2012.pth.tar"