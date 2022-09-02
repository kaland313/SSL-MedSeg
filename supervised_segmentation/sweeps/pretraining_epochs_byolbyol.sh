#!/bin/bash
#SBATCH -o artifacts-acdc/supervised-sweeps/pretraining-epochs-byol-byol/%a.out
#SBATCH -a 0-24
#SBATCH -p gpu
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu

# How to run:
#  sbatch supervised_segmentation/sweep_configs/pretraining_epochs_byolbyol.sh 
# How to sync multiple runs to wandb:
#  cd artifacts-acdc/supervised-sweeps/pretraining-epochs-byol-byol/ 
#  wandb sync --sync-all wandb/
export PYTHONPATH=$(pwd)

pretraining_epochs=(0 1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 22 24 25 50 100 200 300 400)

nr=$SLURM_ARRAY_TASK_ID
ep=${pretraining_epochs[$nr]}
echo "Evaluating artifacts-acdc/pretrain/byol/eix2ba1c/MoreCkpts_BYOL-Imagenet-Acdc_oldaugs-eix2ba1c-ep=${ep}.pth"

python supervised_segmentation/train_acdc.py \
    --model.encoder_weights "artifacts-acdc/pretrain/byol/eix2ba1c/MoreCkpts_BYOL-Imagenet-Acdc_oldaugs-eix2ba1c-ep=${ep}.pth" \
    --artifacts_root "artifacts-acdc/supervised-sweeps/pretraining-epochs-byol-byol" \
    --seed 13
