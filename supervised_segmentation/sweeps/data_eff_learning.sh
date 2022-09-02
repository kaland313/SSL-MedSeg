#!/bin/bash
#SBATCH -o artifacts-acdc/supervised-sweeps/data_eff_learning_2/%a.out
#SBATCH -a 660-859
#SBATCH -p gpu
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu
#SBATCH --time=0-0:40

# How to run:
#  sbatch supervised_segmentation/sweep_configs/data_eff_learning_2.sh 
# How to sync multiple runs to wandb:
#  cd artifacts-acdc/supervised-sweeps/data_eff_learning_2/ 
#  wandb sync --sync-all wandb/
export PYTHONPATH=$(pwd)

run_args=`python supervised_segmentation/sweep_configs/grid_search_helper.py`
echo "Evaluating ${run_args}"

python supervised_segmentation/train_acdc.py \
    ${run_args} \
    --artifacts_root "artifacts-acdc/supervised-sweeps/data_eff_learning_2" \
    --wandb_tag "data_eff_learning_2"