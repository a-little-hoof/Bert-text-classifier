#! /bin/bash

#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=sbatch_example
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=10
#SBATCH --time 12:00:00

conda activate UNETR
python main.py --save_checkpoint --data_dir=./dataset/data/data/