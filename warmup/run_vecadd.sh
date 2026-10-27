#!/bin/bash
#SBATCH -J vecadd
#SBATCH -p a30,h200
#SBATCH --gres=gpu:1
#SBATCH -t 00:05:00
#SBATCH -o vecadd_%j.out

module load cuda/13.0
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
./vecadd
