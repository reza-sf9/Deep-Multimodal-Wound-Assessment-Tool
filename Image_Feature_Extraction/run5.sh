#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -C H100|A100|V100
#SBATCH -t 71:59:00
#SBATCH --mem 16G
#SBATCH -p short
#SBATCH --job-name="rez5"
#SBATCH --exclude=gpu-5-12


source activate envH100 
python3.9 main5.py 