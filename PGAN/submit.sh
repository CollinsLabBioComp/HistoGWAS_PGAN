#!/bin/bash

#SBATCH -J Organoid
#SBATCH -o /data/Collinslab/projects/HistoGWAS/Organoid/eval_logs/stdout_PGAN.txt
#SBATCH --error=/data/Collinslab/projects/HistoGWAS/Organoid/eval_logs/stderr_PGAN.txt
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --mem=160g
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2

source myconda
source ~/.bashrc
conda activate histogwas2
python train.py PGAN -c config/config_Organoid.json -n Organoid --dir /data/Collinslab/projects/HistoGWAS/Organoid --dimEmb 171 --dimOutput 1 --np_vis
