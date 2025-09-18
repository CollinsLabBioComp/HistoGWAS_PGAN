#!/bin/bash

#SBATCH -J Organoid
#SBATCH -o /data/Collinslab/projects/HistoGWAS/Organoid/eval_logs/stdout_PGAN.txt
#SBATCH --error=/data/Collinslab/projects/HistoGWAS/Organoid/eval_logs/stderr_PGAN.txt
#SBATCH --partition=gpu
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:a100:2

source myconda
conda activate Histogwas_PGAN
python train.py PGAN -c config/config_Organoid.json -n Organoid --dir /data/Collinslab/projects/HistoGWAS/Organoid --dimEmb 171 --dimOutput 1 --restart --np_vis
