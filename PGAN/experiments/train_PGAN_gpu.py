import pandas as pd
import numpy as np
import glob
import os
from os.path import join, dirname

import json



def submit_job(command, opts):

    os.makedirs(dirname(opts["stdout"]), exist_ok=True)
    os.makedirs(dirname(opts["stderr"]), exist_ok=True)

    f = open(f"submit.sh", "w")

    f.write(f'#!/bin/bash\n')
    f.write(f'\n')
    f.write(f'#SBATCH -J {opts["name"]}\n')
    f.write(f'#SBATCH -o {opts["stdout"]}\n')
    f.write(f'#SBATCH -e {opts["stderr"]}\n')
    f.write(f'#SBATCH -p {opts["queue"]}\n')
    f.write(f'#SBATCH -t {opts["time"]}\n')
    f.write(f'#SBATCH -c {opts["nodes"]}\n')
    f.write(f'#SBATCH --mem={opts["memory"]}G\n')
    f.write(f'#SBATCH --qos={opts["qos"]}\n')
    f.write(f'#SBATCH --gres={opts["gpu"]}\n')
    f.write(f'#SBATCH --nice=10000\n')
    f.write(f'\n')
    f.write(f'source $HOME/.bashrc\n')
    f.write(f'conda activate {opts["condaenv"]}\n')
    f.write(command)
    f.write(f'\n')
    f.close()

    os.system(f'sbatch submit.sh')
    os.system('rm submit.sh')
    
    
def run_jobs(tissue_hyperparameter):
    opts = {}
    opts['queue'] = 'gpu_p'
    opts["gpu"] = 'gpu:1'
    opts['time'] = '12:00:00'
    opts['qos'] = 'gpu_normal'
    opts['nodes'] = 4
    opts['memory'] = 160
    opts['condaenv'] = 'milgan'
    outdir = tissue_hyperparameter['outdir']
    job_name = tissue_hyperparameter['tissue']
    opts['name'] = job_name
    opts['stdout'] = join(outdir, 'eval_logs', f'stdout_PGAN.txt')
    opts['stderr'] = join(outdir, 'eval_logs', f'stderr_PGAN.txt')
    '''
    command --cluster_mean to be used only when you want take mean across each cluster within a biopsy
    '''

    # import pdb
    # pdb.set_trace()
    
    command = f"python train.py PGAN -c config/config_{tissue_hyperparameter['tissue']}.json -n {tissue_hyperparameter['tissue']} --dir {tissue_hyperparameter['outdir']} --dimEmb 64"
    # os.system(command)
    submit_job(command, opts)

def create_json_file(tissue):

    data_tissue = {
    "pathDB": f"/lustre/groups/casale/datasets/gtex/histology/20230425_v2_tiles/stage2/{tissue}/embedding/summary_scanpy_pc.h5ad",
    "config": {
        "maxIterAtScale": [
        48000,
        96000,
        96000,
        96000,
        96000,
        96000,
        1000000
        ]
    },
    'dimEmb': 64,
    }
    file_path = f'/lustre/groups/casale/code/users/shubham.chaudhary/projects/AIH-SGML/HistoGWAS/Training/PGAN/config/config_{tissue}.json'

    with open(file_path, "w") as json_file:
        json.dump(data_tissue, json_file, indent=4)



tissue_list = [
    
#     'Adipose_Subcutaneous',
#    'Colon_Transverse',
#     'Stomach',
#     'Esophagus_Mucosa',
#     'Skin_Sun_Exposed_Lower_leg',
    'Thyroid',
#     'Osteoarthritis',
# 'Esophagus_Muscularis',
# 'Pancreas',
# 'Artery_Tibial'
]


tissue_hyperparameter = {}
os.chdir('..')
for tissue in tissue_list:

    tissue_hyperparameter['tissue'] = tissue
    tissue_hyperparameter['outdir'] = f'/lustre/groups/casale/code/users/shubham.chaudhary/output/projects/gtex/PGAN/{tissue}_test_test'
    os.makedirs(tissue_hyperparameter['outdir'], exist_ok=True)
    create_json_file(tissue)

    run_jobs(tissue_hyperparameter)
