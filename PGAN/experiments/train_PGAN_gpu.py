import pandas as pd
import numpy as np
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
    f.write(f'#SBATCH --error={opts["stderr"]}\n')
    f.write(f'#SBATCH --partition={opts["partition"]}\n')
    f.write(f'#SBATCH --time={opts["time"]}\n')
    f.write(f'#SBATCH --mem={opts["memory"]}g\n')
    f.write(f'#SBATCH --cpus-per-task={opts["cpu"]}\n')
    f.write(f'#SBATCH --gres={opts["gres"]}\n')
    #f.write(f'#SBATCH --nice=10000\n')
    f.write(f'\n')
    f.write('source myconda\n')
    f.write('source ~/.bashrc\n')
    f.write(f'conda activate {opts["condaenv"]}\n')
    f.write(command)
    f.write(f'\n')
    f.close()

    os.system(f'sbatch submit.sh')
    #os.system('rm submit.sh')
    
    
def run_jobs(tissue_hyperparameter):
    opts = {}
    opts['partition'] = 'gpu'
    opts["gres"] = 'gpu:a100:2'
    opts['time'] = '48:00:00'
    opts['cpu'] = 4
    opts['memory'] = 160
    opts['condaenv'] = 'histogwas2' 
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
    
    command = f"python train.py PGAN -c config/config_{tissue_hyperparameter['tissue']}.json -n {tissue_hyperparameter['tissue']} --dir {tissue_hyperparameter['outdir']} --dimEmb 171 --dimOutput 1 --restart --np_vis"
    # os.system(command)
    submit_job(command, opts)

def create_json_file(tissue):

    data_organoid= {
    "pathDB": f"/data/Collinslab/tcf7l2/organoid_anndata.h5ad",
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
    'dimEmb': 171    
    }
    file_path = f'/data/dennyal/repos/HistoGWAS_PGAN/PGAN/config/config_{tissue}.json'

    with open(file_path, "w") as json_file:
        json.dump(data_organoid, json_file, indent=4)





organoid_hyperparameter = {}
os.chdir('..')
organoid_hyperparameter['tissue'] = 'Organoid'
organoid_hyperparameter['outdir'] = '/data/Collinslab/projects/HistoGWAS/Organoid'
os.makedirs(organoid_hyperparameter['outdir'], exist_ok=True)
create_json_file('Organoid')
run_jobs(organoid_hyperparameter)