import os
import json
import os
from os.path import join, dirname



def make(stdout, stderr):
    print()

    os.makedirs(dirname(stdout), exist_ok=True)
    os.makedirs(dirname(stderr), exist_ok=True)
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



if __name__ == "__main__":
    organoid_hyperparameter = {}
    os.chdir('..')
    organoid_hyperparameter['tissue'] = 'Organoid'
    organoid_hyperparameter['outdir'] = '/data/dennyal/repos/HistoGWAS_PGAN'
    stdout = join(organoid_hyperparameter['outdir'], 'eval_logs', f'stdout_PGAN.txt')
    stderr = join(organoid_hyperparameter['outdir'], 'eval_logs', f'stderr_PGAN.txt' )
    os.makedirs(organoid_hyperparameter['outdir'], exist_ok=True)
    create_json_file('Organoid')

    command = f"python train.py PGAN -c config/config_{organoid_hyperparameter['tissue']}.json -n {organoid_hyperparameter['tissue']} --dir {organoid_hyperparameter['outdir']} --dimEmb 171 --dimOutput 1 --np_vis"
    make(stdout, stderr)
    print(command)
