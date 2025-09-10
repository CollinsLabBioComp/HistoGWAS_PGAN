import glob
import os
from os.path import join, dirname

import json



def create_json_file(tissue, dimEmb):

    data_tissue = {
    "pathDB": f"/data/Collinslab/", #path to anndata Question about whether to 
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
    'dimEmb': dimEmb,
    }
    file_path = f'/data/Collinslab/tcf7l2/HistoGWAS/Training/PGAN/config/config.json'

    with open(file_path, "w") as json_file:
        json.dump(data_tissue, json_file, indent=4)

os.chdir('..')
tissue = 'Thyroid'
dimEmb = 178 # feature dimension
outdir = f'/data/Collinslab/tcf7l2/HistoGWAS_output/{tissue}_test_try'
create_json_file(tissue, dimEmb)

command = f"python train.py PGAN -c config/config.json -n {tissue} --dir {outdir} --dimEmb {dimEmb}"
os.system(command)
