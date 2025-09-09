import glob
import os
from os.path import join, dirname

import json



def create_json_file(tissue, dimEmb):

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
    'dimEmb': dimEmb,
    }
    file_path = f'/lustre/groups/casale/code/users/shubham.chaudhary/projects/AIH-SGML/HistoGWAS/Training/PGAN/config/config.json'

    with open(file_path, "w") as json_file:
        json.dump(data_tissue, json_file, indent=4)

os.chdir('..')
tissue = 'Thyroid'
dimEmb = 64 # feature dimension
outdir = f'/lustre/groups/casale/code/users/shubham.chaudhary/output/projects/gtex/PGAN/{tissue}_test_try'
create_json_file(tissue, dimEmb)

command = f"python train.py PGAN -c config/config.json -n {tissue} --dir {outdir} --dimEmb {dimEmb}"
os.system(command)
