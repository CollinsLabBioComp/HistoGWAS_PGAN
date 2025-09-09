# HistoGWAS Decoder (PGAN)

Progressive GAN (PGAN) decoder in PyTorch for histology-based GWAS.

## Quick Start

### Step 1 — Data
Prepare an AnnData `.h5ad` with:
- `adata.X` → image embeddings
- `adata.obs` → metadata (e.g., tissue/cluster labels). Note:  adata.obs['path'] should contain path of images

Example: `data/low_memory_Thyroid.h5ad`[Link](https://drive.google.com/file/d/1RTny5gKE79x9MEUqu1WHgIplZ958fTJ6/view?usp=sharing)

### Step 2 — Configure
Edit `config/config.json`:
- `"pathDB"`: path to your `.h5ad`
- `"dimEmb"`: embedding dimension (e.g., 64)

### Step 3 — Train
```bash
OUTDIR=out ./train_pgan.sh
```
### Step 4 - Demo
- Once the model is trained follow the steps as done in [`Notebooks/Demo_pgan.ipynb`](Notebooks/Demo_pgan.ipynb) for visualization and interpolation
- Requirements: install [`MTGWAS`](../../mtgwas)


## Acknowledgement
This implementation of PGAN was built on top of amazing repositiory of [Pytorch GAN Zoo](https://github.com/facebookresearch/pytorch_GAN_zoo). We thank the authors for their contribution.
