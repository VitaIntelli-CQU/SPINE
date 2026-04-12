# SPINE

SPINE is a PyTorch package for RNA-to-protein prediction on spatial omics data.

![SPINE model](model.png)

## Environment

Recommended environment for this repository:
- Python `3.12`
- PyTorch `2.7.0`

The current project setup has been validated with:
- Python `3.12.9`
- PyTorch `2.7.0+cu118`

## Organization

The organization of this repository is as follows:
- `app/`: contains the SPINE application entry points
    - `flow/`: training pipeline for SPINE
- `data/`: contains the dataloader for the SPINE model
- `model/`: contains the implementation of denoiser
- `app/preprocessing/`: contains preprocessing scripts and notes
- `io_utils/`: contains shared h5/h5ad utility functions used by the data loaders


## Usage

Install dependencies and the package with:

```bash
pip install -r requirements.txt
pip install -e .
```

Preprocessing notes and scripts are in `spine/app/preprocessing/`.

Training SPINE with the following script:
```bash
python spine/app/flow/train_rna_to_protein.py \
    --dataset DATASET1_RNA_TO_PROTEIN_MINMAX_NOHVG_NOMAP \
    --source_dataroot /path/to/source_dataroot \
    --embed_dataroot /path/to/embed_dataroot \
    --batch_size 2 \
    --epochs 500
```
