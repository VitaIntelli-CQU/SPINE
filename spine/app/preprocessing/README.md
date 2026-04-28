# Preprocessing

`preprocessing.py` converts raw spatial RNA/protein inputs into the directory layout expected by the SPINE RNA-to-protein training pipeline.

Supported dataset ids: `1, 2, 3, 4, 5, 6`.

The script:
- aligns RNA and protein spots by shared barcodes
- applies row-wise min-max normalization to RNA and protein matrices
- writes `rna_gene_list.json` and `protein_list.json`
- creates a train/test split
- saves RNA embedding HDF5 files under `dataset/embed_dataroot/...`
- optionally writes `embeddings_pca` for cosine-based neighborhood features

## Example

```bash
python spine/app/preprocessing/preprocessing.py \
  --dataset_id 1 \
  --dataset_suffix MINMAX \
  --project_root ./spine \
  --raw_data_root ./Data_SpatialGlue
```

Use `--no_pca` only if you do not want to save `embeddings_pca`. The default behavior keeps PCA features because the main training configuration uses cosine-based graph features.
