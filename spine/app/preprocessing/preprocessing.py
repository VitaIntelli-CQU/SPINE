#!/usr/bin/env python3
"""Preprocess spatial RNA-to-protein datasets for SPINE."""

import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from pathlib import Path

from spine.io_utils.file_utils import save_hdf5


def _extract_coords(adata: sc.AnnData) -> np.ndarray:
    possible_keys = ["spatial", "X_spatial", "spatial_coords"]
    for key in possible_keys:
        if key in adata.obsm:
            return np.asarray(adata.obsm[key])
    raise ValueError("Could not find spatial coordinates in AnnData.obsm")


def _to_numpy(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def _minmax_normalize(adata: sc.AnnData, low: float = 1e-8, high: float = 1.0) -> sc.AnnData:
    """Apply row-wise min-max normalization to an AnnData matrix."""
    matrix = _to_numpy(adata.X).astype(np.float32)
    row_min = matrix.min(axis=1, keepdims=True)
    row_max = matrix.max(axis=1, keepdims=True)
    denom = row_max - row_min
    zero_rows = (denom.squeeze(1) == 0)
    denom_safe = np.where(denom == 0, 1.0, denom)
    matrix = low + (matrix - row_min) / denom_safe * (high - low)
    if np.any(zero_rows):
        matrix[zero_rows, :] = low
    matrix = np.where(np.isfinite(matrix), matrix, low)
    adata.X = matrix.astype(np.float32)
    return adata


def _filter_zero_variance_proteins(protein: sc.AnnData) -> sc.AnnData:
    """Drop proteins with zero variance to avoid undefined correlation metrics."""
    matrix = _to_numpy(protein.X).astype(np.float32)
    var = np.var(matrix, axis=0)
    keep_mask = var > 0
    dropped = int((~keep_mask).sum())
    if dropped > 0:
        print(f"  Dropped {dropped} zero-variance proteins")
        protein = protein[:, keep_mask].copy()
    return protein


def process_rna_to_protein_dataset(
    dataset_id: int,
    train_ratio: float = 0.9,
    seed: int = 42,
    dataset_suffix: str = "MINMAX_NOHVG_NOMAP",
    save_pca_embeddings: bool = True,
    pca_dim: int = 256,
    project_root: str | None = None,
    raw_data_root: str | None = None,
):
    """
    Process one spatial RNA-to-protein dataset.

    The output layout matches the SPINE training pipeline:
    - `dataset/<name>/<split>/protein.h5ad`
    - `dataset/embed_dataroot/<name>/<split>/RNA_EMBED/<split>.h5`
    - `rna_gene_list.json`, `protein_list.json`, and split CSVs
    """
    repo_root = Path(__file__).resolve().parents[3]
    if project_root is None:
        project_root = str(repo_root / "spine")
    if raw_data_root is None:
        raw_data_root = str(repo_root / "Data_SpatialGlue")
    
    dataset_map = {
        1: "Dataset1_Mouse_Spleen1",
        2: "Dataset2_Mouse_Spleen2",
        3: "Dataset3_Mouse_Thymus1",
        4: "Dataset4_Mouse_Thymus2",
        5: "Dataset5_Mouse_Thymus3",
        6: "Dataset6_Mouse_Thymus4",
        11: "Dataset11_Human_Lymph_Node_A1",
        12: "Dataset12_Human_Lymph_Node_D1",
    }
    
    if dataset_id not in dataset_map:
        raise ValueError(f"Unsupported dataset_id: {dataset_id}")
    
    raw_dir_name = dataset_map[dataset_id]
    dataset_suffix = (dataset_suffix or "").strip()
    if dataset_suffix and not dataset_suffix.startswith("_"):
        dataset_suffix = f"_{dataset_suffix}"
    dataset_name = f"DATASET{dataset_id}_RNA_TO_PROTEIN{dataset_suffix}"
    raw_dir = os.path.join(raw_data_root, raw_dir_name)
    
    dataset_root = os.path.join(project_root, "dataset", dataset_name)
    embed_root = os.path.join(project_root, "dataset", "embed_dataroot", dataset_name)
    split_dir = os.path.join(dataset_root, "splits")
    os.makedirs(split_dir, exist_ok=True)
    
    print(f"\n{'=' * 70}")
    print(f"Processing dataset: {dataset_name} ({raw_dir_name})")
    print(f"{'=' * 70}")

    rna_file = os.path.join(raw_dir, "adata_RNA.h5ad")
    protein_file = os.path.join(raw_dir, "adata_ADT.h5ad")
    
    if not os.path.exists(rna_file):
        print(f"  Skipping: missing RNA file: {rna_file}")
        return None
    if not os.path.exists(protein_file):
        print(f"  Skipping: missing protein file: {protein_file}")
        return None
    
    print("  Loading RNA and protein inputs")
    rna = sc.read_h5ad(rna_file)
    protein = sc.read_h5ad(protein_file)
    
    
    common_cells = rna.obs_names.intersection(protein.obs_names)
    if len(common_cells) == 0:
        print("  Error: RNA and protein inputs do not share any spots")
        return None
    
    rna = rna[common_cells]
    protein = protein[common_cells]
    coords = _extract_coords(rna)
    protein.obsm["spatial"] = coords

    rna.var_names_make_unique()
    protein.var_names_make_unique()

    print("  Applying row-wise min-max normalization to RNA")
    rna = _minmax_normalize(rna)

    print("  Applying row-wise min-max normalization to protein")
    protein = _minmax_normalize(protein)

    protein = _filter_zero_variance_proteins(protein)
    
    rna_genes = rna.var_names.tolist()
    protein_genes = protein.var_names.tolist()
    
    rna_gene_list_path = os.path.join(dataset_root, "rna_gene_list.json")
    protein_list_path = os.path.join(dataset_root, "protein_list.json")
    with open(rna_gene_list_path, "w") as f:
        json.dump(rna_genes, f)
    with open(protein_list_path, "w") as f:
        json.dump(protein_genes, f)
    
    n_cells = rna.n_obs
    rng = np.random.default_rng(seed)
    indices = np.arange(n_cells)
    rng.shuffle(indices)
    split_idx = int(n_cells * train_ratio)
    train_name = f"Dataset{dataset_id}_train"
    test_name = f"Dataset{dataset_id}_test"
    split_map = {
        train_name: np.sort(indices[:split_idx]),
        test_name: np.sort(indices[split_idx:]),
    }

    pca_model = None
    if save_pca_embeddings:
        try:
            from sklearn.decomposition import PCA, TruncatedSVD
        except Exception as e:
            raise RuntimeError(f"save_pca_embeddings=True but sklearn PCA dependencies are unavailable: {e}")

        X_train = rna[split_map[train_name]].X
        if sp.issparse(X_train):
            pca_model = TruncatedSVD(n_components=int(pca_dim), random_state=seed)
        else:
            pca_model = PCA(n_components=int(pca_dim), random_state=seed)
        pca_model.fit(_to_numpy(X_train).astype(np.float32))
    
    for sample_name, idx in split_map.items():
        sample_dir = os.path.join(dataset_root, sample_name)
        embed_dir = os.path.join(embed_root, sample_name, "RNA_EMBED")
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(embed_dir, exist_ok=True)
        
        subset_rna = rna[idx]
        subset_protein = protein[idx]
        subset_coords = coords[idx]
        subset_barcodes = subset_rna.obs_names.values.astype("S")
        
        rna_features = _to_numpy(subset_rna.X).astype(np.float32)
        embeddings_pca = None
        if pca_model is not None:
            embeddings_pca = pca_model.transform(rna_features).astype(np.float32)
        embed_path = os.path.join(embed_dir, f"{sample_name}.h5")
        asset_dict = {
            "embeddings": rna_features,
            "barcodes": subset_barcodes,
            "coords": subset_coords.astype(np.float32),
        }
        if embeddings_pca is not None:
            asset_dict["embeddings_pca"] = embeddings_pca
        save_hdf5(embed_path, asset_dict=asset_dict, mode="w")
        
        subset_protein.write_h5ad(os.path.join(sample_dir, "protein.h5ad"))
        
        coords_df = pd.DataFrame(subset_coords, columns=["x", "y"], index=subset_rna.obs_names)
        coords_df.to_csv(os.path.join(sample_dir, "coords.csv"))
    
    pd.DataFrame({"sample_id": [train_name]}).to_csv(
        os.path.join(split_dir, "train_0.csv"), index=False
    )
    pd.DataFrame({"sample_id": [test_name]}).to_csv(
        os.path.join(split_dir, "test_0.csv"), index=False
    )
    
    print("\n  Finished dataset")
    print(f"    RNA features: {len(rna_genes)}")
    print(f"    Protein features: {len(protein_genes)}")
    print(f"    Train spots: {len(split_map[train_name])} ({len(split_map[train_name]) / n_cells * 100:.1f}%)")
    print(f"    Test spots: {len(split_map[test_name])} ({len(split_map[test_name]) / n_cells * 100:.1f}%)")
    
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "n_rna_genes": len(rna_genes),
        "n_proteins": len(protein_genes),
        "n_train_cells": len(split_map[train_name]),
        "n_test_cells": len(split_map[test_name]),
        "n_total_cells": n_cells,
    }


def main(
    train_ratio: float = 0.9,
    seed: int = 42,
    dataset_suffix: str = "MINMAX",
    save_pca_embeddings: bool = True,
    pca_dim: int = 256,
    project_root: str | None = None,
    raw_data_root: str | None = None,
):
    """Process every bundled RNA-to-protein dataset."""
    rna_protein_datasets = [1, 2, 3, 4, 5, 6, 11, 12]
    
    print("=" * 70)
    print("Processing bundled RNA-to-protein datasets")
    print("=" * 70)
    print(f"Datasets to process: {len(rna_protein_datasets)}")
    print("Excluded datasets: 7, 8, 9, 10")
    print("=" * 70)
    
    results = []
    for dataset_id in rna_protein_datasets:
        try:
            result = process_rna_to_protein_dataset(
                dataset_id=dataset_id,
                train_ratio=train_ratio,
                seed=seed,
                dataset_suffix=dataset_suffix,
                save_pca_embeddings=save_pca_embeddings,
                pca_dim=pca_dim,
                project_root=project_root,
                raw_data_root=raw_data_root,
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"  Failed on dataset {dataset_id}: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("Finished preprocessing")
    print("=" * 70)
    print(f"\nProcessed datasets: {len(results)}")
    for r in results:
        print(
            f"  - {r['dataset_name']}: "
            f"{r['n_train_cells']} train, {r['n_test_cells']} test, "
            f"{r['n_rna_genes']} RNA, {r['n_proteins']} protein"
        )
    print("=" * 70)


def main_cli():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess spatial RNA-to-protein datasets for SPINE"
    )
    parser.add_argument('--train_ratio', type=float, default=0.9, help='Fraction of spots assigned to the training split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split generation')
    parser.add_argument('--no_pca', action='store_true', help='Do not write embeddings_pca into the RNA embedding HDF5 files')
    parser.add_argument('--pca_dim', type=int, default=256, help='Number of PCA components to save when PCA output is enabled')
    parser.add_argument('--dataset_suffix', type=str, default="MINMAX",
                       help='Suffix appended to the processed dataset name')
    parser.add_argument('--dataset_id', type=int, default=None, 
                       help='Process only one dataset id; process all supported datasets when omitted')
    parser.add_argument('--project_root', type=str, default=None,
                       help='Path to the SPINE package root that contains the dataset directories')
    parser.add_argument('--raw_data_root', type=str, default=None,
                       help='Path to the raw spatial RNA/protein input datasets')
    args = parser.parse_args()
    
    if args.dataset_id:
        process_rna_to_protein_dataset(
            dataset_id=args.dataset_id,
            train_ratio=args.train_ratio,
            seed=args.seed,
            dataset_suffix=args.dataset_suffix,
            save_pca_embeddings=(not args.no_pca),
            pca_dim=args.pca_dim,
            project_root=args.project_root,
            raw_data_root=args.raw_data_root,
        )
    else:
        main(
            train_ratio=args.train_ratio,
            seed=args.seed,
            dataset_suffix=args.dataset_suffix,
            save_pca_embeddings=(not args.no_pca),
            pca_dim=args.pca_dim,
            project_root=args.project_root,
            raw_data_root=args.raw_data_root,
        )


if __name__ == "__main__":
    main_cli()
