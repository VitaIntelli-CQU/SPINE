import os
import json
import numpy as np
from typing import List, Sequence

import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from spine.io_utils.st_dataset import load_adata
from spine.io_utils.file_utils import read_assets_from_h5
from .sampling_utils import PatchSampler
from .normalize_utils import sctranslator_minmax_array

def _maybe_normalize_rna_features(features: np.ndarray, method: str | None):
    if method is None:
        return features
    if method == "sctranslator_minmax":
        return sctranslator_minmax_array(features)
    raise ValueError(f"Unsupported RNA feature normalize method: {method}")


def _load_gene_list(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "genes" in data:
        return data["genes"]
    return data


class SPData:
    features: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    coords: torch.Tensor | None = None
    cosine_features: torch.Tensor | None = None

    def __init__(self, features, labels, coords, cosine_features=None):
        self.features = features
        self.labels = labels
        self.coords = coords
        self.cosine_features = cosine_features

        # decenter
        self.coords[:, 0] = self.coords[:, 0] - self.coords[:, 0].mean()
        self.coords[:, 1] = self.coords[:, 1] - self.coords[:, 1].mean()

    def __len__(self):
        return len(self.features)

    def chunk(self, index):
        return SPData(
            features=self.features[index],
            labels=self.labels[index],
            coords=self.coords[index],
            cosine_features=self.cosine_features[index] if self.cosine_features is not None else None,
        )

class RNAToProteinDatasetPath:
    def __init__(self, name, h5_path, protein_h5ad_path, protein_list_path, rna_gene_list_path, **kwargs):
        self.name = name
        self.h5_path = h5_path
        self.protein_h5ad_path = protein_h5ad_path
        self.protein_list_path = protein_list_path
        self.rna_gene_list_path = rna_gene_list_path
        for k, v in kwargs.items():
            setattr(self, k, v)


class RNAToProteinDataset(Dataset):
    def __init__(self, dataset: RNAToProteinDatasetPath, normalize_method, distribution="beta_3_1", sample_times=5, feature_normalize_method=None):
        super().__init__()
        self.name = dataset.name
        data_dict, _ = read_assets_from_h5(dataset.h5_path)
        barcodes = data_dict["barcodes"].flatten().astype(str).tolist()
        coords = data_dict["coords"]
        rna_features = data_dict["embeddings"]
        rna_features_pca = data_dict.get("embeddings_pca")
        rna_features = _maybe_normalize_rna_features(rna_features, feature_normalize_method)
        if rna_features_pca is not None and rna_features_pca.shape[0] != rna_features.shape[0]:
            raise ValueError("embeddings_pca rows must match embeddings rows")
        rna_genes = _load_gene_list(dataset.rna_gene_list_path)
        if rna_features.shape[1] != len(rna_genes):
            raise ValueError(f"RNA feature dim {rna_features.shape[1]} != gene list ({len(rna_genes)})")
        protein_genes = _load_gene_list(dataset.protein_list_path)
        labels = load_adata(dataset.protein_h5ad_path, genes=protein_genes, barcodes=barcodes, normalize_method=normalize_method).values
        self.gene_list = protein_genes
        self.n_chunks = sample_times
        self.patch_sampler = PatchSampler(distribution)
        self.sp_dataset = SPData(
            features=torch.from_numpy(rna_features).float(),
            labels=torch.from_numpy(labels).float(),
            coords=torch.from_numpy(coords).float(),
            cosine_features=torch.from_numpy(rna_features_pca).float() if rna_features_pca is not None else None,
        )

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        return self.sp_dataset.chunk(self.patch_sampler(self.sp_dataset.coords))

class MultiRNAToProteinDataset(Dataset):
    def __init__(self, dataset_list: List[RNAToProteinDatasetPath], normalize_method, distribution="beta_3_1", sample_times=5, feature_normalize_method=None):
        super().__init__()
        self.dataset_list = dataset_list
        self.sp_datasets = []
        self.n_chunks, self.sample_times = [], sample_times
        self.patch_sampler = PatchSampler(distribution)
        for dataset in self.dataset_list:
            data_dict, _ = read_assets_from_h5(dataset.h5_path)
            barcodes = data_dict["barcodes"].flatten().astype(str).tolist()
            coords = data_dict["coords"]
            rna_features = data_dict["embeddings"]
            rna_features_pca = data_dict.get("embeddings_pca")
            rna_features = _maybe_normalize_rna_features(rna_features, feature_normalize_method)
            if rna_features_pca is not None and rna_features_pca.shape[0] != rna_features.shape[0]:
                raise ValueError("embeddings_pca rows must match embeddings rows")
            rna_genes = _load_gene_list(dataset.rna_gene_list_path)
            if rna_features.shape[1] != len(rna_genes):
                raise ValueError(f"RNA feature dim {rna_features.shape[1]} != gene list ({len(rna_genes)})")
            protein_genes = _load_gene_list(dataset.protein_list_path)
            labels = load_adata(dataset.protein_h5ad_path, genes=protein_genes, barcodes=barcodes, normalize_method=normalize_method).values
            self.n_chunks.append(sample_times)
            self.sp_datasets.append(
                SPData(
                    features=torch.from_numpy(rna_features).float(),
                    labels=torch.from_numpy(labels).float(),
                    coords=torch.from_numpy(coords).float(),
                    cosine_features=torch.from_numpy(rna_features_pca).float() if rna_features_pca is not None else None,
                )
            )


    def __len__(self):
        return sum(self.n_chunks)

    def __getitem__(self, idx):
        for i, n_chunk in enumerate(self.n_chunks):
            if idx < n_chunk:
                return self.sp_datasets[i].chunk(self.patch_sampler(self.sp_datasets[i].coords))
            idx -= n_chunk

class SpatialGlueRNAToProteinDataset(Dataset):
    def __init__(
        self,
        rna_path: str,
        protein_path: str,
        barcodes: Sequence[str],
        normalize_rna=None,
        normalize_protein=None,
        distribution: str = "beta_3_1",
        sample_times: int = 5,
        split_name: str = "train",
    ):
        super().__init__()

        rna_adata = sc.read_h5ad(rna_path)
        protein_adata = sc.read_h5ad(protein_path)

        missing = [b for b in barcodes if b not in rna_adata.obs_names or b not in protein_adata.obs_names]
        if len(missing) > 0:
            raise ValueError(f"Barcodes missing in inputs: {missing[:5]}")

        rna_adata = rna_adata[barcodes].copy()
        protein_adata = protein_adata[barcodes].copy()

        if normalize_rna is not None:
            rna_adata = normalize_rna(rna_adata)
        if normalize_protein is not None:
            protein_adata = normalize_protein(protein_adata)

        coords = rna_adata.obsm.get('spatial')
        if coords is None:
            raise ValueError("RNA AnnData must contain spatial coordinates in obsm['spatial']")
        coords = np.asarray(coords)

        def _to_numpy(matrix):
            if sp.issparse(matrix):
                return matrix.toarray()
            return np.asarray(matrix)

        features = _to_numpy(rna_adata.X)
        labels = _to_numpy(protein_adata.X)

        self.name = f"SpatialGlue_{split_name}"
        self.gene_list = protein_adata.var_names.tolist()
        self.feature_dim = features.shape[1]
        self.label_dim = labels.shape[1]
        self.n_chunks = sample_times
        self.patch_sampler = PatchSampler(distribution)

        self.sp_dataset = SPData(
            features=torch.from_numpy(features).float(),
            labels=torch.from_numpy(labels).float(),
            coords=torch.from_numpy(coords).float(),
        )

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        return self.sp_dataset.chunk(self.patch_sampler(self.sp_dataset.coords))

def padding_batcher():
    def batcher_dev(batch):
        features = [d.features for d in batch]
        labels = [d.labels for d in batch]
        coords = [d.coords for d in batch]
        cosine_features = [getattr(d, "cosine_features", None) for d in batch]

        max_len = max([x.size(0) for x in features])
        features = torch.stack([F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in features])
        labels = torch.stack([F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in labels])
        coords = torch.stack([F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in coords])

        if any(cf is not None for cf in cosine_features):
            if not all(cf is not None for cf in cosine_features):
                raise ValueError("Mixed cosine_features presence in batch; check embeddings_pca availability.")
            cosine_features = torch.stack(
                [F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in cosine_features]
            )
            return features, coords, labels, cosine_features

        return features, coords, labels
    return batcher_dev
