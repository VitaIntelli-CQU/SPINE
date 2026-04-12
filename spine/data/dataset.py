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


# 添加一个新的数据集路径类用于ATAC到RNA任务
class ATACToRNADatasetPath:
    name: str | None = None
    h5_path: str | None = None  # ATAC嵌入文件路径
    h5ad_path: str | None = None  # RNA表达数据路径
    gene_list_path: str | None = None

    def __init__(self, name, h5_path, h5ad_path, gene_list_path, **kwargs):
        self.name = name
        self.h5_path = h5_path
        self.h5ad_path = h5ad_path
        self.gene_list_path = gene_list_path

        for k, v in kwargs.items():
            setattr(self, k, v)






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


# 添加用于ATAC到RNA任务的新数据集类
class ATACToRNADataset(Dataset):
    def __init__(self, dataset: ATACToRNADatasetPath, normalize_method, distribution="beta_3_1", sample_times=5):
        super().__init__()

        self.name = dataset.name
        
        # 读取ATAC嵌入数据（作为输入特征）
        data_dict, _ = read_assets_from_h5(dataset.h5_path)
        barcodes = data_dict["barcodes"].flatten().astype(str).tolist()
        coords = data_dict["coords"]
        # 注意：这里我们使用ATAC的嵌入作为特征
        atac_embeddings = data_dict["embeddings"]

        # 读取基因列表
        with open(os.path.join(dataset.gene_list_path), 'r') as f:
            genes_data = json.load(f)
            # 兼容不同的JSON格式
            if isinstance(genes_data, dict) and 'genes' in genes_data:
                genes = genes_data['genes']
            else:
                genes = genes_data

        self.gene_list = genes
        
        # 读取RNA表达数据（作为标签）
        rna_labels = load_adata(dataset.h5ad_path, genes=genes, barcodes=barcodes, normalize_method=normalize_method)
        rna_labels = rna_labels.values

        self.n_chunks = sample_times
        self.patch_sampler = PatchSampler(distribution)

        self.sp_dataset = SPData(
                features=torch.from_numpy(atac_embeddings).float(),  # ATAC嵌入作为输入特征
                labels=torch.from_numpy(rna_labels).float(),         # RNA表达作为标签
                coords=torch.from_numpy(coords).float()
            )
        
    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        return self.sp_dataset.chunk(self.patch_sampler(self.sp_dataset.coords))


class MultiATACToRNADataset(Dataset):
    def __init__(self, dataset_list: List[ATACToRNADatasetPath], normalize_method, distribution="beta_3_1", sample_times=5):
        super().__init__()

        self.dataset_list = dataset_list
        self.sp_datasets = []
        self.n_chunks, self.sample_times = [], sample_times
        self.patch_sampler = PatchSampler(distribution)

        for i, dataset in enumerate(self.dataset_list):
            # 读取ATAC嵌入数据（作为输入特征）
            data_dict, _ = read_assets_from_h5(dataset.h5_path)
            barcodes = data_dict["barcodes"].flatten().astype(str).tolist()
            coords = data_dict["coords"]
            # 注意：这里我们使用ATAC的嵌入作为特征
            atac_embeddings = data_dict["embeddings"]

            # 读取基因列表
            with open(os.path.join(dataset.gene_list_path), 'r') as f:
                genes_data = json.load(f)
                # 兼容不同的JSON格式
                if isinstance(genes_data, dict) and 'genes' in genes_data:
                    genes = genes_data['genes']
                else:
                    genes = genes_data

            # 读取RNA表达数据（作为标签）
            rna_labels = load_adata(dataset.h5ad_path, genes=genes, barcodes=barcodes, normalize_method=normalize_method)
            rna_labels = rna_labels.values

            self.n_chunks.append(sample_times)

            self.sp_datasets.append(
                SPData(
                    features=torch.from_numpy(atac_embeddings).float(),  # ATAC嵌入作为输入特征
                    labels=torch.from_numpy(rna_labels).float(),         # RNA表达作为标签
                    coords=torch.from_numpy(coords).float()
                )
            )
        
    def __len__(self):
        return sum(self.n_chunks)

    def __getitem__(self, idx):
        for i, n_chunk in enumerate(self.n_chunks):
            if idx < n_chunk:
                return self.sp_datasets[i].chunk(
                        self.patch_sampler(self.sp_datasets[i].coords)
                    )
            idx -= n_chunk




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
        # RNA_EMBED/*.h5 中已经存好了 embeddings/barcodes/coords 三份数据
        data_dict, _ = read_assets_from_h5(dataset.h5_path)
        barcodes = data_dict["barcodes"].flatten().astype(str).tolist()
        coords = data_dict["coords"]
        # embeddings 即为未经降维的 RNA 表达矩阵 (cells × genes)，后续会当作 img_features 输入模型
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
        # 调试用 shape 打印（需要时自行取消注释）
        # print("[RNAToProteinDataset] rna_features:", rna_features.shape,
        #       "labels:", labels.shape,
        #       "coords:", coords.shape)
        # SPData 内部负责完成 patch 采样与 padding，对齐 SPINE 主干的输入格式
        self.sp_dataset = SPData(
            features=torch.from_numpy(rna_features).float(),
            labels=torch.from_numpy(labels).float(),
            coords=torch.from_numpy(coords).float(),
            cosine_features=torch.from_numpy(rna_features_pca).float() if rna_features_pca is not None else None,
        )

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        # 调试用 shape 打印（需要时自行取消注释）
        # chunk = self.sp_dataset.chunk(self.patch_sampler(self.sp_dataset.coords))
        # print("[RNAToProteinDataset.__getitem__] chunk.features:", chunk.features.shape,
        #       "chunk.labels:", chunk.labels.shape,
        #       "chunk.coords:", chunk.coords.shape)
        # return chunk
        return self.sp_dataset.chunk(self.patch_sampler(self.sp_dataset.coords))

# 服务已经有.h5ad文件，不需要再生成.h5文件，可跨多样本随机抽取patch
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
            # 每个样本的 embeddings 都是完整 RNA 表达矩阵，维度一致保证可以堆叠
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
            # 调试用 shape 打印（需要时自行取消注释）
            # print(f"[MultiRNAToProteinDataset] sample {dataset.name}:",
            #       "rna_features:", rna_features.shape,
            #       "labels:", labels.shape,
            #       "coords:", coords.shape)
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
                # 根据 idx 定位到具体样本，再用 patch_sampler 采样一块空间 patch
                # 调试用 shape 打印（需要时自行取消注释）
                # chunk = self.sp_datasets[i].chunk(
                #     self.patch_sampler(self.sp_datasets[i].coords)
                # )
                # print(f"[MultiRNAToProteinDataset.__getitem__] sample_index={i}",
                #       "chunk.features:", chunk.features.shape,
                #       "chunk.labels:", chunk.labels.shape,
                #       "chunk.coords:", chunk.coords.shape)
                # return chunk
                return self.sp_datasets[i].chunk(
                    self.patch_sampler(self.sp_datasets[i].coords)
                )
            idx -= n_chunk


# 服务已经有.h5ad文件，不需要再生成.h5文件，能直接使用spatialglue数据集
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

        # 调试用 shape 打印（需要时自行取消注释）
        # print(f"[SpatialGlueRNAToProteinDataset] split={split_name}",
        #       "features:", features.shape,
        #       "labels:", labels.shape,
        #       "coords:", coords.shape)

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

        # 调试用 shape 打印（需要时自行取消注释）
        # print("[padding_batcher] before padding - batch_size=", len(features))
        # for i, (f, l, c) in enumerate(zip(features, labels, coords)):
        #     print(f"  sample {i}: features={f.shape}, labels={l.shape}, coords={c.shape}")

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

        # 调试用 shape 打印（需要时自行取消注释）
        # print("[padding_batcher] after padding & stack - features:", features.shape,
        #       "labels:", labels.shape,
        #       "coords:", coords.shape)

        return features, coords, labels
    return batcher_dev
