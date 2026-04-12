import scipy
import scprep
import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler


def get_normalize_method(normalize_method, **kwargs):
    if normalize_method is None:
        pass
    elif normalize_method == "log1p":
        return log1p
    elif normalize_method == "stdiff":
        return stdiff_normalize
    elif normalize_method == "scVGAE":
        return scVGAE_normalize
    elif normalize_method == "sctranslator_minmax":
        return sctranslator_minmax
    elif normalize_method == "clr":
        return clr_normalize
    else:
        raise ValueError(f"Unknown normalize method: {normalize_method}")


def identity(adata):
    return adata.copy()


def scale(adata):
    scaler = MaxAbsScaler()
    normalized_data = scaler.fit_transform(adata.X.T).T
    adata.X = normalized_data
    return adata


def log1p(adata):
    process_data = adata.copy()
    sc.pp.log1p(process_data)
    return process_data


# https://github.com/fdu-wangfeilab/stDiff/blob/master/test-stDiff.py#L47
def stdiff_normalize(adata):
    process_adata = adata.copy()
    sc.pp.normalize_total(process_adata, target_sum=1e4)
    sc.pp.log1p(process_adata)
    process_adata = scale(process_adata)
    if isinstance(process_adata.X, scipy.sparse.csr_matrix):
        process_adata.X.data = process_adata.X.data * 2 - 1
    else:
        process_adata.X = process_adata.X * 2 - 1
    return process_adata


def data_augment(adata, fixed, noise_std):
    augmented_adata = adata.copy()
    if fixed:
        augmented_adata.X = augmented_adata.X + np.full(adata.X.shape, noise_std)
    else:
        augmented_adata.X = augmented_adata.X + np.abs(np.random.normal(0, noise_std, adata.X.shape))
    return adata.concatenate(augmented_adata, join='outer')


def scVGAE_normalize(adata):
    process_adata = adata.copy()
    process_adata.X = scprep.normalize.library_size_normalize(process_adata.X)
    process_adata.X = scprep.transform.sqrt(process_adata.X)
    return process_adata


def sctranslator_minmax_array(matrix, low=1e-8, high=1.0):
    """行级 min-max 归一化，参考 scTranslator 的 preprocessing。"""
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"sctranslator_minmax_array expects 2D matrix, got shape {array.shape}")
    min_vals = array.min(axis=1, keepdims=True)
    max_vals = array.max(axis=1, keepdims=True)
    denom = max_vals - min_vals
    safe_denom = np.where(denom < 1e-12, 1.0, denom)
    scaled = low + (array - min_vals) / safe_denom * (high - low)
    flat_mask = (denom < 1e-12).flatten()
    if np.any(flat_mask):
        scaled[flat_mask] = low
    return np.nan_to_num(scaled, nan=low, copy=False)


def sctranslator_minmax(adata):
    """将 AnnData 的每个细胞映射到 (low, high) 区间，匹配 scTranslator 的 scaler。"""
    process_adata = adata.copy()
    matrix = process_adata.X
    if hasattr(matrix, 'toarray'):
        matrix = matrix.toarray()
    process_adata.X = sctranslator_minmax_array(matrix)
    return process_adata


def clr_normalize(adata):
    """
    Seurat 风格 CLR 归一化（按细胞/spot 维度）：
    x_ij' = log1p( x_ij / exp(mean_j(log1p(x_ij))) )
    其中 mean_j 在同一细胞内对所有特征求均值。
    """
    process_adata = adata.copy()
    matrix = process_adata.X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32)
    log1p_x = np.log1p(matrix)
    mean_log1p = log1p_x.mean(axis=1, keepdims=True)
    scale = np.exp(mean_log1p)
    clr = np.log1p(matrix / scale)
    process_adata.X = clr
    return process_adata
