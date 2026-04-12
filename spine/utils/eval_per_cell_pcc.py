import argparse
import json
import os
from pathlib import Path

import numpy as np


EPS = 1e-8


def pcc_cellwise(pred: np.ndarray, truth: np.ndarray) -> float:
    mask = ~np.isnan(truth)
    pccs = []
    for i in range(truth.shape[0]):
        m = mask[i]
        if m.sum() < 2:
            pccs.append(np.nan)
            continue
        t = truth[i, m]
        p = pred[i, m]
        tm, pm = t.mean(), p.mean()
        num = np.sum((t - tm) * (p - pm))
        denom = np.sqrt(np.sum((t - tm) ** 2) * np.sum((p - pm) ** 2))
        pccs.append(np.nan if denom <= EPS else num / denom)
    return float(np.nanmean(pccs))


def cosine_cellwise(pred: np.ndarray, truth: np.ndarray) -> float:
    num = np.sum(pred * truth, axis=1)
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(truth, axis=1)
    denom = np.maximum(denom, EPS)
    cos = num / denom
    cos = np.where(np.isfinite(cos), cos, np.nan)
    return float(np.nanmean(cos))


def mse_mae(pred: np.ndarray, truth: np.ndarray):
    diff = pred - truth
    mask = ~np.isnan(truth)
    mse = float(np.mean(np.square(diff)[mask]))
    mae = float(np.mean(np.abs(diff)[mask]))
    return mse, mae


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-cell PCC / Cosine / MSE / MAE from SPINE pred_target_data.npz outputs"
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        required=True,
        help="Path to pred_target_data.npz (contains pred and target arrays)",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="If set, save metrics to pcc_cellwise.json in the same directory",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz_path)
    data = np.load(npz_path)
    pred = data["pred"]
    truth = data["target"]

    pcc = pcc_cellwise(pred, truth)
    cos = cosine_cellwise(pred, truth)
    mse, mae = mse_mae(pred, truth)

    print(f"Loaded {npz_path}")
    print(f"Shapes pred={pred.shape}, truth={truth.shape}")
    print(f"PCC (cell-wise mean): {pcc:.6f}")
    print(f"Cosine (cell-wise mean): {cos:.6f}")
    print(f"MSE: {mse:.6f}, MAE: {mae:.6f}")

    if args.save_json:
        out = {
            "pcc_cellwise": pcc,
            "cosine_cellwise": cos,
            "mse": mse,
            "mae": mae,
        }
        out_path = npz_path.parent / "pcc_cellwise.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
