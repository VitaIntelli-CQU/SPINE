import os

import numpy as np
import torch
from scipy.stats import pearsonr


def metric_func(preds_all: np.ndarray, y_test: np.ndarray, genes: list):
    """Main paper metrics.

    PCC remains protein-wise. MSE/MAE are reported using the standard
    element-wise definition over all valid entries to align with common
    baseline implementations. For backward compatibility, we also keep the
    legacy per-protein-mean aliases (`mse_mean`, `mae_mean`), which are
    numerically identical when every protein is evaluated on the same number
    of spots and there are no missing entries.
    """
    mse_per_protein = []
    mae_per_protein = []
    pearson_corrs = []
    pearson_genes = []

    n_nan_genes = 0
    for i, target in enumerate(range(y_test.shape[1])):
        preds = preds_all[:, target]
        target_vals = y_test[:, target]

        mse_per_protein.append(float(np.mean((preds - target_vals) ** 2)))
        mae_per_protein.append(float(np.mean(np.abs(preds - target_vals))))

        pred_std = np.std(preds)
        target_std = np.std(target_vals)
        if pred_std == 0 or target_std == 0:
            pearson_corr = np.nan
        else:
            pearson_corr, _ = pearsonr(target_vals, preds)
        pearson_corrs.append(pearson_corr)

        if np.isnan(pearson_corr):
            n_nan_genes += 1

        pearson_genes.append({
            "name": genes[i],
            "pearson_corr": pearson_corr,
        })

    if n_nan_genes > 0:
        print(f"Warning: {n_nan_genes} genes have NaN Pearson correlation")

    diff = preds_all - y_test
    valid_mask = ~np.isnan(y_test)
    mse_overall = float(np.mean(np.square(diff)[valid_mask]))
    mae_overall = float(np.mean(np.abs(diff)[valid_mask]))
    mse_mean = float(np.mean(mse_per_protein))
    mae_mean = float(np.mean(mae_per_protein))

    return {
        "pearson_corrs": pearson_genes,
        "pearson_mean": float(np.nanmean(pearson_corrs)),
        "pearson_std": float(np.nanstd(pearson_corrs)),
        "mse": mse_overall,
        "mae": mae_overall,
        "mse_mean": mse_mean,
        "mae_mean": mae_mean,
    }


@torch.no_grad()
def test(args, diffusier, model, loader_list, return_all=False):
    model.eval()
    all_pred, all_gt = [], []
    res_dict = {}

    for loader in loader_list:
        cur_pred, cur_gt = [], []

        for batch in loader:
            batch = [x.to(args.device) for x in batch]
            if len(batch) == 4:
                rna_features, coords, protein_target, cosine_features = batch
            else:
                rna_features, coords, protein_target = batch
                cosine_features = None
            assert rna_features.shape[0] == 1, "Batch size must be 1 for inference"

            exp_t1 = diffusier.sample_from_prior(protein_target.shape).to(args.device)
            ts = torch.linspace(
                0.01, 1.0, args.n_sample_steps
            )[:, None].expand(args.n_sample_steps, exp_t1.shape[0]).to(args.device)

            for step, (t1, t2) in enumerate(zip(ts[:-1], ts[1:])):
                pred = model.inference(
                    exp_t1,
                    rna_features,
                    coords,
                    t1,
                    cosine_features=cosine_features,
                    predict=True,
                )
                d_t = t2 - t1

                if step == args.n_sample_steps - 2:
                    break
                exp_t1 = diffusier.denoise(pred, exp_t1, t1, d_t)

            cur_pred.append(pred.squeeze(0).cpu().numpy())
            cur_gt.append(protein_target.squeeze(0).cpu().numpy())

        cur_pred = np.concatenate(cur_pred, axis=0)
        cur_gt = np.concatenate(cur_gt, axis=0)
        cur_res_dict = metric_func(cur_pred, cur_gt, loader.dataset.gene_list)
        cur_res_dict.update({"n_test": len(cur_gt)})
        res_dict[loader.dataset.name] = cur_res_dict

        all_pred.append(cur_pred)
        all_gt.append(cur_gt)

    all_pred = np.concatenate(all_pred, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)
    cur_res_dict = metric_func(all_pred, all_gt, loader_list[0].dataset.gene_list)
    cur_res_dict.update({"n_test": len(all_gt)})
    res_dict["all"] = cur_res_dict

    if return_all:
        save_dir = getattr(args, "save_dir", None)
        if save_dir:
            viz_dir = os.path.join(save_dir, "split0", "visualization_data")
            os.makedirs(viz_dir, exist_ok=True)
            np.savez(os.path.join(viz_dir, "pred_target_data.npz"), pred=all_pred, target=all_gt)

        return res_dict, {"preds_all": all_pred, "targets_all": all_gt}
    return res_dict
