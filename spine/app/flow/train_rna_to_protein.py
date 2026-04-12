import os
import json
import argparse
from pathlib import Path
import torch
import pandas as pd

from spine.utils import set_random_seed, get_current_time
from spine.data.dataset import (
    RNAToProteinDatasetPath,
    RNAToProteinDataset,
    MultiRNAToProteinDataset,
    padding_batcher,
)
from spine.data.normalize_utils import get_normalize_method
from spine.model.denoiser import Denoiser
from spine.flow.interpolant import Interpolant
from spine.app.flow.test import test


def _load_gene_list(path: str):
    """读取 json 格式的基因列表，兼容 {"genes": [...]} 与直接数组两种格式。"""
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "genes" in data:
        return data["genes"]
    return data


def _safe_save_checkpoint(state_dict, primary_path: str, fallback_root: str | None = None) -> str:
    """
    Save checkpoint to the requested path. If the primary filesystem is full or
    otherwise fails, automatically retry under a fallback root such as /tmp.
    Returns the final saved path.
    """
    try:
        torch.save(state_dict, primary_path)
        return primary_path
    except Exception as exc:
        if not fallback_root:
            raise

        primary = Path(primary_path)
        fallback_dir = Path(fallback_root) / primary.parent.name
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_path = fallback_dir / primary.name
        print(
            f"⚠️  Failed to save checkpoint to {primary_path}: {exc}\n"
            f"   Falling back to {fallback_path}",
            flush=True,
        )
        torch.save(state_dict, str(fallback_path))
        return str(fallback_path)


def main(args, split_id, train_sample_ids, test_sample_ids, val_save_dir, checkpoint_save_dir):
    #  RNA 默认会用 log1p 归一化，传入 None/字符串 "None" 表示保持原始表达
    normalize_method = None if args.normalize_method in (None, "None") else get_normalize_method(args.normalize_method)
    feature_normalize = args.rna_feature_normalize

    def _build_paths(sample_ids):
        # 依据 sample_id 拼接出嵌入/标签文件路径，确保每个样本都有 “RNA_EMBED + protein.h5ad” 成对数据
        return [
            RNAToProteinDatasetPath(
                name=sample_id,
                h5_path=os.path.join(args.embed_dataroot, args.dataset, sample_id, "RNA_EMBED", f"{sample_id}.h5"),
                protein_h5ad_path=os.path.join(args.source_dataroot, args.dataset, sample_id, "protein.h5ad"),
                protein_list_path=os.path.join(args.source_dataroot, args.dataset, args.protein_list),
                rna_gene_list_path=os.path.join(args.source_dataroot, args.dataset, args.rna_gene_list),
            )
            for sample_id in sample_ids
        ]

    train_paths = _build_paths(train_sample_ids)
    # MultiRNAToProteinDataset 会把多样本的 RNA/蛋白拼在一起，并根据 sample_times 随机采样 patch
    train_dataset = MultiRNAToProteinDataset(
        train_paths,
        normalize_method=normalize_method,
        distribution=args.patch_distribution,
        sample_times=args.sample_times,
        feature_normalize_method=feature_normalize,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=padding_batcher(),
        num_workers=args.num_workers,
        shuffle=True,
    )

    val_paths = _build_paths(test_sample_ids)
    # 验证阶段一律使用 deterministic sampler（distribution="constant_1.0"）保持评估稳定
    val_loaders = [
        torch.utils.data.DataLoader(
            RNAToProteinDataset(
                sample_path,
                normalize_method=normalize_method,
                distribution="constant_1.0",
                sample_times=1,
                feature_normalize_method=feature_normalize,
            ),
            batch_size=1,
            collate_fn=padding_batcher(),
        )
        for sample_path in val_paths
    ]

    device = args.device
    # 设置MLP映射参数（用于渐进式降维）
    args.mlp_intermediate_dim = getattr(args, "mlp_intermediate_dim", 1024)
    args.mlp_num_layers = getattr(args, "mlp_num_layers", 2)
    
    # Denoiser = 时间嵌入 + MLP 输入映射 + SpatialTransformer 主干
    model = Denoiser(args).to(device)

    # Interpolant 负责构造 flow matching 的噪声/先验（例如 ZINB），训练时会随机采样时间步
    # 修复：传递正确的设备参数，确保所有张量在同一设备上
    device_str = f"cuda:{device}" if isinstance(device, int) else device
    diffusier = Interpolant(
        args.prior_sampler,
        device=device_str,
        total_count=torch.tensor([args.zinb_total_count]),
        logits=torch.tensor([args.zinb_logits]),
        zi_logits=args.zinb_zi_logits,
        normalize=args.prior_sampler != "gaussian",
    )
    weight_decay = getattr(args, 'weight_decay', 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)

    # Early stopping metric
    # - mse: lower is better
    # - pcc: higher is better
    early_stop_metric = getattr(args, "early_stop_metric", "pcc")
    if early_stop_metric not in {"mse", "pcc"}:
        raise ValueError(f"Unsupported early_stop_metric: {early_stop_metric}")
    best_val_score = float("inf") if early_stop_metric == "mse" else -float("inf")
    best_val_dict = None
    early_stop_step = 0
    accumulation_steps = args.gradient_accumulation_steps

    for epoch in range(1, args.epochs + 1):
        model.train()
        avg_loss = 0.0
        optimizer.zero_grad()
        model.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = [x.to(device) for x in batch]
            if len(batch) == 4:
                img_features, coords, gene_exp, cosine_features = batch
            else:
                img_features, coords, gene_exp = batch
                cosine_features = None

            # 调试用 shape 打印（需要时自行取消注释）
            # print(f"[train] epoch={epoch} step={step} batch - img_features:", img_features.shape,
            #       "coords:", coords.shape,
            #       "gene_exp:", gene_exp.shape)

            noisy_exp, t_steps = diffusier.corrupt_exp(gene_exp)
            # print("[train] after corrupt_exp - noisy_exp:", noisy_exp.shape,
            #       "t_steps:", t_steps.shape)
            pred_exp, loss = model(
                noisy_target=noisy_exp,
                source_features=img_features,
                coords=coords,
                target=gene_exp,
                t_steps=t_steps,
                cosine_features=cosine_features,
            )
            # print("[train] model output - pred_exp:", pred_exp.shape,
            #       "loss:", float(loss))

            loss = loss / accumulation_steps
            loss.backward()
            # 记录真实损失（需要乘以累积步数）
            avg_loss += loss.item() * accumulation_steps

            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                model.zero_grad()

        # 处理最后一个不完整的累积批次
        if len(train_loader) % accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                model.zero_grad()

        avg_loss /= max(1, len(train_loader))
        recon = getattr(model, "last_gene_recon_loss", None)
        if recon is not None:
            print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f} (gene_recon={float(recon.cpu()):.4f}, w={getattr(args,'gene_recon_weight',0.0):.3g})")
        else:
            print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

        if epoch % args.eval_step == 0 or epoch == args.epochs:
            val_perf_dict, _ = test(args, diffusier, model, val_loaders, return_all=True)
            metrics = val_perf_dict["all"]
            pcc = metrics["pearson_mean"]
            mse = metrics["mse"]
            mae = metrics["mae"]
            print(f"\n[Epoch {epoch}] Validation Metrics:")
            print(f"  PCC (Pearson Correlation): {pcc:.4f} ± {metrics['pearson_std']:.4f}")
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")

            current_score = {
                "mse": float(mse),
                "pcc": float(pcc),
            }[early_stop_metric]

            improved = (
                (current_score < best_val_score - 1e-6)
                if early_stop_metric == "mse"
                else (current_score > best_val_score + 1e-6)
            )

            if improved:
                best_val_score = current_score
                best_val_dict = val_perf_dict
                early_stop_step = 0
                pretty_name = {"mse": "MSE", "pcc": "PCC"}[early_stop_metric]
                print(f"  ✓ New best val {pretty_name}: {best_val_score:.6f}")

                for sample_name, res in val_perf_dict.items():
                    save_path = os.path.join(val_save_dir, f"{sample_name}_results.json")
                    with open(save_path, "w") as f:
                        json.dump(res, f, indent=4, sort_keys=True)

                final_ckpt_path = _safe_save_checkpoint(
                    model.state_dict(),
                    os.path.join(checkpoint_save_dir, "best.pth"),
                    fallback_root=getattr(args, "checkpoint_fallback_root", None),
                )
                if final_ckpt_path != os.path.join(checkpoint_save_dir, "best.pth"):
                    print(f"  ✓ Checkpoint saved to fallback path: {final_ckpt_path}")
            else:
                early_stop_step += 1
                if early_stop_step >= args.early_stop_patience:
                    print(f"\n⚠️  Early stopping after {args.early_stop_patience} epochs without improvement")
                    pretty_name = {"mse": "MSE", "pcc": "PCC"}[early_stop_metric]
                    print(f"   Best val {pretty_name} achieved: {best_val_score:.6f}")
                    break
    
    # 训练结束，打印最终结果
    print("\n" + "="*60)
    print("Training Completed!")
    print("="*60)
    final_res = best_val_dict["all"]
    print(f"Best Validation Metrics:")
    print(f"  PCC (Pearson Correlation): {final_res['pearson_mean']:.4f} ± {final_res['pearson_std']:.4f}")
    print(f"  MSE: {final_res['mse']:.4f}")
    print(f"  MAE: {final_res['mae']:.4f}")
    print("="*60 + "\n")

    return best_val_dict["all"]


def run(args):
    split_dir = os.path.join(args.source_dataroot, args.dataset, "splits")
    train_df = pd.read_csv(os.path.join(split_dir, "train_0.csv"))
    test_df = pd.read_csv(os.path.join(split_dir, "test_0.csv"))
    train_sample_ids = train_df["sample_id"].tolist()
    test_sample_ids = test_df["sample_id"].tolist()

    kfold_save_dir = os.path.join(args.save_dir, "split0")
    checkpoint_save_dir = os.path.join(kfold_save_dir, "checkpoints")
    os.makedirs(checkpoint_save_dir, exist_ok=True)

    results = main(args, 0, train_sample_ids, test_sample_ids, kfold_save_dir, checkpoint_save_dir)
    with open(os.path.join(kfold_save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="DATASET2_RNA_TO_PROTEIN_MINMAX_NOHVG_NOMAP")
    parser.add_argument("--source_dataroot", default=str(project_root / "dataset"))
    parser.add_argument("--embed_dataroot", default=str(project_root / "dataset" / "embed_dataroot"))
    parser.add_argument("--save_dir", default=str(project_root / "results_dir"))
    parser.add_argument(
        "--checkpoint_fallback_root",
        type=str,
        default="/tmp/spine_checkpoints",
        help="当主保存目录写失败时，checkpoint 自动回退保存到该目录",
    )
    parser.add_argument("--protein_list", type=str, default="protein_list.json")
    parser.add_argument("--rna_gene_list", type=str, default="rna_gene_list.json")
    parser.add_argument("--gene_list", type=str, default="protein_list.json")

    parser.add_argument("--feature_dim", type=int, default=-1)
    parser.add_argument("--n_proteins", "--n_genes", dest="n_proteins", type=int, default=51,
                        help="预测蛋白数量")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='Weight decay (L2 regularization)')
    parser.add_argument("--sample_times", type=int, default=10)
    parser.add_argument("--eval_step", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--patch_distribution", type=str, default="uniform")
    parser.add_argument("--normalize_method", type=str, default=None)
    parser.add_argument("--rna_feature_normalize", type=str, default="none",
                        choices=["none", "sctranslator_minmax"],
                        help="RNA输入特征的归一化方式 (none / sctranslator_minmax)")
    parser.add_argument("--early_stop_patience", type=int, default=50)
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="pcc",
        choices=["mse", "pcc"],
        help="早停监控指标：mse(越小越好) / pcc(越大越好)",
    )

    parser.add_argument("--n_sample_steps", type=int, default=10,
                        help="Flow Matching 采样步数，增加步数可提升预测分布准确性")
    parser.add_argument("--prior_sampler", type=str, default="gaussian")
    parser.add_argument("--zinb_logits", type=float, default=0.1)
    parser.add_argument("--zinb_total_count", type=float, default=1.0)
    parser.add_argument("--zinb_zi_logits", type=float, default=0.0)

    parser.add_argument("--backbone", type=str, default="spatial_transformer")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--pairwise_hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--attn_dropout", type=float, default=0.3)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_neighbors", type=int, default=8)
    parser.add_argument("--activation", type=str, default="swiglu")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--exp_code", type=str, default="rna_to_protein")
    
    # MLP映射优化参数
    parser.add_argument("--mlp_intermediate_dim", type=int, default=1024,
                        help='Intermediate dim for RNA embedding (默认 1024，适配 ~3k RNA 基因)')
    parser.add_argument("--mlp_num_layers", type=int, default=2, choices=[2, 3],
                        help='Number of layers in MLP mapping: 2 (feature_dim→intermediate→hidden) or 3 (feature_dim→large→medium→hidden)')
    parser.add_argument(
        "--gene_recon_weight",
        type=float,
        default=0.2,
        help="额外的 RNA/embedding 重建损失权重（Denoiser.gene_decoder），默认 0.2；设为 0 表示关闭",
    )

    args = parser.parse_args()

    if isinstance(args.rna_feature_normalize, str):
        if args.rna_feature_normalize.lower() in ("none", ""):
            args.rna_feature_normalize = None

    set_random_seed(args.seed)

    rna_gene_list_path = os.path.join(args.source_dataroot, args.dataset, args.rna_gene_list)
    rna_genes = _load_gene_list(rna_gene_list_path)
    if args.feature_dim <= 0:
        args.feature_dim = len(rna_genes)
    
    # 自动从protein_list.json读取蛋白数量
    protein_list_path = os.path.join(args.source_dataroot, args.dataset, args.protein_list)
    if os.path.exists(protein_list_path):
        proteins = _load_gene_list(protein_list_path)
        actual_n_proteins = len(proteins)
        if args.n_proteins != actual_n_proteins:
            print(f"⚠️  检测到蛋白数量 ({actual_n_proteins}) 与 --n_proteins 参数 ({args.n_proteins}) 不一致，自动更新为 {actual_n_proteins}")
            args.n_proteins = actual_n_proteins
    else:
        print(f"⚠️  警告: 未找到蛋白列表文件 {protein_list_path}，使用默认 n_proteins={args.n_proteins}")
    
    # Final production defaults for SPINE RNA-to-protein training.
    args.loss_type = "mse"
    args.use_lr_scheduler = False
    args.scheduler_type = "plateau"
    args.use_cosine_edge = True
    args.cosine_edge_fixed_expr_weight = 0.1
    args.use_cosine_graph = True
    args.cosine_graph_extra_k = 2
    args.cosine_graph_mode = "union"
    args.cosine_graph_max_spatial_dist_quantile = 0.95
    args.log_view_gate = False
    args.log_attn_diag = False
    args.log_graph_diag = False
    args.use_feature_mlp = True

    exp_code = args.exp_code + f"_{get_current_time()}"
    save_dir = os.path.join(args.save_dir, exp_code, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    run(args)
