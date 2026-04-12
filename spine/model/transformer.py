import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, SwiGLUPacked
from torch_geometric.utils import to_dense_batch
from einops import rearrange, einsum

from .fa import FrameAveraging


def get_activation(activation="gelu"):
    return {
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "relu": nn.ReLU,
    }[activation]


class GeneUpdate(nn.Module):
    def __init__(
            self, 
            d_model, 
            n_genes,
            proj_drop=0.,
            non_negative=False
        ):
        super(GeneUpdate, self).__init__()    

        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(proj_drop),
            nn.Linear(d_model, n_genes),
            nn.Dropout(proj_drop),
        )
    
    def forward(self, features):
        update = self.output(features) 
        return update


class MLPAttnEdgeAggregation(FrameAveraging):
    def __init__(
            self, 
            d_model, 
            d_edge_model,
            n_genes,
            n_heads=1,
            proj_drop=0.,
            attn_drop=0.,
            activation='gelu',
            use_cosine_edge: bool = False,
            cosine_edge_fixed_expr_weight: float | None = None,
            log_view_gate: bool = False,
            log_attn_diag: bool = False,
        ):
        super(MLPAttnEdgeAggregation, self).__init__(dim=2)
        
        self.d_head, self.d_edge_head, self.n_heads = d_model // n_heads, d_edge_model // n_heads, n_heads
        self.use_cosine_edge = use_cosine_edge
        self.cosine_edge_fixed_expr_weight = (
            None if cosine_edge_fixed_expr_weight is None else float(cosine_edge_fixed_expr_weight)
        )
        if self.cosine_edge_fixed_expr_weight is not None:
            if not (0.0 <= self.cosine_edge_fixed_expr_weight <= 1.0):
                raise ValueError("cosine_edge_fixed_expr_weight must be in [0, 1]")
        self.log_view_gate = log_view_gate
        self.log_attn_diag = log_attn_diag
        self.last_w_expr = None
        self.last_w_spatial = None
        self.last_attn_diag = None
        if use_cosine_edge:
            # Expression-view cosine bias scale (global; model learns magnitude)
            self.cos_scale = nn.Parameter(torch.tensor(0.1))
            if self.cosine_edge_fixed_expr_weight is None:
                # Two-view adaptive fusion gate (spatial-view logits vs expression-view logits)
                # Produces per-node, per-head weights that sum to 1.
                self.view_gate = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, n_heads * 2, bias=True),
                )
                # Initialize to near-uniform weights at start
                nn.init.zeros_(self.view_gate[1].weight)
                nn.init.zeros_(self.view_gate[1].bias)
            else:
                # Fixed fusion ratio: no learnable view gate.
                self.view_gate = None

        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 3),
        )

        # 注意力 MLP 同时接收 query/token/edge 三类特征 + 基因表达差值，用于刻画空间邻域关系
        if activation == "swiglu":
            self.mlp_attn = SwiGLUPacked(
                in_features=self.d_head*2+self.d_edge_head+n_genes, hidden_features=d_model, 
                out_features=1, drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.edge_trans = SwiGLUPacked(
                in_features=self.dim+1, hidden_features=d_edge_model, 
                out_features=d_edge_model, drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.W_output = SwiGLUPacked(
                in_features=d_model+d_edge_model, hidden_features=d_model, 
                out_features=d_model, drop=proj_drop, norm_layer=nn.LayerNorm
            )
        else:
            self.mlp_attn = Mlp(
                in_features=self.d_head*2+self.d_edge_head+n_genes, hidden_features=d_model, 
                out_features=1, drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.edge_trans = Mlp(
                in_features=self.dim+1, hidden_features=d_edge_model, out_features=d_edge_model, 
                act_layer=get_activation(activation), drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.W_output = Mlp(
                in_features=d_model+d_edge_model, hidden_features=d_model, out_features=d_model, 
                act_layer=get_activation(activation), drop=proj_drop, norm_layer=nn.LayerNorm
            )

        self.attn_dropout = nn.Dropout(attn_drop)

    def forward(self, gene_exp, token_embs, coords, neighbor_indices, neighbor_masks=None, cosine_features=None):
        # gene_exp: [N, N_genes], token_embs: [N, -1], geo_token_embs: [N, 3]
        # neighbor_indices: [N, N_neighbor], neighbor_masks: [N, N_neighbor]
        n_tokens, n_neighbors = token_embs.size(0), neighbor_indices.size(1)
        n_heads, d_head, d_edge_head = self.n_heads, self.d_head, self.d_edge_head

        q_s, k_s, v_s = self.layernorm_qkv(token_embs).chunk(3, dim=-1)
        q_s, k_s, v_s = map(lambda x: rearrange(x, 'n (h d) -> n h d', h=n_heads), (q_s, k_s, v_s))

        """build pairwise representation with FA"""
        radial_coords = coords[neighbor_indices] - coords.unsqueeze(dim=1)  # [N, N_neighbor, 2]
        radial_coord_norm = radial_coords.norm(dim=-1).unsqueeze(-1)  # [N, N_neighbor, 1]

        # FrameAveraging expects mask=True for valid entries, but neighbor_masks uses True for invalid/pad.
        fa_mask = None if neighbor_masks is None else ~neighbor_masks
        frame_feats, _, _ = self.create_frame(radial_coords, fa_mask)  # [N*8, N_neighbors, 3]
        frame_feats = frame_feats.view(n_tokens, self.n_frames, n_neighbors, -1)  # [N, 8, N_neighbors, d_model]

        radial_coord_norm = radial_coord_norm.unsqueeze(dim=1).expand(n_tokens, self.n_frames, n_neighbors, -1)
        frame_feats = self.edge_trans(torch.cat([frame_feats, radial_coord_norm], dim=-1)).mean(dim=1)  # [N, N_neighbors, d_edge_model]

        """gene expression features"""
        gene_exp_diff = gene_exp[neighbor_indices] - gene_exp.unsqueeze(dim=1)  # [N, N_neighbor, N_genes]
        gene_exp_feats_expand = gene_exp_diff[..., None, :].expand(n_tokens, n_neighbors, n_heads, -1)  # [N, N_neighbor, n_heads, N_genes+1]

        """attention map"""
        q_s = q_s.unsqueeze(dim=1).expand(n_tokens, n_neighbors, n_heads, d_head)
        frame_feats = frame_feats.view(n_tokens, n_neighbors, n_heads, d_edge_head)
        message = torch.cat([q_s, k_s[neighbor_indices], frame_feats, gene_exp_feats_expand], dim=-1)
        
        spatial_logits = self.mlp_attn(message).squeeze(-1)  # [N, N_neighbor, n_heads]
        if self.use_cosine_edge:
            # Two-view fusion on the same neighborhood (neighbor_indices).
            # - Spatial-view: learned logits from (q,k,edge,expression-diff)
            # - Expression-view: cosine similarity on cosine_features (preferred) or token_embs
            base = cosine_features if cosine_features is not None else token_embs
            expr_logits = F.cosine_similarity(base.unsqueeze(1), base[neighbor_indices], dim=-1)  # [N, N_neighbor]
            expr_logits = expr_logits.unsqueeze(-1).expand(-1, n_neighbors, n_heads)  # [N, N_neighbor, n_heads]

            if self.view_gate is None:
                w_expr = torch.full(
                    (n_tokens, 1, n_heads),
                    float(self.cosine_edge_fixed_expr_weight),
                    device=token_embs.device,
                    dtype=spatial_logits.dtype,
                )
                w_spatial = 1.0 - w_expr
            else:
                gate = self.view_gate(token_embs).view(n_tokens, n_heads, 2)  # [N, n_heads, 2]
                gate = F.softmax(gate, dim=-1)
                w_spatial = gate[..., 0].unsqueeze(1)  # [N, 1, n_heads]
                w_expr = gate[..., 1].unsqueeze(1)     # [N, 1, n_heads]

            if self.log_view_gate or self.log_attn_diag:
                # record per-head mean weights for debugging (avoid excessive printing in forward)
                # shapes: [n_heads]
                w_expr_nh = w_expr.squeeze(1)  # [N, H]
                w_spatial_nh = w_spatial.squeeze(1)  # [N, H]
                self.last_w_spatial = w_spatial_nh.mean(dim=0).detach()
                self.last_w_expr = w_expr_nh.mean(dim=0).detach()

                if self.log_attn_diag:
                    # Summary stats across nodes (per head): mean/std/p10/p50/p90
                    # Quantiles require float tensor on device; keep small and detach.
                    def _q(x, q):
                        return torch.quantile(x, q, dim=0)

                    w_expr_std = w_expr_nh.std(dim=0, unbiased=False)
                    w_expr_p10 = _q(w_expr_nh, 0.10)
                    w_expr_p50 = _q(w_expr_nh, 0.50)
                    w_expr_p90 = _q(w_expr_nh, 0.90)

                    # Logit-scale diagnostics (per head)
                    s = spatial_logits.detach()
                    e = expr_logits.detach()
                    s_flat = s.reshape(-1, n_heads)
                    e_flat = e.reshape(-1, n_heads)
                    s_std = s_flat.std(dim=0, unbiased=False)
                    e_std = e_flat.std(dim=0, unbiased=False)
                    es = (self.cos_scale.detach() * e_flat)
                    es_std = es.std(dim=0, unbiased=False)

                    # Pearson corr between spatial_logits and expr_logits across edges (per head)
                    s_center = s_flat - s_flat.mean(dim=0, keepdim=True)
                    e_center = e_flat - e_flat.mean(dim=0, keepdim=True)
                    denom = (s_center.pow(2).mean(dim=0) * e_center.pow(2).mean(dim=0)).sqrt().clamp_min(1e-8)
                    corr = (s_center * e_center).mean(dim=0) / denom

                    self.last_attn_diag = {
                        "cos_scale": float(self.cos_scale.detach().cpu()),
                        "w_expr_mean": self.last_w_expr.detach().cpu(),
                        "w_expr_std": w_expr_std.detach().cpu(),
                        "w_expr_p10": w_expr_p10.detach().cpu(),
                        "w_expr_p50": w_expr_p50.detach().cpu(),
                        "w_expr_p90": w_expr_p90.detach().cpu(),
                        "spatial_logits_std": s_std.detach().cpu(),
                        "expr_logits_std": e_std.detach().cpu(),
                        "expr_scaled_std": es_std.detach().cpu(),
                        "logits_corr": corr.detach().cpu(),
                    }

            attn_map = w_spatial * spatial_logits + w_expr * (self.cos_scale * expr_logits)
        else:
            attn_map = spatial_logits
        if neighbor_masks is not None:
            attn_map.masked_fill_(neighbor_masks.unsqueeze(dim=-1), -1e9)
        attn_map = self.attn_dropout(nn.Softmax(dim=-1)(attn_map.transpose(1, 2)))  # [N, n_heads, N_neighbor]

        """context aggregation"""
        v_s_neighs = v_s[neighbor_indices].view(n_tokens, -1, n_heads, d_head)  # [N, n_heads, N_neighbor, D]
        scalar_context = einsum(attn_map, v_s_neighs, 'n h m, n m h d -> n h d').view(n_tokens, -1)  # [N, n_heads*D]
        edge_context = einsum(attn_map, frame_feats, 'n h m, n m h d -> n h d').view(n_tokens, -1)  # [N, n_heads*D]
        return self.W_output(torch.cat([scalar_context, edge_context], dim=-1))


class TransformerBlock(nn.Module):
    def __init__(            
            self,
            d_model,
            d_edge_model,
            n_genes,
            n_heads=1,
            activation="gelu",
            attn_drop=0.,
            proj_drop=0.,
            gene_exp_non_negative=True,
            mlp_ratio=4.0,
            use_cosine_edge: bool = False,
            cosine_edge_fixed_expr_weight: float | None = None,
            log_view_gate: bool = False,
            log_attn_diag: bool = False,
        ):
        super(TransformerBlock, self).__init__()

        self.attn = MLPAttnEdgeAggregation(
            d_model=d_model, d_edge_model=d_edge_model, n_genes=n_genes, n_heads=n_heads, 
            proj_drop=proj_drop, attn_drop=attn_drop, activation=activation, use_cosine_edge=use_cosine_edge,
            cosine_edge_fixed_expr_weight=cosine_edge_fixed_expr_weight,
            log_view_gate=log_view_gate,
            log_attn_diag=log_attn_diag,
        )

        if activation == "swiglu":
            self.mlp = SwiGLUPacked(
                in_features=d_model, hidden_features=int(d_model * mlp_ratio), drop=proj_drop, norm_layer=nn.LayerNorm
            )
        else:
            self.mlp = Mlp(
                in_features=d_model, hidden_features=int(d_model * mlp_ratio), 
                act_layer=get_activation(activation), drop=proj_drop, norm_layer=nn.LayerNorm
            )
        
        self.gene_updater = GeneUpdate(d_model, n_genes, proj_drop=proj_drop, non_negative=gene_exp_non_negative)

    def forward(self, gene_exp, token_embs, coords, neighbor_indices, neighbor_masks=None, cosine_features=None):
        # 第一步：根据空间邻域聚合上下文特征
        context_token_embs = self.attn(
            gene_exp,
            token_embs,
            coords,
            neighbor_indices,
            neighbor_masks=neighbor_masks,
            cosine_features=cosine_features,
        )
        token_embs = token_embs + context_token_embs

        # 第二步：MLP 提升 token 表示
        token_embs = token_embs + self.mlp(token_embs)
        # 第三步：GeneUpdate 将 token 表示还原成 n_genes 维表达（相当于每个 token 的预测）
        gene_exp = self.gene_updater(token_embs)

        return gene_exp, token_embs


class SpatialTransformer(nn.Module):
    def __init__(self, config):
        super(SpatialTransformer, self).__init__()

        self.n_neighbors = config.n_neighbors
        self.use_cosine_edge = bool(getattr(config, "use_cosine_edge", False))
        self.cosine_edge_fixed_expr_weight = getattr(config, "cosine_edge_fixed_expr_weight", None)
        self.use_cosine_graph = bool(getattr(config, "use_cosine_graph", False))
        self.cosine_graph_extra_k = int(getattr(config, "cosine_graph_extra_k", 4))
        self.cosine_graph_mode = str(getattr(config, "cosine_graph_mode", "extra"))
        q = getattr(config, "cosine_graph_max_spatial_dist_quantile", None)
        self.cosine_graph_max_spatial_dist_quantile = None if q is None else float(q)
        self.log_view_gate = bool(getattr(config, "log_view_gate", False))
        self.log_attn_diag = bool(getattr(config, "log_attn_diag", False))
        self.log_graph_diag = bool(getattr(config, "log_graph_diag", False))
        self.last_graph_diag = None

        self.blks = nn.ModuleList([
            TransformerBlock(config.d_model, config.d_edge_model, 
                          n_genes=config.n_genes, n_heads=config.n_heads, 
                              activation=config.act, attn_drop=config.attn_dropout, 
                              proj_drop=config.dropout, use_cosine_edge=self.use_cosine_edge,
                              cosine_edge_fixed_expr_weight=self.cosine_edge_fixed_expr_weight,
                              log_view_gate=self.log_view_gate,
                              log_attn_diag=self.log_attn_diag,
                            ) \
                for i in range(config.n_layers)
        ])

    def _build_graph(self, coords, batch_idx, features, n_neighbors, exclude_self=True):
        # coords: [N, 2], batch_idx: [N], features: [N, d_model] or None, n_neighbors: int
        n_tokens = coords.shape[0]
        
        # 特殊处理：当n_neighbors=0时，让每个节点只关注自己（自注意力模式）
        if n_neighbors == 0:
            self_indices = torch.arange(n_tokens, device=coords.device).unsqueeze(1)  # [N, 1]
            return self_indices, torch.zeros_like(self_indices, dtype=torch.bool)

        if n_tokens <= 1:
            idx = torch.arange(n_tokens, device=coords.device).unsqueeze(1)
            return idx, torch.zeros_like(idx, dtype=torch.bool)
        
        exclude_self_mask = torch.eye(n_tokens, dtype=torch.bool, device=coords.device)  # 1: diagonal elements
        batch_mask = batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1)  # [N, N], True if the token is in the same batch

        # calculate relative distance
        rel_pos = rearrange(coords, 'n d -> n 1 d') - rearrange(coords, 'n d -> 1 n d')
        rel_dist = rel_pos.norm(dim = -1).detach()  # [N, N]

        if exclude_self:
            rel_dist.masked_fill_(exclude_self_mask | ~batch_mask, 1e9)
        else:
            rel_dist.masked_fill_(~batch_mask, 1e9)

        k_base = min(n_neighbors, max(1, n_tokens - 1))
        _, euclid_idx = rel_dist.topk(k_base, dim=-1, largest=False)

        # Optional: record diagnostics (overlap between spatial KNN and expression KNN).
        # Note: union/extra mode may change the FINAL neighbor list; we also record effective_k_mean below.
        graph_diag = None
        if self.log_graph_diag and features is not None and n_tokens > 1:
            feat = F.normalize(features, dim=-1)
            cos_sim = torch.einsum('nd,md->nm', feat, feat)  # [N, N]
            cos_sim.masked_fill_(exclude_self_mask | ~batch_mask, -1e9)
            _, cos_topk_for_diag = cos_sim.topk(k_base, dim=-1, largest=True)
            inter = (euclid_idx.unsqueeze(-1) == cos_topk_for_diag.unsqueeze(-2)).any(dim=-1).float().sum(dim=-1)
            recall = (inter / float(k_base)).mean()
            union_size = (k_base * 2 - inter).mean()
            graph_diag = {
                "k_spatial": int(k_base),
                "k_expr": int(k_base),
                "expr_recall_at_k": float(recall.detach().cpu()),
                "union_size_mean": float(union_size.detach().cpu()),
                "overlap_mean": float(inter.mean().detach().cpu()),
                "n_tokens": int(n_tokens),
            }

        if (
            self.use_cosine_graph
            and features is not None
            and self.cosine_graph_extra_k > 0
            and n_tokens > n_neighbors + 1
        ):
            feat = F.normalize(features, dim=-1)
            cos_sim = torch.einsum('nd,md->nm', feat, feat)  # [N, N]
            cos_sim.masked_fill_(exclude_self_mask | ~batch_mask, -1e9)

            # Optional spatial constraint for expression neighbors:
            # Only allow expression edges whose spatial distance is not too large.
            # Threshold is computed from the distribution of spatial KNN distances.
            spatial_thr = None
            if self.cosine_graph_max_spatial_dist_quantile is not None:
                q = self.cosine_graph_max_spatial_dist_quantile
                if not (0.0 < q <= 1.0):
                    raise ValueError("cosine_graph_max_spatial_dist_quantile must be in (0, 1].")
                spatial_dists = rel_dist.gather(1, euclid_idx)  # [N, k_base]
                # rel_dist already masked across batches; euclid_idx distances are finite.
                spatial_thr = torch.quantile(spatial_dists.reshape(-1), q)
                too_far = rel_dist > spatial_thr
                cos_sim = cos_sim.masked_fill(too_far, -1e9)

            if self.cosine_graph_mode not in {"extra", "union"}:
                raise ValueError(f"Unknown cosine_graph_mode={self.cosine_graph_mode!r}, expected 'extra' or 'union'")

            if self.cosine_graph_mode == "extra":
                # mask euclidean neighbors to only add new ones (forces k_extra new edges if possible)
                mask_euclid = torch.zeros_like(cos_sim, dtype=torch.bool)
                mask_euclid.scatter_(1, euclid_idx, True)
                cos_sim.masked_fill_(mask_euclid, -1e9)

                max_extra = max(0, n_tokens - 1 - k_base)
                k_extra = min(self.cosine_graph_extra_k, max_extra)
                if k_extra > 0:
                    _, cos_idx = cos_sim.topk(k_extra, dim=-1, largest=True)
                    idx = torch.cat([euclid_idx, cos_idx], dim=-1)
                    if graph_diag is not None:
                        graph_diag.update(
                            {
                                "mode": "extra",
                                "k_expr_used": int(k_extra),
                                "effective_k_mean": float(idx.shape[1]),
                            }
                        )
                        self.last_graph_diag = graph_diag
                    return idx, torch.zeros_like(idx, dtype=torch.bool)

            # union: spatialKNN ∪ exprKNN (does NOT force-append low-sim edges when overlap is high)
            # Output is padded with self-index and masked to keep rectangular shape.
            k_expr = min(self.cosine_graph_extra_k, max(1, n_tokens - 1))
            _, cos_topk = cos_sim.topk(k_expr, dim=-1, largest=True)  # [N, k_expr]
            max_len = k_base + k_expr
            device = coords.device
            out_idx = torch.empty((n_tokens, max_len), dtype=torch.long, device=device)
            out_mask = torch.ones((n_tokens, max_len), dtype=torch.bool, device=device)
            self_ids = torch.arange(n_tokens, device=device)
            effective_counts = torch.empty((n_tokens,), dtype=torch.float32, device=device)
            for i in range(n_tokens):
                seen = set()
                uniq = []
                for j in euclid_idx[i].tolist():
                    if j not in seen:
                        seen.add(j)
                        uniq.append(j)
                for j in cos_topk[i].tolist():
                    # Skip invalid candidates when spatial constraint masks most entries.
                    # (topk may still return indices with -1e9 scores when not enough candidates exist)
                    if cos_sim[i, j].item() <= -1e8:
                        continue
                    if j not in seen:
                        seen.add(j)
                        uniq.append(j)
                take = min(len(uniq), max_len)
                if take > 0:
                    out_idx[i, :take] = torch.tensor(uniq[:take], device=device)
                    out_mask[i, :take] = False
                effective_counts[i] = float(take)
                # pad with self
                if take < max_len:
                    out_idx[i, take:] = self_ids[i]
                    out_mask[i, take:] = True
            if graph_diag is not None:
                graph_diag.update(
                    {
                        "mode": "union",
                        "k_expr_used": int(k_expr),
                        "effective_k_mean": float(effective_counts.mean().detach().cpu()),
                        "effective_k_min": float(effective_counts.min().detach().cpu()),
                        "effective_k_max": float(effective_counts.max().detach().cpu()),
                    }
                )
                if spatial_thr is not None:
                    graph_diag["spatial_dist_thr"] = float(spatial_thr.detach().cpu())
                    graph_diag["spatial_dist_thr_q"] = float(self.cosine_graph_max_spatial_dist_quantile)
                self.last_graph_diag = graph_diag
            return out_idx, out_mask

        if graph_diag is not None:
            graph_diag.update({"mode": "spatial_only", "k_expr_used": 0, "effective_k_mean": float(euclid_idx.shape[1])})
            self.last_graph_diag = graph_diag
        return euclid_idx, torch.zeros_like(euclid_idx, dtype=torch.bool)

    def forward(self, gene_exp, features, coords, return_token_embs: bool = False, cosine_features=None):
        # gene_exp: [B, N_cells, N_genes], features: [B, N_cells, -1], coords: [B, N_cells, 2]
        # cosine_features: optional [B, N_cells, -1] used only for cosine KNN (e.g., PCA features)
        B, N_cells, N_genes = gene_exp.shape[0], gene_exp.shape[1], gene_exp.shape[-1]
        device = features.device

        # 调试用 shape 打印（需要时自行取消注释）
        # print("[SpatialTransformer.forward] input - gene_exp:", gene_exp.shape,
        #       "features:", features.shape,
        #       "coords:", coords.shape)
        
        pad_source = cosine_features if cosine_features is not None else features
        pad_mask = pad_source.sum(dim=-1) == 0  # [B, N_cells], True if the token is padding
        batch_idx = torch.arange(B, device=device).unsqueeze(-1).repeat(1, N_cells)[~pad_mask]
        # print("[SpatialTransformer.forward] pad_mask:", pad_mask.shape,
        #       "batch_idx:", batch_idx.shape)

        features = features[~pad_mask]  # [-1, 1024]
        coords = coords[~pad_mask]  # [-1, 3]
        gene_exp = gene_exp[~pad_mask]  # [-1, N_genes]
        if cosine_features is not None:
            cosine_features = cosine_features[~pad_mask]
        # print("[SpatialTransformer.forward] after remove padding - gene_exp:", gene_exp.shape,
        #       "features:", features.shape,
        #       "coords:", coords.shape)

        graph_features = cosine_features if cosine_features is not None else features
        nearest_indices, neighbor_masks = self._build_graph(
            coords, batch_idx, graph_features, min(self.n_neighbors, N_cells), exclude_self=True
        )
        # print("[SpatialTransformer.forward] nearest_indices:", nearest_indices.shape)

        # forward pass
        all_gene_exp = []
        for blk_idx, blk in enumerate(self.blks):
            # print(f"[SpatialTransformer.forward] block {blk_idx} input - gene_exp:", gene_exp.shape,
            #       "features:", features.shape)
            gene_exp, features = blk(
                gene_exp,
                features,
                coords,
                nearest_indices,
                neighbor_masks=neighbor_masks,
                cosine_features=cosine_features,
            )
            # print(f"[SpatialTransformer.forward] block {blk_idx} output - gene_exp:", gene_exp.shape,
            #       "features:", features.shape)
            all_gene_exp.append(gene_exp)
        gene_exp = torch.stack(all_gene_exp, dim=0).mean(dim=0)  # [N_tokens, N_genes]
        # print("[SpatialTransformer.forward] stacked & mean gene_exp:", gene_exp.shape)
        
        # average the gene expression among the neighbors
        gene_exp, _ = to_dense_batch(gene_exp, batch=batch_idx, fill_value=0, max_num_nodes=N_cells)  # [B, N_cells, N_genes]
        # print("[SpatialTransformer.forward] to_dense_batch gene_exp:", gene_exp.shape)
        if return_token_embs:
            token_embs, _ = to_dense_batch(features, batch=batch_idx, fill_value=0, max_num_nodes=N_cells)  # [B, N_cells, d_model]
            return gene_exp, token_embs
        return gene_exp
