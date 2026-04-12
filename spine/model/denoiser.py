import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .transformer import SpatialTransformer


class TimestepEmbedder(nn.Module):
    """Embed scalar flow timesteps into the model hidden space."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[..., None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class Denoiser(nn.Module):
    """SPINE flow denoiser with the final RNA-to-protein training path only."""

    def __init__(self, config) -> None:
        super().__init__()

        use_feature_mlp = getattr(config, "use_feature_mlp", False)
        feature_dim = int(config.feature_dim)
        hidden_dim = int(config.hidden_dim)
        output_dim = getattr(config, "n_proteins", None)
        if output_dim is None:
            output_dim = config.n_genes

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.loss_type = "mse"
        self.gene_recon_weight = float(getattr(config, "gene_recon_weight", 0.0))
        self.last_gene_recon_loss = None

        self.backbone = SpatialTransformer(
            ModelConfig(
                n_genes=output_dim,
                d_input=feature_dim,
                d_model=hidden_dim,
                d_edge_model=config.pairwise_hidden_dim,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout,
                n_neighbors=config.n_neighbors,
                act=config.activation,
                use_cosine_edge=getattr(config, "use_cosine_edge", False),
                use_cosine_graph=getattr(config, "use_cosine_graph", False),
                cosine_graph_extra_k=getattr(config, "cosine_graph_extra_k", 2),
                cosine_graph_mode=getattr(config, "cosine_graph_mode", "union"),
                cosine_graph_max_spatial_dist_quantile=getattr(
                    config, "cosine_graph_max_spatial_dist_quantile", 0.95
                ),
                cosine_edge_fixed_expr_weight=getattr(config, "cosine_edge_fixed_expr_weight", 0.1),
                log_view_gate=False,
                log_attn_diag=False,
                log_graph_diag=False,
            )
        )

        self.fourier_proj = TimestepEmbedder(hidden_dim)
        self.rna_transform = self._build_feature_transform(config, use_feature_mlp, feature_dim, hidden_dim)

        if self.gene_recon_weight > 0:
            self.gene_decoder = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, feature_dim),
            )
        else:
            self.gene_decoder = None

    @staticmethod
    def _build_feature_transform(config, use_feature_mlp, feature_dim, hidden_dim):
        if not use_feature_mlp:
            if feature_dim == hidden_dim:
                return nn.Identity()
            return nn.Linear(feature_dim, hidden_dim)

        mlp_intermediate_dim = int(getattr(config, "mlp_intermediate_dim", 1024))
        mlp_num_layers = int(getattr(config, "mlp_num_layers", 2))
        dropout = float(getattr(config, "dropout", 0.3))

        if mlp_num_layers == 1:
            return nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim, bias=True),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim, bias=True),
            )

        if mlp_num_layers == 2:
            return nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, mlp_intermediate_dim, bias=True),
                nn.LayerNorm(mlp_intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_intermediate_dim, hidden_dim, bias=True),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim, bias=True),
            )

        if mlp_num_layers == 3:
            dim1 = mlp_intermediate_dim
            dim2 = mlp_intermediate_dim // 2
            return nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, dim1, bias=True),
                nn.LayerNorm(dim1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim1, dim2, bias=True),
                nn.LayerNorm(dim2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim2, hidden_dim, bias=True),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim, bias=True),
            )

        raise ValueError(f"mlp_num_layers must be 1, 2 or 3, got {mlp_num_layers}")

    def inference(
        self,
        noisy_target,
        source_features,
        coords,
        t_steps,
        cosine_features=None,
        predict: bool = False,
        return_features: bool = False,
        return_token_embs: bool = False,
    ):
        del predict
        source_features_mapped = self.rna_transform(source_features)
        time_emb = self.fourier_proj(t_steps)[:, None].expand(
            noisy_target.shape[0], noisy_target.shape[1], -1
        )
        features = source_features_mapped + time_emb

        if cosine_features is None:
            cosine_features = source_features

        backbone_out = self.backbone(
            gene_exp=noisy_target,
            features=features,
            coords=coords,
            cosine_features=cosine_features,
            return_token_embs=return_token_embs,
        )
        if return_token_embs:
            prediction, token_embs = backbone_out
        else:
            prediction = backbone_out
            token_embs = None

        if return_features:
            return prediction, source_features_mapped, features, token_embs
        if return_token_embs:
            return prediction, token_embs
        return prediction

    def forward(self, noisy_target, source_features, coords, target, t_steps, cosine_features=None):
        need_token_embs = self.gene_decoder is not None
        prediction, _, _, token_embs = self.inference(
            noisy_target,
            source_features,
            coords,
            t_steps,
            cosine_features=cosine_features,
            return_features=True,
            return_token_embs=need_token_embs,
        )

        valid_mask = source_features.sum(-1) != 0
        loss = F.mse_loss(prediction[valid_mask], target[valid_mask])

        self.last_gene_recon_loss = None
        if self.gene_decoder is not None and self.gene_recon_weight > 0:
            if token_embs is None:
                raise RuntimeError("gene_recon_weight>0 but token_embs is None")
            gene_pred = self.gene_decoder(token_embs)
            gene_recon_loss = F.mse_loss(gene_pred[valid_mask], source_features[valid_mask])
            self.last_gene_recon_loss = gene_recon_loss.detach()
            loss = loss + self.gene_recon_weight * gene_recon_loss

        return prediction, loss

    def load_state_dict(self, state_dict, strict=True):
        """Backward compatibility for old checkpoints using `image_transform.*` keys."""
        has_legacy_keys = any(k.startswith("image_transform.") for k in state_dict.keys())
        if has_legacy_keys:
            remapped = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("image_transform."):
                    new_k = "rna_transform." + k[len("image_transform.") :]
                    remapped[new_k] = v
                else:
                    remapped[k] = v
            if hasattr(state_dict, "_metadata"):
                remapped._metadata = state_dict._metadata
            state_dict = remapped
        return super().load_state_dict(state_dict, strict=strict)
