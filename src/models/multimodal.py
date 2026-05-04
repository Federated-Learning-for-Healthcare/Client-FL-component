"""
multimodal.py — MultiModal model for federated healthcare learning.

Architecture per modality:
  ECGEncoder  : FastKAN (flat CSV mode) or 1D-ResNet + FastKAN (raw waveform mode)
  MRIEncoder  : 2D-ResNet + FastKAN projector
  EHREncoder  : Full FastKAN end-to-end (ideal for compact tabular data)

Fusion:
  Learnable null tokens replace absent-modality embeddings.
  Masked attention (null slots → -inf before softmax) produces a stable
  fused embedding regardless of which modalities are present.
  FastKAN classification head over the fused embedding.

Parameter helpers:
  get_multimodal_parameters / set_multimodal_parameters
  Only the components a client owns (per modal_mask) are sent to/from the server.
  Shared components (modal_attn, fusion_head) are always included.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models.kan import FastKAN

logger = logging.getLogger(__name__)

MODALITIES = ["ecg", "mri", "ehr"]


# ── Modality config dataclasses ───────────────────────────────────────────────

@dataclass
class ECGModalityConfig:
    n_features: int  = 187    # flat feature count for CSV mode (MIT-BIH = 187)
    n_leads:    int  = 1      # number of ECG leads for waveform mode
    mode:       str  = "flat" # "flat" (CSV features) | "waveform" (raw 1D signal)
    embed_dim:  int  = 128


@dataclass
class MRIModalityConfig:
    in_channels: int = 1
    img_size:    int = 64
    embed_dim:   int = 128


@dataclass
class EHRModalityConfig:
    n_features: int = 13
    embed_dim:  int = 128


# ── Residual blocks ───────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """1D residual block for raw ECG waveform backbone."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.skip(x))


class ResBlock2D(nn.Module):
    """2D residual block for MRI image backbone."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.skip(x))


# ── Modality encoders ─────────────────────────────────────────────────────────

class ECGEncoder(nn.Module):
    """
    ECG encoder with two modes:
      flat     — FastKAN over flat feature vector (e.g. MIT-BIH CSV, 187-d input).
                 No CNN needed; the KAN learns non-linear beat representations directly.
      waveform — 1D ResNet backbone + FastKAN projector for raw (B, n_leads, timesteps).
                 CNN captures local temporal patterns; KAN projects to embed_dim.
    """

    def __init__(self, cfg: ECGModalityConfig):
        super().__init__()
        self.cfg  = cfg
        self.mode = cfg.mode

        if cfg.mode == "flat":
            self.encoder = FastKAN([cfg.n_features, 256, cfg.embed_dim])
        else:
            stem_k = 7
            self.stem = nn.Sequential(
                nn.Conv1d(cfg.n_leads, 32, kernel_size=stem_k,
                          padding=stem_k // 2, bias=False),
                nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            )
            self.backbone = nn.Sequential(
                ResBlock1D(32,  64,  stride=2), ResBlock1D(64,  64),
                ResBlock1D(64,  128, stride=2), ResBlock1D(128, 128),
                ResBlock1D(128, 256, stride=2), ResBlock1D(256, 256),
            )
            self.gap       = nn.AdaptiveAvgPool1d(1)
            self.projector = FastKAN([256, cfg.embed_dim])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "flat":
            return self.encoder(x)
        feat = self.gap(self.backbone(self.stem(x))).squeeze(-1)
        return self.projector(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class MRIEncoder(nn.Module):
    """
    2D ResNet backbone + FastKAN projector for MRI images (B, C, H, W).
    Pure KAN on raw 2D images is infeasible (flattened dim too large);
    CNN first reduces spatial dims to a 256-d global average pooled vector.
    """

    def __init__(self, cfg: MRIModalityConfig):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.backbone = nn.Sequential(
            ResBlock2D(32,  64,  stride=2), ResBlock2D(64,  64),
            ResBlock2D(64,  128, stride=2), ResBlock2D(128, 128),
            ResBlock2D(128, 256, stride=2), ResBlock2D(256, 256),
        )
        self.gap       = nn.AdaptiveAvgPool2d(1)
        self.projector = FastKAN([256, cfg.embed_dim])
        self.cfg       = cfg

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.gap(self.backbone(self.stem(x))).squeeze(-1).squeeze(-1)
        return self.projector(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class EHREncoder(nn.Module):
    """
    Full FastKAN end-to-end for tabular EHR (B, n_features).
    KANs outperform MLPs on compact tabular data with clinical non-linearities.
    """

    def __init__(self, cfg: EHRModalityConfig):
        super().__init__()
        self.encoder = FastKAN([cfg.n_features, 256, cfg.embed_dim])
        self.cfg      = cfg

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


# ── MultiModal model ──────────────────────────────────────────────────────────

class MultiModalModel(nn.Module):
    """
    Multimodal fusion model for federated healthcare learning.

    Args:
        ecg_cfg    : ECGModalityConfig or None — instantiates ECGEncoder if provided
        mri_cfg    : MRIModalityConfig or None — instantiates MRIEncoder if provided
        ehr_cfg    : EHRModalityConfig or None — instantiates EHREncoder if provided
        n_classes  : number of output classes (shared across all modalities)
        embed_dim  : shared embedding dimension all encoders project into
        modal_mask : [has_ecg, has_mri, has_ehr] — which modalities this client owns.
                     Defaults to [1,0,0] / [0,0,1] etc. based on which cfgs are given.

    Forward:
        batch        — dict with keys "ecg", "mri", "ehr" (only owned keys needed)
        modal_mask   — override instance mask (optional)
        modal_drop_p     — probability of randomly dropping each modality during training
                           (0.3 recommended for multi-modal clients, 0.0 for uni-modal)
        fusion_head_type — "kan" (default) or "mlp".
                           KAN learns non-linear classification boundaries explicitly.
                           MLP is faster and simpler; use when overfitting is a concern.
    """

    def __init__(
        self,
        ecg_cfg:          Optional[ECGModalityConfig],
        mri_cfg:          Optional[MRIModalityConfig],
        ehr_cfg:          Optional[EHRModalityConfig],
        n_classes:        int,
        embed_dim:        int       = 128,
        modal_mask:       List[int] = None,
        fusion_head_type: str       = "kan",
    ):
        super().__init__()

        self.ecg_encoder = ECGEncoder(ecg_cfg) if ecg_cfg else None
        self.mri_encoder = MRIEncoder(mri_cfg) if mri_cfg else None
        self.ehr_encoder = EHREncoder(ehr_cfg) if ehr_cfg else None

        # Null tokens initialised near zero → low attention weight early on.
        # They learn to signal "absent modality" without pulling probability mass.
        self.null_ecg = nn.Parameter(torch.zeros(1, embed_dim))
        self.null_mri = nn.Parameter(torch.zeros(1, embed_dim))
        self.null_ehr = nn.Parameter(torch.zeros(1, embed_dim))

        # Shared across all clients — aggregated from the full client pool
        self.modal_attn = nn.Linear(embed_dim, 1, bias=False)

        fht = fusion_head_type.lower()
        if fht == "kan":
            self.fusion_head = FastKAN([embed_dim, 64, n_classes])
        elif fht == "mlp":
            self.fusion_head = nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, n_classes),
            )
        else:
            raise ValueError(
                f"fusion_head_type must be 'kan' or 'mlp', got '{fusion_head_type}'"
            )
        self.fusion_head_type = fht

        self.embed_dim  = embed_dim
        self.n_classes  = n_classes
        self.modal_mask = modal_mask or [
            int(ecg_cfg is not None),
            int(mri_cfg is not None),
            int(ehr_cfg is not None),
        ]

        logger.info(
            "MultiModalModel — mask=%s  n_classes=%d  embed_dim=%d  "
            "fusion=%s  encoders=[ecg=%s, mri=%s, ehr=%s]",
            self.modal_mask, n_classes, embed_dim, self.fusion_head_type,
            self.ecg_encoder is not None,
            self.mri_encoder is not None,
            self.ehr_encoder is not None,
        )

    def _get_embedding(
        self,
        encoder:      Optional[nn.Module],
        null_token:   nn.Parameter,
        data_key:     str,
        batch:        dict,
        mask_val:     int,
        modal_drop_p: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (embedding, is_real_flag) for one modality."""
        B = next(v for v in batch.values() if isinstance(v, torch.Tensor)).size(0)
        use_real = (
            mask_val == 1
            and encoder is not None
            and data_key in batch
            and not (self.training and torch.rand(1).item() < modal_drop_p)
        )
        if use_real:
            emb     = encoder.encode(batch[data_key])
            is_real = torch.ones(B, 1, device=emb.device)
        else:
            dev     = next(self.parameters()).device
            emb     = null_token.expand(B, -1).to(dev)
            is_real = torch.zeros(B, 1, device=dev)
        return emb, is_real

    def forward(
        self,
        batch:        dict,
        modal_mask:   List[int] = None,
        modal_drop_p: float     = 0.0,
    ) -> torch.Tensor:
        if modal_mask is None:
            modal_mask = self.modal_mask

        ecg_emb, ecg_real = self._get_embedding(
            self.ecg_encoder, self.null_ecg, "ecg", batch, modal_mask[0], modal_drop_p)
        mri_emb, mri_real = self._get_embedding(
            self.mri_encoder, self.null_mri, "mri", batch, modal_mask[1], modal_drop_p)
        ehr_emb, ehr_real = self._get_embedding(
            self.ehr_encoder, self.null_ehr, "ehr", batch, modal_mask[2], modal_drop_p)

        embs  = torch.stack([ecg_emb, mri_emb, ehr_emb], dim=1)   # (B, 3, D)
        reals = torch.stack([ecg_real, mri_real, ehr_real], dim=1) # (B, 3, 1)

        # Mask null slots to -inf before softmax so they contribute zero weight
        attn_logits  = self.modal_attn(embs)                       # (B, 3, 1)
        attn_logits  = attn_logits.masked_fill(reals == 0, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1)

        fused = (attn_weights * embs).sum(dim=1)                   # (B, D)
        return self.fusion_head(fused)                             # (B, n_classes)


# ── Modality-aware parameter helpers ─────────────────────────────────────────

def _encoder_for(model: MultiModalModel, modality: str) -> Optional[nn.Module]:
    return {
        "ecg": model.ecg_encoder,
        "mri": model.mri_encoder,
        "ehr": model.ehr_encoder,
    }.get(modality)


def _null_token_for(model: MultiModalModel, modality: str) -> nn.Parameter:
    return {
        "ecg": model.null_ecg,
        "mri": model.null_mri,
        "ehr": model.null_ehr,
    }[modality]


def get_multimodal_parameters(
    model:      MultiModalModel,
    modal_mask: List[int],
) -> List[np.ndarray]:
    """
    Return only owned-modality parameters as a flat list of numpy arrays.

    Order: for each owned modality → encoder params → null token
           then modal_attn → fusion_head (always included, shared).
    """
    params = []
    for mod, owned in zip(MODALITIES, modal_mask):
        if owned:
            enc = _encoder_for(model, mod)
            if enc is not None:
                params.extend(p.cpu().detach().numpy() for p in enc.parameters())
            params.append(_null_token_for(model, mod).cpu().detach().numpy())
    params.extend(p.cpu().detach().numpy() for p in model.modal_attn.parameters())
    params.extend(p.cpu().detach().numpy() for p in model.fusion_head.parameters())
    return params


def set_multimodal_parameters(
    model:      MultiModalModel,
    modal_mask: List[int],
    params:     List[np.ndarray],
) -> None:
    """
    Load server-broadcast parameters into owned-modality components only.
    Unowned encoder weights are untouched — they stay at their server-global value.
    """
    idx = 0

    def _load_module(module: nn.Module) -> None:
        nonlocal idx
        sd     = module.state_dict()
        new_sd = OrderedDict()
        for k, v in sd.items():
            new_sd[k] = torch.tensor(params[idx], dtype=v.dtype)
            idx += 1
        module.load_state_dict(new_sd, strict=True)

    for mod, owned in zip(MODALITIES, modal_mask):
        if owned:
            enc = _encoder_for(model, mod)
            if enc is not None:
                _load_module(enc)
            nt = _null_token_for(model, mod)
            nt.data.copy_(torch.tensor(params[idx], dtype=nt.dtype))
            idx += 1

    _load_module(model.modal_attn)
    _load_module(model.fusion_head)
