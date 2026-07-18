# -*- coding: utf-8 -*-
"""ResNet50 + IVAF（Inter-View Attention Fusion）雙視角模型。

把同一病灶的 CC / MLO 兩個視角一起判斷:共享 ResNet50 encoder 各抽 2048-d 特徵,
用 multi-head self-attention 讓兩視角互相注意（inter-view attention）後融合,直接輸出
單一 case-level logit。單視角的 case 以同一張影像餵兩次（退化為單視角）。

與 V2-S 相同的防洩漏 patient-disjoint split;IVAF 天生就是 case-level 預測（不需再平均 CC/MLO）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

from data import get_transforms


def build_paired_cases(image_manifest: pd.DataFrame) -> pd.DataFrame:
    """把 image-level manifest 收成 case-level 配對表。

    每個 case 取一張 CC、一張 MLO（缺的用另一張補）。回傳欄位:
    case_id, patient_id, abnormality, label, cc_path, mlo_path, has_pair
    """
    rows = []
    for case_id, g in image_manifest.groupby("case_id"):
        views = {}
        for _, r in g.iterrows():
            views.setdefault(str(r["image_view"]).upper(), r["filepath"])
        cc = views.get("CC"); mlo = views.get("MLO")
        has_pair = cc is not None and mlo is not None
        if cc is None:
            cc = mlo
        if mlo is None:
            mlo = cc
        r0 = g.iloc[0]
        rows.append({"case_id": case_id, "patient_id": r0["patient_id"],
                     "abnormality": r0["abnormality"], "label": int(r0["label"]),
                     "cc_path": cc, "mlo_path": mlo, "has_pair": has_pair})
    return pd.DataFrame(rows).reset_index(drop=True)


class PairedDataset(Dataset):
    """回傳 (img_cc, img_mlo, label, index)。"""

    def __init__(self, paired_df: pd.DataFrame, transform=None):
        self.df = paired_df.reset_index(drop=True)
        self.cc = self.df["cc_path"].tolist()
        self.mlo = self.df["mlo_path"].tolist()
        self.labels = self.df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _load(self, path):
        img = Image.open(path).convert("RGB")
        return self.transform(img) if self.transform else img

    def __getitem__(self, i):
        return self._load(self.cc[i]), self._load(self.mlo[i]), self.labels[i], i


class IVAFModel(nn.Module):
    """共享 ResNet50 encoder + inter-view multi-head attention 融合 + 單 logit 頭。"""

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True,
                 num_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        from torchvision import models
        if backbone == "resnet50":
            w = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            enc = models.resnet50(weights=w); enc.fc = nn.Identity(); embed_dim = 2048
        elif backbone == "efficientnet_v2_s":
            w = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            enc = models.efficientnet_v2_s(weights=w); enc.classifier = nn.Identity(); embed_dim = 1280
        else:
            raise ValueError(f"不支援的 IVAF backbone: {backbone}")
        self.encoder = enc
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, 1))

    def forward(self, cc, mlo):
        f_cc = self.encoder(cc)              # [B, 2048]
        f_mlo = self.encoder(mlo)            # [B, 2048]
        tokens = torch.stack([f_cc, f_mlo], dim=1)   # [B, 2, 2048]
        attn_out, _ = self.attn(tokens, tokens, tokens)   # inter-view attention
        fused = self.norm(tokens + attn_out).mean(dim=1)  # 殘差 + 融合兩視角
        return self.head(fused)              # [B, 1]


def make_paired_split(image_manifest, split_dfs):
    """給 (train_df, val_df, test_df) 的 image-level 切分,回傳各自的 case-level 配對表。"""
    return tuple(build_paired_cases(df) for df in split_dfs)
