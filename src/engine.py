# -*- coding: utf-8 -*-
"""訓練引擎：AdamW + cosine 退火，依驗證 case-level ROC-AUC 選最佳 epoch 並早停。

對齊 data.py 的 API：split 為 DataFrame；CBISDataset 以 return_index=True 回傳
(image, label, index)，index 用來對回該 split DataFrame 取 case_id / abnormality。
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import CBISDataset, get_transforms
from metrics import aggregate_by_case, evaluate_predictions, select_threshold
from model import build_model
from losses import dml_bce_with_logits


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 24
    lr: float = 3e-4
    weight_decay: float = 1e-4
    img_size: int = 224
    num_workers: int = 0
    early_stop_patience: int = 5
    seed: int = 42
    smoke: bool = False          # True：極小子集 + 1 epoch（不可作為交付 best.pt）


def compute_pos_weight(train_df) -> float:
    """pos_weight = #neg / #pos，只用訓練 fold。"""
    n_pos = int((train_df["label"] == 1).sum())
    n_neg = int((train_df["label"] == 0).sum())
    return n_neg / max(n_pos, 1)


def _make_loader(df, cfg: TrainConfig, train: bool):
    tf = get_transforms(train=train, img_size=cfg.img_size)
    ds = CBISDataset(df, transform=tf, return_index=True)
    if train:
        g = torch.Generator().manual_seed(cfg.seed)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, generator=g)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                      num_workers=cfg.num_workers)


@torch.no_grad()
def infer(model, loader, ref_df, device):
    """回傳 (y_true, y_prob, case_ids, abnormality)。index 對回 ref_df（已 reset）。"""
    model.eval()
    case_col = ref_df["case_id"].to_numpy()
    abn_col = ref_df["abnormality"].to_numpy()
    ys, ps, idxs = [], [], []
    for x, y, idx in loader:
        logit = model(x.to(device))
        prob = torch.sigmoid(logit).squeeze(1)
        ys.append(np.asarray(y)); ps.append(prob.detach().cpu().numpy())
        idxs.append(np.asarray(idx))
    y_true = np.concatenate(ys); y_prob = np.concatenate(ps); idx = np.concatenate(idxs)
    return y_true, y_prob, case_col[idx], abn_col[idx]


def train(train_df, val_df, cfg: TrainConfig, device, log=print):
    """回傳 (model, history, best)。best 內含最佳 epoch、驗證 case-AUC 與凍結閾值。"""
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    if cfg.smoke:
        train_df = train_df.groupby("label", group_keys=False).head(24).reset_index(drop=True)
        val_df = val_df.groupby("label", group_keys=False).head(24).reset_index(drop=True)
    else:
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    train_loader = _make_loader(train_df, cfg, train=True)
    val_loader = _make_loader(val_df, cfg, train=False)

    model = build_model(pretrained=True).to(device)
    pos_weight = torch.tensor([compute_pos_weight(train_df)], device=device)  # DML 友善 BCE（見 losses.py）
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    epochs = 1 if cfg.smoke else cfg.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    history = {"train_loss": [], "val_case_auc": [], "val_image_auc": [], "lr": []}
    best_auc, best_state, best = -1.0, copy.deepcopy(model.state_dict()), None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train(); t0 = time.time(); running = torch.zeros((), device=device)
        for x, y, _ in train_loader:
            x = x.to(device); y = y.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            loss = dml_bce_with_logits(model(x), y, pos_weight)
            loss.backward(); optimizer.step()
            running += loss.detach() * x.size(0)
        scheduler.step()
        train_loss = (running / len(train_loader.dataset)).item()

        yt, yp, cids, abn = infer(model, val_loader, val_df, device)
        ct, cp = aggregate_by_case(cids, yt, yp)
        thr = select_threshold(ct, cp)                     # 閾值只在驗證 case-level 上選
        rep = evaluate_predictions(yt, yp, cids, threshold=thr, abnormality=abn)
        case_auc = rep.case["roc_auc"]

        history["train_loss"].append(train_loss)
        history["val_case_auc"].append(case_auc)
        history["val_image_auc"].append(rep.image["roc_auc"])
        history["lr"].append(optimizer.param_groups[0]["lr"])
        log(f"Epoch {epoch:2d}/{epochs} | {time.time()-t0:5.1f}s | loss {train_loss:.4f} | "
            f"val_case_auc {case_auc:.4f} val_img_auc {rep.image['roc_auc']:.4f} thr {thr:.3f}")

        if np.isfinite(case_auc) and case_auc > best_auc:
            best_auc = case_auc
            best_state = copy.deepcopy(model.state_dict())
            best = {"epoch": epoch, "case_auc": float(case_auc),
                    "threshold": float(thr), "report": rep}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                log(f"  早停：連續 {no_improve} epoch 無提升")
                break

    model.load_state_dict(best_state)
    return model, history, best


@torch.no_grad()
def evaluate_split(model, df, cfg: TrainConfig, device, threshold):
    """在指定 split（含 abnormality）上算 image + case + subgroup 指標。"""
    df = df.reset_index(drop=True)
    loader = _make_loader(df, cfg, train=False)
    yt, yp, cids, abn = infer(model, loader, df, device)
    return evaluate_predictions(yt, yp, cids, threshold=threshold, abnormality=abn)
