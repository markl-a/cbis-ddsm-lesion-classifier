# -*- coding: utf-8 -*-
"""v2 實驗訓練器：可切換 backbone / 解析度 / 損失 / TTA / k-fold 集成。

與 baseline 完全相同的防洩漏切分與 case-level 評估協定，確保公平比較。
所有指標都在官方測試集上以 case-level ROC-AUC 為主。

範例：
    python train_v2.py --backbone efficientnet_v2_s --img-size 320 --tta
    python train_v2.py --backbone convnext_tiny --img-size 288 --folds 0 1 2 3 4 --tta
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from data import (build_manifest, make_patient_splits, CBISDataset,
                  get_transforms, manifest_fingerprint)
from metrics import aggregate_by_case, classification_metrics, evaluate_predictions, select_threshold
from model import get_device
from backbones import build_backbone, count_params
from losses import dml_bce_with_logits

HERE = Path(__file__).parent
DATA = HERE / "data"
SEED = 42


class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=1.5, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        return ((1 - p_t) ** self.gamma * bce).mean()


import os as _os
# 平行資料載入（0=單執行緒，最穩）。注意：多 fold 迴圈用 persistent_workers=True 會外洩 worker
# 造成 Windows 下 crash，故此處固定 persistent_workers=False。
NUM_WORKERS = int(_os.environ.get("NUM_WORKERS", "0"))


def make_loader(df, img_size, batch_size, train, seed=SEED):
    tf = get_transforms(train=train, img_size=img_size)
    ds = CBISDataset(df.reset_index(drop=True), transform=tf, return_index=True)
    kw = dict(batch_size=batch_size, num_workers=NUM_WORKERS, persistent_workers=False,
              prefetch_factor=4 if NUM_WORKERS > 0 else None, pin_memory=False)
    if train:
        g = torch.Generator().manual_seed(seed)
        return DataLoader(ds, shuffle=True, generator=g, **kw)
    return DataLoader(ds, shuffle=False, **kw)


@torch.no_grad()
def infer(model, loader, ref_df, device, tta=False):
    """回傳 (y_true, y_prob, case_ids, abnormality)。tta=True 時對水平翻轉平均。"""
    model.eval()
    case_col = ref_df["case_id"].to_numpy(); abn_col = ref_df["abnormality"].to_numpy()
    ys, ps, idxs = [], [], []
    for x, y, idx in loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze(1)
        if tta:
            prob_f = torch.sigmoid(model(torch.flip(x, dims=[3]))).squeeze(1)
            prob = (prob + prob_f) / 2
        ys.append(np.asarray(y)); ps.append(prob.detach().cpu().numpy()); idxs.append(np.asarray(idx))
    yt = np.concatenate(ys); yp = np.concatenate(ps); idx = np.concatenate(idxs)
    return yt, yp, case_col[idx], abn_col[idx]


def train_one_fold(train_df, val_df, cfg, device, log=print):
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    train_loader = make_loader(train_df, cfg["img_size"], cfg["batch_size"], True, cfg["seed"])
    val_loader = make_loader(val_df, cfg["img_size"], cfg["batch_size"], False)

    model = build_backbone(cfg["backbone"], pretrained=True).to(device)
    n_pos = int((train_df["label"] == 1).sum()); n_neg = int((train_df["label"] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    if cfg["loss"] == "focal":
        criterion = BCEFocalLoss(gamma=cfg["focal_gamma"], pos_weight=pos_weight)
    else:   # DML 友善 BCE（見 src/losses.py）,數值等價 BCEWithLogitsLoss、避開 CPU fallback
        criterion = lambda logits, targets: dml_bce_with_logits(logits, targets, pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg["epochs"], 1))

    best_auc, best_state, best = -1.0, copy.deepcopy(model.state_dict()), None
    no_improve = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train(); t0 = time.time(); running = torch.zeros((), device=device)
        for x, y, _ in train_loader:
            x = x.to(device); y = y.float().unsqueeze(1).to(device)
            optimizer.zero_grad(); loss = criterion(model(x), y)
            loss.backward(); optimizer.step(); running += loss.detach() * x.size(0)
        scheduler.step()
        tr_loss = (running / len(train_loader.dataset)).item()
        yt, yp, cids, _ = infer(model, val_loader, val_df, device, tta=cfg["tta"])
        ct, cp = aggregate_by_case(cids, yt, yp)
        thr = select_threshold(ct, cp)
        cauc = classification_metrics(ct, cp, thr)["roc_auc"]
        log(f"  [{cfg['backbone']}@{cfg['img_size']} f{cfg['fold']}] ep{epoch:2d}/{cfg['epochs']} "
            f"{time.time()-t0:5.1f}s loss {tr_loss:.4f} val_case_auc {cauc:.4f}")
        if np.isfinite(cauc) and cauc > best_auc:
            best_auc = cauc
            best_state = copy.deepcopy(model.state_dict())
            best = {"epoch": epoch, "case_auc": float(cauc), "threshold": float(thr)}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                log(f"  早停 @ep{epoch}"); break
    model.load_state_dict(best_state)
    return model, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="efficientnet_v2_s")
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--loss", choices=["bce", "focal"], default="bce")
    ap.add_argument("--focal-gamma", type=float, default=1.5)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args()

    t_start = time.time()
    man = build_manifest(DATA, require_files=True, verbose=False)
    device = get_device(verbose=True)
    print(f"device: {device} | backbone {args.backbone} img {args.img_size} tta {args.tta} "
          f"loss {args.loss} folds {args.folds}")

    # 固定測試集（fold 不影響 test），逐 fold 訓練並集成 test case 機率
    fold_test_probs = []   # list of (case_true, case_prob, case_abn) aligned by case order
    pooled_val_true, pooled_val_prob = [], []
    per_fold = []
    ref_test = None
    for fold in args.folds:
        train_df, val_df, test_df, split_audit = make_patient_splits(man, n_splits=5, fold=fold, seed=SEED)
        cfg = dict(backbone=args.backbone, img_size=args.img_size, epochs=args.epochs,
                   batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                   loss=args.loss, focal_gamma=args.focal_gamma, patience=args.patience,
                   tta=args.tta, fold=fold, seed=SEED)
        print(f"\n=== fold {fold} ===")
        model, best = train_one_fold(train_df, val_df, cfg, device)

        # pooled OOF val for threshold
        yt, yp, cids, _ = infer(model, make_loader(val_df, args.img_size, args.batch_size, False),
                                val_df, device, tta=args.tta)
        cvt, cvp = aggregate_by_case(cids, yt, yp)
        pooled_val_true.append(cvt); pooled_val_prob.append(cvp)

        # test case probs
        tyt, typ, tcids, tabn = infer(model, make_loader(test_df, args.img_size, args.batch_size, False),
                                      test_df, device, tta=args.tta)
        ct, cp, ca = aggregate_by_case(tcids, tyt, typ, abnormality=tabn)
        fold_test_probs.append(cp)
        ref_test = (ct, ca)   # case_true/abn identical across folds (same test set/order)
        per_fold.append({"fold": fold, "val_best_case_auc": best["case_auc"],
                         "test_case_auc": classification_metrics(ct, cp, 0.5)["roc_auc"]})
        print(f"  fold {fold}: val_case_auc {best['case_auc']:.4f} | test_case_auc(this fold) "
              f"{per_fold[-1]['test_case_auc']:.4f}")

    # 集成：對 test case 機率平均
    ens_prob = np.mean(np.stack(fold_test_probs, 0), axis=0)
    ct, ca = ref_test
    # 閾值：pooled OOF val
    vt = np.concatenate(pooled_val_true); vp = np.concatenate(pooled_val_prob)
    thr = select_threshold(vt, vp)
    ens = classification_metrics(ct, ens_prob, thr)
    # subgroup
    subs = {}
    ca = np.asarray(ca)
    for name in ("mass", "calc"):
        mmask = ca == name
        if mmask.sum():
            subs[name] = classification_metrics(ct[mmask], ens_prob[mmask], thr)["roc_auc"]

    print("\n=== 集成結果（官方測試集，case-level）===")
    print(f"backbone {args.backbone} img {args.img_size} tta {args.tta} loss {args.loss} folds {args.folds}")
    print(f"  ENSEMBLE test case-AUC: {ens['roc_auc']:.4f} | 敏感度 {ens['sensitivity']:.4f} "
          f"| 特異度 {ens['specificity']:.4f} | 閾值 {thr:.3f}")
    print(f"  subgroup case-AUC: {{'mass': {subs.get('mass', float('nan')):.4f}, "
          f"'calc': {subs.get('calc', float('nan')):.4f}}}")
    print(f"  baseline 對照: test case-AUC 0.748")
    print(f"  per-fold: {per_fold}")
    print(f"  用時 {(time.time()-t_start)/60:.1f} 分")

    out = {"config": vars(args), "ensemble_test": ens, "subgroups": subs,
           "per_fold": per_fold, "threshold": thr,
           "elapsed_min": round((time.time()-t_start)/60, 1)}
    (HERE / "artifacts").mkdir(exist_ok=True)
    (HERE / "artifacts" / f"v2_{args.tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("RESULT_JSON:", json.dumps({"backbone": args.backbone, "img": args.img_size,
                                      "tta": args.tta, "loss": args.loss, "folds": args.folds,
                                      "ensemble_test_case_auc": round(ens["roc_auc"], 4)}))
    print("V2_DONE")


if __name__ == "__main__":
    main()
