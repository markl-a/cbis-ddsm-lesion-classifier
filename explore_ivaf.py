# -*- coding: utf-8 -*-
"""IVAF 模型探索:在同一防洩漏 split 上訓練 ResNet50+IVAF,與 V2-S 誠實比較。

只有 IVAF 的 case-level AUC 真的贏過 V2-S(5-fold mean 0.767 / 單一 0.794)才值得採用。
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from data import build_manifest, make_patient_splits, get_transforms
from metrics import classification_metrics, select_threshold
from model import get_device
from ivaf import build_paired_cases, PairedDataset, IVAFModel
from losses import dml_bce_with_logits

DATA = HERE / "data"
ART = HERE / "artifacts"; ART.mkdir(exist_ok=True)
SEED = int(os.environ.get("SEED", "42"))
FOLD = int(os.environ.get("FOLD", "0"))
IMG = int(os.environ.get("IMG_SIZE", "288"))
BATCH = int(os.environ.get("BATCH", "8"))       # pairs per batch（=2x 影像）
EPOCHS = int(os.environ.get("EPOCHS", "16"))
ENC_LR = float(os.environ.get("ENC_LR", "5e-5"))
FUSE_LR = float(os.environ.get("FUSE_LR", "5e-4"))
WD = float(os.environ.get("WD", "1e-4"))
PATIENCE = int(os.environ.get("PATIENCE", "4"))
TTA = os.environ.get("TTA", "1") == "1"
SMOKE = os.environ.get("SMOKE", "0") == "1"
TAG = os.environ.get("TAG", "ivaf")
BACKBONE = os.environ.get("BACKBONE", "resnet50")

import random
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))   # Windows 下 workers spawn 慢,預設 0


def loader(df, train):
    ds = PairedDataset(df, transform=get_transforms(train=train, img_size=IMG))
    g = torch.Generator().manual_seed(SEED)
    return DataLoader(ds, batch_size=BATCH, shuffle=train, generator=g if train else None,
                      num_workers=NUM_WORKERS, persistent_workers=False,
                      prefetch_factor=4 if NUM_WORKERS > 0 else None)


@torch.no_grad()
def infer(model, ld, ref_df, device, tta=TTA):
    model.eval()
    labels = ref_df["label"].to_numpy()
    ys, ps, idxs = [], [], []
    for cc, mlo, y, idx in ld:
        cc = cc.to(device); mlo = mlo.to(device)
        prob = torch.sigmoid(model(cc, mlo)).squeeze(1)
        if tta:
            prob = (prob + torch.sigmoid(model(torch.flip(cc, [3]), torch.flip(mlo, [3]))).squeeze(1)) / 2
        ys.append(np.asarray(y)); ps.append(prob.detach().cpu().numpy()); idxs.append(np.asarray(idx))
    return np.concatenate(ys), np.concatenate(ps)


def main():
    t0 = time.time()
    man = build_manifest(DATA, require_files=True, verbose=False)
    tr_i, va_i, te_i, _ = make_patient_splits(man, n_splits=5, fold=FOLD, seed=SEED)
    tr, va, te = build_paired_cases(tr_i), build_paired_cases(va_i), build_paired_cases(te_i)
    if SMOKE:
        tr = tr.groupby("label", group_keys=False).head(16).reset_index(drop=True)
        va = va.groupby("label", group_keys=False).head(16).reset_index(drop=True)
    print(f"paired cases — train {len(tr)} (pair {tr.has_pair.mean()*100:.0f}%) | "
          f"val {len(va)} | test {len(te)} (pair {te.has_pair.mean()*100:.0f}%)")

    device = get_device(verbose=True)
    print(f"device {device} | IVAF {BACKBONE} img {IMG} batch {BATCH} enc_lr {ENC_LR} fuse_lr {FUSE_LR} tta {TTA}")

    model = IVAFModel(backbone=BACKBONE, pretrained=True).to(device)
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder")]
    fuse = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
    opt = torch.optim.AdamW([{"params": enc, "lr": ENC_LR}, {"params": fuse, "lr": FUSE_LR}], weight_decay=WD)
    n_pos = int((tr["label"] == 1).sum()); n_neg = int((tr["label"] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    epochs = 1 if SMOKE else EPOCHS
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    train_loader, val_loader = loader(tr, True), loader(va, False)
    best_auc, best_state, best = -1.0, copy.deepcopy(model.state_dict()), None
    no_improve = 0
    for ep in range(1, epochs + 1):
        model.train(); ts = time.time(); run = torch.zeros((), device=device)
        for cc, mlo, y, _ in train_loader:
            cc, mlo = cc.to(device), mlo.to(device); y = y.float().unsqueeze(1).to(device)
            opt.zero_grad(); loss = dml_bce_with_logits(model(cc, mlo), y, pos_weight); loss.backward(); opt.step()
            run += loss.detach() * cc.size(0)          # 累加在 GPU 上，避免每步 .item() 同步
        sched.step()
        tl = (run / len(train_loader.dataset)).item()  # 一個 epoch 只同步一次
        yv, pv = infer(model, val_loader, va, device, tta=TTA)
        thr = select_threshold(yv, pv)
        auc = classification_metrics(yv, pv, thr)["roc_auc"]
        print(f"Epoch {ep:2d}/{epochs} | {time.time()-ts:5.1f}s | loss {tl:.4f} | val case-AUC {auc:.4f}")
        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc; best_state = copy.deepcopy(model.state_dict())
            best = {"epoch": ep, "val_auc": float(auc), "threshold": float(thr)}; no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  早停 @ep{ep}"); break
    model.load_state_dict(best_state)

    yt, pt = infer(model, loader(te, False), te, device, tta=TTA)
    m = classification_metrics(yt, pt, best["threshold"])
    print("\n=== IVAF 測試集（case-level）===")
    print(f"  IVAF test case-AUC: {m['roc_auc']:.4f} | 敏感度 {m['sensitivity']:.4f} | 特異度 {m['specificity']:.4f}")
    print(f"  對照 V2-S: 5-fold mean 0.767 / 單一 fold 0.794 / 集成 0.791")
    verdict = "勝出" if m["roc_auc"] > 0.794 else ("持平" if m["roc_auc"] > 0.767 else "未勝出")
    print(f"  判定:IVAF {m['roc_auc']:.4f} vs V2-S → 【{verdict}】")
    out = {"model": f"ivaf_{BACKBONE}", "fold": FOLD, "img": IMG, "tta": TTA,
           "val_auc": best["val_auc"], "best_epoch": best["epoch"],
           "test": m, "vs_v2s": {"mean5": 0.767, "single": 0.794, "ensemble": 0.791},
           "elapsed_min": round((time.time()-t0)/60, 1)}
    (ART / f"{TAG}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    # 存權重供之後 5-fold / 集成（只有勝出才會納入交付）
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, HERE / f"_{TAG}.pt")
    print(f"RESULT_JSON: {json.dumps({'ivaf_test_auc': round(m['roc_auc'],4), 'verdict': verdict})}")
    print(f"用時 {(time.time()-t0)/60:.1f} 分 | IVAF_DONE")


if __name__ == "__main__":
    main()
