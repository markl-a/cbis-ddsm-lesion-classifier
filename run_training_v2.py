# -*- coding: utf-8 -*-
"""交付用 v2 訓練：EfficientNetV2-S @288 + TTA（單模型），存 best_v2.pt。

與 baseline 完全相同的防洩漏切分與 case-level 評估協定；差別只有更強 backbone、
更高解析度、推論時水平翻轉 TTA。閾值只在驗證集（含 TTA）選定後凍結。
"""
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from data import build_manifest, make_patient_splits, CBISDataset, get_transforms, manifest_fingerprint
from metrics import aggregate_by_case, classification_metrics, evaluate_predictions, select_threshold
from model import get_device
from backbones import build_backbone, count_params
from checkpoint import save_checkpoint, load_for_inference, predict_image

DATA = HERE / "data"
OUT = Path(os.environ.get("OUT_PATH", HERE / "best_v2.pt"))
ART = HERE / "artifacts"; ART.mkdir(exist_ok=True)
TAG = os.environ.get("TAG", "v2")

BACKBONE = os.environ.get("BACKBONE", "efficientnet_v2_s")
IMG_SIZE = int(os.environ.get("IMG_SIZE", "288"))
BATCH = int(os.environ.get("BATCH", "12"))
EPOCHS = int(os.environ.get("EPOCHS", "16"))
LR = float(os.environ.get("LR", "3e-4"))
WD = 1e-4
PATIENCE = int(os.environ.get("PATIENCE", "4"))
SCHED = os.environ.get("SCHED", "cosine")     # cosine | warmrestart
RESTART_T0 = int(os.environ.get("RESTART_T0", "8"))
SEED = int(os.environ.get("SEED", "42"))
FOLD = int(os.environ.get("FOLD", "0"))
TTA = os.environ.get("TTA", "1") == "1"


def make_loader(df, train):
    tf = get_transforms(train=train, img_size=IMG_SIZE)
    ds = CBISDataset(df.reset_index(drop=True), transform=tf, return_index=True)
    if train:
        g = torch.Generator().manual_seed(SEED)
        return DataLoader(ds, batch_size=BATCH, shuffle=True, generator=g)
    return DataLoader(ds, batch_size=BATCH, shuffle=False)


@torch.no_grad()
def infer(model, loader, ref_df, device, tta=TTA):
    model.eval()
    case_col = ref_df["case_id"].to_numpy(); abn_col = ref_df["abnormality"].to_numpy()
    ys, ps, idxs = [], [], []
    for x, y, idx in loader:
        x = x.to(device)
        prob = torch.sigmoid(model(x)).squeeze(1)
        if tta:
            prob = (prob + torch.sigmoid(model(torch.flip(x, dims=[3]))).squeeze(1)) / 2
        ys.append(np.asarray(y)); ps.append(prob.detach().cpu().numpy()); idxs.append(np.asarray(idx))
    yt = np.concatenate(ys); yp = np.concatenate(ps); idx = np.concatenate(idxs)
    return yt, yp, case_col[idx], abn_col[idx]


def main():
    t0 = time.time()
    print("=== 1. manifest + 防洩漏切分 ===")
    man = build_manifest(DATA, require_files=True, verbose=False)
    train_df, val_df, test_df, split_audit = make_patient_splits(man, n_splits=5, fold=FOLD, seed=SEED)
    print("rows:", split_audit["rows"], "| 清除跨集病患:", len(split_audit["purged_train_patients"]))

    device = get_device(verbose=True)
    print(f"device: {device} | backbone {BACKBONE} img {IMG_SIZE} tta {TTA}")

    print("\n=== 2. 訓練 ===")
    import random; random.seed(SEED)
    torch.manual_seed(SEED); np.random.seed(SEED)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    train_loader = make_loader(train_df, True)
    val_loader = make_loader(val_df, False)
    model = build_backbone(BACKBONE, pretrained=True).to(device)
    print("可訓練參數:", f"{count_params(model)/1e6:.1f}M")
    n_pos = int((train_df["label"] == 1).sum()); n_neg = int((train_df["label"] == 0).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    if SCHED == "warmrestart":
        # SGDR：每 RESTART_T0 回合把學習率重新拉高再退火，給機會跳出平原
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=RESTART_T0, T_mult=1)
        print(f"schedule: CosineAnnealingWarmRestarts(T_0={RESTART_T0})")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        print("schedule: CosineAnnealingLR")

    history = {"train_loss": [], "val_case_auc": [], "val_image_auc": [], "lr": []}
    best_auc, best_state, best = -1.0, copy.deepcopy(model.state_dict()), None
    no_improve = 0
    for epoch in range(1, EPOCHS + 1):
        model.train(); te = time.time(); running = 0.0
        for x, y, _ in train_loader:
            x = x.to(device); y = y.float().unsqueeze(1).to(device)
            optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step()
            running += loss.item() * x.size(0)
        scheduler.step()
        tr_loss = running / len(train_loader.dataset)
        yt, yp, cids, abn = infer(model, val_loader, val_df, device, tta=TTA)
        ct, cp = aggregate_by_case(cids, yt, yp)
        thr = select_threshold(ct, cp)
        rep = evaluate_predictions(yt, yp, cids, thr, abnormality=abn)
        cauc = rep.case["roc_auc"]
        history["train_loss"].append(tr_loss); history["val_case_auc"].append(cauc)
        history["val_image_auc"].append(rep.image["roc_auc"]); history["lr"].append(optimizer.param_groups[0]["lr"])
        print(f"Epoch {epoch:2d}/{EPOCHS} | {time.time()-te:5.1f}s | loss {tr_loss:.4f} | "
              f"val_case_auc {cauc:.4f} thr {thr:.3f}")
        if np.isfinite(cauc) and cauc > best_auc:
            best_auc = cauc; best_state = copy.deepcopy(model.state_dict())
            best = {"epoch": epoch, "case_auc": float(cauc), "threshold": float(thr)}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  早停 @ep{epoch}"); break
    model.load_state_dict(best_state)
    print("最佳 epoch:", best["epoch"], "| val case-AUC:", round(best["case_auc"], 4),
          "| 凍結閾值:", round(best["threshold"], 4))

    print("\n=== 3. 官方測試集評估（TTA, 凍結閾值）===")
    yt, yp, cids, abn = infer(model, make_loader(test_df, False), test_df, device, tta=TTA)
    report = evaluate_predictions(yt, yp, cids, best["threshold"], abnormality=abn)
    print(f"test case-AUC {report.case['roc_auc']:.4f} | image-AUC {report.image['roc_auc']:.4f} | "
          f"敏感度 {report.case['sensitivity']:.4f} 特異度 {report.case['specificity']:.4f}")
    print("subgroup case-AUC:", {k: round(v['roc_auc'], 4) for k, v in report.subgroups.items()})
    print("baseline 對照: test case-AUC 0.748 / 敏感度 0.646")

    print("\n=== 4. 存 best_v2.pt ===")
    fp = manifest_fingerprint(train_df, val_df, test_df)
    save_checkpoint(OUT, model, img_size=IMG_SIZE, class_names=["benign", "malignant"],
                    threshold=best["threshold"], best_epoch=best["epoch"],
                    best_metric=best["case_auc"], best_metric_name="val_case_auc",
                    split_seed=SEED, fold=FOLD, history=history, manifest_fp=fp,
                    arch=BACKBONE, tta=TTA,
                    extra={"split_audit": split_audit,
                           "test_metrics": {"image": report.image, "case": report.case,
                                            "subgroups": report.subgroups}})
    print("已存:", OUT, "| 大小(MB):", round(OUT.stat().st_size / 1e6, 1))

    print("\n=== 5. CPU round-trip ===")
    m2, ckpt = load_for_inference(OUT, device=torch.device("cpu"))
    demo = test_df.iloc[0]
    print("推論:", predict_image(m2, ckpt, demo["filepath"], device=torch.device("cpu")), "| 真實:", demo["pathology"])

    (ART / f"metrics_{TAG}.json").write_text(json.dumps({
        "backbone": BACKBONE, "img_size": IMG_SIZE, "tta": TTA,
        "epochs": EPOCHS, "sched": SCHED, "seed": SEED,
        "val_case_auc": best["case_auc"], "best_epoch": best["epoch"], "threshold": best["threshold"],
        "test": {"image": report.image, "case": report.case, "subgroups": report.subgroups},
        "baseline_test_case_auc": 0.748, "elapsed_min": round((time.time()-t0)/60, 1),
    }, indent=2), encoding="utf-8")
    print(f"\n用時 {(time.time()-t0)/60:.1f} 分")
    print("V2_TRAIN_DONE")


if __name__ == "__main__":
    main()
