# -*- coding: utf-8 -*-
"""完整（非 smoke）訓練，產出交付用 best.pt 與 artifacts/metrics.json。

用法:
    ..\.venv\Scripts\python.exe run_training.py
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import torch
from data import build_manifest, make_patient_splits, manifest_fingerprint, split_audit_json
from model import get_device
from engine import TrainConfig, train, evaluate_split
from checkpoint import save_checkpoint, load_for_inference, predict_image

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "best.pt"
ART = HERE / "artifacts"
ART.mkdir(exist_ok=True)

SEED, FOLD = 42, 0


def main():
    t_start = time.time()
    print("=== 1. 建立 manifest ===")
    man = build_manifest(DATA, require_files=True, verbose=True)

    print("\n=== 2. 防洩漏切分 ===")
    train_df, val_df, test_df, split_audit = make_patient_splits(man, seed=SEED, fold=FOLD)
    print("rows:", split_audit["rows"], "| 清除跨集病患:", len(split_audit["purged_train_patients"]))

    print("\n=== 3. 裝置 ===")
    device = get_device(verbose=True)
    print("device:", device)

    print("\n=== 4. 訓練（EfficientNet-B0, 依 val case-AUC 選最佳）===")
    cfg = TrainConfig(epochs=15, batch_size=32, lr=3e-4, img_size=224,
                      early_stop_patience=4, seed=SEED, smoke=False)
    model, history, best = train(train_df, val_df, cfg, device)
    print("最佳 epoch:", best["epoch"], "| val case-AUC:", round(best["case_auc"], 4),
          "| 凍結閾值:", round(best["threshold"], 4))

    print("\n=== 5. 官方測試集單次評估（凍結閾值）===")
    rep = evaluate_split(model, test_df, cfg, device, threshold=best["threshold"])
    print("test image-AUC:", round(rep.image["roc_auc"], 4), "| case-AUC:", round(rep.case["roc_auc"], 4))
    print("test case sensitivity:", round(rep.case["sensitivity"], 4),
          "| specificity:", round(rep.case["specificity"], 4),
          "| balanced_acc:", round(rep.case["balanced_accuracy"], 4))
    print("subgroup case-AUC:", {k: round(v["roc_auc"], 4) for k, v in rep.subgroups.items()})

    print("\n=== 6. 存 best.pt ===")
    fp = manifest_fingerprint(train_df.assign(assigned_split="train"),
                              val_df.assign(assigned_split="validation"),
                              test_df.assign(assigned_split="test"))
    save_checkpoint(OUT, model, img_size=cfg.img_size, class_names=["benign", "malignant"],
                    threshold=best["threshold"], best_epoch=best["epoch"],
                    best_metric=best["case_auc"], best_metric_name="val_case_auc",
                    split_seed=SEED, fold=FOLD, history=history, manifest_fp=fp,
                    extra={"split_audit": split_audit,
                           "test_metrics": {"image": rep.image, "case": rep.case,
                                            "subgroups": rep.subgroups}})
    print("已存:", OUT)

    print("\n=== 7. CPU round-trip 驗證 ===")
    m2, ckpt = load_for_inference(OUT, device=torch.device("cpu"))
    demo = test_df.iloc[0]
    res = predict_image(m2, ckpt, demo["filepath"], device=torch.device("cpu"))
    print("推論:", res, "| 真實:", demo["pathology"])

    metrics = {
        "val_case_auc": best["case_auc"], "best_epoch": best["epoch"],
        "threshold": best["threshold"],
        "test": {"image": rep.image, "case": rep.case, "subgroups": rep.subgroups},
        "history": history,
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (ART / "split_audit.json").write_text(split_audit_json(split_audit), encoding="utf-8")
    print("\n=== 完成，用時 %.1f 分 ===" % ((time.time() - t_start) / 60))
    print("TRAINING_DONE")


if __name__ == "__main__":
    main()
