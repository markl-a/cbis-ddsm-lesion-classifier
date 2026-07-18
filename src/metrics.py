# -*- coding: utf-8 -*-
"""評估指標：image-/case-level 聚合、閾值選擇、subgroup 指標。

case-level：同一 case（abnormality|patient|side|abnormality_id）的多視角(CC/MLO)
機率取平均後再算指標。閾值只在驗證集上決定，凍結後套用到測試集。

回傳的指標 dict 欄位（供 notebook / checkpoint / 測試使用）：
    roc_auc, pr_auc, balanced_accuracy, f1, sensitivity, specificity,
    threshold, confusion{tn,fp,fn,tp}, n
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def aggregate_by_case(case_ids, y_true, y_prob, abnormality=None):
    """把 image-level 聚合成 case-level。

    同一 case 內 label 必須一致（否則丟 ValueError）；prob 取平均。
    若給 abnormality，同一 case 內 abnormality 也必須一致，並回傳對齊的 abnormality。
    回傳 (case_true, case_prob) 或 (case_true, case_prob, case_abn)，皆依 case 名穩定排序。
    """
    df = pd.DataFrame({
        "case": np.asarray(case_ids),
        "y": np.asarray(y_true, dtype=float),
        "p": np.asarray(y_prob, dtype=float),
    })
    if abnormality is not None:
        df["abn"] = np.asarray(abnormality)

    # label 一致性檢查
    label_nunique = df.groupby("case")["y"].nunique()
    if (label_nunique > 1).any():
        bad = label_nunique[label_nunique > 1].index.tolist()
        raise ValueError(f"case 內 label 不一致: {bad}")

    agg = {"y": ("y", "first"), "p": ("p", "mean")}
    if abnormality is not None:
        abn_nunique = df.groupby("case")["abn"].nunique()
        if (abn_nunique > 1).any():
            bad = abn_nunique[abn_nunique > 1].index.tolist()
            raise ValueError(f"case 內 abnormality 不一致: {bad}")
        agg["abn"] = ("abn", "first")

    g = df.groupby("case", sort=True).agg(**agg)
    case_true = g["y"].values.astype(int)
    case_prob = g["p"].values.astype(float)
    if abnormality is not None:
        return case_true, case_prob, g["abn"].values
    return case_true, case_prob


def _single_class(y_true) -> bool:
    return len(np.unique(np.asarray(y_true))) < 2


def select_threshold(y_true, y_prob) -> float:
    """Youden J (sensitivity+specificity-1) 最大化選閾值；平手取較高（較 specific）閾值。"""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if _single_class(y_true):
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    jmax = j.max()
    # 平手時取最高的有限閾值（roc_curve 首元素可能為 inf）
    candidates = thr[j == jmax]
    candidates = candidates[np.isfinite(candidates)]
    if len(candidates) == 0:
        return 1.0
    t = float(candidates.max())
    return float(min(max(t, 0.0), 1.0))


def classification_metrics(y_true, y_prob, threshold: float) -> dict:
    """給定閾值算一組分類指標。單類別時相關指標回 nan。"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    single = _single_class(y_true)
    return {
        "roc_auc": float("nan") if single else float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float("nan") if single else float(average_precision_score(y_true, y_prob)),
        "balanced_accuracy": float("nan") if single else float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "threshold": float(threshold),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
    }


@dataclass
class EvalReport:
    image: dict
    case: dict
    subgroups: dict = field(default_factory=dict)   # {'mass': {..case metrics..}, 'calc': {...}}

    def flat(self) -> dict:
        out = {
            "image_roc_auc": self.image["roc_auc"],
            "case_roc_auc": self.case["roc_auc"],
            "case_sensitivity": self.case["sensitivity"],
            "case_specificity": self.case["specificity"],
            "case_balanced_accuracy": self.case["balanced_accuracy"],
        }
        for k, m in self.subgroups.items():
            out[f"subgroup_{k}_roc_auc"] = m["roc_auc"]
        return out


def evaluate_predictions(y_true, y_prob, case_ids, threshold: float,
                         abnormality=None) -> EvalReport:
    """從 image-level 預測產生 image + case 指標；有 abnormality 時附各 subgroup 的 case 指標。"""
    img_m = classification_metrics(y_true, y_prob, threshold)

    if abnormality is not None:
        case_true, case_prob, case_abn = aggregate_by_case(
            case_ids, y_true, y_prob, abnormality=abnormality)
    else:
        case_true, case_prob = aggregate_by_case(case_ids, y_true, y_prob)
        case_abn = None
    case_m = classification_metrics(case_true, case_prob, threshold)

    subs = {}
    if case_abn is not None:
        case_abn = np.asarray(case_abn)
        for name in ("mass", "calc"):
            mask = case_abn == name
            if mask.sum() == 0:
                continue
            subs[name] = classification_metrics(case_true[mask], case_prob[mask], threshold)
    return EvalReport(image=img_m, case=case_m, subgroups=subs)
