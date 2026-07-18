# -*- coding: utf-8 -*-
"""用既有的 best_v1.pt / best.pt，在同一 fold-0 測試集上做統計檢定：
- 每個模型 case-level ROC-AUC 的 bootstrap 95% CI
- v2 - v1 的配對差異 bootstrap CI 與 p 值
- DeLong 配對檢定 p 值（兩條相關 ROC 曲線）

不需重訓；用來把「0.794 vs 0.748」變成「附信賴區間與顯著性」的可辯護結論。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from data import build_manifest, make_patient_splits, CBISDataset, get_transforms
from checkpoint import load_for_inference

DATA = HERE / "data"
SEED = 42
N_BOOT = 2000


@torch.no_grad()
def case_probs(ckpt_path, test_df, device):
    """用某個 checkpoint 對測試集推論，回傳依 case_id 排序的 (case_ids, true, prob)。"""
    model, ck = load_for_inference(ckpt_path, device=device)
    img_size = ck.get("img_size", 224); tta = ck.get("tta", False)
    df = test_df.reset_index(drop=True)
    ds = CBISDataset(df, transform=get_transforms(False, img_size), return_index=True)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    case_col = df["case_id"].to_numpy(); lab_col = df["label"].to_numpy()
    probs = np.zeros(len(df));
    for x, y, idx in loader:
        x = x.to(device)
        p = torch.sigmoid(model(x)).squeeze(1)
        if tta:
            p = (p + torch.sigmoid(model(torch.flip(x, dims=[3]))).squeeze(1)) / 2
        probs[np.asarray(idx)] = p.detach().cpu().numpy()
    g = pd.DataFrame({"case": case_col, "y": lab_col, "p": probs}).groupby("case", sort=True).agg(
        y=("y", "first"), p=("p", "mean"))
    return g.index.to_numpy(), g["y"].values.astype(int), g["p"].values.astype(float)


# ---------- DeLong（fast, Sun & Xu 2014）----------
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def _fast_delong(preds, m):
    n = preds.shape[1] - m
    pos = preds[:, :m]; neg = preds[:, m:]; k = preds.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1) / 2 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, cov


def delong_test(y, pa, pb):
    order = np.argsort(-y)              # 正例(1)排前面
    m = int(y.sum())
    preds = np.vstack((pa, pb))[:, order]
    aucs, cov = _fast_delong(preds, m)
    l = np.array([[1.0, -1.0]])
    var = (l @ cov @ l.T).item()
    from scipy.stats import norm
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * norm.sf(abs(z))
    return aucs, z, p


def bootstrap(y, pa, pb, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    N = len(y); a_s, b_s, d_s = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        a = roc_auc_score(yb, pa[idx]); b = roc_auc_score(yb, pb[idx])
        a_s.append(a); b_s.append(b); d_s.append(b - a)
    a_s, b_s, d_s = map(np.array, (a_s, b_s, d_s))
    def ci(v): return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    p_one = float(np.mean(d_s <= 0))          # v2 不優於 v1 的比例（單尾）
    return ci(a_s), ci(b_s), ci(d_s), p_one


def main():
    device = torch.device("cpu")
    man = build_manifest(DATA, require_files=True, verbose=False)
    _, _, test_df, _ = make_patient_splits(man, n_splits=5, fold=0, seed=SEED)

    c1, y1, p1 = case_probs(HERE / "best_v1.pt", test_df, device)
    c2, y2, p2 = case_probs(HERE / "best.pt", test_df, device)
    assert np.array_equal(c1, c2) and np.array_equal(y1, y2), "case 對齊失敗"
    y = y1

    auc1 = roc_auc_score(y, p1); auc2 = roc_auc_score(y, p2)
    print(f"cases: {len(y)} | malignant: {int(y.sum())} benign: {int((1-y).sum())}")
    print(f"v1 (b0@224)      case-AUC: {auc1:.4f}")
    print(f"v2 (effv2s@288+TTA) case-AUC: {auc2:.4f}")
    print(f"ΔAUC (v2-v1): {auc2-auc1:+.4f}")

    (a_ci, b_ci, d_ci, p_boot) = bootstrap(y, p1, p2)
    print(f"\nbootstrap 95% CI ({N_BOOT}x):")
    print(f"  v1 AUC: [{a_ci[0]:.4f}, {a_ci[1]:.4f}]")
    print(f"  v2 AUC: [{b_ci[0]:.4f}, {b_ci[1]:.4f}]")
    print(f"  ΔAUC:   [{d_ci[0]:+.4f}, {d_ci[1]:+.4f}]  (含 0 = 不顯著)")
    print(f"  bootstrap 單尾 p(v2<=v1): {p_boot:.4f}")

    aucs, z, p_delong = delong_test(y, p1, p2)
    print(f"\nDeLong 配對檢定: z={z:.3f}  two-sided p={p_delong:.4f}")

    out = {"n_cases": int(len(y)), "auc_v1": float(auc1), "auc_v2": float(auc2),
           "delta": float(auc2 - auc1),
           "v1_ci95": a_ci, "v2_ci95": b_ci, "delta_ci95": d_ci,
           "bootstrap_p_one_sided": p_boot, "delong_z": float(z), "delong_p_two_sided": float(p_delong)}
    (HERE / "artifacts").mkdir(exist_ok=True)
    (HERE / "artifacts" / "significance.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    verdict = "顯著" if (d_ci[0] > 0 and p_delong < 0.05) else ("邊緣" if p_delong < 0.1 else "不顯著")
    print(f"\n結論: v2 相對 v1 的提升在統計上【{verdict}】")
    print("SIGNIFICANCE_DONE")


if __name__ == "__main__":
    main()
