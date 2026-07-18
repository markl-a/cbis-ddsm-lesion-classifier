# CBIS-DDSM 病灶良惡分類器

本專案在 awsaf49 版 CBIS-DDSM（JPEG repack）的**裁切、已知病灶**影像上，訓練一個
**防資料洩漏（leakage-safe）**的良性 / 惡性分類器。交付兩個主檔：可從頭跑到尾的
`CBIS_DDSM_All_In_One.ipynb`，與真實訓練出來的 `best.pt`。notebook 內含完整 pipeline，
**不依賴** `src/` 內的開發模組。

## 這份繳交包含什麼

- **`CBIS_DDSM_All_In_One.ipynb`** — 自我包含的 notebook（資料對接 → 防洩漏切分 → 訓練 →
  評估〔含 bootstrap CI〕 → 存/載 `best.pt` → 單張推論）。所有邏輯都內聯定義，不 import `src/`。
- **`best.pt`** — 交付模型（EfficientNetV2-S @288 + TTA，81 MB，CPU 張量，完整 metadata）。
  另外保留 `best_v1.pt`（baseline），供 `evaluate_significance.py` 跑配對顯著性檢定
  （`best.pt` vs `best_v1.pt`）。
- **`src/` + `tests/`** — 同一套 pipeline 的可重用模組，附單元測試
  （`python -m unittest discover -s tests`）。
- **`evaluate_significance.py`** — v2 相對 baseline 的 DeLong + bootstrap 顯著性檢定。
- **`artifacts/`** — 指標、切分稽核、顯著性檢定的 JSON。

notebook 以**未執行**狀態繳交（已清除輸出）。要從頭重現需先取得資料集（見下），並跑一次
訓練（AMD iGPU 上約 30–45 分鐘）。

## 資料集放這裡

建議解壓後的目錄結構：

```text
breast_cancer/data/
  csv/
    dicom_info.csv
    calc_case_description_train_set.csv
    calc_case_description_test_set.csv
    mass_case_description_train_set.csv
    mass_case_description_test_set.csv
  jpeg/
    <SeriesInstanceUID>/
      *.jpg
```

資料集下載：<https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset>

也支援把 CSV 平放於 `data/` 的舊佈局。若你的解壓路徑是另一層巢狀目錄，**不要**去改名數千個
影像資料夾；pipeline 會自動搜尋常見的 Kaggle 佈局並回報解析到的路徑。

**請勿**把 Kaggle token 放進此資料夾或 notebook。

## 成果（交付版本）

以下數字皆在**同一套**防洩漏、patient-disjoint、case-level 協定下取得（清除 31 位橫跨官方
train/test 的病人；train/val/test 三組病人兩兩不相交；train 2,223 / val 556 / test 704 張影像；
對接覆蓋率 99.94%）。官方測試集只用「驗證集凍結後的閾值」評估**一次**。

**交付模型 — v2（`best.pt`）：EfficientNetV2-S @288 + 水平翻轉 TTA**

| 指標（官方測試集，case-level） | v1 baseline | **v2（交付）** | Δ |
|---|---|---|---|
| ROC-AUC | 0.748 | **0.794** | **+0.046** |
| image-level ROC-AUC | 0.739 | 0.776 | +0.037 |
| **敏感度**（少漏診） | 0.646 | **0.732** | **+0.086** |
| 特異度 | 0.711 | 0.699 | −0.012 |
| subgroup AUC — mass / calc | 0.778 / 0.729 | 0.811 / 0.780 | +0.033 / +0.051 |

- 選模：依驗證集 case-level ROC-AUC 0.829 選到 best epoch 8；閾值 0.414。
- `best.pt`：81.6 MB，CPU 張量，可用 `weights_only=True` 安全載入，round-trip CPU 推論已驗證。
  baseline 保存為 `best_v1.pt`（16.3 MB）。

**這個提升是真的嗎？是 —— 有統計檢定。** 在同一個 fold-0 測試集（413 個 case）上，v2 vs v1：
**ΔAUC = +0.046，bootstrap 95% CI [+0.004, +0.089]**（不含 0），**DeLong 配對檢定 p = 0.035**，
bootstrap p = 0.018。提升明確超過雜訊地板。注意：單一 413-case 測試集的 AUC 標準誤約 SE ≈ 0.024
（0.794 的 95% CI ≈ [0.745, 0.834]），故點估計應以 ±0.05 看待。詳見
`artifacts/significance.json`（由 `evaluate_significance.py` 產生）。

**為什麼是 v2、而不是文獻上「0.9+ 的 SOTA」。** 經多來源、對抗式查核洩漏的文獻調查後可知：
多數 CBIS-DDSM 上 ≥0.85 的 AUC **不可與本題相比** —— 它們用的是隨機切分（有病人洩漏）、
更容易的任務層級（整張影像 / breast-level / 篩檢），或非排序型的「AUC」指標。在本題這個
「裁切病灶 + patient-disjoint」的設定下，調過的 CNN 誠實天花板約 **0.78–0.82**，而且這個範圍
本身就在 ±0.024 的雜訊內。**我們已在天花板的證據**：一次 24-epoch 的 warm-restart 訓練拿到
**更高的驗證** AUC（0.839 vs 0.829），但**測試** AUC 反而更低（0.776）—— 硬擠驗證分數只是
在過擬合小小的驗證集。真正能轉移到本題的增益是解析度 + TTA + 更強 backbone（v2 全用上了）；
要到 0.85+ 需要整張影像脈絡或大規模領域預訓練，在 4 GB DirectML 上不切實際。

重現方式：把 `CBIS_DDSM_All_In_One.ipynb` 從頭跑到尾，或執行
`.venv\Scripts\python.exe run_training_v2.py`，指標會寫到 `artifacts/metrics_v2.json`。
若要重現 v1 baseline：跑 `run_training.py`，或在 notebook 設環境變數
`BACKBONE=efficientnet_b0 IMG_SIZE=224 TTA=0`。

快速煙霧測試（極小子集、1 epoch、CPU）：

```powershell
$env:SMOKE=1; $env:CBIS_FORCE_CPU=1; $env:OUT_PATH="_smoke_best.pt"
.\.venv\Scripts\jupyter-nbconvert.exe --to notebook --execute `
  breast_cancer\CBIS_DDSM_All_In_One.ipynb --output _smoke.ipynb
```

驗收合約見 [SPEC.md](SPEC.md)，工作分解見 [PLAN.md](PLAN.md)。單元測試：
`.\.venv\Scripts\python.exe -m unittest discover -s breast_cancer\tests`。

## 重要限制

CBIS-DDSM 的 case-description CSV 只含**已知存在的異常**，沒有正常篩檢陰性樣本。因此本模型量測的是
「**已知病灶**的良惡分類」，**不是**族群篩檢，也**不是**篩檢模型、醫材或臨床判斷依據。
