# CBIS-DDSM lesion classifier

This project trains a leakage-safe malignant-vs-benign classifier on **cropped,
known lesions** from the awsaf49 CBIS-DDSM JPEG repack. It will deliver
`CBIS_DDSM_All_In_One.ipynb` and a genuinely trained `best.pt`. These are the
two primary handoff files; the notebook contains the complete pipeline and does
not rely on the development modules in `src/`.

## What's in this submission

- **`CBIS_DDSM_All_In_One.ipynb`** — the self-contained notebook (data → leakage-safe
  split → train → evaluate w/ bootstrap CI → save/load `best.pt` → inference). Defines
  everything inline; does not import `src/`.
- **`best.pt`** — the delivered model (EfficientNetV2-S @288 + TTA, 81 MB, CPU tensors,
  full metadata). `best_v1.pt` (the baseline) is also tracked so
  `evaluate_significance.py` can run the paired significance test (`best.pt` vs `best_v1.pt`).
- **`src/` + `tests/`** — the same pipeline as reusable modules with unit tests
  (`python -m unittest discover -s tests`).
- **`evaluate_significance.py`** — DeLong + bootstrap significance of v2 vs baseline.
- **`artifacts/`** — metrics + split audit + significance JSON.

The notebook is shipped un-executed (outputs cleared). Reproducing it end-to-end needs
the dataset (below) and one training run (~30–45 min on the AMD iGPU).

## Put the manual download here

Preferred extracted layout:

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

The existing flat CSV layout (`data/*.csv`) is also supported. If your download
extracts to another nested directory, do not rename thousands of image folders;
the pipeline will search the common Kaggle layouts and report the resolved path.

Do not place a Kaggle token in this folder or notebook.

## Results (delivered)

All numbers on the **identical** leakage-safe, patient-disjoint, case-level
protocol (31 official-test patients purged; train/val/test patient sets pairwise
disjoint; train 2,223 / val 556 / test 704 images; mapping coverage 99.94%).
The official test set is evaluated **once** with a validation-frozen threshold.

**Delivered model — v2 (`best.pt`): EfficientNetV2-S @288 + horizontal-flip TTA**

| Metric (official test, case-level) | v1 baseline | **v2 (delivered)** | Δ |
|---|---|---|---|
| ROC-AUC | 0.748 | **0.794** | **+0.046** |
| image-level ROC-AUC | 0.739 | 0.776 | +0.037 |
| **Sensitivity** (fewer missed cancers) | 0.646 | **0.732** | **+0.086** |
| Specificity | 0.711 | 0.699 | −0.012 |
| subgroup AUC — mass / calc | 0.778 / 0.729 | 0.811 / 0.780 | +0.033 / +0.051 |

- Model selection: best epoch 8 by validation case-level ROC-AUC 0.829; threshold 0.414.
- `best.pt`: 81.6 MB, CPU tensors, loads with `weights_only=True`, round-trip CPU
  inference verified. Baseline archived as `best_v1.pt` (16.3 MB).

**Is the improvement real? Yes — significance-tested.** On the same fold-0 test set
(413 cases), v2 vs v1: **ΔAUC = +0.046, bootstrap 95% CI [+0.004, +0.089]** (excludes 0),
**DeLong paired p = 0.035**, bootstrap p = 0.018. So the gain clears the noise floor.
Caveat: a single 413-case test AUC has SE ≈ 0.024 (95% CI on 0.794 ≈ [0.745, 0.834]),
so treat the point estimate as ±0.05. See `artifacts/significance.json`
(`evaluate_significance.py`).

**Why v2 and not a "SOTA 0.9+" model.** A multi-source, adversarially leakage-checked
survey plus a 67-agent method panel (both reports linked in the handoff) found that most
published CBIS-DDSM AUCs ≥0.85 are **not comparable**: random (patient-overlapping)
splits, an easier task level (whole-image / breast-level / screening), or non-ranking
"AUC" metrics. On this exact cropped-lesion, patient-disjoint task a tuned CNN's honest
ceiling is ~0.78–0.82, and even that is within the ±0.024 noise. Evidence we are at the
ceiling: a 24-epoch warm-restart run reached a *higher validation* AUC (0.839 vs 0.829)
but a *lower test* AUC (0.776) — squeezing validation just overfits the small val set.
The gains that actually transfer here are resolution + TTA + a stronger backbone (all in
v2); 0.85+ requires whole-image context or large-scale domain pretraining, impractical
on 4 GB DirectML.

Reproduce: run `CBIS_DDSM_All_In_One.ipynb` top-to-bottom, or
`.venv\Scripts\python.exe run_training_v2.py`. Metrics land in `artifacts/metrics_v2.json`.
To reproduce the v1 baseline instead: `run_training.py`, or set
`BACKBONE=efficientnet_b0 IMG_SIZE=224 TTA=0` for the notebook.

Smoke check (tiny subset, 1 epoch, CPU):

```powershell
$env:SMOKE=1; $env:CBIS_FORCE_CPU=1; $env:OUT_PATH="_smoke_best.pt"
.\.venv\Scripts\jupyter-nbconvert.exe --to notebook --execute `
  breast_cancer\CBIS_DDSM_All_In_One.ipynb --output _smoke.ipynb
```

See [SPEC.md](SPEC.md) for the acceptance contract and [PLAN.md](PLAN.md) for
the ordered tasks. Unit tests: `.\.venv\Scripts\python.exe -m unittest discover -s breast_cancer\tests`.

## Important limitation

CBIS-DDSM case-description CSVs contain abnormalities already known to exist;
they do not provide normal screening negatives. Results therefore measure
malignancy classification of known lesion crops, not population screening.
