# Spec: CBIS-DDSM lesion malignancy classifier

## Objective

Build a reproducible PyTorch notebook that classifies a **known lesion crop** as
malignant (`1`) or benign-like (`0`) from the awsaf49 JPEG repack of CBIS-DDSM.
The final deliverables are:

- `CBIS_DDSM_Training.ipynb`: data validation, leakage-safe splitting, EDA,
  training, evaluation, checkpoint loading, and single-image inference.
- `best.pt`: the checkpoint from the epoch with the highest validation
  case-level ROC-AUC.
- Tests and documentation needed to reproduce and audit both artifacts.

This is a lesion-level research classifier. It is **not** a normal-vs-cancer
screening model, a medical device, or a substitute for clinical judgement.

## Data contract

- Expected root: `data/`.
- Required metadata: the four `*_case_description_*_set.csv` files and
  `dicom_info.csv`, either directly under `data/` or under `data/csv/`.
- Required images: `data/jpeg/<SeriesInstanceUID>/*.jpg` (the alternative
  `data/CBIS-DDSM/jpeg/` layout is also accepted).
- Labels: `MALIGNANT -> 1`; `BENIGN` and `BENIGN_WITHOUT_CALLBACK -> 0`.
- Input: the JPEG identified as `cropped images` in `dicom_info.csv`; ROI masks
  and full mammograms are excluded.
- The DICOM paths in the case CSVs are never converted by merely replacing the
  extension. Their SeriesInstanceUID is joined to `dicom_info.csv.image_path`.
- Every manifest build reports unmatched and missing files and refuses to train
  below the documented mapping-coverage threshold.

### Leakage-safe split

The mass and calcification splits were created independently. After combining
them, 31 `patient_id` values occur in both official train and official test.
Therefore:

1. Preserve every official-test row as the final test set.
2. Remove every official-test patient from the combined training pool.
3. Create train/validation from the remaining pool with
   `StratifiedGroupKFold`, grouping on `patient_id`.
4. Assert that train, validation, and test patient sets are pairwise disjoint.
5. Never tune a threshold, epoch, or hyperparameter on official test data.

An evaluation case is identified by abnormality type, patient, breast side,
and abnormality id. CC/MLO image probabilities for the same case are averaged
before case-level metrics are computed.

## Model and training

- Stack: Python 3.11, PyTorch 2.4.1, Torchvision 0.19.1.
- Backbone: ImageNet-pretrained `efficientnet_b0`; replace the classifier with
  one output logit.
- Input: grayscale JPEG converted to three channels, aspect ratio preserved by
  padding, then resized to the configured square size and normalized with the
  documented ImageNet statistics.
- Augmentation: mild horizontal flip, rotation/translation/scale, and contrast
  changes. No vertical flip or heavy blur.
- Loss: `BCEWithLogitsLoss` with `pos_weight` computed from the **training fold
  only**.
- Optimizer/schedule: AdamW with cosine decay; early stopping on validation
  case-level ROC-AUC.
- Device order: DirectML probe -> DirectML when the probe passes -> CPU
  fallback. CUDA is supported if the notebook is moved to a CUDA machine.
- Reproducibility: seed Python, NumPy, PyTorch, DataLoader generators and record
  dependency/device versions. Exact bitwise equality across devices is not
  promised.

## Metrics and checkpoint contract

- Primary selection metric: validation **case-level ROC-AUC**.
- Secondary metrics: image- and case-level PR-AUC, balanced accuracy, F1,
  sensitivity, specificity, confusion matrix, plus mass/calc subgroup AUC.
- The decision threshold is selected on validation only and frozen for test.
- `best.pt` is a dictionary containing CPU `state_dict` tensors, architecture,
  image size, class names, threshold, best epoch/metric, split seed/fold,
  training history, package versions, and manifest fingerprint.
- Loading `best.pt` must reconstruct the model and run CPU inference without
  importing notebook state.

## Commands

Run from `C:\Users\m4932\Desktop\test0718`:

```powershell
# Tests
.\.venv\Scripts\python.exe -m unittest discover -s breast_cancer\tests -v

# Generate the notebook from its Jupytext source
.\.venv\Scripts\jupytext.exe --to ipynb breast_cancer\CBIS_DDSM_Training.py

# Execute a smoke configuration after data is present
.\.venv\Scripts\jupyter-nbconvert.exe --to notebook --execute `
  breast_cancer\CBIS_DDSM_Training.ipynb `
  --output CBIS_DDSM_Training.executed.ipynb `
  --ExecutePreprocessor.timeout=1800
```

## Project structure

```text
breast_cancer/
  data/                         local dataset; ignored by Git
  tests/                        fast unit and integration tests
  src/                          importable pipeline code
  CBIS_DDSM_Training.py         Jupytext source of the full notebook
  CBIS_DDSM_Training.ipynb      final notebook deliverable
  best.pt                       final trained checkpoint
  README.md                     usage and results
  SPEC.md                       this contract
  PLAN.md                       ordered implementation tasks
  DATASET_NOTICE.md             attribution and usage restrictions
```

## Testing strategy

- Unit tests use tiny synthetic CSV/JPEG fixtures for UID mapping, label
  mapping, grouped splitting, case aggregation, metric calculation, and
  checkpoint round-trip.
- A DirectML probe performs one EfficientNet forward/backward pass; lack of
  DirectML skips the accelerator path rather than failing CPU use.
- Notebook smoke execution uses a very small subset and one short epoch. The
  delivered `best.pt` must come from the full configured run, never smoke mode.
- Final verification loads `best.pt` on CPU, validates metadata and finite
  outputs, and confirms notebook JSON has no stored credentials.

## Boundaries

- Always: validate external CSV/image data, keep patient sets disjoint, select
  on validation only, save checkpoint tensors on CPU, and report missing data.
- Ask first: switch to full mammograms, add radiologist metadata as features,
  train separate mass/calc models, or change the label definition.
- Never: store Kaggle credentials, silently drop large fractions of data, use
  official test data for model selection, claim clinical readiness, or attempt
  patient re-identification.

## Success criteria

- All unit tests pass.
- At least 99.5% of description rows map to the intended crop metadata, and all
  retained image files exist.
- No patient appears in more than one of train/validation/test.
- The notebook executes end-to-end in smoke mode with no cell error.
- `best.pt` is produced by a non-smoke training run and passes CPU checkpoint
  round-trip/inference validation.
- Final metrics and limitations are recorded in the notebook and README.

## Official implementation references

- Torchvision EfficientNet-B0 0.19 API:
  https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.efficientnet_b0.html
- PyTorch 2.4 BCEWithLogitsLoss:
  https://docs.pytorch.org/docs/2.4/generated/torch.nn.BCEWithLogitsLoss.html
- PyTorch 2.4 reproducibility:
  https://docs.pytorch.org/docs/2.4/notes/randomness.html
- scikit-learn StratifiedGroupKFold:
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
- Microsoft PyTorch with DirectML:
  https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows

