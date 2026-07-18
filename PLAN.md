# Implementation Plan

## Architecture decisions

- Use `breast_cancer/` as the single authoritative project because it already
  contains the metadata and the correct UID-based JPEG bridge.
- Keep `cbis_ddsm_project/` untouched as an earlier draft; it is not a source of
  truth because its path resolver cannot map the Kaggle JPEG layout.
- Keep tested logic in `src/`; make the notebook an end-to-end presentation and
  execution surface with all configuration and outputs visible.
- Risk first: validate UID coverage, patient overlap, and DirectML
  forward/backward before long training.

## Phase 1: foundation and data contract

- [ ] Task 1: Freeze the project contract and environment
  - Acceptance: objective, boundaries, exact deliverables, commands, and local
    dependency snapshot are documented.
  - Verify: all documentation links and paths resolve locally.
  - Files: `SPEC.md`, `PLAN.md`, `README.md`, `requirements.txt`, `.gitignore`.

- [ ] Task 2: Harden metadata-to-JPEG manifest construction
  - Acceptance: both CSV layouts work; crop images are selected by
    SeriesDescription; case/view fields and unmatched reasons are retained.
  - Verify: synthetic RED/GREEN tests and real-metadata mapping >= 99.5%.
  - Files: `src/data.py`, `tests/test_data.py`.

- [ ] Task 3: Add leakage-safe grouped splitting
  - Acceptance: official-test patient union is purged from training and all
    split patient sets are disjoint while retaining both labels.
  - Verify: synthetic overlap regression test plus real-metadata split audit.
  - Files: `src/data.py`, `tests/test_data.py`.

### Checkpoint: data foundation

- [ ] All data tests pass.
- [ ] A manifest audit CSV can be generated without loading image pixels.
- [ ] Image-presence validation is ready for the manual download.

## Phase 2: model, training, and evaluation

- [ ] Task 4: Implement transforms, model, and device probing
  - Acceptance: EfficientNet-B0 produces one logit; CPU always works;
    DirectML is selected only after a real forward/backward probe.
  - Verify: shape/finite-gradient tests on CPU and optional DirectML.
  - Files: `src/model.py`, `tests/test_model.py`.

- [ ] Task 5: Implement grouped metrics and threshold selection
  - Acceptance: view probabilities aggregate by case; threshold is derived
    from validation only; subgroup metrics handle single-class edge cases.
  - Verify: deterministic metric fixtures with hand-checkable expectations.
  - Files: `src/metrics.py`, `tests/test_metrics.py`.

- [ ] Task 6: Implement training and checkpoint round-trip
  - Acceptance: best epoch is selected by case ROC-AUC and `best.pt` stores CPU
    tensors plus complete reconstruction metadata.
  - Verify: tiny synthetic training run and CPU load/inference round-trip.
  - Files: `src/engine.py`, `src/checkpoint.py`, related tests.

### Checkpoint: trainable core

- [ ] Full unit suite passes.
- [ ] One batch trains on CPU and DirectML probe passes or cleanly falls back.
- [ ] A temporary checkpoint reloads with identical logits.

## Phase 3: notebook and real run

- [ ] Task 7: Build the full Jupytext notebook
  - Acceptance: notebook covers setup, audit, EDA, split proof, training,
    plots, test evaluation, checkpoint loading, and inference.
  - Verify: generated `.ipynb` parses and smoke-executes without cell error.
  - Files: `CBIS_DDSM_Training.py`, `CBIS_DDSM_Training.ipynb`.

- [ ] Task 8: Train the production checkpoint
  - Acceptance: non-smoke run completes, saves the best validation checkpoint,
    and evaluates the untouched official test once.
  - Verify: run log, metrics JSON, and `best.pt` metadata agree.
  - Files: `best.pt`, `artifacts/metrics.json`, executed notebook.

- [ ] Task 9: Final review and handoff
  - Acceptance: no credentials/absolute user-specific paths, license notice is
    present, all artifacts open, and limitations are explicit.
  - Verify: tests, notebook execution, checkpoint audit, secret scan, and
    independent five-axis code review all pass.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Crop/ROI metadata inconsistencies | Wrong pixels or dropped rows | Select by SeriesDescription, retain audit reasons, validate dimensions/content |
| 31 cross-task patient overlaps | Inflated test score | Purge union of official-test patients before validation split |
| DirectML unsupported/fallback ops | Slow or failed training | Real forward/backward probe, conservative batch size, CPU fallback |
| 4 GiB reported graphics memory | OOM | EfficientNet-B0, 224 baseline, adaptive batch reduction |
| JPEG conversion loses 16-bit intensity fidelity | Limits scientific claims | Document repack limitation; do not claim radiomics fidelity |
| No normal cases | Misleading screening claim | Name and document task as known-lesion malignancy classification |

