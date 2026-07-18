# CBIS-DDSM lesion classifier

This project trains a leakage-safe malignant-vs-benign classifier on **cropped,
known lesions** from the awsaf49 CBIS-DDSM JPEG repack. It will deliver
`CBIS_DDSM_Training.ipynb` and a genuinely trained `best.pt`.

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

## Current preparation status

- Dataset metadata is present and complete enough for schema/split preparation.
- Combined metadata has 2,864 train rows and 704 test rows before mapping;
  malignant counts are 1,181 and 276 respectively.
- Combining mass and calc official splits creates 31 overlapping patient IDs;
  the implementation will purge them from the training pool.
- A local DirectML EfficientNet-B0 forward/backward probe passed. One
  `BCEWithLogitsLoss` operator falls back to CPU, so performance will be
  benchmarked against CPU before the long run.
- The full JPEG directory and final `best.pt` are not yet present.

See [SPEC.md](SPEC.md) for the acceptance contract and [PLAN.md](PLAN.md) for
the ordered tasks.

## Important limitation

CBIS-DDSM case-description CSVs contain abnormalities already known to exist;
they do not provide normal screening negatives. Results therefore measure
malignancy classification of known lesion crops, not population screening.

