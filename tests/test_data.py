from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import build_manifest, make_patient_splits  # noqa: E402


class ManifestTests(unittest.TestCase):
    def test_build_manifest_joins_series_uid_and_selects_crop_not_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_dir = root / "csv"
            jpeg_dir = root / "jpeg" / "SERIES_CROP"
            csv_dir.mkdir()
            jpeg_dir.mkdir(parents=True)

            crop_path = jpeg_dir / "2-crop.jpg"
            mask_path = jpeg_dir / "1-mask.jpg"
            Image.new("L", (17, 11), color=127).save(crop_path)
            Image.new("L", (101, 89), color=255).save(mask_path)

            pd.DataFrame(
                [
                    {
                        "SeriesInstanceUID": "SERIES_CROP",
                        "SeriesDescription": "ROI mask images",
                        "image_path": "CBIS-DDSM/jpeg/SERIES_CROP/1-mask.jpg",
                        "Rows": 89,
                        "Columns": 101,
                        "BitsAllocated": 8,
                    },
                    {
                        "SeriesInstanceUID": "SERIES_CROP",
                        "SeriesDescription": "cropped images",
                        "image_path": "CBIS-DDSM/jpeg/SERIES_CROP/2-crop.jpg",
                        "Rows": 11,
                        "Columns": 17,
                        "BitsAllocated": 16,
                    },
                ]
            ).to_csv(csv_dir / "dicom_info.csv", index=False)

            pd.DataFrame(
                [
                    {
                        "patient_id": "P_00001",
                        "left or right breast": "LEFT",
                        "image view": "CC",
                        "abnormality id": 1,
                        "pathology": "MALIGNANT",
                        "cropped image file path": (
                            "Mass-Training_P_00001_LEFT_CC_1/STUDY/"
                            "SERIES_CROP/000000.dcm"
                        ),
                    }
                ]
            ).to_csv(csv_dir / "mass_case_description_train_set.csv", index=False)

            manifest = build_manifest(root, require_files=True, verbose=False)

            self.assertEqual(len(manifest), 1)
            row = manifest.iloc[0]
            self.assertEqual(Path(row.filepath), crop_path)
            self.assertNotEqual(Path(row.filepath), mask_path)
            self.assertEqual(row.label, 1)
            self.assertEqual(row.official_split, "train")
            self.assertEqual(row.abnormality, "mass")
            self.assertEqual(row.image_view, "CC")
            # A few real CBIS-DDSM abnormality ids are swapped between CC/MLO
            # and carry contradictory pathologies.  Qualifying the evaluation
            # key by pathology prevents a benign and malignant image from being
            # averaged into one case; pathology is never a model input.
            self.assertEqual(row.case_id, "mass|P_00001|LEFT|1|MALIGNANT")
            self.assertGreaterEqual(manifest.attrs["mapping_coverage"], 0.995)


class PatientSplitTests(unittest.TestCase):
    def _manifest(self) -> pd.DataFrame:
        rows = []
        for i in range(20):
            patient = f"P_{i:05d}"
            label = i % 2
            for view in ("CC", "MLO"):
                rows.append(
                    {
                        "patient_id": patient,
                        "label": label,
                        "official_split": "train",
                        "case_id": f"mass|{patient}|LEFT|1",
                        "image_view": view,
                    }
                )

        # P_00000 occurs in both author-provided task splits. Its training rows
        # must be purged before validation splitting.
        rows.extend(
            [
                {
                    "patient_id": "P_00000",
                    "label": 0,
                    "official_split": "test",
                    "case_id": "calc|P_00000|RIGHT|1",
                    "image_view": "CC",
                },
                {
                    "patient_id": "P_TEST_BENIGN",
                    "label": 0,
                    "official_split": "test",
                    "case_id": "mass|P_TEST_BENIGN|LEFT|1",
                    "image_view": "CC",
                },
                {
                    "patient_id": "P_TEST_MALIGNANT",
                    "label": 1,
                    "official_split": "test",
                    "case_id": "mass|P_TEST_MALIGNANT|LEFT|1",
                    "image_view": "CC",
                },
            ]
        )
        return pd.DataFrame(rows)

    def test_split_purges_official_test_patients_and_is_group_disjoint(self):
        train, val, test, audit = make_patient_splits(
            self._manifest(), n_splits=5, fold=0, seed=42
        )

        train_patients = set(train.patient_id)
        val_patients = set(val.patient_id)
        test_patients = set(test.patient_id)

        self.assertTrue(train_patients.isdisjoint(val_patients))
        self.assertTrue(train_patients.isdisjoint(test_patients))
        self.assertTrue(val_patients.isdisjoint(test_patients))
        self.assertNotIn("P_00000", train_patients | val_patients)
        self.assertIn("P_00000", test_patients)
        self.assertEqual(audit["purged_train_patients"], ["P_00000"])
        self.assertEqual(set(train.label), {0, 1})
        self.assertEqual(set(val.label), {0, 1})
        self.assertEqual(set(test.label), {0, 1})

    def test_split_is_reproducible_for_same_seed_and_fold(self):
        first = make_patient_splits(self._manifest(), n_splits=5, fold=1, seed=7)
        second = make_patient_splits(self._manifest(), n_splits=5, fold=1, seed=7)

        self.assertEqual(first[0].index.tolist(), second[0].index.tolist())
        self.assertEqual(first[1].index.tolist(), second[1].index.tolist())
        self.assertEqual(first[2].index.tolist(), second[2].index.tolist())


if __name__ == "__main__":
    unittest.main()
