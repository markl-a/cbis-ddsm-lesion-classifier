from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from metrics import (  # noqa: E402
    aggregate_by_case,
    classification_metrics,
    evaluate_predictions,
    select_threshold,
)


class CaseAggregationTests(unittest.TestCase):
    def test_probabilities_are_averaged_per_case_in_stable_case_order(self):
        labels, probabilities = aggregate_by_case(
            case_ids=["case-b", "case-a", "case-a"],
            y_true=[0, 1, 1],
            y_prob=[0.1, 0.2, 0.8],
        )

        np.testing.assert_array_equal(labels, np.array([1, 0]))
        np.testing.assert_allclose(probabilities, np.array([0.5, 0.1]))

    def test_conflicting_labels_within_a_case_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "label"):
            aggregate_by_case(
                case_ids=["same-case", "same-case"],
                y_true=[0, 1],
                y_prob=[0.2, 0.8],
            )

    def test_conflicting_abnormalities_within_a_case_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "abnormality"):
            evaluate_predictions(
                y_true=[1, 1],
                y_prob=[0.2, 0.8],
                case_ids=["same-case", "same-case"],
                threshold=0.5,
                abnormality=["mass", "calc"],
            )


class ThresholdSelectionTests(unittest.TestCase):
    def test_youden_tie_uses_highest_threshold_deterministically(self):
        # Thresholds 0.8 and 0.4 both have Youden J == 0.5.  The documented
        # tie-break chooses the higher (more specific) threshold.
        threshold = select_threshold(
            y_true=[0, 0, 1, 1],
            y_prob=[0.1, 0.4, 0.4, 0.8],
        )

        self.assertEqual(threshold, 0.8)

    def test_validation_threshold_is_reused_unchanged_for_test_metrics(self):
        validation_threshold = select_threshold(
            y_true=[0, 0, 1, 1],
            y_prob=[0.1, 0.4, 0.4, 0.8],
        )

        test_metrics = classification_metrics(
            y_true=[0, 1],
            y_prob=[0.7, 0.9],
            threshold=validation_threshold,
        )

        self.assertEqual(validation_threshold, 0.8)
        self.assertEqual(test_metrics["threshold"], validation_threshold)


class ClassificationMetricTests(unittest.TestCase):
    def test_binary_metrics_and_confusion_counts_are_hand_checkable(self):
        metrics = classification_metrics(
            y_true=[0, 0, 1, 1],
            y_prob=[0.1, 0.6, 0.4, 0.9],
            threshold=0.5,
        )

        self.assertAlmostEqual(metrics["roc_auc"], 0.75)
        self.assertAlmostEqual(metrics["pr_auc"], 5.0 / 6.0)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["sensitivity"], 0.5)
        self.assertAlmostEqual(metrics["specificity"], 0.5)
        self.assertEqual(
            metrics["confusion"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
        )

    def test_report_contains_overall_and_full_mass_calc_case_metrics(self):
        report = evaluate_predictions(
            y_true=[0, 0, 1, 1, 0, 0],
            y_prob=[0.1, 0.3, 0.7, 0.9, 0.1, 0.6],
            case_ids=["m0", "m0", "m1", "m1", "c0", "c1"],
            threshold=0.5,
            abnormality=["mass", "mass", "mass", "mass", "calc", "calc"],
        )

        self.assertEqual(set(report.subgroups), {"mass", "calc"})
        self.assertEqual(report.case["n"], 4)
        self.assertEqual(report.subgroups["mass"]["n"], 2)
        self.assertEqual(report.subgroups["mass"]["roc_auc"], 1.0)

        calc = report.subgroups["calc"]
        self.assertEqual(calc["n"], 2)
        self.assertTrue(math.isnan(calc["roc_auc"]))
        self.assertTrue(math.isnan(calc["pr_auc"]))
        self.assertTrue(math.isnan(calc["balanced_accuracy"]))
        self.assertTrue(math.isnan(calc["sensitivity"]))
        self.assertAlmostEqual(calc["specificity"], 0.5)
        self.assertEqual(
            calc["confusion"], {"tn": 1, "fp": 1, "fn": 0, "tp": 0}
        )


if __name__ == "__main__":
    unittest.main()
