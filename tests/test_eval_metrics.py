"""Unit tests for eval_audit pure metric functions."""

from __future__ import annotations

import unittest

import pandas as pd

from eval_audit_metrics_detection import compute_review_level_detection
from eval_audit_metrics_product import compute_product_level_metrics
from eval_audit_metrics_sentiment import round_clip_rating, compute_review_level_sentiment


class TestRoundClip(unittest.TestCase):
    def test_round_clip(self) -> None:
        self.assertEqual(round_clip_rating(1.2), 1.0)
        self.assertEqual(round_clip_rating(4.7), 5.0)
        self.assertEqual(round_clip_rating(3.4), 3.0)


class TestDetection(unittest.TestCase):
    def test_macro_prf(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "nm_id": 1,
                    "id": "r1",
                    "true_labels_parsed": {"A": 5.0, "B": 4.0},
                    "rating": 5,
                },
            ]
        )
        pipeline = {
            1: {
                "aspects": ["pA", "pB"],
                "per_review": {"r1": {"pA": 3.0, "pB": 4.0}},
            }
        }
        mapping = {1: {"pA": "A", "pB": "B"}}
        res = compute_review_level_detection(df, pipeline, mapping)
        self.assertAlmostEqual(res.macro_precision, 1.0)
        self.assertAlmostEqual(res.macro_recall, 1.0)
        self.assertAlmostEqual(res.macro_f1, 1.0)


class TestSentiment(unittest.TestCase):
    def test_mae_intersection(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "nm_id": 1,
                    "id": "r1",
                    "true_labels_parsed": {"Качество": 5.0},
                    "rating": 5,
                },
            ]
        )
        pipeline = {
            1: {
                "aspects": ["Качество"],
                "per_review": {"r1": {"Качество": 4.0}},
            }
        }
        mapping = {1: {"Качество": "Качество"}}
        res = compute_review_level_sentiment(df, pipeline, mapping)
        self.assertEqual(res.n_matched_pairs, 1)
        self.assertAlmostEqual(res.mae_continuous_review_macro, 1.0)


class TestProductMatched(unittest.TestCase):
    def test_same_review_ids(self) -> None:
        df = pd.DataFrame(
            [
                {"nm_id": 1, "id": "a", "true_labels_parsed": {"X": 5.0}, "rating": 5},
                {"nm_id": 1, "id": "b", "true_labels_parsed": {"X": 3.0}, "rating": 3},
            ]
        )
        pipeline = {
            1: {
                "aspects": ["pX"],
                "per_review": {
                    "a": {"pX": 5.0},
                    "b": {"pX": 3.0},
                },
            }
        }
        mapping = {1: {"pX": "X"}}
        res = compute_product_level_metrics(df, pipeline, mapping)
        self.assertIsNotNone(res.product_mae_matched_all)
        self.assertAlmostEqual(res.product_mae_matched_all, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
