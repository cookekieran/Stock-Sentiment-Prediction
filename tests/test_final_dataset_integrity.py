from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = ROOT / "data" / "processed" / "final_study"


class FinalDatasetIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        if not FINAL_ROOT.exists():
            self.skipTest("Build data/processed/final_study before running integrity tests.")

    def test_availability_audit_passes(self) -> None:
        audit = json.loads((FINAL_ROOT / "availability_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["fundamental_availability_leaks"], 0)
        self.assertIn("16:00 America/New_York", audit["prediction_cutoff"])

    def test_fold_integrity(self) -> None:
        for fold_dir in sorted(FINAL_ROOT.glob("fold_*")):
            daily = pd.read_parquet(fold_dir / "daily.parquet")
            schema = json.loads((fold_dir / "schema.json").read_text(encoding="utf-8"))
            self.assertFalse(daily.duplicated(["ticker", "anchor_trading_date"]).any())
            self.assertFalse(daily[["ticker", "anchor_trading_date", "target_observation_date"]].isna().any().any())
            self.assertTrue((daily["target_observation_date"] > daily["anchor_trading_date"]).all())
            self.assertEqual(set(daily["split"]), {"train", "validation", "test"})
            self.assertFalse(any(column.startswith("macro_") for column in schema["feature_columns"]))
            self.assertFalse(any(column.startswith("rule_") for column in schema["feature_columns"]))
            self.assertTrue(set(schema["feature_columns"]).issubset(daily.columns))
            fundamentals = pd.to_datetime(daily["fundamental_available_from_date"])
            self.assertFalse((fundamentals > daily["anchor_trading_date"]).fillna(False).any())

    def test_purged_split_boundaries(self) -> None:
        for fold_dir in sorted(FINAL_ROOT.glob("fold_*")):
            daily = pd.read_parquet(fold_dir / "daily.parquet")
            train = daily[daily["split"] == "train"]
            validation = daily[daily["split"] == "validation"]
            test = daily[daily["split"] == "test"]
            self.assertLess(train["target_observation_date"].max(), validation["anchor_trading_date"].min())
            self.assertLess(validation["target_observation_date"].max(), test["anchor_trading_date"].min())


if __name__ == "__main__":
    unittest.main()
