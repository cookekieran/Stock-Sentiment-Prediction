from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "src" / "data"
sys.path.insert(0, str(DATA_DIR))

from build_ltn_dataset import add_signal_dates


class InformationCutoffTests(unittest.TestCase):
    def test_utc_news_is_converted_to_new_york_cutoff(self) -> None:
        examples = pd.DataFrame(
            {
                "time_published": [
                    "2025-07-01 19:59:00",
                    "2025-07-01 20:00:00",
                ]
            }
        )
        result = add_signal_dates(examples, 16, "UTC", "America/New_York")
        self.assertEqual(result.loc[0, "signal_date"], pd.Timestamp("2025-07-01"))
        self.assertEqual(result.loc[1, "signal_date"], pd.Timestamp("2025-07-02"))

    def test_dst_conversion_uses_market_timezone(self) -> None:
        examples = pd.DataFrame({"time_published": ["2025-01-02 21:00:00"]})
        result = add_signal_dates(examples, 16, "UTC", "America/New_York")
        self.assertEqual(result.loc[0, "signal_date"], pd.Timestamp("2025-01-03"))


if __name__ == "__main__":
    unittest.main()
