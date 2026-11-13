"""Offline tests for the economy-mood variant: keyword filter and hourly features."""
import unittest

import pandas as pd

from collector.macro_backfill import ECONOMY_LABELS, matches_macro
from processing.macro import build_macro_features, build_macro_hourly, MACRO_FEATURES

T0 = pd.Timestamp("2026-09-08 00:00", tz="UTC")


class KeywordTests(unittest.TestCase):
    def test_matches_economy_terms(self):
        for text in ["Fed raises rates", "FED day tomorrow", "CPI comes in hot",
                     "Interest rates are crushing me", "Is a recession coming?",
                     "10-year Treasury yields hit a 24-year high", "Powell speaks", "rate cuts priced in"]:
            self.assertTrue(matches_macro(text), text)

    def test_ignores_lookalikes(self):
        for text in ["I'm fed up with my job", "Fed up with this market", "FED UP",
                     "federal holiday on Monday", "my dog got fed", "pacer cpis"]:
            self.assertFalse(matches_macro(text), text)

    def test_finbert_classes_become_economy_labels(self):
        self.assertEqual(ECONOMY_LABELS, {"bullish": "good", "bearish": "bad", "neutral": "neutral"})


def posts(*rows):
    """rows: (minutes after T0, finbert_score, finbert_label)."""
    return pd.DataFrame([{"post_id": f"p{i}", "created_utc": T0 + pd.Timedelta(minutes=m),
                          "finbert_score": s, "finbert_label": lab}
                         for i, (m, s, lab) in enumerate(rows)])


GRID = pd.date_range(T0, periods=6, freq="h", tz="UTC", name="bucket_utc")


class HourlyTests(unittest.TestCase):
    def test_aggregates_and_forward_fills_without_backfilling(self):
        h = build_macro_hourly(posts((70, -0.8, "bad"), (80, 0.2, "neutral"), (250, 0.6, "good")), GRID)
        self.assertEqual(h["macro_post_count"].tolist(), [0, 2, 0, 0, 1, 0])
        self.assertEqual(h["macro_had_posts"].tolist(), [False, True, False, False, True, False])
        self.assertEqual(h.loc[GRID[0], "macro_mood"], 0.0)            # before any post: neutral, not back-filled
        self.assertAlmostEqual(h.loc[GRID[1], "macro_mood"], -0.3)
        self.assertAlmostEqual(h.loc[GRID[3], "macro_mood"], -0.3)     # carried forward
        self.assertAlmostEqual(h.loc[GRID[1], "macro_bad_ratio"], 0.5)

    def test_features_have_no_nans_and_no_look_ahead(self):
        p = posts((10, 0.5, "good"), (70, -0.8, "bad"), (190, 0.1, "neutral"), (300, -0.4, "bad"))
        full = build_macro_features(build_macro_hourly(p, GRID))[MACRO_FEATURES]
        self.assertFalse(full.isna().any().any())
        cut = GRID[2]
        short = build_macro_features(build_macro_hourly(p[p["created_utc"] < cut + pd.Timedelta(hours=1)],
                                                        GRID[GRID <= cut]))[MACRO_FEATURES]
        pd.testing.assert_frame_equal(full.loc[:cut], short)


if __name__ == "__main__":
    unittest.main()
