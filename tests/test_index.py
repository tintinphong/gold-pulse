"""Offline tests for processing/index.py: trailing-only scores, bands, complete days."""
import unittest

import numpy as np
import pandas as pd

from processing.index import ATTENTION_BANDS, MOOD_BANDS, band, build_index, trailing_score


def posts(per_day, start="2025-01-01", sentiment=0.2):
    rows, k = [], 0
    for d, n in enumerate(per_day):
        for i in range(n):
            rows.append({"post_id": f"p{k}", "sentiment": sentiment,
                         "created_utc": pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=d, minutes=10 * i)})
            k += 1
    return pd.DataFrame(rows)


class ScoreTests(unittest.TestCase):
    def test_score_ignores_later_days(self):
        s = pd.Series(np.random.default_rng(0).normal(size=400))
        full, short = trailing_score(s, 100, 50), trailing_score(s.iloc[:250], 100, 50)
        pd.testing.assert_series_equal(full.iloc[:250], short)

    def test_new_high_scores_100(self):
        s = pd.Series(list(range(1, 60)))
        self.assertEqual(trailing_score(s, 30, 10).iloc[-1], 100)

    def test_bands(self):
        self.assertEqual(band(96, ATTENTION_BANDS), "Frenzy")
        self.assertEqual(band(50, ATTENTION_BANDS), "Normal")
        self.assertEqual(band(5, MOOD_BANDS), "Gloomy")
        self.assertIsNone(band(np.nan, MOOD_BANDS))


class BuildTests(unittest.TestCase):
    def test_partial_last_day_is_not_published_and_empty_days_are_zero(self):
        idx = build_index(posts([3, 0, 2, 5]))  # the 4th day is the last one with posts: treated as partial
        self.assertEqual(idx.index.max(), pd.Timestamp("2025-01-03"))
        self.assertEqual(idx["posts"].tolist(), [3, 0, 2])
        self.assertTrue(np.isnan(idx.loc["2025-01-02", "mean_sentiment"]))

    def test_mood_is_post_weighted(self):
        p = pd.concat([posts([1] * 8, sentiment=1.0), posts([3] * 8, start="2025-01-01", sentiment=0.0)
                       .assign(post_id=lambda d: "q" + d["post_id"])])
        idx = build_index(p)
        self.assertAlmostEqual(idx["mood_7d"].iloc[-1], 0.25)


if __name__ == "__main__":
    unittest.main()
