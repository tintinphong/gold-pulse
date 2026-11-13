"""Offline tests for processing/features.py: target alignment and no look-ahead."""
import unittest

import numpy as np
import pandas as pd

from processing.features import PRICE_FEATURES, SENTIMENT_FEATURES, build_features

T0 = pd.Timestamp("2026-09-08 00:00", tz="UTC")


def hourly(prices, sentiments=None):
    n = len(prices)
    grid = pd.date_range(T0, periods=n, freq="h", tz="UTC", name="bucket_utc")
    sentiments = sentiments if sentiments is not None else [0.1 * i for i in range(n)]
    return pd.DataFrame({
        "post_count": [i % 3 for i in range(n)],
        "mean_sentiment": sentiments,
        "bullish_ratio": [0.5] * n,
        "bearish_ratio": [0.2] * n,
        "mean_engagement": [np.nan if i % 4 == 0 else 10.0 * i for i in range(n)],
        "stale_share": [0.0] * n,
        "had_posts": [i % 3 != 0 for i in range(n)],
        "xau_usd": prices,
        "market_open": [not np.isnan(p) for p in prices],
    }, index=grid)


class TargetTests(unittest.TestCase):
    def test_target_is_move_from_t_plus_1_to_t_plus_2(self):
        # t=0 -> prices at 1 and 2 are 101, 100: down, even though 0 -> 1 is up.
        f = build_features(hourly([100.0, 101.0, 100.0, 105.0]))
        self.assertEqual(f["price_direction"].iloc[0], 0.0)
        self.assertEqual(f["price_direction"].iloc[1], 1.0)
        self.assertAlmostEqual(f["next_return"].iloc[1], 105.0 / 100.0 - 1)

    def test_target_invalid_when_either_future_price_missing(self):
        # Row t needs prices at t+1 and t+2 only; its own price may be missing (row 2).
        f = build_features(hourly([100.0, 101.0, np.nan, 103.0, 104.0, 105.0]))
        self.assertEqual(f["target_valid"].tolist(), [False, False, True, True, False, False])
        self.assertTrue(f["price_direction"][~f["target_valid"]].isna().all())

    def test_tie_counts_as_not_up(self):
        f = build_features(hourly([100.0, 101.0, 101.0]))
        self.assertEqual(f["price_direction"].iloc[0], 0.0)


class NoLookAheadTests(unittest.TestCase):
    def test_features_at_t_ignore_later_rows(self):
        rng = np.random.default_rng(0)
        prices = list(4000 + rng.normal(0, 5, 48).cumsum())
        prices[20:26] = [np.nan] * 6  # a closure
        full = hourly(prices, list(rng.uniform(-1, 1, 48)))
        cut = 30
        a = build_features(full)[PRICE_FEATURES + SENTIMENT_FEATURES].iloc[:cut]
        b = build_features(full.iloc[:cut])[PRICE_FEATURES + SENTIMENT_FEATURES]
        pd.testing.assert_frame_equal(a, b)

    def test_return_is_nan_across_a_closure(self):
        f = build_features(hourly([100.0, np.nan, 102.0, 103.0]))
        self.assertTrue(np.isnan(f["last_return"].iloc[2]))
        self.assertTrue(f["return_missing"].iloc[2])
        self.assertAlmostEqual(f["last_return"].iloc[3], 103.0 / 102.0 - 1)

    def test_rolling_std_skips_forward_filled_hours(self):
        # had_posts False at i % 3 == 0: those values must not count.
        f = build_features(hourly([100.0] * 6, [9.0, 1.0, 3.0, 9.0, 5.0, 7.0]))
        self.assertAlmostEqual(f["sent_roll_std_3h"].iloc[3], np.std([1.0, 3.0], ddof=1))


if __name__ == "__main__":
    unittest.main()
