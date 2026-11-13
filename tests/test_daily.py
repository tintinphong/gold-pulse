"""Offline tests for processing/daily.py: day cut-off, target and no look-ahead."""
import unittest

import numpy as np
import pandas as pd

from processing.daily import (DAILY_CHATTER_FEATURES, DAILY_MARKET_FEATURES, DAILY_MOOD_FEATURES,
                              assign_days, build_daily)

# Thu Sept 3 .. Fri Oct 2 2026, weekdays only
DAYS = pd.bdate_range("2026-09-03", "2026-10-02", name="date")


def utc(s):
    return pd.Timestamp(s, tz="UTC")


def market(n=len(DAYS)):
    rng = np.random.default_rng(1)
    cols = ["XAU", "DXY", "USGGT10Y", "GVZ", "VIX", "SPX"]
    m = pd.DataFrame({f"{c}_PX_LAST": 100 + np.cumsum(rng.normal(0, 1, len(DAYS))) for c in cols}, index=DAYS)
    return m.iloc[:n]


def etf(n=len(DAYS)):
    return pd.DataFrame({"GLD_FUND_FLOW": np.arange(len(DAYS), dtype=float), "IAU_FUND_FLOW": 1.0,
                         "GLD_FUND_TOTAL_ASSETS": 1000.0, "IAU_FUND_TOTAL_ASSETS": 1000.0}, index=DAYS).iloc[:n]


def news(n=len(DAYS)):
    return pd.DataFrame({f"{t}_NEWS_SENTIMENT_DAILY_AVG": np.linspace(-1, 1, len(DAYS))
                         for t in ["NEM", "AEM", "KGC", "WPM", "FNV"]}, index=DAYS).iloc[:n]


def posts():
    times = pd.date_range(utc("2026-09-03 21:00"), utc("2026-10-02 19:00"), freq="5h")
    return pd.DataFrame({"post_id": [f"p{i}" for i in range(len(times))], "created_utc": times,
                         "sentiment": np.sin(np.arange(len(times)))})


class CutoffTests(unittest.TestCase):
    def test_post_after_cutoff_goes_to_next_trading_day(self):
        got = assign_days(pd.Series([utc("2026-09-08 19:59"), utc("2026-09-08 20:00"), utc("2026-09-08 20:30")]), DAYS)
        self.assertEqual(list(got), [pd.Timestamp("2026-09-08")] * 2 + [pd.Timestamp("2026-09-09")])

    def test_weekend_posts_roll_into_monday(self):
        got = assign_days(pd.Series([utc("2026-09-05 12:00"), utc("2026-09-06 23:00")]), DAYS)
        self.assertEqual(set(got), {pd.Timestamp("2026-09-07")})

    def test_first_day_and_after_last_cutoff_are_dropped(self):
        got = assign_days(pd.Series([utc("2026-09-01 00:00"), utc("2026-10-02 21:00")]), DAYS)
        self.assertTrue(got.isna().all())


class TargetAndLeakTests(unittest.TestCase):
    def setUp(self):
        self.f = build_daily(posts(), market(), etf(), news(), last_day=DAYS[-1])

    def test_target_is_next_close_vs_today(self):
        close = self.f["xau_close"]
        self.assertEqual(self.f["up"].iloc[3], float(close.iloc[4] > close.iloc[3]))
        self.assertFalse(self.f["target_valid"].iloc[-1])

    def test_etf_flow_is_lagged_one_day(self):
        pct = self.f["etf_flow_pct"]
        self.assertAlmostEqual(self.f["etf_flow_lag1"].iloc[5], pct.iloc[4])

    def test_monday_counts_are_per_24h(self):
        mon = self.f.loc["2026-09-07"]
        self.assertEqual(mon["window_hours"], 72)
        self.assertAlmostEqual(mon["posts_per_day"], mon["post_count"] / 3)

    def test_features_at_d_ignore_later_rows(self):
        cut = 15
        short = build_daily(posts(), market(cut), etf(cut), news(cut), last_day=DAYS[cut - 1])
        cols = DAILY_MARKET_FEATURES + DAILY_MOOD_FEATURES + DAILY_CHATTER_FEATURES
        pd.testing.assert_frame_equal(short[cols].iloc[:cut], self.f[cols].iloc[:cut])


if __name__ == "__main__":
    unittest.main()
