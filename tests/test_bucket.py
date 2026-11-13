"""Offline tests for processing/bucket.py's cleaning and merge rules."""
import unittest

import pandas as pd

import tempfile
from pathlib import Path

from processing.bucket import build_hourly, dedupe_posts, load_price_file, normalize_text, select_posts

T0 = pd.Timestamp("2026-09-08 00:00", tz="UTC")


def posts(*rows):
    """rows: (minutes after T0, text, sentiment, engagement, stale)."""
    return pd.DataFrame(
        [
            {"post_id": f"p{i}", "created_utc": T0 + pd.Timedelta(minutes=m), "text": text,
             "sentiment": s, "engagement": e, "engagement_stale": stale}
            for i, (m, text, s, e, stale) in enumerate(rows)
        ]
    )


def prices(*hours):
    return pd.DataFrame({"bucket_utc": [T0 + pd.Timedelta(hours=h) for h in hours],
                         "xau_usd": [4000.0 + h for h in hours]})


class DedupeTests(unittest.TestCase):
    def test_normalization_ignores_case_punctuation_and_urls(self):
        self.assertEqual(normalize_text("Buy the DIP, they said! https://i.redd.it/x.jpg"),
                         "buy the dip they said")

    def test_cross_post_minutes_apart_is_dropped_keeping_earliest(self):
        kept = dedupe_posts(posts((5, "Same story", 0, 0, 0), (1, "same story!", 0, 0, 0)))
        self.assertEqual(kept["post_id"].tolist(), ["p1"])

    def test_same_generic_text_days_apart_is_kept(self):
        kept = dedupe_posts(posts((0, "Any info?", 0, 0, 0), (3 * 24 * 60, "Any info?", 0, 0, 0)))
        self.assertEqual(len(kept), 2)


class ExcludeSubredditTests(unittest.TestCase):
    def test_excluded_posts_vanish_and_cross_post_keeps_surviving_copy(self):
        rows = posts((1, "Same story", 0, 0, 0), (5, "same story!", 0, 0, 0), (9, "other", 0, 0, 0))
        rows["subreddit"] = ["Silverbugs", "Gold", "Silverbugs"]
        selected = select_posts(rows, T0, T0, exclude_subreddits=("Silverbugs",))
        kept = dedupe_posts(selected)
        self.assertEqual(kept["post_id"].tolist(), ["p1"])  # the Gold copy, not dropped as a repeat
        self.assertEqual(dedupe_posts(select_posts(rows, T0, T0))["post_id"].tolist(), ["p0", "p2"])


class BuildHourlyTests(unittest.TestCase):
    def build(self, post_rows, price_hours, hours=4):
        return build_hourly(posts(*post_rows), prices(*price_hours),
                            T0, T0 + pd.Timedelta(hours=hours - 1))

    def test_every_hour_is_present(self):
        hourly = self.build([(10, "a", 0.5, 1, 0)], [], hours=5)
        self.assertEqual(len(hourly), 5)

    def test_aggregates_and_ratios(self):
        hourly = self.build([(1, "a", 0.6, 10, 0), (2, "b", -0.4, 20, 0), (3, "c", 0.0, 30, 0)], [])
        row = hourly.iloc[0]
        self.assertEqual(row["post_count"], 3)
        self.assertAlmostEqual(row["mean_sentiment"], 0.2 / 3)
        self.assertAlmostEqual(row["bullish_ratio"], 1 / 3)
        self.assertAlmostEqual(row["bearish_ratio"], 1 / 3)

    def test_engagement_uses_settled_rows_only(self):
        hourly = self.build([(1, "a", 0, 100, 0), (2, "b", 0, 0, 1)], [])
        self.assertEqual(hourly.iloc[0]["mean_engagement"], 100)
        self.assertEqual(hourly.iloc[0]["stale_share"], 0.5)

    def test_sentiment_forward_filled_never_back_filled(self):
        hourly = self.build([(70, "a", 0.8, 1, 0)], [])  # only hour 1 has a post
        self.assertTrue(pd.isna(hourly.iloc[0]["mean_sentiment"]))  # no back-fill
        self.assertEqual(hourly.iloc[2]["mean_sentiment"], 0.8)      # forward-filled
        self.assertEqual(hourly.iloc[2]["post_count"], 0)
        self.assertEqual(hourly["had_posts"].tolist(), [False, True, False, False])

    def test_price_is_never_filled(self):
        hourly = self.build([(1, "a", 0, 1, 0)], [0, 2])
        self.assertEqual(hourly["market_open"].tolist(), [True, False, True, False])
        self.assertTrue(pd.isna(hourly.iloc[1]["xau_usd"]))


class PriceFileTests(unittest.TestCase):
    def test_price_file_feeds_build_hourly_like_price_ticks(self):
        spot = prices(0, 1)
        spot["bucket_utc"] = spot["bucket_utc"].dt.tz_localize(None).dt.tz_localize("UTC")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "spot.parquet"
            spot.assign(extra=1).to_parquet(path, index=False)
            loaded = load_price_file(path)
        self.assertEqual(loaded.columns.tolist(), ["bucket_utc", "xau_usd"])
        hourly = build_hourly(posts((1, "a", 0, 1, 0)), loaded, T0, T0 + pd.Timedelta(hours=2))
        self.assertEqual(hourly["xau_usd"].tolist()[:2], [4000.0, 4001.0])


if __name__ == "__main__":
    unittest.main()
