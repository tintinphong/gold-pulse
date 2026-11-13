"""Offline tests for the hourly price series: bar -> row transform, upsert, prune.

fetch_hourly_bars is patched out, and storage points at a temp database, so
nothing here touches Yahoo or the committed data/goldpulse.sqlite3.
"""
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pandas as pd

import storage.db as db
from price import price_feed

NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


def bars(*hours_utc, tz="America/New_York"):
    """Hourly bars with Open = 4000 + hour index, indexed like yfinance returns."""
    index = pd.DatetimeIndex(pd.to_datetime(list(hours_utc), utc=True)).tz_convert(tz)
    opens = [4000.0 + i for i in range(len(index))]
    return pd.DataFrame({"Open": opens, "Close": [o + 0.5 for o in opens]}, index=index)


class BarsToRowsTests(unittest.TestCase):
    def test_keys_by_utc_hour_and_uses_open(self):
        rows = price_feed.bars_to_rows(bars("2026-09-08 13:00", "2026-09-08 14:00"), "x")
        self.assertEqual(
            [(r["bucket_utc"], r["xau_usd"]) for r in rows],
            [("2026-09-08T13:00:00+00:00", 4000.0), ("2026-09-08T14:00:00+00:00", 4001.0)],
        )

    def test_empty_and_nan_bars_give_no_rows(self):
        self.assertEqual(price_feed.bars_to_rows(pd.DataFrame(), "x"), [])
        frame = bars("2026-09-08 13:00")
        frame["Open"] = float("nan")
        self.assertEqual(price_feed.bars_to_rows(frame, "x"), [])


class SyncRangeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name)
        self.patches = [
            mock.patch.object(db, "DATA_DIR", path),
            mock.patch.object(db, "DB_PATH", path / "test.sqlite3"),
        ]
        for p in self.patches:
            p.start()
        db.init_db()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def stored(self):
        with db.get_conn() as conn:
            return dict(conn.execute("SELECT bucket_utc, xau_usd FROM price_ticks ORDER BY 1"))

    def sync(self, frame, **kwargs):
        with mock.patch.object(price_feed, "fetch_hourly_bars", return_value=frame):
            return price_feed.sync_range(NOW, NOW, now=NOW, **kwargs)

    def test_fills_gaps_and_corrects_old_snapshot(self):
        # An old-style mid-hour snapshot for 13:00 that disagrees with the bar.
        db.upsert_price_ticks([{"bucket_utc": "2026-09-08T13:00:00+00:00",
                                "xau_usd": 3999.0, "collected_at": "old"}])
        stats = self.sync(bars("2026-09-08 13:00", "2026-09-08 14:00"))
        self.assertEqual((stats["inserted"], stats["updated"]), (1, 1))
        self.assertEqual(list(self.stored().values()), [4000.0, 4001.0])

    def test_resync_is_a_no_op(self):
        frame = bars("2026-09-08 13:00", "2026-09-08 14:00")
        self.sync(frame)
        stats = self.sync(frame)
        self.assertEqual((stats["inserted"], stats["updated"]), (0, 0))

    def test_prune_drops_closed_hours_only_inside_bar_span(self):
        for hour in ("12", "15", "17"):  # 15:00 has no bar; 12 and 17 are outside the span
            db.upsert_price_ticks([{"bucket_utc": f"2026-09-08T{hour}:00:00+00:00",
                                    "xau_usd": 1.0, "collected_at": "old"}])
        stats = self.sync(bars("2026-09-08 13:00", "2026-09-08 14:00", "2026-09-08 16:00"),
                          prune_closed=True)
        self.assertEqual(stats["pruned"], 1)
        self.assertNotIn("2026-09-08T15:00:00+00:00", self.stored())
        self.assertIn("2026-09-08T12:00:00+00:00", self.stored())
        self.assertIn("2026-09-08T17:00:00+00:00", self.stored())

    def test_empty_response_writes_and_prunes_nothing(self):
        db.upsert_price_ticks([{"bucket_utc": "2026-09-08T13:00:00+00:00",
                                "xau_usd": 1.0, "collected_at": "old"}])
        stats = self.sync(pd.DataFrame(), prune_closed=True)
        self.assertEqual(stats["pruned"], 0)
        self.assertEqual(len(self.stored()), 1)


if __name__ == "__main__":
    unittest.main()
