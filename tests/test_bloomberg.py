"""Offline tests for processing/bloomberg.py: block parsing and the time-zone shift."""
import unittest

import numpy as np
import pandas as pd

from processing.bloomberg import intraday_to_utc, read_blocks, shift_scan, to_wide

D1, D2 = pd.Timestamp("2026-09-01").to_pydatetime(), pd.Timestamp("2026-09-02").to_pydatetime()
ROWS = [
    ("title", None, None, None, None, None),
    (None,) * 6,
    ("XAU Curncy", None, None, "DXY Curncy", None, None),
    ("date", "PX_LAST", None, "date", "PX_LAST", "PX_VOLUME"),
    (D1, 4000.5, None, D1, 99.1, "#N/A N/A"),
    (D2, "#N/A N/A", None, D2, 99.3, 12),
]


class BlockTests(unittest.TestCase):
    def test_blocks_become_one_column_per_ticker_field(self):
        wide = to_wide(read_blocks(ROWS, 2, 3, 4))
        self.assertEqual(sorted(wide.columns), ["DXY_PX_LAST", "DXY_PX_VOLUME", "XAU_PX_LAST"])
        self.assertEqual(wide.loc[D1, "XAU_PX_LAST"], 4000.5)
        self.assertEqual(wide.loc[D2, "DXY_PX_VOLUME"], 12)

    def test_na_strings_become_nan(self):
        wide = to_wide(read_blocks(ROWS, 2, 3, 4))
        self.assertTrue(np.isnan(wide.loc[D2, "XAU_PX_LAST"]))
        self.assertTrue(np.isnan(wide.loc[D1, "DXY_PX_VOLUME"]))


class TimeZoneTests(unittest.TestCase):
    def test_bangkok_bar_close_becomes_utc_price_at_hour(self):
        # The bar starting 10:00 Bangkok closes at 11:00 Bangkok = 04:00 UTC.
        local = pd.Series([1.0], index=pd.DatetimeIndex([pd.Timestamp("2026-09-01 10:00")]))
        out = intraday_to_utc(local)
        self.assertEqual(out["bucket_utc"].iloc[0], pd.Timestamp("2026-09-01 04:00", tz="UTC"))

    def test_shift_scan_peaks_where_series_align(self):
        grid = pd.date_range("2026-09-01", periods=200, freq="h", tz="UTC")
        price = 4000 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 1e-3, 200)))
        ref = pd.DataFrame({"bucket_utc": grid, "xau_usd": price})
        scan = shift_scan(ref, ref, hours=range(-3, 4))
        self.assertEqual(scan.idxmax(), 0)
        self.assertAlmostEqual(scan[0], 1.0)


if __name__ == "__main__":
    unittest.main()
