"""Bloomberg terminal pull (extra_BBL/) -> tidy parquet files in data/bbg/.

    python -m processing.bloomberg

The workbooks and everything written here are git-ignored: terminal data is
licensed to us, not redistributable. Each sheet holds side-by-side blocks,
one per ticker: the ticker name above a `date` (or `bar start`) column, its
fields to the right, then a blank separator column. Every "#N/A ..." cell
becomes NaN.

Outputs (wide, one row per date, columns "<SHORT>_<FIELD>" e.g. XAU_PX_LAST):
  market_daily.parquet      3_Market_Daily  (trading days, 2022 -> 2026-10-05)
  etf_flows.parquet         5_ETF_Flows     (GLD, IAU, SLV)
  news_sentiment.parquet    2_Sentiment_Daily (miners only; no gold-level field exists)
  gold_news_weekly.parquet  gldstuf.xlsx, weekly count of gold news stories
  xau_spot_hourly.parquet   4_Intraday_1h XAU, as bucket_utc / xau_usd (price at HH:00 UTC),
                            the same shape as the DB's price_ticks
"""
import sqlite3
import sys

import numpy as np
import pandas as pd

import config

RAW_DIR = config.BASE_DIR / "extra_BBL"
PULL_PATH = RAW_DIR / "goldpulse_bloomberg_pull.xlsx"
GLDSTUF_PATH = RAW_DIR / "gldstuf.xlsx"
OUT_DIR = config.DATA_DIR / "bbg"

# The intraday sheet's time-zone cell was left at its placeholder. Matching
# hourly XAU returns against our GC=F price_ticks (pitfall #33) puts the
# terminal at UTC+7 (Bangkok): only that shift correlates (r ~ 0.84; every
# other shift ~ 0). A bar labelled L (local) closes at L+1h local, which is
# the price at L+1h-7h UTC.
TERMINAL_UTC_OFFSET = 7
BAR_CLOSE_SHIFT = pd.Timedelta(hours=1 - TERMINAL_UTC_OFFSET)

# (sheet, ticker row, field row, first data row), 0-based row indices
SHEETS = {
    "market_daily": ("3_Market_Daily", 2, 3, 4),
    "etf_flows": ("5_ETF_Flows", 2, 3, 4),
    "news_sentiment": ("2_Sentiment_Daily", 2, 3, 4),
    "intraday": ("4_Intraday_1h", 3, 4, 5),
}
DATE_HEADERS = {"date", "bar start"}
# Well covered in 2_Sentiment_Daily; GFI, AU and PAAS are mostly blank or 0.
NEWS_TICKERS = ["NEM", "AEM", "KGC", "WPM", "FNV"]


def short_name(ticker: str) -> str:
    return ticker.split()[0]


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else np.nan


def read_blocks(rows: list[tuple], ticker_row: int, field_row: int, first_data_row: int) -> pd.DataFrame:
    """Side-by-side ticker blocks -> long frame (date, ticker, field, value)."""
    tickers, fields = rows[ticker_row], rows[field_row]
    frames = []
    for j, header in enumerate(fields):
        if header not in DATE_HEADERS:
            continue
        ticker = tickers[j]
        k = j + 1
        while k < len(fields) and fields[k] not in (None, *DATE_HEADERS):
            dates = [r[j] if j < len(r) else None for r in rows[first_data_row:]]
            values = [_number(r[k]) if k < len(r) else np.nan for r in rows[first_data_row:]]
            frames.append(pd.DataFrame({"date": dates, "ticker": ticker, "field": fields[k], "value": values}))
            k += 1
    long = pd.concat(frames, ignore_index=True)
    long = long[pd.notna(long["date"])]
    long["date"] = pd.to_datetime(long["date"])
    return long


def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    long = long.assign(column=long["ticker"].map(short_name) + "_" + long["field"].str.split().str[0])
    wide = long.pivot_table(index="date", columns="column", values="value", aggfunc="first", dropna=False)
    wide = wide.dropna(how="all").sort_index()
    wide.columns.name = None
    return wide


def _rows(path, sheet) -> list[tuple]:
    import openpyxl  # only the parser needs it; notebooks import this module's constants

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return list(wb[sheet].iter_rows(values_only=True))
    finally:
        wb.close()


def load_sheet(key: str, path=PULL_PATH) -> pd.DataFrame:
    sheet, *rows_at = SHEETS[key]
    return to_wide(read_blocks(_rows(path, sheet), *rows_at))


def intraday_to_utc(local: pd.Series) -> pd.DataFrame:
    """Bar-close prices indexed by local bar start -> price_ticks shape (price at HH:00 UTC)."""
    local = local.dropna()
    utc = (local.index + BAR_CLOSE_SHIFT).tz_localize("UTC")
    return pd.DataFrame({"bucket_utc": utc, "xau_usd": local.to_numpy()})


def gold_news_weekly(path=GLDSTUF_PATH) -> pd.DataFrame:
    rows = _rows(path, "XAU Curncy NT <GO>")
    weekly = pd.DataFrame([(r[0], _number(r[1])) for r in rows[1:] if r[0] is not None],
                          columns=["week", "story_count"])
    weekly["week"] = pd.to_datetime(weekly["week"])
    return weekly.sort_values("week").reset_index(drop=True)


def shift_scan(spot: pd.DataFrame, reference: pd.DataFrame, hours=range(-12, 13)) -> pd.Series:
    """Correlation of hourly returns, spot shifted by h hours vs. reference, for each h."""
    ref = reference.set_index("bucket_utc")["xau_usd"]
    ref_ret = ref.pct_change(fill_method=None)
    out = {}
    for h in hours:
        s = spot.set_index("bucket_utc")["xau_usd"]
        s.index = s.index + pd.Timedelta(hours=h)
        joined = pd.concat([s.pct_change(fill_method=None), ref_ret], axis=1, join="inner").dropna()
        out[h] = joined.iloc[:, 0].corr(joined.iloc[:, 1])
    return pd.Series(out, name="return_corr")


def check_alignment(spot: pd.DataFrame, db_path=config.DB_PATH) -> pd.Series:
    """The converted spot series must line up with GC=F at shift 0, and nowhere else."""
    with sqlite3.connect(db_path) as conn:
        ref = pd.read_sql("SELECT bucket_utc, xau_usd FROM price_ticks", conn)
    ref["bucket_utc"] = pd.to_datetime(ref["bucket_utc"], utc=True, format="ISO8601")
    scan = shift_scan(spot, ref)
    assert scan.idxmax() == 0 and scan[0] > 0.7, f"spot/GC=F alignment off:\n{scan}"
    return scan


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for key in ["market_daily", "etf_flows", "news_sentiment"]:
        wide = load_sheet(key)
        wide.to_parquet(OUT_DIR / f"{key}.parquet")
        summary[key] = f"{len(wide)} rows, {wide.index.min():%Y-%m-%d} -> {wide.index.max():%Y-%m-%d}"
    intraday = load_sheet("intraday")
    spot = intraday_to_utc(intraday["XAU_LAST_PRICE"])
    spot.to_parquet(OUT_DIR / "xau_spot_hourly.parquet", index=False)
    scan = check_alignment(spot)
    summary["xau_spot_hourly"] = (f"{len(spot)} hours, {spot['bucket_utc'].min()} -> {spot['bucket_utc'].max()}; "
                                  f"corr with GC=F at shift 0 = {scan[0]:.2f}, next best {scan.drop(0).max():.2f}")
    weekly = gold_news_weekly()
    weekly.to_parquet(OUT_DIR / "gold_news_weekly.parquet", index=False)
    summary["gold_news_weekly"] = f"{len(weekly)} weeks, {weekly['week'].min():%Y-%m-%d} -> {weekly['week'].max():%Y-%m-%d}"
    return summary


def main():
    for key, value in run().items():
        print(f"{key:>16}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
