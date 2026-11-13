"""XAU/USD reference series: one row per hour the market traded.

Each bucket_utc row holds the Open of that hour's GC=F bar, i.e. the price at
HH:00. Every run re-reads the last few days of hourly bars and upserts them,
so a skipped or cancelled cron run is filled in by the next one that lands,
and hours the market was closed (weekends, the daily 21:00 UTC break) never
get a row at all.

The original feed stored "latest 1-minute close at poll time" under the
current hour instead. GitHub cron fired on average ~30 min late, so those
values were snapshots from random minutes, and on closed hours they were the
last close repeated. backfill_price.py rewrites that history onto this basis.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

import config
from storage.db import delete_price_ticks, select_price_buckets, upsert_price_ticks

logger = logging.getLogger("goldpulse.price")


def fetch_hourly_bars(start: datetime, end: datetime) -> pd.DataFrame:
    return yf.Ticker(config.PRICE_TICKER).history(start=start, end=end, interval="1h")


def bars_to_rows(bars: pd.DataFrame, collected_at: str) -> list[dict]:
    """Hourly OHLC bars -> price_ticks rows keyed by UTC hour. Pure, for tests."""
    if bars.empty:
        return []
    opens = bars["Open"].dropna()
    index = opens.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    index = index.tz_convert("UTC").floor("h")
    rows = {}
    for ts, value in zip(index, opens.to_numpy()):
        rows[ts.to_pydatetime().isoformat()] = {
            "bucket_utc": ts.to_pydatetime().isoformat(),
            "xau_usd": float(value),
            "collected_at": collected_at,
        }
    return list(rows.values())


def sync_range(
    start: datetime,
    end: datetime,
    *,
    prune_closed: bool = False,
    dry_run: bool = False,
    now: Optional[datetime] = None,
) -> dict:
    """Upserts every hourly bar in [start, end). With prune_closed, also drops
    stored rows inside the covered span that have no bar (market closed)."""
    now = now or datetime.now(tz=timezone.utc)
    rows = bars_to_rows(fetch_hourly_bars(start, end), now.isoformat())
    stats = {"bars": len(rows), "inserted": 0, "updated": 0, "pruned": 0}
    if not rows:
        logger.warning("no hourly bars for %s in [%s, %s)", config.PRICE_TICKER, start, end)
        return stats

    if dry_run:
        return stats

    stats["inserted"], stats["updated"] = upsert_price_ticks(rows)

    if prune_closed:
        # Only inside the span the bars actually cover: an empty or truncated
        # response must never be read as "the market was closed".
        bar_buckets = {row["bucket_utc"] for row in rows}
        first, last = min(bar_buckets), max(bar_buckets)
        stored = select_price_buckets(first, last)
        closed = sorted(set(stored) - bar_buckets)
        stats["pruned"] = delete_price_ticks(closed)

    return stats


def collect_once(now: Optional[datetime] = None) -> bool:
    """Live cron entry point: re-syncs the trailing lookback window."""
    now = now or datetime.now(tz=timezone.utc)
    start = now - timedelta(days=config.PRICE_LOOKBACK_DAYS)
    stats = sync_range(start, now + timedelta(hours=1), now=now)
    logger.info(
        "price sync: bars=%d inserted=%d updated=%d",
        stats["bars"], stats["inserted"], stats["updated"],
    )
    return stats["bars"] > 0
