"""Backfills the hourly XAU/USD series from Yahoo Finance GC=F hourly bars.

The live price cron lost most of its runs (they shared a concurrency group
with the 15-minute collector, which kept cancelling them while they waited),
leaving ~5 ticks a day. Yahoo serves ~2 years of hourly history, so the gap
is recoverable -- this rewrites the window onto the same basis the live feed
now uses: the Open of each traded hour (see price/price_feed.py).

    python backfill_price.py --start 2026-09-03 --end now --dry-run
    python backfill_price.py --start 2026-09-03 --end now --prune-closed

Safe to re-run: unchanged hours are left alone.
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

from backfill import parse_when
from price.price_feed import sync_range
from storage.db import init_db


def main():
    parser = argparse.ArgumentParser(description="Backfill hourly XAU/USD (GC=F) prices.")
    parser.add_argument("--start", required=True, help="ISO date/datetime, e.g. 2026-09-03")
    parser.add_argument("--end", default="now", help="ISO date/datetime or 'now' (default)")
    parser.add_argument(
        "--prune-closed", action="store_true",
        help="delete stored hours that have no bar (market closed; the old "
             "feed stored the last close repeated for those)",
    )
    parser.add_argument("--dry-run", action="store_true", help="fetch only, store nothing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    start = datetime.fromtimestamp(parse_when(args.start), tz=timezone.utc)
    end = datetime.fromtimestamp(parse_when(args.end), tz=timezone.utc)
    if start >= end:
        parser.error("--start must be before --end")
    if args.end == "now":
        end += timedelta(hours=1)  # yfinance's end is exclusive; keep the current hour

    if not args.dry_run:
        init_db()

    stats = sync_range(start, end, prune_closed=args.prune_closed, dry_run=args.dry_run)
    verb = "would upsert" if args.dry_run else "upserted"
    print(
        f"\nPrice backfill: {stats['bars']} hourly bars; {verb} "
        f"(inserted={stats['inserted']} updated={stats['updated']} pruned={stats['pruned']})."
    )
    return 0 if stats["bars"] else 1


if __name__ == "__main__":
    sys.exit(main())
