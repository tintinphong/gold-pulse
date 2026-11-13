"""Backfills historical posts from Arctic Shift into data/goldpulse.sqlite3.

This is what the old PRAW collector could never do: .new() only ever showed
the most recent page, so a gap in collection was gone for good. Arctic Shift
serves history, so a missed window can be recovered after the fact.

    python backfill.py --start 2026-09-08 --end now
    python backfill.py --start 2026-09-08 --end now --subreddits Gold --dry-run

Safe to re-run: post_id is the primary key and inserts use INSERT OR IGNORE,
so overlapping runs just skip what's already stored.
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone

import requests

import config
from collector.arctic_shift_collector import fetch_posts, ingest, new_stats
from storage.db import init_db

logger = logging.getLogger("goldpulse.backfill")

_DRY_RUN_SAMPLES = 3


def parse_when(value: str) -> int:
    """Accepts 'now' or an ISO date/datetime, returns unix seconds (UTC)."""
    if value == "now":
        return int(datetime.now(tz=timezone.utc).timestamp())
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def backfill_subreddit(
    subreddit: str,
    start_ts: int,
    end_ts: int,
    *,
    needs_keyword_match: bool,
    limit: int = 100,
    pause: float = 1.0,
    dry_run: bool = False,
) -> dict:
    """Walks a subreddit's history backwards from end_ts to start_ts."""
    stats = new_stats()
    stats["pages"] = 0
    cursor = end_ts
    seen_ids: set[str] = set()

    while cursor > start_ts:
        items = fetch_posts(
            subreddit, limit=limit, before=cursor, after=start_ts, sort="desc"
        )
        stats["pages"] += 1
        if not items:
            break

        # ingest() tolerates malformed items, so the paging logic has to as
        # well -- one item missing an id or timestamp must not end the walk.
        page_ids = {item["id"] for item in items if "id" in item}
        if page_ids and page_ids <= seen_ids:
            # Cursor stopped advancing (every id already seen) -- bail out
            # rather than page over the same window forever.
            logger.warning("r/%s: cursor stalled at %d, stopping", subreddit, cursor)
            break
        seen_ids |= page_ids

        before = stats["inserted"]
        accepted = ingest(
            items, needs_keyword_match=needs_keyword_match, stats=stats, store=not dry_run
        )
        if dry_run and before < _DRY_RUN_SAMPLES:
            for row in accepted[:_DRY_RUN_SAMPLES - before]:
                print(f"  sample: [{row['created_utc']}] r/{row['subreddit']} "
                      f"sentiment={row['sentiment']:+.3f} "
                      f"stale={row['engagement_stale']} :: {row['text'][:90]}")

        timestamps = [item["created_utc"] for item in items if "created_utc" in item]
        if not timestamps:
            logger.warning("r/%s: page had no usable timestamps, stopping", subreddit)
            break
        oldest = min(timestamps)
        if len(items) < limit or oldest <= start_ts:
            break
        # +1 so posts sharing the oldest timestamp are re-requested rather
        # than skipped; INSERT OR IGNORE absorbs the duplicates, whereas
        # stepping past a tie would silently drop posts.
        cursor = oldest + 1
        time.sleep(pause)

    logger.info(
        "r/%s done: pages=%d seen=%d inserted=%d skipped_lang=%d "
        "skipped_removed=%d skipped_keyword=%d",
        subreddit, stats["pages"], stats["seen"], stats["inserted"],
        stats["skipped_lang"], stats["skipped_removed"], stats["skipped_keyword"],
    )
    return stats


def main():
    parser = argparse.ArgumentParser(description="Backfill Reddit posts from Arctic Shift.")
    parser.add_argument("--start", required=True, help="ISO date/datetime, e.g. 2026-09-08")
    parser.add_argument("--end", default="now", help="ISO date/datetime or 'now' (default)")
    parser.add_argument("--subreddits", nargs="*", help="defaults to the configured set")
    parser.add_argument("--limit", type=int, default=config.BACKFILL_PAGE_LIMIT)
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between pages")
    parser.add_argument("--dry-run", action="store_true", help="fetch and score, store nothing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    start_ts = parse_when(args.start)
    end_ts = parse_when(args.end)
    if start_ts >= end_ts:
        parser.error("--start must be before --end")

    # Asking for more than the API serves per page would make every page look
    # short, which the paging loop reads as "history exhausted" after one page.
    limit = min(args.limit, config.BACKFILL_PAGE_LIMIT)
    if args.limit > limit:
        logger.warning("--limit %d exceeds the API page cap, using %d", args.limit, limit)

    configured = config.FULL_COVERAGE_SUBREDDITS + config.KEYWORD_FILTERED_SUBREDDITS
    subreddits = args.subreddits or configured

    if not args.dry_run:
        init_db()

    totals = {"seen": 0, "inserted": 0}
    for subreddit in subreddits:
        try:
            stats = backfill_subreddit(
                subreddit,
                start_ts,
                end_ts,
                needs_keyword_match=subreddit in config.KEYWORD_FILTERED_SUBREDDITS,
                limit=limit,
                pause=args.pause,
                dry_run=args.dry_run,
            )
        except (requests.RequestException, OSError, ValueError, KeyError):
            # Keep going: rows already inserted for this subreddit are kept,
            # and the remaining subreddits still get their turn.
            logger.exception("backfill failed for r/%s", subreddit)
            continue
        totals["seen"] += stats["seen"]
        totals["inserted"] += stats["inserted"]

    verb = "would store" if args.dry_run else "stored"
    print(f"\nBackfill complete: saw {totals['seen']} posts, {verb} {totals['inserted']}.")
    if args.dry_run:
        print("Dry run -- nothing was written. Re-run without --dry-run to store.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
