"""Long daily study: backfill r/Gold, r/Silverbugs and r/investing from Dec 2021,
into separate files (never the live DB).

    python -m collector.history_backfill              # fetch every missing month (resumable)
    python -m collector.history_backfill --combine    # -> data/clean/history_posts.parquet

Each subreddit-month is fetched once and saved as
data/clean/history/{subreddit}_{YYYY-MM}.parquet, so an interrupted run picks
up where it stopped. Posts go through the live collector's transform(): same
removal and language rules, same clean_text, same VADER score, and r/investing
still has to match config.KEYWORDS. History is long settled, so engagement is
real (engagement_stale = 0).
"""
import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config
from collector.arctic_shift_collector import transform
from collector.macro_backfill import fetch_subreddit

SUBREDDITS = config.FULL_COVERAGE_SUBREDDITS + config.KEYWORD_FILTERED_SUBREDDITS
WINDOW = ("2021-12-01T00:00", "2026-10-05T16:00")  # a month of warm-up before Jan 2022; end exclusive
MONTH_DIR = config.DATA_DIR / "clean" / "history"
OUTPUT_PATH = config.DATA_DIR / "clean" / "history_posts.parquet"
STATS_PATH = MONTH_DIR / "_stats.csv"
STATS_COLUMNS = ["subreddit", "month", "seen", "ok", "removed", "no_keyword", "lang", "malformed"]
COLUMNS = ["post_id", "created_utc", "subreddit", "text", "score", "num_comments",
           "lang", "sentiment", "engagement", "engagement_stale"]


def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)


def months(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """[month_start, month_end) pairs covering [start, end), clipped to it."""
    edges = pd.date_range(start.replace(day=1), end, freq="MS", tz="UTC").to_pydatetime().tolist()
    edges = [max(e, start) for e in edges] + [end]
    return [(a, b) for a, b in zip(edges, edges[1:]) if a < b]


def month_path(subreddit: str, month_start: datetime) -> Path:
    return MONTH_DIR / f"{subreddit}_{month_start:%Y-%m}.parquet"


def filter_items(items: list[dict], needs_keyword_match: bool, now: datetime) -> tuple[list[dict], Counter]:
    rows, stats = [], Counter()
    for item in items:
        stats["seen"] += 1
        try:
            row, reason = transform(item, needs_keyword_match=needs_keyword_match, now=now)
        except (KeyError, TypeError, ValueError, OSError):
            stats["malformed"] += 1
            continue
        stats[reason] += 1
        if row is not None:
            rows.append(row)
    return rows, stats


def fetch_month(subreddit: str, start: datetime, end: datetime, now: datetime) -> Counter:
    items = fetch_subreddit(subreddit, int(start.timestamp()), int(end.timestamp()))
    rows, stats = filter_items(items, subreddit in config.KEYWORD_FILTERED_SUBREDDITS, now)
    posts = pd.DataFrame(rows, columns=COLUMNS)
    path = month_path(subreddit, start)
    tmp = path.with_suffix(".tmp")
    posts.to_parquet(tmp, index=False)
    tmp.replace(path)  # only a finished month ever gets its final name
    return stats


def fetch_all(window=WINDOW) -> None:
    MONTH_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=timezone.utc)
    for subreddit in SUBREDDITS:
        for start, end in months(_utc(window[0]), _utc(window[1])):
            if month_path(subreddit, start).exists():
                continue
            stats = fetch_month(subreddit, start, end, now)
            print(f"\r  r/{subreddit} {start:%Y-%m}: {dict(stats)}", flush=True)
            row = pd.DataFrame([{"subreddit": subreddit, "month": f"{start:%Y-%m}", **stats}])
            # Fixed column order: Counter keys come in first-seen order, which differs by month.
            row.reindex(columns=STATS_COLUMNS, fill_value=0).to_csv(
                STATS_PATH, mode="a", header=not STATS_PATH.exists(), index=False)


def combine(output=OUTPUT_PATH) -> pd.DataFrame:
    parts = sorted(MONTH_DIR.glob("*.parquet"))
    posts = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    posts = posts.drop_duplicates("post_id").sort_values("created_utc").reset_index(drop=True)
    posts.to_parquet(output, index=False)
    return posts


def main():
    parser = argparse.ArgumentParser(description="Backfill Reddit history for the long daily study.")
    parser.add_argument("--combine", action="store_true", help=f"merge the monthly files into {OUTPUT_PATH}")
    args = parser.parse_args()
    if args.combine:
        posts = combine()
        print(f"wrote {OUTPUT_PATH} ({len(posts)} posts)")
        print(posts.groupby("subreddit").size().to_string())
    else:
        fetch_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
