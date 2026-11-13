"""Economy-talk variant: backfill posts about inflation, interest rates and the
Fed from finance communities, into a separate file (never the gold DB).

    python -m collector.macro_backfill            # -> data/clean/macro_posts.parquet

Posts go through the same transform() as the gold collector (removed/deleted
and non-English posts dropped, same clean_text), then must match MACRO_RE.
The keyword list was fixed before any labels or results were seen. Kept posts
are de-duplicated with the Phase 3 rule and scored with FinBERT, whose
positive/negative here means good/bad news for the economy.
"""
import argparse
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config
from collector.arctic_shift_collector import fetch_posts, transform

SUBREDDITS = ["economy", "Economics", "wallstreetbets", "stocks", "CanadianInvestor"]
WINDOW = ("2026-09-08T00:00", "2026-10-05T16:00")  # same frozen window as the gold data, end exclusive
OUTPUT_PATH = config.DATA_DIR / "clean" / "macro_posts.parquet"
ECONOMY_LABELS = {"bullish": "good", "bearish": "bad", "neutral": "neutral"}

_CI_TERMS = (r"inflation(?:ary)?|deflation(?:ary)?|disinflation|stagflation|cpi|pce|"
             r"interest rates?|rate (?:cuts?|hikes?)|federal reserve|fomc|powell|"
             r"(?:treasury|bond) yields?|recession(?:ary)?")
_CI_RE = re.compile(rf"\b(?:{_CI_TERMS})\b", re.IGNORECASE)
# "Fed" only as the capitalised name, and never "Fed up" / "FED UP".
_FED_RE = re.compile(r"\b(?:Fed|FED)\b(?!\s+(?:up|Up|UP)\b)")


def matches_macro(text: str) -> bool:
    return bool(_CI_RE.search(text) or _FED_RE.search(text))


def _ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp())


def fetch_subreddit(subreddit: str, start: int, end: int) -> list[dict]:
    """Every submission in [start, end), oldest first, de-duplicated by id."""
    items, seen, after = [], set(), start
    while True:
        page = fetch_posts(subreddit, limit=config.BACKFILL_PAGE_LIMIT, after=after, before=end, sort="asc")
        new = [i for i in page if "id" in i and i["id"] not in seen]
        if not new:
            return items
        seen.update(i["id"] for i in new)
        items += new
        after = max(int(i["created_utc"]) for i in page if "created_utc" in i)
        print(f"\r  r/{subreddit}: {len(items)} posts", end="", flush=True)
        if len(page) < config.BACKFILL_PAGE_LIMIT:
            return items


def filter_items(items: list[dict], now: datetime) -> tuple[list[dict], Counter]:
    rows, stats = [], Counter()
    for item in items:
        stats["seen"] += 1
        row, reason = transform(item, now=now)
        if row is None:
            stats[reason] += 1
        elif not matches_macro(row["text"]):
            stats["no_macro_keyword"] += 1
        else:
            stats["kept"] += 1
            rows.append(row)
    return rows, stats


def run(output=OUTPUT_PATH, score=True) -> pd.DataFrame:
    from processing.bucket import dedupe_posts

    now = datetime.now(tz=timezone.utc)
    start, end = _ts(WINDOW[0]), _ts(WINDOW[1])
    rows = []
    for sub in SUBREDDITS:
        items = fetch_subreddit(sub, start, end)
        kept, stats = filter_items(items, now)
        print(f"\r  r/{sub}: {dict(stats)}")
        rows += kept
    posts = pd.DataFrame(rows).rename(columns={"sentiment": "vader"})
    posts["created_utc"] = pd.to_datetime(posts["created_utc"], utc=True, format="ISO8601")
    deduped = dedupe_posts(posts).reset_index(drop=True)
    print(f"macro posts kept: {len(posts)}, after de-duplication: {len(deduped)}")
    if score:
        from sentiment.finbert import score_texts
        deduped = pd.concat([deduped, score_texts(deduped["text"].tolist())], axis=1)
        # Here FinBERT's positive/negative reads as good/bad news for the economy.
        deduped["finbert_label"] = deduped["finbert_label"].map(ECONOMY_LABELS)
    output.parent.mkdir(parents=True, exist_ok=True)
    deduped.to_parquet(output, index=False)
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Backfill and score economy-talk posts.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help=f"default {OUTPUT_PATH}")
    args = parser.parse_args()
    posts = run(output=args.output)
    print(f"wrote {args.output} ({len(posts)} posts)")
    print(posts.groupby("subreddit")["finbert_label"].value_counts().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
