"""Phase 3: raw posts + price ticks -> one clean row per UTC hour.

    python -m processing.bucket                       # Sept 8 -> last fully polled hour
    python -m processing.bucket --start 2026-09-08 --end 2026-09-24T23:00

Writes data/clean/hourly_merged.parquet. Rules, all decided here so the
notebook, Track B and the report describe the same thing:

* Grid. One row per UTC hour from --start to --end, every hour present. A
  row is never dropped for lack of posts or lack of a price.
* De-duplication. Posts whose normalized text (or its text_hash) (lowercased, URLs and
  punctuation stripped) matches an earlier post within DEDUP_WINDOW are
  dropped, keeping the earliest. That removes double-posts and the same story
  cross-posted to another subreddit minutes later, but keeps generic titles
  like "Any info?" that different people post days apart.
* Sentiment. bullish_ratio / bearish_ratio use VADER's usual +/-0.05
  neutral band. With --sentiment-file (e.g. FinBERT scores from
  sentiment/finbert.py) the stored VADER score is swapped for that file's
  score and the ratios come from its own label column instead; the DB is
  never changed.
* Engagement. mean_engagement is over settled rows only
  (engagement_stale = 0): Arctic Shift reports placeholder counts until it
  re-crawls a post, which would drag the mean toward zero. stale_share
  records how much of each hour was excluded.
* Sparse hours. mean_sentiment, bullish_ratio and bearish_ratio are
  forward-filled across hours with no posts, never back-filled (that would
  leak the future into the past). post_count is 0 there, not filled, and
  had_posts marks which rows are real.
* Price. xau_usd is the hour's GC=F Open, present only when the market
  traded (market_open). It is never filled: a carried-forward price on a
  closed hour would read as "no change" to a direction-of-move target.
  With --price-file (e.g. Bloomberg spot from processing/bloomberg.py, same
  bucket_utc / xau_usd shape) that file replaces price_ticks.
"""
import argparse
import hashlib
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

import config

WINDOW_START = "2026-09-08"  # collection window opens (docs/COLLECTION_LOG.md)
DEDUP_WINDOW = pd.Timedelta(hours=24)
NEUTRAL_BAND = 0.05
OUTPUT_PATH = config.DATA_DIR / "clean" / "hourly_merged.parquet"

SENTIMENT_COLUMNS = ["mean_sentiment", "bullish_ratio", "bearish_ratio"]

_URL = re.compile(r"https?://\S+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def load_raw(db_path: Path = config.DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(db_path) as conn:
        posts = pd.read_sql("SELECT * FROM posts", conn)
        prices = pd.read_sql("SELECT * FROM price_ticks", conn)
    posts["created_utc"] = pd.to_datetime(posts["created_utc"], utc=True, format="ISO8601")
    posts["collected_at"] = pd.to_datetime(posts["collected_at"], utc=True, format="ISO8601")
    prices["bucket_utc"] = pd.to_datetime(prices["bucket_utc"], utc=True, format="ISO8601")
    return posts, prices


def apply_sentiment_file(posts: pd.DataFrame, scores: pd.DataFrame,
                         score_col: str = "finbert_score",
                         label_col: str = "finbert_label") -> pd.DataFrame:
    """Replaces each post's sentiment with an alternative scorer's score and
    adds its bullish/neutral/bearish `label`. Every post must be scored, so the
    two scales are never mixed in one hourly mean."""
    alt = scores.set_index("post_id")[[score_col, label_col]]
    missing = ~posts["post_id"].isin(alt.index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} posts have no score in the sentiment file")
    return posts.assign(sentiment=posts["post_id"].map(alt[score_col]),
                        label=posts["post_id"].map(alt[label_col]))


def normalize_text(text: str) -> str:
    text = _URL.sub(" ", text.lower())
    return " ".join(_NON_ALNUM.sub(" ", text).split())


def text_hash(text: str) -> str:
    """SHA-1 of the normalized text. The published data stores this instead of
    the post text, and de-duplication gives the same result on either."""
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def dedupe_posts(posts: pd.DataFrame, window: pd.Timedelta = DEDUP_WINDOW) -> pd.DataFrame:
    """Drops repeats of the same normalized text within `window` of the
    previous kept copy; keeps the earliest. Uses `text` when present, else
    the `text_hash` column of the published data."""
    key = posts["text"].map(text_hash) if "text" in posts else posts["text_hash"]
    posts = posts.assign(_norm=key).sort_values("created_utc")
    keep = []
    last_kept: dict[str, pd.Timestamp] = {}
    for norm, created in zip(posts["_norm"], posts["created_utc"]):
        previous = last_kept.get(norm)
        if previous is not None and created - previous <= window:
            keep.append(False)
            continue
        last_kept[norm] = created
        keep.append(True)
    return posts[keep].drop(columns="_norm")


def aggregate_hourly(posts: pd.DataFrame) -> pd.DataFrame:
    if "label" in posts:  # alternative scorer: use its own class decision
        bullish, bearish = posts["label"] == "bullish", posts["label"] == "bearish"
    else:
        bullish, bearish = posts["sentiment"] > NEUTRAL_BAND, posts["sentiment"] < -NEUTRAL_BAND
    posts = posts.assign(
        bucket_utc=posts["created_utc"].dt.floor("h"),
        bullish=bullish,
        bearish=bearish,
        settled_engagement=posts["engagement"].where(posts["engagement_stale"] == 0),
    )
    return posts.groupby("bucket_utc").agg(
        post_count=("post_id", "size"),
        mean_sentiment=("sentiment", "mean"),
        bullish_ratio=("bullish", "mean"),
        bearish_ratio=("bearish", "mean"),
        mean_engagement=("settled_engagement", "mean"),  # NaN-skipping: settled only
        stale_share=("engagement_stale", "mean"),
    )


def build_hourly(
    posts: pd.DataFrame, prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    grid = pd.date_range(start, end, freq="h", tz="UTC", name="bucket_utc")
    hourly = aggregate_hourly(posts).reindex(grid)
    hourly["had_posts"] = hourly["post_count"].notna()
    hourly["post_count"] = hourly["post_count"].fillna(0).astype(int)
    hourly[SENTIMENT_COLUMNS] = hourly[SENTIMENT_COLUMNS].ffill()

    price = prices.set_index("bucket_utc")["xau_usd"]
    hourly["xau_usd"] = price.reindex(grid)
    hourly["market_open"] = hourly["xau_usd"].notna()
    return hourly


def default_end(posts: pd.DataFrame) -> pd.Timestamp:
    """Last hour a collect run fully covered: the one before the latest poll."""
    return posts["collected_at"].max().floor("h") - pd.Timedelta(hours=1)


def _utc_hour(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    return ts.floor("h")


def select_posts(posts: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                 exclude_subreddits=()) -> pd.DataFrame:
    """Posts in [start, end + 1h), minus excluded subreddits.

    Exclusion happens before de-duplication, so a story cross-posted to an
    excluded and a kept community keeps the kept community's copy, exactly as
    if the excluded one had never been collected."""
    in_window = posts[(posts["created_utc"] >= start)
                      & (posts["created_utc"] < end + pd.Timedelta(hours=1))]
    return in_window[~in_window["subreddit"].isin(exclude_subreddits)]


def load_price_file(path) -> pd.DataFrame:
    prices = pd.read_parquet(path)[["bucket_utc", "xau_usd"]]
    prices["bucket_utc"] = pd.to_datetime(prices["bucket_utc"], utc=True)
    return prices


def run(start=WINDOW_START, end=None, db_path=config.DB_PATH, output=OUTPUT_PATH,
        exclude_subreddits=(), sentiment_file=None, price_file=None):
    posts, prices = load_raw(db_path)
    if price_file is not None:
        prices = load_price_file(price_file)
    if sentiment_file is not None:
        posts = apply_sentiment_file(posts, pd.read_parquet(sentiment_file))
    start = _utc_hour(start)
    end = default_end(posts) if end is None else _utc_hour(end)

    in_window = select_posts(posts, start, end)
    selected = select_posts(posts, start, end, exclude_subreddits)
    deduped = dedupe_posts(selected)
    hourly = build_hourly(deduped, prices, start, end)

    output.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(output)
    summary = {
        "start": start, "end": end, "hours": len(hourly),
        "posts_in_window": len(in_window),
        "excluded_subreddits": ", ".join(exclude_subreddits) or "none",
        "sentiment": sentiment_file or "VADER (stored in DB)",
        "prices": price_file or "GC=F price_ticks (DB)",
        "posts_excluded": len(in_window) - len(selected),
        "duplicates_dropped": len(selected) - len(deduped),
        "posts_kept": len(deduped),
        "hours_with_posts": int(hourly["had_posts"].sum()),
        "market_open_hours": int(hourly["market_open"].sum()),
        "output": output,
    }
    return hourly, summary


def main():
    parser = argparse.ArgumentParser(description="Bucket posts to the hour and merge with prices.")
    parser.add_argument("--start", default=WINDOW_START, help=f"ISO, default {WINDOW_START}")
    parser.add_argument("--end", default=None, help="ISO hour; default last fully polled hour")
    parser.add_argument("--exclude-subreddits", nargs="*", default=[], metavar="NAME",
                        help="drop these subreddits before de-duplication (variant datasets)")
    parser.add_argument("--sentiment-file", type=Path, default=None, metavar="PARQUET",
                        help="post_id -> finbert_score/finbert_label; replaces the stored VADER score")
    parser.add_argument("--price-file", type=Path, default=None, metavar="PARQUET",
                        help="bucket_utc -> xau_usd; replaces the DB's GC=F price_ticks")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help=f"default {OUTPUT_PATH}")
    args = parser.parse_args()
    _, summary = run(args.start, args.end, output=args.output,
                     exclude_subreddits=tuple(args.exclude_subreddits),
                     sentiment_file=args.sentiment_file, price_file=args.price_file)
    for key, value in summary.items():
        print(f"{key:>20}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
