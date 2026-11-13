"""The GoldPulse Index: a daily read of how much, and how warmly, Reddit talks about gold.

    python -m processing.index                    # -> data/clean/goldpulse_index.parquet
    python -m processing.index --refresh-prices   # also re-download daily GC=F from Yahoo

Built only from our own Reddit data and free Yahoo prices (no Bloomberg), so
it can be committed and shown on the public dashboard. Rules (docs/PRODUCT.md):

* Day. A UTC calendar day; only complete days are published.
* Posts. data/clean/history_posts.parquet plus any newer posts in the live DB,
  de-duplicated with the Phase 3 rule (processing/bucket.py:dedupe_posts).
* Attention score (headline). The 7-day average of posts per day divided by
  the average of the 90 days before that week (so the community's growth
  since 2022 doesn't read as permanent excitement), then ranked against the
  trailing 365 days: 0 = the quietest week relative to normal in the past
  year, 100 = the most unusual surge. Needs no sentiment scorer.
* Mood score (experimental). The 7-day post-weighted mean VADER score, ranked
  the same way. VADER failed our hand-label check, so this is labelled
  experimental everywhere it appears.
* Trailing only. A day's scores use that day and earlier, never later ones,
  and need at least MIN_HISTORY_DAYS of history.
* It describes the crowd; it is not a price forecast (notebooks 04 and 06).
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from processing.bucket import NEUTRAL_BAND, dedupe_posts

HISTORY_PATH = config.DATA_DIR / "clean" / "history_posts.parquet"
PRICES_PATH = config.DATA_DIR / "clean" / "gold_daily_yahoo.parquet"
OUTPUT_PATH = config.DATA_DIR / "clean" / "goldpulse_index.parquet"

SMOOTH_DAYS = 7
BASELINE_DAYS = 90
RANK_WINDOW = 365
MIN_HISTORY_DAYS = 180
ATTENTION_BANDS = [(0, "Quiet"), (20, "Normal"), (80, "Busy"), (95, "Frenzy")]
MOOD_BANDS = [(0, "Gloomy"), (20, "Typical"), (80, "Upbeat")]


def load_posts(history_path=HISTORY_PATH, db_path=config.DB_PATH) -> pd.DataFrame:
    cols = ["post_id", "created_utc", "subreddit", "text_hash", "sentiment"]
    history = pd.read_parquet(history_path, columns=cols)
    frames = [history]
    if Path(db_path).exists():
        with sqlite3.connect(db_path) as conn:
            frames.append(pd.read_sql(f"SELECT {', '.join(cols)} FROM posts", conn))
    posts = pd.concat(frames, ignore_index=True).drop_duplicates("post_id")
    posts["created_utc"] = pd.to_datetime(posts["created_utc"], utc=True, format="ISO8601")
    return dedupe_posts(posts)


def trailing_score(series: pd.Series, window=RANK_WINDOW, min_periods=MIN_HISTORY_DAYS) -> pd.Series:
    """Percentile (0-100) of each value among itself and the previous window-1 days."""
    return series.rolling(window, min_periods=min_periods).rank(pct=True) * 100


def band(score: float, bands) -> str | None:
    if pd.isna(score):
        return None
    return [name for floor, name in bands if score >= floor][-1]


def build_index(posts: pd.DataFrame, prices: pd.DataFrame | None = None) -> pd.DataFrame:
    last_full_day = posts["created_utc"].max().floor("D") - pd.Timedelta(days=1)
    posts = posts.assign(day=posts["created_utc"].dt.floor("D").dt.tz_localize(None),
                         bullish=posts["sentiment"] > NEUTRAL_BAND)
    days = pd.date_range(posts["day"].min(), last_full_day.tz_localize(None), freq="D", name="date")
    daily = posts.groupby("day").agg(posts=("post_id", "size"), sentiment_sum=("sentiment", "sum"),
                                     bullish_share=("bullish", "mean")).reindex(days)
    daily["posts"] = daily["posts"].fillna(0).astype(int)
    daily["sentiment_sum"] = daily["sentiment_sum"].fillna(0.0)
    daily["mean_sentiment"] = daily["sentiment_sum"] / daily["posts"].replace(0, np.nan)

    roll_posts = daily["posts"].rolling(SMOOTH_DAYS, min_periods=SMOOTH_DAYS).sum()
    daily["attention_7d"] = roll_posts / SMOOTH_DAYS
    baseline = daily["posts"].shift(SMOOTH_DAYS).rolling(BASELINE_DAYS, min_periods=BASELINE_DAYS).mean()
    daily["attention_ratio"] = daily["attention_7d"] / baseline
    daily["mood_7d"] = daily["sentiment_sum"].rolling(SMOOTH_DAYS, min_periods=SMOOTH_DAYS).sum() / roll_posts
    daily["attention_score"] = trailing_score(daily["attention_ratio"])
    daily["mood_score"] = trailing_score(daily["mood_7d"])
    daily["attention_band"] = daily["attention_score"].map(lambda s: band(s, ATTENTION_BANDS))
    daily["mood_band"] = daily["mood_score"].map(lambda s: band(s, MOOD_BANDS))
    daily = daily.drop(columns="sentiment_sum")
    if prices is not None:
        daily["gold_close"] = prices.set_index("date")["close"].reindex(days)  # NaN on non-trading days
    return daily


def download_prices(start="2021-12-01") -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(config.PRICE_TICKER, start=start, interval="1d", progress=False, auto_adjust=False)
    close = raw["Close"].squeeze("columns")
    return pd.DataFrame({"date": close.index.tz_localize(None), "close": close.to_numpy()}).dropna()


def run(refresh_prices=False, output=OUTPUT_PATH) -> pd.DataFrame:
    if refresh_prices or not PRICES_PATH.exists():
        download_prices().to_parquet(PRICES_PATH, index=False)
    index = build_index(load_posts(), pd.read_parquet(PRICES_PATH))
    index.to_parquet(output)
    published = index.dropna(subset=["attention_score"])
    latest = published.iloc[-1]
    print(f"Wrote {output}: {len(index)} days, scores from {published.index.min():%Y-%m-%d} "
          f"to {published.index.max():%Y-%m-%d}")
    print(f"Latest ({published.index.max():%Y-%m-%d}): attention {latest['attention_score']:.0f} "
          f"({latest['attention_band']}), mood {latest['mood_score']:.0f} ({latest['mood_band']})")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh-prices", action="store_true", help="re-download daily GC=F from Yahoo")
    args = parser.parse_args()
    run(args.refresh_prices)
    return 0


if __name__ == "__main__":
    sys.exit(main())
