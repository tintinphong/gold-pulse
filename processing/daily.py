"""Long daily study: Reddit history + Bloomberg market data -> one row per gold trading day.

    python -m processing.daily        # -> data/bbg/daily_features.parquet (git-ignored)

Rules (pitfalls #34-#36):

* Days. One row per day Bloomberg has an XAU close, up to LAST_FULL_DAY. The
  2026-10-05 row was pulled mid-session, so it is not a close and is dropped.
* Close time. Bloomberg's daily XAU close is New York 17:00: it equals the
  hourly spot price at 21:00 UTC in summer time (22:00 UTC in winter).
* Reddit day. Day d gets every post made after the previous trading day's
  cut-off and up to CUTOFF_HOUR:00 UTC on d, so it is complete before d's
  close in both summer and winter time. Weekend and holiday posts roll into
  the next trading day. Counts are per 24h of window, so a Monday's three-day
  window doesn't look like a chatter spike.
* Target. up = close[d+1] > close[d] (a tie is "not up"). It and next_return
  are the only forward-looking columns; never feed them to a model.
* Market features are all known by d's close. ETF flows and miner news
  sentiment have no published cut-off time, so they are lagged one day.
  US-holiday gaps (SPX, VIX, GVZ) are forward-filled, so their change is 0.
* Trailing only: every lag, rolling mean and spike looks backward.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from processing.bloomberg import NEWS_TICKERS, OUT_DIR as BBG_DIR
from processing.bucket import NEUTRAL_BAND, dedupe_posts

POSTS_PATH = config.DATA_DIR / "clean" / "history_posts.parquet"
OUTPUT_PATH = BBG_DIR / "daily_features.parquet"
LAST_FULL_DAY = pd.Timestamp("2026-10-02")
CUTOFF_HOUR = 20
SPIKE_WINDOW = 20

DAILY_PRICE_FEATURES = ["xau_ret_1d", "xau_ret_5d"]
DAILY_MARKET_FEATURES = DAILY_PRICE_FEATURES + [
    "dxy_ret_1d", "real_yield_chg_1d", "gvz_level", "gvz_chg_1d", "vix_chg_1d", "spx_ret_1d",
    "etf_flow_lag1", "miner_news_lag1", "day_of_week",
]
DAILY_MOOD_FEATURES = ["mean_sentiment", "bullish_ratio", "bearish_ratio",
                       "sent_roll_mean_5d", "mean_sentiment_lag1"]
DAILY_CHATTER_FEATURES = ["log_posts_per_day", "log_posts_per_day_lag1", "attention_spike"]


def cutoffs(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return (days + pd.Timedelta(hours=CUTOFF_HOUR)).tz_localize("UTC")


def assign_days(created_utc: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """Trading day each post counts toward: the first whose cut-off is at or after
    it. Posts before the first day's window or after the last cut-off get NaT."""
    cuts = cutoffs(days).as_unit("ns")
    created = pd.DatetimeIndex(created_utc).tz_convert("UTC").as_unit("ns")
    pos = np.searchsorted(cuts.asi8, created.asi8, side="left")
    out = pd.Series(pd.NaT, index=created_utc.index, dtype="datetime64[ns]")
    ok = (pos > 0) & (pos < len(days))  # day 0 has no previous cut-off, so its window is open-ended
    out[ok] = days[pos[ok]]
    return out


def reddit_daily(posts: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    posts = posts.assign(day=assign_days(posts["created_utc"], days)).dropna(subset=["day"])
    posts = posts.assign(bullish=posts["sentiment"] > NEUTRAL_BAND, bearish=posts["sentiment"] < -NEUTRAL_BAND)
    daily = posts.groupby("day").agg(
        post_count=("post_id", "size"),
        mean_sentiment=("sentiment", "mean"),
        bullish_ratio=("bullish", "mean"),
        bearish_ratio=("bearish", "mean"),
    ).reindex(days)
    window_hours = cutoffs(days).to_series().diff().dt.total_seconds().div(3600).to_numpy()
    daily["window_hours"] = window_hours
    daily["had_posts"] = daily["post_count"].notna()
    daily["post_count"] = daily["post_count"].fillna(0).astype(int)
    mood = ["mean_sentiment", "bullish_ratio", "bearish_ratio"]
    daily[mood] = daily[mood].ffill()  # never back-filled
    return daily


def market_features(market: pd.DataFrame, etf: pd.DataFrame, news: pd.DataFrame,
                    days: pd.DatetimeIndex) -> pd.DataFrame:
    m = market.reindex(days).ffill()  # US-holiday gaps only; XAU itself is never missing on `days`
    f = pd.DataFrame(index=days)
    xau = market["XAU_PX_LAST"].reindex(days)
    f["xau_close"] = xau
    f["xau_ret_1d"] = xau / xau.shift(1) - 1
    f["xau_ret_5d"] = xau / xau.shift(5) - 1
    f["dxy_ret_1d"] = m["DXY_PX_LAST"].pct_change(fill_method=None)
    f["real_yield_chg_1d"] = m["USGGT10Y_PX_LAST"].diff()
    f["gvz_level"] = m["GVZ_PX_LAST"]
    f["gvz_chg_1d"] = m["GVZ_PX_LAST"].diff()
    f["vix_chg_1d"] = m["VIX_PX_LAST"].diff()
    f["spx_ret_1d"] = m["SPX_PX_LAST"].pct_change(fill_method=None)

    e = etf.reindex(days)
    flow = e[["GLD_FUND_FLOW", "IAU_FUND_FLOW"]].sum(axis=1, min_count=1)
    assets = e[["GLD_FUND_TOTAL_ASSETS", "IAU_FUND_TOTAL_ASSETS"]].sum(axis=1, min_count=1).ffill()
    f["etf_flow_pct"] = (flow / assets.shift(1) * 100)
    f["etf_flow_lag1"] = f["etf_flow_pct"].shift(1).fillna(0.0)

    cols = [f"{t}_NEWS_SENTIMENT_DAILY_AVG" for t in NEWS_TICKERS]
    f["miner_news"] = news.reindex(days)[cols].mean(axis=1)
    f["miner_news_lag1"] = f["miner_news"].ffill().shift(1).fillna(0.0)
    f["day_of_week"] = days.dayofweek
    return f


def build_daily(posts: pd.DataFrame, market: pd.DataFrame, etf: pd.DataFrame, news: pd.DataFrame,
                last_day: pd.Timestamp = LAST_FULL_DAY) -> pd.DataFrame:
    days = market.index[market["XAU_PX_LAST"].notna() & (market.index <= last_day)]
    days = pd.DatetimeIndex(days, name="date")
    f = market_features(market, etf, news, days).join(reddit_daily(posts, days))

    f["sent_roll_mean_5d"] = f["mean_sentiment"].rolling(5, min_periods=1).mean()
    f["mean_sentiment_lag1"] = f["mean_sentiment"].shift(1)
    per_day = f["post_count"] / f["window_hours"] * 24
    f["posts_per_day"] = per_day
    f["log_posts_per_day"] = np.log1p(per_day)
    f["log_posts_per_day_lag1"] = f["log_posts_per_day"].shift(1)
    trailing = per_day.shift(1).rolling(SPIKE_WINDOW, min_periods=SPIKE_WINDOW).mean()
    f["attention_spike"] = np.log1p(per_day) - np.log1p(trailing)

    # The only look-ahead in this file.
    nxt = f["xau_close"].shift(-1)
    f["target_valid"] = nxt.notna()
    f["next_return"] = (nxt / f["xau_close"] - 1).where(f["target_valid"])
    f["up"] = (nxt > f["xau_close"]).astype(float).where(f["target_valid"])
    return f


def load_inputs(posts_path=POSTS_PATH):
    posts = pd.read_parquet(posts_path)
    posts["created_utc"] = pd.to_datetime(posts["created_utc"], utc=True, format="ISO8601")
    posts = dedupe_posts(posts)
    market, etf, news = (pd.read_parquet(BBG_DIR / f"{k}.parquet")
                         for k in ["market_daily", "etf_flows", "news_sentiment"])
    return posts, market, etf, news


def run(posts_path=POSTS_PATH, output=OUTPUT_PATH) -> pd.DataFrame:
    daily = build_daily(*load_inputs(posts_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output)
    valid = daily["target_valid"]
    print(f"Wrote {output}: {len(daily)} trading days {daily.index.min():%Y-%m-%d} -> {daily.index.max():%Y-%m-%d}, "
          f"{int(valid.sum())} target-valid, {daily.loc[valid, 'up'].mean():.1%} up, "
          f"{int(daily['post_count'].sum())} posts assigned")
    return daily


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--posts", type=Path, default=POSTS_PATH, help=f"default {POSTS_PATH}")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help=f"default {OUTPUT_PATH}")
    args = parser.parse_args()
    run(args.posts, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
