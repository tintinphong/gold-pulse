"""Phase 4: hourly_merged.parquet -> features.parquet (Track B).

    python -m processing.features

Writes data/clean/features.parquet, the single feature table read by
notebooks/04_model.ipynb and Track C's dashboard. Rules:

* Grid. Computed on the full contiguous hourly grid (all 664 rows) and kept
  that way. Filtering to tradable hours first would make "lag 1" jump across
  weekends. The model filters on target_valid itself.
* Trailing only. Every lag and rolling window looks backward
  (rolling() default, never center=True). The target is the only column
  built with shift(-k); tests/test_features.py checks that features at hour t
  don't change when later rows are dropped.
* Target (pitfall #11). xau_usd at row t is the price at t:00, but row t's
  sentiment covers posts from t:00 to t:59, so it's only fully known at
  (t+1):00. The target is therefore the next move after that:
  price_direction = xau_usd[t+2] > xau_usd[t+1]. A tie counts as "not up".
  target_valid requires both prices, so closures never produce a target.
* Volatility. Rolling std of sentiment uses observed hours only
  (mean_sentiment where had_posts): forward-filled hours repeat a value and
  would understate volatility. Rolling means use the forward-filled series.
* Engagement. log1p, NaN -> 0 with an engagement_missing flag (no settled
  posts that hour).
* Price-only feature. last_return = xau_usd[t] / xau_usd[t-1] - 1, NaN when
  either hour was closed. last_return_filled / return_missing are the
  model-ready versions.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config

INPUT_PATH = config.DATA_DIR / "clean" / "hourly_merged.parquet"
OUTPUT_PATH = config.DATA_DIR / "clean" / "features.parquet"

ROLL_WINDOWS = (3, 6)
LAGS = (1, 2)
LAGGED_COLUMNS = ["mean_sentiment", "bullish_ratio", "post_count"]

PRICE_FEATURES = ["last_return_filled", "return_missing", "hour_sin", "hour_cos"]
SENTIMENT_FEATURES = [
    "mean_sentiment",
    *[f"sent_roll_mean_{w}h" for w in ROLL_WINDOWS],
    *[f"sent_roll_std_{w}h_filled" for w in ROLL_WINDOWS],
    *[f"{c}_lag{k}" for c in ["mean_sentiment", "bullish_ratio"] for k in LAGS],
    "log_post_count",
    *[f"log_post_count_lag{k}" for k in LAGS],
    "log_engagement", "engagement_missing", "day_of_week",
]


def build_features(hourly: pd.DataFrame) -> pd.DataFrame:
    f = hourly.copy()
    sentiment = f["mean_sentiment"]
    observed = sentiment.where(f["had_posts"])

    for w in ROLL_WINDOWS:
        f[f"sent_roll_mean_{w}h"] = sentiment.rolling(w, min_periods=1).mean()
        f[f"sent_roll_std_{w}h"] = observed.rolling(w, min_periods=2).std()
        # Fewer than two observed posts-hours in the window: no spread measurable.
        f[f"sent_roll_std_{w}h_filled"] = f[f"sent_roll_std_{w}h"].fillna(0.0)

    for column in LAGGED_COLUMNS:
        for k in LAGS:
            f[f"{column}_lag{k}"] = f[column].shift(k)

    f["log_post_count"] = np.log1p(f["post_count"])
    for k in LAGS:
        f[f"log_post_count_lag{k}"] = np.log1p(f[f"post_count_lag{k}"])

    f["engagement_missing"] = f["mean_engagement"].isna()
    f["log_engagement"] = np.log1p(f["mean_engagement"]).fillna(0.0)

    hour = f.index.hour
    f["day_of_week"] = f.index.dayofweek
    f["hour_of_day"] = hour
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    price = f["xau_usd"]
    f["last_return"] = price / price.shift(1) - 1
    f["return_missing"] = f["last_return"].isna()
    f["last_return_filled"] = f["last_return"].fillna(0.0)

    # The only look-ahead in this file: the label and its continuous version
    # (next_return, for EDA plot 3). Neither may be used as a model input.
    p1, p2 = price.shift(-1), price.shift(-2)
    f["target_valid"] = p1.notna() & p2.notna()
    f["next_return"] = (p2 / p1 - 1).where(f["target_valid"])
    f["price_direction"] = (p2 > p1).astype(float).where(f["target_valid"])
    return f


def run(input_path=INPUT_PATH, output=OUTPUT_PATH) -> pd.DataFrame:
    features = build_features(pd.read_parquet(input_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output)
    valid = features["target_valid"]
    print(f"Wrote {output}: {len(features)} rows, {int(valid.sum())} target-valid, "
          f"{features.loc[valid, 'price_direction'].mean():.1%} up")
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help=f"default {INPUT_PATH}")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help=f"default {OUTPUT_PATH}")
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
