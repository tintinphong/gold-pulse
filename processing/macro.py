"""Economy-mood variant: macro posts -> hourly features joined onto the main table.

    python -m processing.macro sample      # writes the two labelling CSVs (only if absent)
    python -m processing.macro features    # -> data/clean/features_macro.parquet

Same rules as Phase 3/4: one row per UTC hour on the frozen grid, mood
forward-filled over hours with no posts (never back-filled), counts 0 there,
and every feature trailing. The target and price columns come unchanged from
data/clean/features.parquet, so the test hours are identical to 04_model.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config

POSTS_PATH = config.DATA_DIR / "clean" / "macro_posts.parquet"
BASE_FEATURES_PATH = config.DATA_DIR / "clean" / "features.parquet"
OUTPUT_PATH = config.DATA_DIR / "clean" / "features_macro.parquet"
SAMPLE_PATH = config.DATA_DIR / "validation" / "macro_sample.csv"
OVERLAP_PATH = config.DATA_DIR / "validation" / "macro_overlap.csv"
OVERLAP_ROWS = 15

ROLL_WINDOWS = (3, 6)
LAGS = (1, 2)
MACRO_FEATURES = [
    "macro_mood",
    *[f"macro_mood_roll_mean_{w}h" for w in ROLL_WINDOWS],
    *[f"macro_mood_lag{k}" for k in LAGS],
    "macro_bad_ratio",
    "log_macro_count",
    *[f"log_macro_count_lag{k}" for k in LAGS],
]


def build_macro_hourly(posts: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    posts = posts.assign(bucket_utc=posts["created_utc"].dt.floor("h"),
                         bad=posts["finbert_label"] == "bad")
    hourly = posts.groupby("bucket_utc").agg(
        macro_post_count=("post_id", "size"),
        macro_mood=("finbert_score", "mean"),
        macro_bad_ratio=("bad", "mean"),
    ).reindex(grid)
    hourly["macro_had_posts"] = hourly["macro_post_count"].notna()
    hourly["macro_post_count"] = hourly["macro_post_count"].fillna(0).astype(int)
    # Carry the last mood forward; before the first post there is none, so neutral (0).
    hourly[["macro_mood", "macro_bad_ratio"]] = hourly[["macro_mood", "macro_bad_ratio"]].ffill().fillna(0.0)
    return hourly


def build_macro_features(hourly: pd.DataFrame) -> pd.DataFrame:
    f = hourly.copy()
    for w in ROLL_WINDOWS:
        f[f"macro_mood_roll_mean_{w}h"] = f["macro_mood"].rolling(w, min_periods=1).mean()
    f["log_macro_count"] = np.log1p(f["macro_post_count"])
    for k in LAGS:
        f[f"macro_mood_lag{k}"] = f["macro_mood"].shift(k).fillna(0.0)
        f[f"log_macro_count_lag{k}"] = f["log_macro_count"].shift(k).fillna(0.0)
    return f


def load_posts(path=POSTS_PATH) -> pd.DataFrame:
    posts = pd.read_parquet(path)
    posts["created_utc"] = pd.to_datetime(posts["created_utc"], utc=True)
    return posts


def run_features(posts_path=POSTS_PATH, base_path=BASE_FEATURES_PATH, output=OUTPUT_PATH) -> pd.DataFrame:
    base = pd.read_parquet(base_path)
    macro = build_macro_features(build_macro_hourly(load_posts(posts_path), base.index))
    out = base.join(macro)
    out.to_parquet(output)
    print(f"Wrote {output}: {len(out)} rows, {int(out['macro_had_posts'].sum())} hours with macro posts, "
          f"{int(out['target_valid'].sum())} target-valid")
    return out


def write_samples(posts_path=POSTS_PATH, sample_path=SAMPLE_PATH, overlap_path=OVERLAP_PATH):
    """45 posts, 15 per FinBERT class, for blind hand-labelling. The CSVs carry
    no scorer columns, and are never rewritten once they exist (they hold labels)."""
    if sample_path.exists() or overlap_path.exists():
        print(f"{sample_path} / {overlap_path} already exist, so they are left untouched.")
        return
    posts = load_posts(posts_path)
    sample = (posts.groupby("finbert_label", group_keys=False)
                   .sample(n=15, random_state=42)
                   .sample(frac=1, random_state=42))[["post_id", "subreddit", "text"]]
    sample = sample.assign(human_label="", labelled_by="")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(sample_path, index=False, encoding="utf-8")
    sample.head(OVERLAP_ROWS).to_csv(overlap_path, index=False, encoding="utf-8")
    print(f"wrote {sample_path} ({len(sample)} posts) and {overlap_path} ({OVERLAP_ROWS} posts)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["sample", "features"])
    args = parser.parse_args()
    if args.step == "sample":
        write_samples()
    else:
        run_features()
    return 0


if __name__ == "__main__":
    sys.exit(main())
