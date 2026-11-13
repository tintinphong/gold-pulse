# GoldPulse — technical guide

How to install, run and rebuild each part of the pipeline. For an overview of the project and where every
deliverable is, start with the [README](../README.md).

## The collector

Polls r/Gold, r/Silverbugs and (keyword-filtered) r/investing every 15 minutes
via the [Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift) archive
API, scores each post's sentiment, and keeps an hourly gold price reference
series (GC=F futures) in sync, all into `data/goldpulse.sqlite3`.

Our Reddit API application was rejected, so we collect through Arctic Shift
instead. It needs no credentials and no app approval, and unlike Reddit's own
API it serves history, so gaps in collection can be recovered after the fact
(see Backfill below).

## Setup

```bash
cd GoldPulse-repo
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

No API keys or `.env` file are needed.

Price reference is `GC=F` (COMEX gold futures) via Yahoo Finance — the free
tier has no true XAU/USD spot ticker, and futures track spot closely enough
for this signal. Swap `PRICE_TICKER` in `config.py` if you get access to a
real spot feed later.

`price_ticks` holds one row per hour the market traded, valued at the Open of
that hour's bar (the price at HH:00). Each run re-syncs the last
`PRICE_LOOKBACK_DAYS` of hourly bars, so missed runs heal themselves, and
closed hours (weekends, the daily 21:00 UTC break) simply have no row.

## Run

```bash
python main.py
```

Runs forever: one Reddit poll every 15 minutes, one price sync every hour.
(On GitHub Actions, `collect.yml` did both every 15 minutes in one job until live collection ended on
5 October 2026; its schedule is now removed and it runs only when started by hand.)
Logs go to `logs/goldpulse.log` and stdout. Stop with Ctrl+C; it's safe to
restart — `post_id` is the primary key, so re-polled posts are skipped.

## Backfill

Recovers posts from a past window — use it to fill a collection gap:

```bash
python backfill.py --start 2026-09-08 --end now --dry-run   # look first
python backfill.py --start 2026-09-08 --end now             # then store
```

Re-running is safe; already-stored posts are skipped. It can also be run from
the Actions tab via the "Backfill posts from Arctic Shift" workflow, which
commits the result for you. If you run it locally instead, disable the cron
workflow first (`gh workflow disable collect.yml`) so the bot
doesn't race you on the sqlite file, and re-enable it after pushing.

Prices have their own backfill, from Yahoo's hourly history (~2 years deep):

```bash
python backfill_price.py --start 2026-09-03 --end now --dry-run
python backfill_price.py --start 2026-09-03 --end now --prune-closed
```

or the "Backfill hourly XAU/USD prices" workflow in the Actions tab, which
likewise commits the result for you.

## A caveat on `engagement`

Arctic Shift archives a post the moment it appears and only fills in its real
score and comment count on a later crawl (~36h). Rows stored before then carry
placeholder numbers and are flagged `engagement_stale = 1`; the collector
re-fetches and settles them once they age past that window. **Downstream
aggregates over `engagement` should filter to `engagement_stale = 0`.**

## Tests

```bash
python -m unittest discover tests
```

Covers the post → row transform (filtering, scoring, staleness). No network
needed, so it runs anywhere.

## Inspecting the data

```bash
sqlite3 data/goldpulse.sqlite3 "select subreddit, count(*) from posts group by 1;"
sqlite3 data/goldpulse.sqlite3 "select engagement_stale, count(*) from posts group by 1;"
```

## Processing (Phase 3)

```bash
python -m processing.bucket          # Sept 8 -> last fully polled hour
```

Writes `data/clean/hourly_merged.parquet`: one row per UTC hour, indexed by
`bucket_utc`, with no hours missing. The rules are documented at the top of
`processing/bucket.py`; in short:

| Column | Meaning |
|---|---|
| `post_count` | posts that hour after de-duplication (0 if none) |
| `mean_sentiment`, `bullish_ratio`, `bearish_ratio` | VADER compound mean; share > +0.05 / < −0.05. Forward-filled over empty hours, never back-filled |
| `mean_engagement` | mean over settled rows only (`engagement_stale = 0`); NaN if none |
| `stale_share` | fraction of that hour's posts still awaiting settled engagement |
| `had_posts` | False where sentiment columns are forward-filled |
| `xau_usd` | GC=F Open at HH:00; NaN when the market was closed, never filled |
| `market_open` | whether `xau_usd` is present |

De-duplication drops a post whose normalized text matches an earlier one
within 24h (double-posts, cross-posts), keeping the earliest.

## Features and models (Phase 4–5)

```bash
python -m processing.features       # -> data/clean/features.parquet (lags, rolling windows, target)
python notebooks/build.py 03 04 09  # EDA, hourly models, course-method checks
```

## Bloomberg data and the long daily study

The Bloomberg terminal pull lives in `extra_BBL/` and everything built from it
in `data/bbg/`. **Both are git-ignored**: terminal data is licensed to us, not
redistributable. Get the workbooks from a teammate, never commit them, and keep
them off the Streamlit dashboard. Notebooks, figures and `reports/model_results.csv`
(summaries only) are committed.

```bash
python -m processing.bloomberg              # workbooks -> data/bbg/*.parquet (+ time-zone check)
python -m collector.history_backfill        # Reddit Dec 2021 -> Oct 2026, resumable, ~4h
python -m collector.history_backfill --combine
python -m processing.daily                  # -> data/bbg/daily_features.parquet
python -m processing.bucket --end 2026-10-05T15:00 --price-file data/bbg/xau_spot_hourly.parquet --output data/bbg/hourly_merged_bbg.parquet
python -m processing.features --input data/bbg/hourly_merged_bbg.parquet --output data/bbg/features_bbg.parquet
python notebooks/build.py 06 07
```

The Reddit history (`data/clean/history/`, `history_posts.parquet`) is our own
collected data and is committed. Timing rules (time zone, close time, day
cut-off) are in docs/PITFALLS.md #33–#38.

## Dashboard

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

To publish it on Streamlit Community Cloud: sign in at share.streamlit.io with
the GitHub account that owns this repository, choose **Create app**, pick this
repository, branch `main` and main file `dashboard/app.py`. It reads only
committed, non-Bloomberg files, so it needs no secrets.

## Team tracks

The project was split into three tracks: data collection, modelling, and product and report.
