# Collection Log

Running record of collection uptime, gaps and how they were resolved, kept for the report's
"Data Collection" section. Newest entries at the top.

Planned window: **Sept 8 – Sept 24, 2026**, later extended to **Oct 5, 2026** (the window used in the
report). Collector runs every 15 min (`collect.yml`), price
feed hourly (`price.yml`).

---

## Oct 5 — Silent per-subreddit failures on Arctic Shift 422 timeouts; fixed

**What went wrong.** While verifying the price backfill, the post table turned out to be missing
**39 posts** that Arctic Shift had: 26 from r/Gold and 13 from r/Silverbugs, between Oct 4 13:19
and Oct 5 15:41 UTC. r/Gold had nothing stored after 05:02 UTC on Oct 5, even though collect runs
at 08:xx and 12:38 stored r/Silverbugs posts.

**Cause.** When Arctic Shift is overloaded it answers **HTTP 422** with
`{"error": "Timeout. Maybe slow down a bit"}`. The collector retried only 429 and 5xx, so a 422
counted the subreddit as failed for that run and moved on. A run fails loudly only if *every*
subreddit fails, so these runs stayed green. That makes it the third "green checks, missing data"
incident, and this time the failure was per subreddit.

**Fix.** 422 responses whose error mentions a timeout now go through the same backoff-and-retry
path as 429/5xx. Other 422s still raise.

**Recovery.** "Backfill posts from Arctic Shift" workflow from Oct 4 → now.

---

## Oct 5 — Price feed was losing ~80% of its runs; fixed and backfilled

**What went wrong.** By Oct 5, `price_ticks` held **143 rows** for Sept 3 → Oct 5. Gold futures
trade ~23 h/day on weekdays, so the expected count was ~500. Two separate problems:

1. **Runs were being cancelled.** `price.yml` (hourly) and `collect.yml` (every 15 min) shared the
   `goldpulse-db` concurrency group so they wouldn't race on the sqlite file. But GitHub keeps only
   *one* pending run per group and cancels the older pending run when a new one queues. Price runs
   waiting at :00 were routinely cancelled by the :15 collect run. That left ~5 ticks a day, and
   fewer during the busy UTC daytime hours. Cancelled runs aren't red, so this was the
   "green checks, missing data" failure again, now for prices.
2. **The ticks that did land were inconsistent.** Each one was "latest 1-minute close at poll
   time". GitHub cron fired on average **31 min late** (range 2–60), so the values were snapshots
   from random minutes. On closed hours the feed returned the last close repeated: **42 of the 143
   rows** were weekend or daily-break hours with frozen prices, which a direction-of-move target
   would read as "flat".

**Fix.** The price series is now defined as the **Open of each traded hour's GC=F bar**, i.e. the
price at HH:00. Each sync re-reads the last 5 days of hourly bars and upserts them, so any run that
lands fills every hour missed since the previous one. Closed hours get no row. The sync moved into
`collect.yml` as a step, so there is a single writer and nothing to cancel. `price.yml` is now
manual-only.

**Recovery.** `backfill_price.py --start 2026-09-03 --end now --prune-closed` (also available as
the "Backfill hourly XAU/USD prices" workflow). Verified on a copy of the Oct 5 database:

| Metric | Result |
|---|---|
| Hourly bars from Yahoo | 495 |
| New hours inserted | 394 |
| Old mid-hour snapshots corrected to the hour's Open | 99 |
| Closed-market rows removed | 42 |
| Weekday coverage after | 21–24 bars/day (full session) |
| Remaining gaps | Weekends, plus Sept 4 21:00 → Sept 8 04:00 UTC (Labor Day, Sept 7: no bars) |

**Outcome:** price series complete for every traded hour in the window. Downstream note:
`price_ticks` rows exist only for traded hours. In the hourly merge, hours with posts but no price
are closed-market hours, not collection gaps.

---

## Sept 10 — Reddit API rejected; pivoted to Arctic Shift and recovered the window

**What went wrong.** Our Reddit API application was rejected, so the PRAW-based collector could
never authenticate. This failed silently in a way we didn't catch for two days: the `collect.yml`
workflow ran on schedule the whole time, but every run exited without writing anything. On Sept 10
the `posts` table held **0 rows**, while all 31 database commits in git history were `price:`
commits. Price collection was never affected.

**Why it went unnoticed.** The workflow's commit step is `git diff --cached --quiet && exit 0` —
"commit only if the file changed" — so a run that collected nothing looked identical to a run with
no new posts. Green checkmarks, no data. Worth stating in the report: uptime monitoring by
workflow status alone was insufficient; row counts are the real health check.

**Fix.** Replaced PRAW with the [Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift)
archive API, which needs no credentials or app approval. Crucially it also serves *history*, so
the two lost days were recoverable — Reddit's own API only exposes a live "newest posts" view, so
under PRAW this gap would have been permanent.

**Recovery.** Ran the backfill workflow for Sept 8 → now (commit `899b0a8`, 09:08 UTC).

| Metric | Result |
|---|---|
| Posts recovered | **189** |
| r/Silverbugs | 94 |
| r/Gold | 92 |
| r/investing | 3 (keyword-filtered, so most posts there are correctly excluded) |
| Coverage | Sept 8 00:26 UTC → Sept 10 08:15 UTC |
| Sentiment range | −0.999 to +0.997, mean +0.311 |
| `price_ticks` | 29 (unaffected throughout) |

**Outcome:** gap fully closed, no data lost against the declared window.

**Known caveat carried forward.** Arctic Shift archives a post when it appears and only fills in
its real score/comment count on a later crawl (~36h). At time of backfill, 129 of 189 rows were
flagged `engagement_stale = 1` and 60 were settled. The collector's refresh pass clears these
automatically as posts age out of the window. **Downstream aggregates over `engagement` must
filter to `engagement_stale = 0`**; `sentiment` is unaffected and usable immediately.

---

## Sept 8 — Collection window opens

`collect.yml` (15 min) and `price.yml` (hourly) scheduled and running. Price data collecting
normally from this point on. Reddit collection was silently failing — see the Sept 10 entry.
