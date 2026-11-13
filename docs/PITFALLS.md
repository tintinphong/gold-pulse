# Pitfalls & Lessons Learned

Every problem we hit, why it happened, and what we did about it. This is raw material for the
report's Data Collection, Data Processing, Scope & Limitations, and Conclusion sections. Incidents
with full numbers are written up in [COLLECTION_LOG.md](COLLECTION_LOG.md); this file is the
complete list, including the smaller ones.

**Recurring theme:** four separate failures showed green checkmarks while data was missing
(#1, #2, #4, #6). Workflow status alone is not a health check; row counts and coverage are.

---

## A. Data collection

### 1. Reddit API rejected → collector silently wrote nothing for 2 days (Sept 8–10)
- **What:** the PRAW collector could never authenticate, but `collect.yml` stayed green because
  "nothing changed, nothing to commit" looks the same as "no new posts".
- **Fix:** switched to the Arctic Shift archive API (no credentials). It serves history, so the
  gap was backfilled (189 posts).
- **Lesson:** live-only APIs make gaps permanent; an archive source makes them recoverable.

### 2. Price runs cancelled by GitHub's concurrency queue (Sept 3 – Oct 5)
- **What:** `price.yml` (hourly) and `collect.yml` (every 15 min) shared one concurrency group to
  avoid racing on the sqlite file. GitHub keeps only **one pending run per group** and cancels the
  older pending run when a new one queues, so the 15-minute job kept cancelling the hourly price
  job. Result: 143 price rows instead of ~495 (~5/day instead of ~23).
- **Fix:** price sync moved into `collect.yml` as a step (single writer). Each run re-syncs the
  last 5 days, so missed runs heal themselves. History was backfilled from Yahoo hourly bars.
- **Lesson:** a concurrency group is not a queue. Cancelled runs are grey, not red.

### 3. Live price values were inconsistent snapshots
- **What:** GitHub cron fired on average **31 min late** (range 2–60 min), so each "hourly" price
  was taken at a random minute. On closed hours (weekends, the daily 21:00 UTC break) the feed
  returned the **last close repeated**: 42 of 143 rows were frozen prices that a direction
  target would read as "flat".
- **Fix:** each price is now defined as the **Open of that hour's GC=F bar** (the price at HH:00).
  Closed hours have no row. 99 snapshots were corrected and 42 frozen rows removed.
- **Lesson:** don't trust cron timing for time-sensitive sampling. Read timestamped bars instead
  of "the latest price right now".

### 4. Arctic Shift HTTP 422 "Timeout" silently skipped a subreddit (~Oct 4–5)
- **What:** under load Arctic Shift returns **HTTP 422** `{"error": "Timeout. Maybe slow down a
  bit"}`. The collector retried only 429/5xx, so the subreddit was marked failed for that run. A
  run only fails loudly if *all* subreddits fail, so runs stayed green. r/Gold went ~10 h without
  a stored post; 39 posts were missing overall.
- **Fix:** 422-timeouts now retry with backoff like 429/5xx. Recovery is via the posts backfill
  workflow.
- **Lesson:** an unofficial API's error codes must be observed, not assumed. Partial failure
  needs its own alarm.

### 5. Engagement numbers are placeholders for ~36 h
- **What:** Arctic Shift archives a post at creation and fills in its real score and comment
  count on a later crawl.
- **Fix:** an `engagement_stale` flag plus a refresh pass. Processing averages engagement over
  settled rows only and reports `stale_share`.
- **Lesson:** an archive's snapshot time matters as much as its content.

### 6. Collector produced no commits after 12:38 UTC on Oct 5 — *under investigation*
- **What:** no `collect:` commits for 4+ hours while Arctic Shift had new posts. This overlaps
  with #4; the cause is not yet confirmed.
- **Status:** awaiting the Actions run logs. The posts backfill recovers the data either way.

### 7. GC=F futures, not true XAU/USD spot
- **What:** Yahoo's free tier has no spot gold ticker, so COMEX gold futures are used as a proxy.
- **Report:** state this as a limitation. Futures track spot closely but trade on CME hours and
  close for US holidays. **Labor Day (Sept 7) has no bars**, so Sept 4 21:00 → Sept 8 04:00 UTC is
  a market closure, not a collection gap.

### 8. yfinance / Actions deprecation warnings
- `Timestamp.utcnow` Pandas4Warning from inside yfinance, and a GitHub "Node.js 20 is deprecated"
  annotation on `actions/checkout@v4` / `setup-python@v5`. Both are harmless now; they belong in
  the report only as maintenance notes.

---

## B. Data processing & modeling

### 9. Duplicate posts: cross-posts vs. genuinely different posts
- **What:** 62 groups of identical text (133 posts, final Sept 8 – Oct 5 window). Some were real duplicates (double-submits seconds apart,
  the same post in r/Gold and r/Silverbugs minutes apart). Others were different people posting
  generic titles ("Any info? [removed]") days apart.
- **Fix:** drop a post only if its normalized text matches an earlier post **within 24 h**, keeping
  the earliest. 58 duplicates dropped (2,477 → 2,419 posts).

### 10. Sparse hours: forward-fill sentiment, never back-fill, never fill price
- Back-filling would leak future sentiment into past hours. A forward-filled *price* would
  recreate pitfall #3's "flat" artefact. `had_posts` and `market_open` flag which rows are real.

### 11. Look-ahead risk when building the target (for Track B)
- **What:** `xau_usd` at hour *t* is the price at **t:00**, but hour *t*'s sentiment covers posts
  from **t:00 to t:59**. So "price at t+1 vs price at t" moves *during* the same hour the
  sentiment was measured. That's concurrent, not predictive.
- **Fix (Track B):** with features from hour *t*, predict the move from **(t+1):00 to (t+2):00**,
  i.e. `price_direction = xau_usd.shift(-2) > xau_usd.shift(-1)`. Alternatively, use lagged
  features only.

### 12. Calendar time vs. trading time
- Prices exist for only 450 of 664 hours (final window). Lags and rolling windows on the hourly grid span
  weekends, and "next hour" on Friday 20:00 has no price. Track B should build the target only
  where both prices exist and say how many rows that leaves.

---

## C. Tooling & process

### 13. Working copy was not a git clone
- The local `GoldPulse` folder was a zip export with no `.git` and no branches visible. Work moved
  to a real clone (`GoldPulse-repo`). It matched `main` exactly, so nothing was lost.

### 14. Line endings and encoding on Windows
- Every file "differed" from the repo until CRLF/LF was ignored (`core.autocrlf=true`). Python's
  default file encoding on this machine (cp874) can't read the README's UTF-8 characters, so read
  and write with `encoding="utf-8"`. Related: the console prints characters like `…` as `�`.
  That's a display problem only; the database and notebooks store the correct characters.

### 15. Private repo hides Actions from the API
- Without `gh` or a token, the run history isn't visible, so diagnosis depended on screenshots.
  Installing `gh` and running `gh auth login` would make monitoring scriptable.

### 16. Workflow UI shows input descriptions, not names
- `prune_closed` appears in the Run-workflow form as "Delete stored hours with no bar (market
  closed)". Refer to inputs by their visible label in instructions.

### 17. Similar workflow names: the price backfill was run instead of the posts backfill
- "Backfill hourly XAU/USD prices" and "Backfill posts from Arctic Shift" sit next to each other.
  The run log's last line ("Price backfill: …" vs "Backfill complete: saw N posts…") shows which
  one ran.

### 18. Dry-run counts aren't "missing" counts
- `backfill.py --dry-run` reports every post it *would* accept, without checking what's already
  stored. To measure a gap, compare post IDs against the database.

### 19. `workflow_dispatch` workflows appear only once they're on the default branch
- The new backfill workflow couldn't be run until its commit reached `main`.

### 20. Schedule slip
- The plan's submission date was Oct 1. As of Oct 5, Phase 3 onward hadn't started. The final
  deadline is Oct 5, so later phases were compressed. Report "planned vs. actual" honestly on the
  timeline slide.

---

## D. Features & modeling (Track B)

### 21. The obvious target leaks: shift by two hours, not one
- **What:** "price at t+1 > price at t" moves *during* the hour whose posts make up row t's
  sentiment (see #11), so it measures co-movement.
- **Fix:** `price_direction = xau_usd[t+2] > xau_usd[t+1]`, valid only when both prices exist:
  431 of 664 hours. `tests/test_features.py` and a truncation check in `02_features.ipynb` show
  that no feature at hour t changes when later rows are removed.
- **Lesson:** align the label with the moment the features are *fully known*, not the hour they're stamped with.

### 22. The test week's base rate differs from the full sample's
- **What:** across all 431 rows "up" is 48.0%, so majority-class looks like ~52%. But the test week
  (Sept 28 → Oct 5) was 50.8% up, so the train-fitted majority baseline scores **49.2%** on test.
- **Lesson:** quote the baseline *on the test set*, fitted on train only. Never quote the full-sample class share.

### 23. Returns and lags across market closures
- **What:** the first open hour after a weekend or the 21:00 break has no previous price, so
  `last_return` is NaN for 38 target-valid rows.
- **Fix:** NaN → 0 plus a `return_missing` flag. Lags are computed on the full calendar grid
  first, then filtered, so "lag 1" never silently jumps over a weekend.

### 24. Forward-filled sentiment fakes low volatility
- **Fix:** rolling std uses observed hours only (`mean_sentiment.where(had_posts)`, ≥2 needed),
  and NaN → 0 for the model. Rolling means use the forward-filled series.

### 25. Small test set: a seven-point gain is still noise
- **What:** with 124 test hours the 95% Wilson interval is about ±9 points. Price + sentiment
  logistic regression scored 56.5% [47.7–64.9] vs. 48.4% price-only and 49.2% majority class.
  Exact McNemar p = 0.19 vs. price-only and p = 0.31 vs. majority class.
- **Lesson:** report intervals and a paired test, not just accuracy. H3 is **inconclusive**.

### 26. Overfitting is visible even in a depth-3 tree
- Tree train 68.1% → test 51.6%. Logistic 62.2% → 56.5%. Four weeks of hourly data support only
  very simple supervised models, which is also the course constraint (supervised learning only:
  no clustering, PCA or other unsupervised steps anywhere).

### 27. Hand labels must be protected from re-runs
- **What:** regenerating the validation sample would overwrite the human labels.
- **Fix:** `02_features.ipynb` writes `data/validation/sentiment_sample.csv` only if it doesn't
  exist. The code prepares the sample and fills in no labels. A person fills `human_label` and `labelled_by`.

### 28. Dropping a community: the order of de-duplication matters
- **What:** stories cross-posted to r/Gold and r/Silverbugs were de-duplicated keeping the earliest
  copy, sometimes the Silverbugs one. Dropping Silverbugs *after* de-duplication would lose those
  r/Gold copies too.
- **Fix:** `processing/bucket.py --exclude-subreddits` drops the community **before**
  de-duplication, as if it had never been collected. r/Gold keeps 1,131 posts (vs. 1,124 before).
  This is unit-tested.

### 29. A more on-topic dataset is also a thinner one
- Without r/Silverbugs: 1,146 posts, 1.7 per hour, and **22% of hours have no posts** (vs. 6%),
  so their mood is carried forward. Human labels in the partial sample didn't show r/Gold as
  more market-focused: 70% neutral vs. 61% for Silverbugs, on tiny n.

### 30. Trying more variants makes a lucky "win" more likely
- **What:** without Silverbugs, the depth-3 tree scored **62.1%** [53.3–70.2], the best number
  in the project. Its McNemar p vs. price-only was 0.046. That passes 0.05 but fails the Bonferroni
  bar of 0.0125 for the four sentiment models tried.
- **Check:** depth 2 and 4, and splits two days earlier or later, give 43–56%. The 62% appears at one
  setting only, and the same features made the logistic model *worse* (56.5% → 50.0%).
  The tree's rules lean on post counts and "is it Monday", not on mood.
- **Lesson:** a single good number among many tries is not a finding. Report the robustness check
  alongside it. Changing the setting after seeing the test score would be tuning on the test set.

### 31. A "better" scorer can win by saying neutral to everything
- **What:** FinBERT (trained on financial news) labels 96% of posts neutral. Most hand-labelled posts
  are neutral, so its raw agreement (60.0%) beats VADER (37.8%), yet it stays **below** an
  always-neutral guess (62.2%) and finds 1 of 13 bullish posts (all 45 labels).
- **Fix:** always report the always-neutral yardstick and a balanced score (mean agreement per human
  class) next to raw agreement (`02_features.ipynb` §4). Don't tune FinBERT thresholds on the 45 labels,
  because they are the exam.
- **Other limits:** posts over 512 tokens are truncated (a handful; median is 20 words). It has no notion
  of silver vs. gold, and it reads "added a couple of ounces" or a complaint about the feed as
  market sentiment.
- **Reproducibility:** model revision pinned in `sentiment/finbert.py`; scores stored per post and
  never written to the DB. Unchanged VADER outputs were checked to be identical after the
  `--sentiment-file` change.

### 32. News lags the market, and keywords catch passers-by
- **What:** the economy-mood variant used Reddit posts about the Fed, inflation and yields. Markets
  price such news within minutes, usually before anyone posts it, so next-hour prediction is a hard
  test by design. The keyword filter also keeps some off-topic posts that mention a term in passing.
- **Fix:** "Fed" matches only as a capitalised word and never "Fed up" (unit-tested). The success
  criteria were written down before results (`05_macro.ipynb`), and the model learns the sign of the
  economy-gold link instead of assuming "bad economy = gold up". Result: null (Track B results).

### 33. The Bloomberg hourly sheet had no time zone
- **What:** intraday bars come back in the *terminal's* local time, and the sheet's time-zone cell
  was left at its placeholder. A wrong guess would shift every hourly price and make any lead/lag
  result an artefact.
- **Fix:** hourly XAU returns were lined up against our GC=F `price_ticks` at every shift from −12h
  to +12h. Only one shift correlates (r = 0.84; the next best is 0.13): the terminal is **UTC+7
  (Bangkok)**, and a bar labelled L closes at L+1h local = L−6h UTC. `processing/bloomberg.py`
  re-runs this scan and refuses to write if the best shift moves.

### 34. A "daily close" has a clock time, and the last row wasn't one
- **What:** Bloomberg's daily XAU close is New York 17:00: it equals the hourly spot price at
  21:00 UTC exactly on 38 of 44 checked days (22:00 UTC in winter time). The 2026-10-05 row was
  pulled mid-session, so its "close" is an intraday price.
- **Fix:** the daily table stops at 2026-10-02, the last full close (`LAST_FULL_DAY`).

### 35. A Reddit "day" must end before the price it predicts
- **What:** a calendar day of posts (to 23:59 UTC) ends *after* the 21:00 UTC close, so the last
  three hours would leak the next session into today's features.
- **Fix:** Reddit day *d* ends at **20:00 UTC**, before the close in summer and winter time. Weekend
  and holiday posts roll into the next trading day, and counts are per 24h of window, so Monday's
  three-day window isn't a fake chatter spike (unit-tested in `tests/test_daily.py`).

### 36. Data without a timestamp is lagged a day
- **What:** ETF fund flows and Bloomberg's daily news sentiment have no published cut-off time;
  flows in particular are usually reported the next morning.
- **Fix:** both enter the daily model one trading day late. That may cost a little signal, but it
  can't leak the future.

### 37. Bloomberg has no sentiment for gold itself
- **What:** every news/Twitter sentiment field returned "Field Not Applicable" for XAU, XAG, GC1 and
  the gold ETFs. Only mining stocks have a daily news score, and three of eight (GFI, AU, PAAS) are
  mostly blank or zero. XAU spot also has no volume.
- **Fix:** the professional-mood benchmark is the mean of the five well-covered miners (NEM, AEM,
  KGC, WPM, FNV). It is a proxy: miner news also covers costs, mines and management.

### 38. A progress log can lie even when the data is right
- **What:** the history backfill appended each month's counts to a CSV in the order the reasons were
  first met. That order differed between months, so columns shifted: May 2022 read as "45 kept,
  564 removed" when it was 564 kept, 45 non-English.
- **Fix:** fixed column order; the shuffled rows were rebuilt from the run log, whose lines label each
  number. The saved posts were never affected. Same lesson as #1–#6: check numbers that look odd
  before trusting or "fixing" the data.
