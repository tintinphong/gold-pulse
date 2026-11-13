"""Builds and executes the analysis notebooks headless (no Jupyter UI needed).

    python notebooks/build.py all        # notebooks 02-09
    python notebooks/build.py 03 04      # only these

Notebooks 06-09 need the licensed Bloomberg tables in data/bbg/, which are not in the repository.

Each notebook is defined here as a list of cells, written with nbformat and
executed in place with nbclient, so the committed .ipynb files always carry
outputs from the frozen dataset. Run with PYTHONUTF8=1 on this machine.
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent

SETUP = """\
import os
from pathlib import Path
if Path.cwd().name == "notebooks":
    os.chdir("..")

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

FIGURES = Path("reports/figures")
FIGURES.mkdir(parents=True, exist_ok=True)
WINDOW = ("2026-09-08", "2026-10-05T16:00")  # frozen collection window (handoff §2)
"""


def md(text):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text):
    return nbformat.v4.new_code_cell(text.strip())


# --------------------------------------------------------------------------- 02
NB02 = [
    md("""
# 02 · Features & sentiment-scorer validation (Phase 4)

Input: `data/clean/hourly_merged.parquet` (Phase 3, frozen: 664 hours, Sept 8 00:00 → Oct 5 15:00 UTC).
Output: `data/clean/features.parquet`, the **single feature table** used by `04_model.ipynb` and the
Track C dashboard, and `data/validation/sentiment_sample.csv` for hand-labelling.

The logic lives in `processing/features.py` (unit-tested in `tests/test_features.py`). This notebook
runs it and shows the checks.
"""),
    code(SETUP),
    md("""
## 1. The target: why it is shifted by two hours

`xau_usd` at row *t* is the GC=F price at **t:00**. Row *t*'s sentiment summarises posts from
**t:00 to t:59**, so it is only fully known at **(t+1):00**.

* ❌ `xau_usd[t+1] > xau_usd[t]`: that move happens *while* the posts are written. It measures
  co-movement, and accuracy would look better than anything tradable.
* ✅ `price_direction = xau_usd[t+2] > xau_usd[t+1]`: features known at (t+1):00 predict the
  *next* hour's move. A tie counts as "not up".

A row is `target_valid` only when both prices exist (the market is closed on weekends, during the daily
21:00 UTC break and on US holidays). See pitfall #11.
"""),
    code("""
from processing.features import run, build_features, PRICE_FEATURES, SENTIMENT_FEATURES

features = run()
valid = features[features["target_valid"]]

assert len(features) == 664 and features.index.is_unique
assert (features.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
assert len(valid) == 431
assert str(valid.index.min()) == "2026-09-08 03:00:00+00:00"
assert str(valid.index.max()) == "2026-10-05 13:00:00+00:00"
print(f"target-valid rows: {len(valid)}  up: {valid['price_direction'].mean():.1%}  "
      f"-> majority-class baseline ≈ {1 - valid['price_direction'].mean():.1%}")
print("valid rows per ISO week:", valid.groupby(valid.index.isocalendar().week).size().tolist())
"""),
    md("""
## 2. Features

All computed on the **full 664-hour calendar grid** (so "lag 1" never jumps across a weekend), and
all **trailing**: no `center=True`, and no `shift(-k)` anywhere except the target.

| Feature | Definition |
|---|---|
| `sent_roll_mean_3h/6h` | rolling mean of `mean_sentiment` (forward-filled series) |
| `sent_roll_std_3h/6h` | rolling std over **observed** hours only (`mean_sentiment.where(had_posts)`, ≥2 needed). Forward-filled hours repeat a value and would fake low volatility. `_filled` = NaN→0 for the model |
| `*_lag1/2` | `mean_sentiment`, `bullish_ratio`, `post_count` at t-1, t-2 |
| `log_post_count(_lag1/2)` | `log1p(post_count)` |
| `log_engagement` | `log1p(mean_engagement)`, NaN→0, plus `engagement_missing` flag (76 hours have no settled posts) |
| `day_of_week`, `hour_of_day`, `hour_sin/cos` | calendar; gold trading is strongly intraday |
| `last_return` | `xau_usd[t]/xau_usd[t-1] - 1`, the **price-only** feature for the H3 comparison. NaN across closures; `last_return_filled` = NaN→0, plus `return_missing` flag |
"""),
    code("""
model_cols = PRICE_FEATURES + SENTIMENT_FEATURES
print("model input columns:", len(model_cols))
print("NaNs in model inputs on target-valid rows:", int(valid[model_cols].isna().sum().sum()))
print("target-valid rows whose last_return spans a closure:", int(valid["return_missing"].sum()))
valid[model_cols + ["price_direction"]].describe().T.round(3)
"""),
    md("""
### No-look-ahead check
If a feature at hour *t* used anything after *t*, recomputing the features on a frame **cut off**
at some hour would change the earlier rows. They must be identical.
"""),
    code("""
hourly = pd.read_parquet("data/clean/hourly_merged.parquet")
for cut in ["2026-09-15 12:00", "2026-09-28 00:00", "2026-10-03 07:00"]:
    full = build_features(hourly)[model_cols].loc[:cut]
    short = build_features(hourly.loc[:cut])[model_cols]
    pd.testing.assert_frame_equal(full, short)
print("features at t are unchanged when all rows after t are removed ✔")
"""),
    md("""
## 3. Sentiment-scorer validation (hand-labelled sample)

The scorer (`sentiment/scorer.py`: VADER + `FINANCE_LEXICON`) is a fixed, rule-based lexicon
scorer, **not a trained model**. Checking it against human labels is how we measure its accuracy.

* Population: posts in the frozen window, de-duplicated with the same rule as Phase 3
  (`processing.bucket.dedupe_posts`).
* Sample: **15 per scorer class** (bullish > +0.05, neutral, bearish < −0.05), `random_state=42`.
  We stratify because otherwise about a third would be neutral and bearish would be rare (283 of 2,419).
  Agreement is therefore reported per class and as a balanced figure, not as a population estimate.
* **A person labels `human_label`** (bullish / neutral / bearish) without looking at the scorer's
  label first, and writes their name in `labelled_by`. The code prepares the file but fills in no labels.
* The CSV is written **only if it doesn't exist**, so re-running this notebook never wipes labels.
"""),
    code("""
from processing.bucket import load_raw, dedupe_posts, NEUTRAL_BAND

def scorer_label(s):
    return np.where(s > NEUTRAL_BAND, "bullish", np.where(s < -NEUTRAL_BAND, "bearish", "neutral"))

posts, _ = load_raw()
posts = posts[(posts["created_utc"] >= WINDOW[0]) & (posts["created_utc"] < WINDOW[1])]
posts = dedupe_posts(posts).assign(scorer_label=lambda d: scorer_label(d["sentiment"]))
print(len(posts), "de-duplicated posts in window")
print(posts["scorer_label"].value_counts().to_string())

SAMPLE = Path("data/validation/sentiment_sample.csv")
if SAMPLE.exists():
    print(f"\\n{SAMPLE} already exists, so it is left untouched (it may hold human labels).")
else:
    sample = (posts.groupby("scorer_label", group_keys=False)
                   .sample(n=15, random_state=42)
                   .sample(frac=1, random_state=42))  # shuffle so classes aren't in blocks
    sample = sample[["post_id", "subreddit", "text", "sentiment", "scorer_label"]].assign(
        human_label="", labelled_by="")
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE, index=False, encoding="utf-8")
    print(f"\\nwrote {SAMPLE} ({len(sample)} posts)")
"""),
    code("""
from sklearn.metrics import confusion_matrix

labels = ["bearish", "neutral", "bullish"]
val = pd.read_csv(SAMPLE, encoding="utf-8", keep_default_na=False)
val["human_label"] = val["human_label"].str.strip().str.lower()
done = val[val["human_label"].isin(labels)]

if done.empty:
    print(f"Awaiting human labels: 0 of {len(val)} rows labelled. Fill human_label in {SAMPLE} "
          "and re-run this notebook.")
else:
    agree = (done["human_label"] == done["scorer_label"])
    who = ", ".join(sorted(set(done["labelled_by"]) - {""})) or "(labeller not recorded)"
    print(f"Labelled: {len(done)} of {len(val)} by {who}")
    print(f"Agreement (scorer sign at ±{NEUTRAL_BAND} vs human): {agree.mean():.1%} ({agree.sum()}/{len(done)})")
    print("Per scorer class:", done.assign(a=agree).groupby("scorer_label")["a"].mean().round(2).to_dict())
    by_sub = done.assign(a=agree).groupby("subreddit")["a"].agg(labelled="size", agreement="mean")
    print("Per subreddit (tiny n, indicative only):")
    print(by_sub.round(2).to_string())
    no_sb = agree[done["subreddit"] != "Silverbugs"]
    print(f"Without r/Silverbugs: {no_sb.mean():.1%} ({no_sb.sum()}/{len(no_sb)})")
    print("Human label share by subreddit:")
    print(pd.crosstab(done["subreddit"], done["human_label"], normalize="index").round(2).to_string())
    cm = pd.DataFrame(confusion_matrix(done["human_label"], done["scorer_label"], labels=labels),
                      index=[f"human {l}" for l in labels], columns=[f"scorer {l}" for l in labels])
    display(cm)
    print("\\nDisagreements (look for missing lexicon terms or sarcasm):")
    display(done[~agree][["subreddit", "sentiment", "scorer_label", "human_label", "text"]])
"""),
    md("""
**Lexicon follow-up.** If the disagreements show an obvious missing finance term, the finding is
written up as *"adding X to `FINANCE_LEXICON` would fix N of the disagreements"*. With collection
frozen, the database is **not** re-scored, so every stored sentiment value stays as it is.
"""),
    md("""
## 4. Second scorer: FinBERT, side by side with VADER

VADER measures emotional tone, not a view on the gold price. **FinBERT** (`ProsusAI/finbert`, pinned
revision, see `sentiment/finbert.py`) is a free, ready-made classifier trained on financial news. It
is used **as is**: no fine-tuning and no thresholds fitted to these labels (they are the exam, so
tuning on them would inflate the score). Its label is simply its most likely class
(positive → bullish, negative → bearish). Scores live in `data/clean/post_sentiment_finbert.parquet`;
the database and this CSV are only read.

Both scorers are compared on **the same labelled rows**. The exact McNemar test asks whether one is
right more often than the other by more than chance. Most hand-labelled posts are neutral, so raw
agreement flatters a scorer that says "neutral" to everything. The table therefore also shows that
**always-neutral yardstick** and a **balanced** score (mean agreement per human class).
**Pre-commitment:** the scorer that agrees better with the human labels *and* beats the yardstick is
the recommended one. This is decided on label agreement, never on
model accuracy. Per-post FinBERT labels for the sample are shown only once all 45 rows are labelled,
so they can't influence the remaining labelling.
"""),
    code("""
from scipy.stats import binomtest

fin = pd.read_parquet("data/clean/post_sentiment_finbert.parquet")
assert fin["post_id"].is_unique and fin.notna().all().all()
print("FinBERT label share, all de-duplicated posts in window:")
print(posts.merge(fin, on="post_id")["finbert_label"].value_counts(normalize=True).round(3).to_string())

if done.empty:
    print("Awaiting human labels.")
else:
    both = done.merge(fin[["post_id", "finbert_score", "finbert_label"]], on="post_id", validate="1:1")
    v_ok, f_ok = both["human_label"] == both["scorer_label"], both["human_label"] == both["finbert_label"]
    only_v, only_f = int((v_ok & ~f_ok).sum()), int((f_ok & ~v_ok).sum())
    p = binomtest(only_f, only_v + only_f, 0.5).pvalue if only_v + only_f else 1.0
    no_sb = both["subreddit"] != "Silverbugs"
    n_ok = both["human_label"] == "neutral"  # yardstick: a "scorer" that always says neutral
    by_class = lambda ok: ok.groupby(both["human_label"]).mean().mean()  # balanced: mean agreement per human class
    print(f"\\nLabelled rows: {len(both)} of {len(val)}" + ("" if len(both) == len(val) else "  (PREVIEW: labelling not finished)"))
    display(pd.DataFrame({
        "agreement": [v_ok.mean(), f_ok.mean()],
        "right / labelled": [f"{v_ok.sum()}/{len(both)}", f"{f_ok.sum()}/{len(both)}"],
        "without r/Silverbugs": [v_ok[no_sb].mean(), f_ok[no_sb].mean()],
        "balanced (mean per human class)": [by_class(v_ok), by_class(f_ok)],
    }, index=["VADER (±0.05 band)", "FinBERT (most likely class)"]).round(3))
    print(f"Yardstick, always 'neutral': agreement {n_ok.mean():.3f} ({n_ok.sum()}/{len(both)}), "
          f"balanced {1 / both['human_label'].nunique():.3f}. A useful scorer must beat this.")
    print(f"only VADER right: {only_v}   only FinBERT right: {only_f}   exact McNemar p = {p:.3f}")
    print("Agreement by HUMAN label (how well each scorer finds each kind of post):")
    print(pd.DataFrame({"n": both.groupby("human_label").size(),
                        "VADER": v_ok.groupby(both["human_label"]).mean(),
                        "FinBERT": f_ok.groupby(both["human_label"]).mean()}).round(2).to_string())
    cm_f = pd.DataFrame(confusion_matrix(both["human_label"], both["finbert_label"], labels=labels),
                        index=[f"human {l}" for l in labels], columns=[f"FinBERT {l}" for l in labels])
    display(cm_f)
    if len(both) == len(val):
        print("\\nFinBERT disagreements:")
        display(both[~f_ok][["subreddit", "finbert_score", "finbert_label", "human_label", "text"]])
"""),
]

# --------------------------------------------------------------------------- 03
NB03 = [
    md("""
# 03 · Exploratory data analysis: the four proposal plots

Plots 1–3 read `data/clean/features.parquet` (the single feature table). Plot 4 reads raw,
de-duplicated posts from the frozen window. Each figure is saved to `reports/figures/` for the
report and slides.
"""),
    code(SETUP + """
import seaborn as sns
from scipy import stats
features = pd.read_parquet("data/clean/features.parquet")
print(features.shape, features.index.min(), "→", features.index.max())
"""),
    md("## 1. Hourly mean sentiment vs. gold price"),
    code("""
fig, ax1 = plt.subplots(figsize=(13, 4.8))
ax1.plot(features.index, features["mean_sentiment"], color="tab:blue", alpha=0.25, lw=0.8,
         label="mean sentiment (hourly)")
ax1.plot(features.index, features["sent_roll_mean_6h"], color="tab:blue", lw=1.8,
         label="mean sentiment (6h rolling mean)")
ax1.set_ylabel("mean VADER compound", color="tab:blue")
ax1.axhline(0, color="tab:blue", lw=0.5, ls=":")
ax2 = ax1.twinx()
ax2.plot(features.index, features["xau_usd"], color="goldenrod", lw=1.6, label="GC=F gold futures (USD/oz)")
ax2.set_ylabel("gold price, USD/oz (GC=F)", color="darkgoldenrod")
lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
ax1.legend(lines, [l.get_label() for l in lines], loc="lower left", fontsize=9)
import matplotlib.dates as mdates
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.set_title("Hourly Reddit gold sentiment vs. gold price, Sept 8 – Oct 5, 2026 (UTC)")
fig.text(0.01, -0.02, "Gaps in the price line are market closures (weekends, daily 21:00 UTC break), "
         "not missing data. Sentiment is forward-filled over hours with no posts.", fontsize=8.5)
fig.tight_layout()
fig.savefig(FIGURES / "eda_1_sentiment_vs_price.png", dpi=150, bbox_inches="tight")
r = features[["sent_roll_mean_6h", "xau_usd"]].dropna().corr().iloc[0, 1]
print(f"level correlation, 6h sentiment vs price (descriptive only, both trend): r = {r:.2f}")
"""),
    md("## 2. Posts per hour"),
    code("""
pc = features["post_count"]
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.hist(pc, bins=range(0, pc.max() + 2), color="slategray", edgecolor="white", align="left")
ax.axvline(pc.median(), color="crimson", ls="--", label=f"median = {pc.median():.0f}")
ax.set_xlabel("posts in the hour (after de-duplication)")
ax.set_ylabel("number of hours")
ax.set_title(f"Posts per hour, n = {len(pc)} hours (max {pc.max()}, {int((pc == 0).sum())} hours with none)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "eda_2_posts_per_hour.png", dpi=150, bbox_inches="tight")
print(pc.describe().round(2).to_string())
"""),
    md("""
## 3. Lead–lag: engagement at hour *t* vs. the next tradable price move (H1)

The y-axis uses the same alignment as the model target: the return from **(t+1):00 to (t+2):00**,
the first move that happens after hour *t*'s posts are complete. Only hours with a valid target
and at least one settled-engagement post are shown.
"""),
    code("""
d = features[features["target_valid"] & features["mean_engagement"].notna()]
x, y = d["mean_engagement"] + 1, d["next_return"] * 100  # +1 so 0-engagement hours fit a log axis
pear = stats.pearsonr(np.log(x), y)
spear = stats.spearmanr(x, y)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, y, s=14, alpha=0.5, color="teal")
ax.set_xscale("log")
ax.axhline(0, color="gray", lw=0.6)
ax.set_xlabel("1 + mean engagement at hour t (score + 2×comments, settled posts; log scale)")
ax.set_ylabel("gold return (t+1):00 → (t+2):00, %")
ax.set_title(f"Engagement vs. next-hour gold move   n = {len(d)}\\n"
             f"Pearson r (log engagement) = {pear.statistic:.3f} (p = {pear.pvalue:.2f}),  "
             f"Spearman ρ = {spear.statistic:.3f} (p = {spear.pvalue:.2f})", fontsize=10)
fig.tight_layout()
fig.savefig(FIGURES / "eda_3_engagement_leadlag.png", dpi=150, bbox_inches="tight")
print(f"n={len(d)}  pearson={pear.statistic:.3f} p={pear.pvalue:.3f}  spearman={spear.statistic:.3f} p={spear.pvalue:.3f}")
"""),
    md("## 4. Post sentiment by subreddit"),
    code("""
from processing.bucket import load_raw, dedupe_posts
posts, _ = load_raw()
posts = dedupe_posts(posts[(posts["created_utc"] >= WINDOW[0]) & (posts["created_utc"] < WINDOW[1])])
order = posts.groupby("subreddit")["sentiment"].median().sort_values().index.tolist()
counts = posts["subreddit"].value_counts()
fig, ax = plt.subplots(figsize=(8, 4.8))
sns.boxplot(data=posts, x="subreddit", y="sentiment", order=order, ax=ax,
            hue="subreddit", palette="Set2", legend=False, showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5})
ax.set_xticks(range(len(order)), [f"r/{s}\\n(n = {counts[s]})" for s in order])
ax.axhline(0, color="gray", lw=0.6, ls=":")
ax.set_xlabel("")
ax.set_ylabel("post sentiment (VADER compound)")
ax.set_title("Post sentiment by subreddit (de-duplicated posts, Sept 8 – Oct 5); ◆ = mean")
fig.text(0.01, -0.02, f"r/investing is keyword-filtered and has only n = {counts.get('investing', 0)} posts, "
         "so its box is not reliable.", fontsize=8.5)
fig.tight_layout()
fig.savefig(FIGURES / "eda_4_sentiment_by_subreddit.png", dpi=150, bbox_inches="tight")
summary = posts.groupby("subreddit")["sentiment"].agg(n="size", mean="mean", median="median")
summary["bullish_share"] = posts.assign(b=posts["sentiment"] > 0.05).groupby("subreddit")["b"].mean()
summary["zero_score_share"] = posts.assign(z=posts["sentiment"] == 0).groupby("subreddit")["z"].mean()
summary.round(3)
"""),
    md("""
## Extra (not one of the four proposal plots): mood with vs. without r/Silverbugs
Supports the variant comparison in `04_model.ipynb`. Both lines are 6h rolling means.
"""),
    code("""
fv = pd.read_parquet("data/clean/features_no_silverbugs.parquet")
fig, ax1 = plt.subplots(figsize=(13, 4.8))
ax1.plot(features.index, features["sent_roll_mean_6h"], color="tab:blue", lw=1.4, label="all communities (6h)")
ax1.plot(fv.index, fv["sent_roll_mean_6h"], color="tab:red", lw=1.4, alpha=0.8, label="without r/Silverbugs (6h)")
ax1.set_ylabel("mean VADER compound")
ax1.axhline(0, color="gray", lw=0.5, ls=":")
ax2 = ax1.twinx()
ax2.plot(features.index, features["xau_usd"], color="goldenrod", lw=1.6, label="GC=F gold futures (USD/oz)")
ax2.set_ylabel("gold price, USD/oz (GC=F)", color="darkgoldenrod")
lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
ax1.legend(lines, [l.get_label() for l in lines], loc="lower left", fontsize=9)
import matplotlib.dates as mdates
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.set_title("Hourly mood with and without r/Silverbugs vs. gold price")
fig.text(0.01, -0.02, f"Without r/Silverbugs, {(~fv['had_posts']).mean():.0%} of hours have no posts "
         f"(vs {(~features['had_posts']).mean():.0%}); their mood is carried forward from the last post.", fontsize=8.5)
fig.tight_layout()
fig.savefig(FIGURES / "eda_1b_sentiment_no_silverbugs.png", dpi=150, bbox_inches="tight")
r = features[["sent_roll_mean_6h"]].join(fv[["sent_roll_mean_6h"]], rsuffix="_v").corr().iloc[0, 1]
print(f"correlation between the two 6h mood lines: r = {r:.2f}")
"""),
    md("""
## Extra: VADER vs. FinBERT mood
Same posts, same hours, two scorers. FinBERT's hourly mood is the mean of P(positive) − P(negative),
on the same −1…+1 scale as VADER. Both lines are 6h rolling means.
"""),
    code("""
ff = pd.read_parquet("data/clean/features_finbert.parquet")
fig, ax1 = plt.subplots(figsize=(13, 4.8))
ax1.plot(features.index, features["sent_roll_mean_6h"], color="tab:blue", lw=1.4, label="VADER (6h)")
ax1.plot(ff.index, ff["sent_roll_mean_6h"], color="tab:purple", lw=1.4, alpha=0.85, label="FinBERT (6h)")
ax1.set_ylabel("mean sentiment score (−1 … +1)")
ax1.axhline(0, color="gray", lw=0.5, ls=":")
ax2 = ax1.twinx()
ax2.plot(features.index, features["xau_usd"], color="goldenrod", lw=1.6, label="GC=F gold futures (USD/oz)")
ax2.set_ylabel("gold price, USD/oz (GC=F)", color="darkgoldenrod")
lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
ax1.legend(lines, [l.get_label() for l in lines], loc="lower left", fontsize=9)
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.set_title("Hourly mood by scorer (VADER vs. FinBERT) vs. gold price")
fig.text(0.01, -0.02, "FinBERT calls most posts neutral, so its line sits near 0; VADER reads "
         "friendly hobby posts as positive.", fontsize=8.5)
fig.tight_layout()
fig.savefig(FIGURES / "eda_1c_vader_vs_finbert.png", dpi=150, bbox_inches="tight")
r = features[["sent_roll_mean_6h"]].join(ff[["sent_roll_mean_6h"]], rsuffix="_f").corr().iloc[0, 1]
print(f"correlation between VADER and FinBERT 6h mood lines: r = {r:.2f}")

fin = pd.read_parquet("data/clean/post_sentiment_finbert.parquet")
print("\\nFinBERT label share by subreddit (de-duplicated posts in window):")
pd.crosstab(posts.merge(fin, on="post_id")["subreddit"], posts.merge(fin, on="post_id")["finbert_label"],
            normalize="index").round(3)
"""),
]

# --------------------------------------------------------------------------- 04
NB04 = [
    md("""
# 04 · Baseline vs. supervised models: does sentiment add anything over price? (H3)

**Supervised learning only.** Every model here is a supervised classifier fitted to the labelled
target `price_direction` (1 = gold rises from (t+1):00 to (t+2):00). No clustering, dimensionality
reduction or other unsupervised steps are used anywhere in the pipeline.

| # | Model | Inputs |
|---|---|---|
| 1 | Majority class (`DummyClassifier`) | none |
| 2 | Logistic regression, **price-only** | last return (+ missing flag), hour of day |
| 3 | Logistic regression, **price + sentiment** | (2) + sentiment, rolling means/stds, lags, volume, engagement, weekday |
| 4 | Decision tree, depth 3 (fixed) | same as (3) |

H3 is the comparison between (2) and (3). No hyperparameter search and no boosting: defaults,
fixed in advance.
"""),
    code(SETUP + """
from scipy.stats import binomtest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from processing.features import PRICE_FEATURES, SENTIMENT_FEATURES

features = pd.read_parquet("data/clean/features.parquet")
data = features[features["target_valid"]].copy()
data["y"] = data["price_direction"].astype(int)
ALL_FEATURES = PRICE_FEATURES + SENTIMENT_FEATURES
assert data[ALL_FEATURES].notna().all().all(), "model inputs must have no NaNs"
"""),
    md("""
## Time-ordered split
Train on everything before **Sept 28 00:00 UTC** and test on Sept 28 → Oct 5. No shuffling.
The test week never informs training, including the feature scaling (the scaler is fitted inside
each pipeline, on training rows only).
"""),
    code("""
SPLIT = pd.Timestamp("2026-09-28 00:00", tz="UTC")
train, test = data[data.index < SPLIT], data[data.index >= SPLIT]
assert train.index.max() < test.index.min()          # no look-ahead across the split
assert (len(train), len(test)) == (307, 124)
for name, part in [("train", train), ("test", test)]:
    print(f"{name}: {len(part)} rows, {part.index.min():%b %d %H:%M} → {part.index.max():%b %d %H:%M}, "
          f"up = {part['y'].mean():.1%}")
"""),
    code("""
def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half

def make_models():
    return {
        "1 majority class": (DummyClassifier(strategy="most_frequent"), []),
        "2 logistic, price-only": (make_pipeline(StandardScaler(), LogisticRegression()), PRICE_FEATURES),
        "3 logistic, price + sentiment": (make_pipeline(StandardScaler(), LogisticRegression()), ALL_FEATURES),
        "4 decision tree (depth 3), price + sentiment": (DecisionTreeClassifier(max_depth=3, random_state=0), ALL_FEATURES),
    }

def fit_all(train, test):
    \"\"\"Fits the four supervised models on train, scores them on test.\"\"\"
    models, rows, preds = make_models(), [], {}
    for name, (model, cols) in models.items():
        X_tr = train[cols] if cols else np.zeros((len(train), 1))
        X_te = test[cols] if cols else np.zeros((len(test), 1))
        model.fit(X_tr, train["y"])
        preds[name] = model.predict(X_te)
        k = int((preds[name] == test["y"]).sum())
        lo, hi = wilson(k, len(test))
        rows.append({"model": name, "n_features": len(cols), "n_train": len(train), "n_test": len(test),
                     "train_acc": accuracy_score(train["y"], model.predict(X_tr)),
                     "test_acc": k / len(test), "ci95_low": lo, "ci95_high": hi,
                     "pred_up_share": preds[name].mean()})
    return pd.DataFrame(rows), preds, models

results, preds, MODELS = fit_all(train, test)
results.round(3)
"""),
    md("## Confusion matrices (test set, Sept 28 → Oct 5)"),
    code("""
labels_txt = ["down/flat", "up"]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
for ax, (name, p) in zip(axes, preds.items()):
    ConfusionMatrixDisplay(confusion_matrix(test["y"], p, labels=[0, 1]), display_labels=labels_txt
                           ).plot(ax=ax, colorbar=False, cmap="Blues")
    acc = results.set_index("model").loc[name, "test_acc"]
    ax.set_title(f"{name}\\naccuracy {acc:.1%}", fontsize=10)
fig.suptitle("Next-hour gold direction, test set (n = 124). Rows = actual, columns = predicted", y=1.03)
fig.tight_layout()
fig.savefig(FIGURES / "model_cm_all.png", dpi=150, bbox_inches="tight")

for i, (name, p) in enumerate(preds.items(), start=1):
    f1, a1 = plt.subplots(figsize=(4.2, 3.8))
    ConfusionMatrixDisplay(confusion_matrix(test["y"], p, labels=[0, 1]), display_labels=labels_txt
                           ).plot(ax=a1, colorbar=False, cmap="Blues")
    a1.set_title(f"{name}\\naccuracy {(p == test['y']).mean():.1%}", fontsize=9.5)
    f1.tight_layout()
    f1.savefig(FIGURES / f"model_cm_{i}.png", dpi=150, bbox_inches="tight")
    plt.close(f1)
"""),
    md("""
## H3: does adding sentiment beat price-only?
Exact McNemar test on the test set: among hours where exactly one of models 2 and 3 is right, is
either one right more often than chance?
"""),
    code("""
y = test["y"].to_numpy()
p2, p3 = preds["2 logistic, price-only"], preds["3 logistic, price + sentiment"]
only2, only3 = int(((p2 == y) & (p3 != y)).sum()), int(((p3 == y) & (p2 != y)).sum())
mcnemar_p = binomtest(only3, only2 + only3, 0.5).pvalue if only2 + only3 else 1.0
print(f"only price-only right: {only2}   only price+sentiment right: {only3}   exact McNemar p = {mcnemar_p:.3f}")
p1 = preds["1 majority class"]
b1, b3 = int(((p1 == y) & (p3 != y)).sum()), int(((p3 == y) & (p1 != y)).sum())
print(f"model 3 vs majority class: only majority right {b1}, only model 3 right {b3}, "
      f"exact McNemar p = {binomtest(b3, b1 + b3, 0.5).pvalue:.3f}")

base = results.loc[0, "test_acc"]
for _, r in results.iterrows():
    print(f"{r['model']:<46} test {r['test_acc']:.1%}  [95% CI {r['ci95_low']:.1%} – {r['ci95_high']:.1%}]  "
          f"train {r['train_acc']:.1%}  Δ vs majority {100 * (r['test_acc'] - base):+.1f} pts")
"""),
    md("### What the price + sentiment logistic regression leaned on (standardised coefficients)"),
    code("""
lr3 = MODELS["3 logistic, price + sentiment"][0]
coef = pd.Series(lr3[-1].coef_[0], index=ALL_FEATURES).sort_values(key=abs, ascending=False)
coef.round(3).to_frame("coef (per 1 sd)")
"""),
    code("""
print(export_text(MODELS["4 decision tree (depth 3), price + sentiment"][0], feature_names=ALL_FEATURES))
"""),
    md("""
## How to read this honestly
* The bar to beat is the majority class learned from training: always predicting "down/flat". Over
  all 431 rows that would score 52%, but the **test week happened to be 50.8% up**, so on the test set
  it scores **49.2%**. The baseline is fitted on training data only, as it must be.
* With **124 test hours**, a 95% interval on accuracy is roughly **±9 points** (the Wilson intervals
  above). An accuracy of 55% against a 52% baseline is **not** evidence of skill.
* H3 is supported only if model 3 beats model 2 by more than noise (McNemar p < 0.05) *and* beats
  the baseline. A negative or inconclusive result is a valid finding and is reported as such.
* Limitations: four weeks of data, one test week, GC=F futures as a proxy for spot (pitfall #7),
  VADER scores on short Reddit titles (about a third score exactly 0), and the test week (Sept 28 →
  Oct 5) may not look like the training weeks.
"""),
    md("""
## Variant: without r/Silverbugs

Hand-labelling showed that r/Silverbugs is mostly **silver collecting** with no view on the gold
price. This variant rebuilds the hourly table from the **same stored posts** minus r/Silverbugs,
keeping r/Gold and r/investing:

    python -m processing.bucket --end 2026-10-05T15:00 --exclude-subreddits Silverbugs \\
        --output data/clean/hourly_merged_no_silverbugs.parquet
    python -m processing.features --input data/clean/hourly_merged_no_silverbugs.parquet \\
        --output data/clean/features_no_silverbugs.parquet

Nothing was re-collected or re-scored. Silverbugs is dropped **before** de-duplication, so a story
cross-posted to both communities keeps its r/Gold copy.

The target depends only on price, so **both variants are tested on exactly the same 124 hours**.
That allows a paired comparison. Models 1 and 2 use no sentiment, so they must score identically.
"""),
    code("""
features_v = pd.read_parquet("data/clean/features_no_silverbugs.parquet")
data_v = features_v[features_v["target_valid"]].copy()
data_v["y"] = data_v["price_direction"].astype(int)
assert data_v.index.equals(data.index) and (data_v["y"] == data["y"]).all()
assert data_v[ALL_FEATURES].notna().all().all()
train_v, test_v = data_v[data_v.index < SPLIT], data_v[data_v.index >= SPLIT]
assert train_v.index.max() < test_v.index.min() and (len(train_v), len(test_v)) == (307, 124)

for label, f in [("all communities", features), ("no Silverbugs", features_v)]:
    v = f[f["target_valid"]]
    print(f"{label:<16} posts/hour {f['post_count'].mean():.2f}   hours with no posts "
          f"{(~f['had_posts']).mean():.0%}   target-valid hours with posts {v['had_posts'].mean():.0%}")

results_v, preds_v, MODELS_V = fit_all(train_v, test_v)
for name in ["1 majority class", "2 logistic, price-only"]:
    assert (preds_v[name] == preds[name]).all(), f"{name} should not depend on sentiment"

combined = pd.concat([results.assign(dataset="all communities"),
                      results_v.assign(dataset="no Silverbugs")], ignore_index=True)
combined = combined[["dataset"] + [c for c in combined.columns if c != "dataset"]]
Path("reports").mkdir(exist_ok=True)   # reports/model_results.csv is written once, below
combined.pivot(index="model", columns="dataset", values="test_acc").map("{:.1%}".format)
"""),
    code("""
def mcnemar(a, b, y):
    \"\"\"Exact McNemar on paired predictions: (only a right, only b right, p).\"\"\"
    only_a, only_b = int(((a == y) & (b != y)).sum()), int(((b == y) & (a != y)).sum())
    return only_a, only_b, (binomtest(only_b, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0)

m3, m3v = preds["3 logistic, price + sentiment"], preds_v["3 logistic, price + sentiment"]
m4, m4v = preds["4 decision tree (depth 3), price + sentiment"], preds_v["4 decision tree (depth 3), price + sentiment"]
p2_, p1_ = preds["2 logistic, price-only"], preds["1 majority class"]
for label, a, b in [("model 3: all communities vs no Silverbugs", m3, m3v),
                    ("model 4: all communities vs no Silverbugs", m4, m4v),
                    ("no Silverbugs: price-only vs model 3", p2_, m3v),
                    ("no Silverbugs: majority vs model 3", p1_, m3v),
                    ("no Silverbugs: price-only vs model 4", p2_, m4v),
                    ("no Silverbugs: majority vs model 4", p1_, m4v)]:
    oa, ob, p = mcnemar(a, b, y)
    print(f"{label:<44} only first right {oa:>3}   only second right {ob:>3}   exact McNemar p = {p:.3f}")
# Four sentiment models were tried against price-only (2 models x 2 datasets): more tries mean
# more chances of a lucky win. Bonferroni: require p < 0.05 / 4.
print("\\nmultiple-comparison threshold (Bonferroni, 4 sentiment models vs price-only): p < 0.0125")
print()
for _, r in results_v.iterrows():
    print(f"{r['model']:<46} test {r['test_acc']:.1%}  [95% CI {r['ci95_low']:.1%} – {r['ci95_high']:.1%}]  "
          f"train {r['train_acc']:.1%}")
"""),
    code("""
fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.4))
for row, (label, p) in enumerate([("all communities", preds), ("no Silverbugs", preds_v)]):
    for col, name in enumerate(["3 logistic, price + sentiment", "4 decision tree (depth 3), price + sentiment"]):
        ax = axes[row, col]
        ConfusionMatrixDisplay(confusion_matrix(y, p[name], labels=[0, 1]), display_labels=labels_txt
                               ).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{label}: {name.split(',')[0]}\\naccuracy {(p[name] == y).mean():.1%}", fontsize=9.5)
fig.suptitle("Sentiment models with and without r/Silverbugs, same 124 test hours", y=1.01)
fig.tight_layout()
fig.savefig(FIGURES / "model_cm_all_vs_no_silverbugs.png", dpi=150, bbox_inches="tight")

lr3v = MODELS_V["3 logistic, price + sentiment"][0]
pd.DataFrame({"all communities": pd.Series(lr3[-1].coef_[0], index=ALL_FEATURES),
              "no Silverbugs": pd.Series(lr3v[-1].coef_[0], index=ALL_FEATURES)}
             ).sort_values("no Silverbugs", key=abs, ascending=False).round(3)
"""),
    md("""
### Sensitivity check: is the variant tree's score stable?
**This is not tuning.** The reported model stays fixed at depth 3 and the Sept 28 split. The check
only asks whether the score survives small, arbitrary changes. A real signal should; a lucky one
won't.
"""),
    code("""
rows_s = []
for depth in [2, 3, 4]:
    for split in ["2026-09-26", "2026-09-28", "2026-09-30"]:
        s = pd.Timestamp(split, tz="UTC")
        for label, d in [("all communities", data), ("no Silverbugs", data_v)]:
            tr, te = d[d.index < s], d[d.index >= s]
            tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(tr[ALL_FEATURES], tr["y"])
            rows_s.append({"dataset": label, "depth": depth, "split": split, "n_test": len(te),
                           "tree_acc": (tree.predict(te[ALL_FEATURES]) == te["y"]).mean(),
                           "majority_acc": (te["y"] == tr["y"].mode()[0]).mean()})
sens = pd.DataFrame(rows_s)
sens.pivot_table(index=["dataset", "depth"], columns="split", values="tree_acc").map("{:.1%}".format)
"""),
    md("""
### Reading the variant
* Dropping r/Silverbugs halves the posts (2,419 → 1,146), so far more hours have no post and
  carry a forward-filled mood. The mood clues get **more on-topic but thinner**.
* The same rule applies as above: a difference only counts if McNemar p < 0.05. Because we have now
  tried **four** sentiment models (2 models × 2 datasets) against price-only, the stricter
  Bonferroni bar is **p < 0.0125**. Trying more combinations makes a lucky win more likely.
* Watch for the logistic and tree models disagreeing. If removing Silverbugs helps one model and
  hurts the other, that points to noise, not a real signal.
* `data/clean/features.parquet` (all communities) remains the main feature table for Track C. The
  variant file exists only for this comparison.
"""),
    md("""
## Second scorer: FinBERT (with and without r/Silverbugs)

Same posts, same de-duplication, same features, same split. Only the per-post score changes:
VADER's compound is replaced by FinBERT's P(positive) − P(negative), and the bullish/bearish ratios come
from FinBERT's most likely class (see `02_features.ipynb` §4 and `sentiment/finbert.py`):

    python -m sentiment.finbert
    python -m processing.bucket --end 2026-10-05T15:00 --sentiment-file data/clean/post_sentiment_finbert.parquet \\
        --output data/clean/hourly_merged_finbert.parquet
    python -m processing.features --input data/clean/hourly_merged_finbert.parquet \\
        --output data/clean/features_finbert.parquet
    # + the same two steps with --exclude-subreddits Silverbugs -> *_finbert_no_silverbugs.parquet

The test hours are identical, so every comparison is paired. With **8 sentiment models** now tried
against price-only (2 models × 2 datasets × 2 scorers), the Bonferroni bar is **p < 0.05 / 8 = 0.00625**.
"""),
    code("""
datasets = {"VADER, all communities": (data, results, preds),
            "VADER, no Silverbugs": (data_v, results_v, preds_v)}
for label, path in [("FinBERT, all communities", "data/clean/features_finbert.parquet"),
                    ("FinBERT, no Silverbugs", "data/clean/features_finbert_no_silverbugs.parquet")]:
    f = pd.read_parquet(path)
    d = f[f["target_valid"]].copy()
    d["y"] = d["price_direction"].astype(int)
    assert d.index.equals(data.index) and (d["y"] == data["y"]).all()
    assert d[ALL_FEATURES].notna().all().all()
    tr, te = d[d.index < SPLIT], d[d.index >= SPLIT]
    assert (len(tr), len(te)) == (307, 124)
    r, p, _ = fit_all(tr, te)
    for name in ["1 majority class", "2 logistic, price-only"]:
        assert (p[name] == preds[name]).all(), f"{name} should not depend on sentiment"
    datasets[label] = (d, r, p)

combined = pd.concat([r.assign(dataset=k) for k, (_, r, _) in datasets.items()], ignore_index=True)
combined = combined[["dataset"] + [c for c in combined.columns if c != "dataset"]]
# Replace only this notebook's datasets: the file also holds rows from notebooks 05-07
# (economy mood, daily, Bloomberg), and the daily/Bloomberg ones need data not in the repository.
RESULTS = Path("reports/model_results.csv")
others = pd.read_csv(RESULTS) if RESULTS.exists() else combined.iloc[:0]
others = others[~others["dataset"].isin(combined["dataset"])]
pd.concat([combined.round(4), others], ignore_index=True).to_csv(RESULTS, index=False)
combined.pivot(index="model", columns="dataset", values="test_acc")[list(datasets)].map("{:.1%}".format)
"""),
    code("""
BONFERRONI = 0.05 / 8
M3, M4 = "3 logistic, price + sentiment", "4 decision tree (depth 3), price + sentiment"
rows_t = []
for label, (_, _, p) in datasets.items():
    for m in [M3, M4]:
        oa, ob, pv = mcnemar(p2_, p[m], y)
        ma, mb, pm = mcnemar(p1_, p[m], y)
        rows_t.append({"dataset": label, "model": m.split(",")[0], "test_acc": (p[m] == y).mean(),
                       "vs price-only: only price right": oa, "only sentiment right": ob, "p": pv,
                       "vs majority p": pm, f"beats price-only at p<{BONFERRONI:.4f}": pv < BONFERRONI and ob > oa})
display(pd.DataFrame(rows_t).round(3))
for m in [M3, M4]:
    oa, ob, pv = mcnemar(datasets["VADER, all communities"][2][m], datasets["FinBERT, all communities"][2][m], y)
    print(f"{m.split(',')[0]}: VADER vs FinBERT (all communities)  only VADER right {oa}  only FinBERT right {ob}  p = {pv:.3f}")
"""),
    code("""
rows_s = []
for depth in [2, 3, 4]:
    for split in ["2026-09-26", "2026-09-28", "2026-09-30"]:
        s = pd.Timestamp(split, tz="UTC")
        for label, (d, _, _) in datasets.items():
            tr, te = d[d.index < s], d[d.index >= s]
            tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(tr[ALL_FEATURES], tr["y"])
            rows_s.append({"dataset": label, "depth": depth, "split": split,
                           "tree_acc": (tree.predict(te[ALL_FEATURES]) == te["y"]).mean()})
print("Tree sensitivity across all four datasets (not tuning; the reported model stays depth 3, Sept 28):")
pd.DataFrame(rows_s).pivot_table(index=["dataset", "depth"], columns="split", values="tree_acc").map("{:.1%}".format)
"""),
    md("""
### Reading the FinBERT comparison
* Which scorer to recommend was decided **before** this table, on agreement with the human labels
  (`02_features.ipynb` §4), not on which one happens to score higher here.
* FinBERT calls most posts neutral, so its hourly features are much flatter (many hours with a
  bullish/bearish ratio of 0). A better-measured mood can still carry no next-hour price signal.
* The same rules apply: a difference only counts at **p < 0.00625**, and only if it also survives the
  sensitivity grid above. Logistic and tree models moving in opposite directions point to noise.
"""),
]

# --------------------------------------------------------------------------- 05
NB05 = [
    md("""
# 05 · Economy-mood variant: does talk about inflation, rates and the Fed predict gold?

Gold is a safe-haven asset, so economy worries may move it. This notebook tests posts about
**inflation, interest rates, the Fed, yields and recession** from r/economy, r/Economics,
r/wallstreetbets, r/stocks and r/CanadianInvestor, over the same frozen window.

* Posts: `data/clean/macro_posts.parquet`, built by `python -m collector.macro_backfill` (same removal,
  language and de-duplication rules as the gold data; keyword list fixed in advance).
* Scorer: FinBERT used as is. Here positive / negative = **good / bad news for the economy**.
* Features: `python -m processing.macro features`, which joins hourly economy features onto the main
  `features.parquet`, so the target and test hours are unchanged.
* **The model learns the direction.** Bad-economy talk can mean gold up (panic, inflation) or down
  (rate hikes, high yields). We don't hard-code either; the coefficients show what was learned.

**Supervised learning only.** Two new models: **5** logistic regression and **6** depth-3 tree, both on
price + economy features, against the same majority-class and price-only baselines.

### Success criteria, fixed before any labels or results were seen
The economy variant becomes the headline result **only if all three hold**:
1. **Labellers agree with each other:** Cohen's κ ≥ 0.40 on the 15 posts labelled twice.
2. **FinBERT passes validation:** agreement above always guessing the most common human label,
   *and* balanced score (mean agreement per human class) ≥ 0.50 (chance = 0.33).
3. **A model beats price-only by more than luck:** exact McNemar p < 0.005 (Bonferroni: 10 sentiment
   models tried in the project) on the 124 test hours, *and* it is robust. The tree must beat price-only
   in ≥ 7 of 9 cells (depth 2/3/4 × split Sept 26/28/30). The logistic model has no depth, so it must
   beat price-only at all 3 split dates.
"""),
    code(SETUP + """
from processing.features import PRICE_FEATURES, SENTIMENT_FEATURES
from processing.macro import MACRO_FEATURES

base = pd.read_parquet("data/clean/features.parquet")
features = pd.read_parquet("data/clean/features_macro.parquet")
pd.testing.assert_frame_equal(features[base.columns], base)   # main table carried over unchanged
macro_posts = pd.read_parquet("data/clean/macro_posts.parquet")
print(f"{len(macro_posts)} de-duplicated economy posts; {int(features['macro_had_posts'].sum())} of "
      f"{len(features)} hours have at least one ({features['macro_had_posts'].mean():.0%})")
"""),
    md("## 1. What the keyword filter kept"),
    code("""
print(pd.crosstab(macro_posts["subreddit"], macro_posts["finbert_label"], margins=True).to_string())
print("\\nRandom examples (by eye: is it economy talk?):")
if "text" in macro_posts:
    for _, r in macro_posts.sample(10, random_state=0).iterrows():
        print(f"  [{r['subreddit']}] {r['text'][:110]}")
else:  # the published table keeps only a hash of each post's text
    print("  (post text is not published; data/validation/macro_sample.csv has 45 examples with text)")
"""),
    md("""
## 2. Validation: do people agree with each other, and with FinBERT?
`data/validation/macro_sample.csv` (45 posts, 15 per FinBERT class, labelled blind with no scorer
column; rule in `data/validation/MACRO_LABELLING.md`) and `macro_overlap.csv` (its first 15 posts,
labelled a second time by a different person).
"""),
    code("""
from sklearn.metrics import cohen_kappa_score, confusion_matrix

LABELS = ["bad", "neutral", "good"]
def read_labels(path):
    d = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
    d["human_label"] = d["human_label"].str.strip().str.lower()
    return d

sample, overlap = read_labels("data/validation/macro_sample.csv"), read_labels("data/validation/macro_overlap.csv")
labelled = sample["human_label"].isin(LABELS).all() and overlap["human_label"].isin(LABELS).all()
kappa = agree_f = yard = balanced = None
if not labelled:
    print(f"Awaiting labels: {sample['human_label'].isin(LABELS).sum()}/45 in the sample, "
          f"{overlap['human_label'].isin(LABELS).sum()}/15 in the overlap file.")
else:
    pair = sample.merge(overlap, on="post_id", suffixes=("_1", "_2"), validate="1:1")
    kappa = cohen_kappa_score(pair["human_label_1"], pair["human_label_2"])
    print(f"Labeller agreement on {len(pair)} double-labelled posts: {(pair['human_label_1'] == pair['human_label_2']).mean():.0%} "
          f"identical, Cohen's kappa = {kappa:.2f}  ({', '.join(sorted(set(pair['labelled_by_1']) | set(pair['labelled_by_2'])))})")
    both = sample.merge(macro_posts[["post_id", "finbert_score", "finbert_label"]], on="post_id", validate="1:1")
    ok = both["human_label"] == both["finbert_label"]
    agree_f = ok.mean()
    majority = both["human_label"].value_counts()
    yard = majority.iloc[0] / len(both)
    balanced = ok.groupby(both["human_label"]).mean().mean()
    print(f"FinBERT vs human labels: agreement {agree_f:.1%} ({ok.sum()}/{len(both)}), balanced {balanced:.2f}")
    print(f"Yardstick, always '{majority.index[0]}': agreement {yard:.1%}, balanced {1 / both['human_label'].nunique():.2f}")
    print("Agreement by human label:", ok.groupby(both["human_label"]).mean().round(2).to_dict())
    display(pd.DataFrame(confusion_matrix(both["human_label"], both["finbert_label"], labels=LABELS),
                         index=[f"human {l}" for l in LABELS], columns=[f"FinBERT {l}" for l in LABELS]))
    print("FinBERT disagreements:")
    display(both[~ok][["subreddit", "finbert_score", "finbert_label", "human_label", "text"]])
"""),
    md("## 3. Economy mood vs. gold price"),
    code("""
import matplotlib.dates as mdates
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax1.plot(features.index, features["macro_mood_roll_mean_6h"], color="tab:red", lw=1.4, label="economy mood, FinBERT (6h)")
ax1.axhline(0, color="gray", lw=0.5, ls=":")
ax1.set_ylabel("economy mood (−1 bad … +1 good)")
ax2 = ax1.twinx()
ax2.plot(features.index, features["xau_usd"], color="goldenrod", lw=1.6, label="GC=F gold futures (USD/oz)")
ax2.set_ylabel("gold price, USD/oz (GC=F)", color="darkgoldenrod")
lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
ax1.legend(lines, [l.get_label() for l in lines], loc="lower left", fontsize=9)
ax1.set_title("Economy talk on Reddit (inflation, rates, Fed) vs. gold price, Sept 8 – Oct 5, 2026 (UTC)")
ax3.bar(features.index, features["macro_post_count"], width=1 / 24, color="slategray")
ax3.set_ylabel("posts / hour")
ax3.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig.tight_layout()
fig.savefig(FIGURES / "eda_5_economy_mood_vs_price.png", dpi=150, bbox_inches="tight")
"""),
    md("""
## 4. Models: price-only vs. price + economy features
Same split as `04_model.ipynb`: train before Sept 28 00:00 UTC (307 hours), test Sept 28 → Oct 5 (124).
"""),
    code("""
from scipy.stats import binomtest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

data = features[features["target_valid"]].copy()
data["y"] = data["price_direction"].astype(int)
ECON = PRICE_FEATURES + MACRO_FEATURES
assert data[ECON].notna().all().all()
SPLIT = pd.Timestamp("2026-09-28 00:00", tz="UTC")
train, test = data[data.index < SPLIT], data[data.index >= SPLIT]
assert train.index.max() < test.index.min() and (len(train), len(test)) == (307, 124)

M5, M6 = "5 logistic, price + economy", "6 decision tree (depth 3), price + economy"
def make_models(depth=3):
    return {"1 majority class": (DummyClassifier(strategy="most_frequent"), []),
            "2 logistic, price-only": (make_pipeline(StandardScaler(), LogisticRegression()), PRICE_FEATURES),
            M5: (make_pipeline(StandardScaler(), LogisticRegression()), ECON),
            M6: (DecisionTreeClassifier(max_depth=depth, random_state=0), ECON)}

def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half

def fit_all(train, test, depth=3):
    models, rows, preds = make_models(depth), [], {}
    for name, (model, cols) in models.items():
        X_tr = train[cols] if cols else np.zeros((len(train), 1))
        X_te = test[cols] if cols else np.zeros((len(test), 1))
        model.fit(X_tr, train["y"])
        preds[name] = model.predict(X_te)
        k = int((preds[name] == test["y"]).sum())
        lo, hi = wilson(k, len(test))
        rows.append({"model": name, "n_features": len(cols), "n_train": len(train), "n_test": len(test),
                     "train_acc": (model.predict(X_tr) == train["y"]).mean(), "test_acc": k / len(test),
                     "ci95_low": lo, "ci95_high": hi, "pred_up_share": preds[name].mean()})
    return pd.DataFrame(rows), preds, models

def mcnemar(a, b, y):
    only_a, only_b = int(((a == y) & (b != y)).sum()), int(((b == y) & (a != y)).sum())
    return only_a, only_b, (binomtest(only_b, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0)

results, preds, MODELS = fit_all(train, test)
prior = pd.read_csv("reports/model_results.csv")
prior = prior[prior["dataset"] == "VADER, all communities"].set_index("model")["test_acc"]
for name in ["1 majority class", "2 logistic, price-only"]:   # baselines must match 04_model exactly
    assert round(results.set_index("model").loc[name, "test_acc"], 4) == round(prior[name], 4), name
results.round(3)
"""),
    code("""
y = test["y"].to_numpy()
BONFERRONI = 0.005
p2, p1 = preds["2 logistic, price-only"], preds["1 majority class"]
tests = {}
for m in [M5, M6]:
    oa, ob, pv = mcnemar(p2, preds[m], y)
    ma, mb, pm = mcnemar(p1, preds[m], y)
    tests[m] = (oa, ob, pv)
    print(f"{m:<44} test {(preds[m] == y).mean():.1%}   vs price-only: only price right {oa}, "
          f"only economy right {ob}, p = {pv:.3f}   vs majority p = {pm:.3f}")

coef = pd.Series(MODELS[M5][0][-1].coef_[0], index=ECON)
print("\\nLogistic coefficients on economy features (per 1 sd; + means higher value -> gold more likely up):")
print(coef[MACRO_FEATURES].sort_values(key=abs, ascending=False).round(3).to_string())
"""),
    code("""
grid, wins = [], {M5: 0, M6: 0}
for split in ["2026-09-26", "2026-09-28", "2026-09-30"]:
    s = pd.Timestamp(split, tz="UTC")
    tr, te = data[data.index < s], data[data.index >= s]
    for depth in [2, 3, 4]:
        r, p, _ = fit_all(tr, te, depth)
        acc = r.set_index("model")["test_acc"]
        grid.append({"split": split, "depth": depth, "price-only": acc["2 logistic, price-only"],
                     "tree (economy)": acc[M6], "logistic (economy)": acc[M5]})
        wins[M6] += acc[M6] > acc["2 logistic, price-only"]
        if depth == 3:  # logistic has no depth: count each split once
            wins[M5] += acc[M5] > acc["2 logistic, price-only"]
print(f"Sensitivity: tree beats price-only in {wins[M6]} of 9 cells; logistic in {wins[M5]} of 3 splits")
pd.DataFrame(grid).set_index(["split", "depth"]).map("{:.1%}".format)
"""),
    code("""
out = pd.read_csv("reports/model_results.csv")
out = pd.concat([out[out["dataset"] != "economy mood"], results.assign(dataset="economy mood")[out.columns]],
                ignore_index=True)
out.round(4).to_csv("reports/model_results.csv", index=False)
"""),
    md("## 5. Verdict against the pre-set criteria"),
    code("""
def verdict(ok):
    return "PENDING (labels)" if ok is None else ("PASS" if ok else "FAIL")
c1 = None if kappa is None else kappa >= 0.40
c2 = None if agree_f is None else (agree_f > yard and balanced >= 0.50)
robust = {M5: wins[M5] == 3, M6: wins[M6] >= 7}
c3 = any(tests[m][2] < BONFERRONI and tests[m][1] > tests[m][0] and robust[m] for m in [M5, M6])
print(f"1. labellers agree (kappa >= 0.40):          {verdict(c1)}" + ("" if kappa is None else f"   kappa = {kappa:.2f}"))
print(f"2. FinBERT validated (beats yardstick, bal >= 0.50): {verdict(c2)}"
      + ("" if agree_f is None else f"   {agree_f:.1%} vs {yard:.1%}, balanced {balanced:.2f}"))
print(f"3. model beats price-only (p < 0.005, robust):  {verdict(c3)}")
print("\\nOVERALL:", "economy mood becomes the headline result" if (c1 and c2 and c3)
      else "PENDING" if None in (c1, c2) else "reported as a tested, null variant; the VADER table stays the main one")
"""),
]

# --------------------------------------------------------------------------- 06
NB06 = [
    md("""
# 06 · Long daily study: does Reddit predict tomorrow's gold move, 2022–2026?

The hourly study (`04_model.ipynb`) tested on **124 hours from one week**, too few to tell a real effect
from luck. Here the question moves to **days** and the history to **Jan 2022 → Oct 2 2026**:

* **Reddit:** r/Gold, r/Silverbugs and keyword-filtered r/investing, backfilled from Arctic Shift
  (`python -m collector.history_backfill`). Same cleaning, language filter, de-duplication and VADER score
  as the live collector. *VADER is not validated against our hand labels* (see `02_features.ipynb`), so mood
  features come from an unvalidated scorer. **Chatter** features (how many posts) need no scorer at all.
* **Market:** the Bloomberg terminal pull (`python -m processing.bloomberg`, git-ignored, licensed): spot gold
  (XAU), dollar index, 10-year real yield, gold volatility (GVZ), VIX, S&P 500, GLD+IAU ETF flows and
  Bloomberg's news sentiment for five gold miners.
* **Table:** `python -m processing.daily` → `data/bbg/daily_features.parquet`, one row per trading day.

**Timing (pitfalls #34–#36).** Bloomberg's daily gold close is New York 17:00 (21:00 UTC in summer). Reddit day
*d* is every post up to **20:00 UTC on d**, so it is complete before that close. **Target:** does gold close
higher on d+1 than on d? ETF flows and miner news have no published cut-off, so they are lagged a day.

**Walk-forward test.** Train on everything before a month, predict that month, add it to the training data,
repeat. That is about 940 test days from Jan 2023 → Oct 2026, every prediction made only with the past.

**Models (supervised only):**
1 majority class · 2 logistic, gold returns only · **3 logistic, market baseline** (what professionals already
see) · 7 logistic, market + Reddit mood + chatter · 8 decision tree (depth 3), market + mood + chatter ·
9 logistic, market + chatter only (no sentiment scorer).

### Success criteria, fixed before any results were seen
Reddit "adds something" **only if one model among 7, 8, 9 meets both**:

* **A. Better than the market baseline by more than luck:** more test days right than model 3, exact McNemar
  p **< 0.004** (Bonferroni: 13 Reddit models tried in the project so far).
* **B. Robust:** the same model also beats model 3 in **both halves** of the test period (2023–24 and 2025–26)
  **and** with **quarterly** instead of monthly refits. For the tree (8), it must also beat model 3 at depth 2
  **and** depth 4.

If no model meets both, the answer is: *over ~4 years of daily data, Reddit adds nothing measurable to what
the market already shows.*
"""),
    code(SETUP + """
from processing.daily import (DAILY_CHATTER_FEATURES, DAILY_MARKET_FEATURES, DAILY_MOOD_FEATURES,
                              DAILY_PRICE_FEATURES)
daily = pd.read_parquet("data/bbg/daily_features.parquet")
posts = pd.read_parquet("data/clean/history_posts.parquet")
print(f"{len(posts):,} Reddit posts backfilled, {posts['created_utc'].min()[:10]} -> {posts['created_utc'].max()[:10]}")
print(posts.groupby("subreddit").size().to_string())
print(f"\\n{len(daily)} trading days {daily.index.min():%Y-%m-%d} -> {daily.index.max():%Y-%m-%d}; "
      f"{int(daily['post_count'].sum()):,} de-duplicated posts assigned; gold up on {daily['up'].mean():.1%} of days")
"""),
    md("## 1. Checks: the backfill is complete and the day cut-off is safe"),
    code("""
stats = pd.read_csv("data/clean/history/_stats.csv")
print("Posts kept per subreddit-month (min / median / max):")
print(stats.groupby("subreddit")["ok"].agg(["min", "median", "max"]).to_string())
assert (stats.loc[stats["subreddit"] != "investing", "ok"] > 0).all(), "a month with no posts: backfill gap"

# Bloomberg's daily XAU close should equal the hourly spot price at 21:00 UTC (NY 17:00 summer time).
spot = pd.read_parquet("data/bbg/xau_spot_hourly.parquet").set_index("bucket_utc")["xau_usd"]
close = pd.read_parquet("data/bbg/market_daily.parquet")["XAU_PX_LAST"]
match = {}
for h in [19, 20, 21, 22]:
    at = (close.index + pd.Timedelta(hours=h)).tz_localize("UTC")
    both = pd.DataFrame({"close": close.to_numpy(), "spot": spot.reindex(at).to_numpy()}).dropna()
    match[f"{h}:00 UTC"] = f"{((both['close'] - both['spot']).abs() < 0.01).mean():.0%} of {len(both)} days identical"
print(pd.Series(match).to_string())
print("Reddit cut-off: 20:00 UTC, one hour before the earliest close time.")
"""),
    md("## 2. Reddit and gold, four years at a glance"),
    code("""
import matplotlib.dates as mdates
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
ax1.plot(daily.index, daily["mean_sentiment"].rolling(20).mean(), color="tab:blue", lw=1.3,
         label="Reddit mood, VADER (20-day mean)")
ax1.set_ylabel("Reddit mood (VADER)")
ax2 = ax1.twinx()
ax2.plot(daily.index, daily["xau_close"], color="goldenrod", lw=1.6, label="spot gold (XAU, USD/oz)")
ax2.set_ylabel("gold price, USD/oz", color="darkgoldenrod")
lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9)
ax1.set_title("Reddit gold communities vs. spot gold, Jan 2022 – Oct 2026 (daily)")
ax3.plot(daily.index, daily["posts_per_day"].rolling(20).mean(), color="slategray")
ax3.set_ylabel("posts / day (20-day mean)")
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.tight_layout()
fig.savefig(FIGURES / "daily_1_reddit_vs_gold.png", dpi=150, bbox_inches="tight")
"""),
    md("## 3. Walk-forward models"),
    code("""
from scipy.stats import binomtest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

REDDIT = DAILY_MARKET_FEATURES + DAILY_MOOD_FEATURES + DAILY_CHATTER_FEATURES
CHATTER = DAILY_MARKET_FEATURES + DAILY_CHATTER_FEATURES
data = daily[daily["target_valid"]].dropna(subset=REDDIT).copy()   # drops the warm-up weeks only
data["y"] = data["up"].astype(int)
TEST_START, HALF = pd.Timestamp("2023-01-01"), pd.Timestamp("2025-01-01")
print(f"{len(data)} usable days from {data.index.min():%Y-%m-%d}; "
      f"{(data.index >= TEST_START).sum()} test days from {TEST_START:%Y-%m-%d}")

M1, M2, M3 = "1 majority class", "2 logistic, gold returns only", "3 logistic, market baseline"
M7, M8, M9 = ("7 logistic, market + Reddit mood + chatter", "8 decision tree, market + Reddit mood + chatter",
              "9 logistic, market + chatter (no scorer)")
REDDIT_MODELS = [M7, M8, M9]

def make_models(depth=3):
    logit = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return {M1: (DummyClassifier(strategy="most_frequent"), []), M2: (logit(), DAILY_PRICE_FEATURES),
            M3: (logit(), DAILY_MARKET_FEATURES), M7: (logit(), REDDIT),
            M8: (DecisionTreeClassifier(max_depth=depth, random_state=0), REDDIT), M9: (logit(), CHATTER)}

def walk_forward(data, freq="MS", depth=3):
    \"\"\"Refit every model at the start of each period on all earlier days; predict that period.\"\"\"
    starts = pd.date_range(TEST_START, data.index.max(), freq=freq).union([TEST_START])
    preds, last = {m: [] for m in make_models(depth)}, None
    for a, b in zip(starts, list(starts[1:]) + [data.index.max() + pd.Timedelta(days=1)]):
        train, test = data[data.index < a], data[(data.index >= a) & (data.index < b)]
        if test.empty:
            continue
        assert train.index.max() < test.index.min()
        last = make_models(depth)
        for name, (model, cols) in last.items():
            X_tr = train[cols] if cols else np.zeros((len(train), 1))
            X_te = test[cols] if cols else np.zeros((len(test), 1))
            model.fit(X_tr, train["y"])
            preds[name].append(pd.Series(model.predict(X_te), index=test.index))
    return pd.DataFrame({m: pd.concat(p) for m, p in preds.items()}), last

def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half

def mcnemar(a, b, y):
    only_a, only_b = int(((a == y) & (b != y)).sum()), int(((b == y) & (a != y)).sum())
    return only_a, only_b, (binomtest(only_b, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0)

preds, final_models = walk_forward(data)
y = data.loc[preds.index, "y"]
rows = []
for m in preds:
    k = int((preds[m] == y).sum())
    lo, hi = wilson(k, len(y))
    rows.append({"model": m, "n_features": len(make_models()[m][1]), "n_train": int((data.index < TEST_START).sum()),
                 "n_test": len(y), "train_acc": np.nan, "test_acc": k / len(y), "ci95_low": lo, "ci95_high": hi,
                 "pred_up_share": preds[m].mean()})
results = pd.DataFrame(rows)
print("n_train = days before the first test month (it grows by a month at each refit)")
results.round(3)
"""),
    code("""
BONFERRONI = 0.004
tests = {}
for m in REDDIT_MODELS:
    oa, ob, pv = mcnemar(preds[M3], preds[m], y)
    tests[m] = (oa, ob, pv)
    print(f"{m:<50} {(preds[m] == y).mean():.1%}   vs market baseline: only baseline right {oa}, "
          f"only Reddit model right {ob}, p = {pv:.3f}")
for m in [M2, M3]:
    print(f"{m:<50} {(preds[m] == y).mean():.1%}   vs majority p = {mcnemar(preds[M1], preds[m], y)[2]:.3f}")

coef = pd.Series(final_models[M7][0][-1].coef_[0], index=REDDIT)
print("\\nModel 7, last refit: coefficients per 1 sd (+ means higher value -> gold more likely up tomorrow)")
print(coef.sort_values(key=abs, ascending=False).round(3).to_string())
"""),
    code("""
fig, ax = plt.subplots(figsize=(13, 4.2))
for m, colour in [(M1, "lightgray"), (M3, "black"), (M7, "tab:blue"), (M8, "tab:green"), (M9, "tab:orange")]:
    ax.plot(preds.index, (preds[m] == y).astype(float).rolling(120).mean(), label=m, color=colour, lw=1.3)
ax.axhline(0.5, color="gray", ls=":", lw=0.8)
ax.set_ylabel("accuracy, trailing 120 test days")
ax.set_title("Walk-forward accuracy: Reddit models vs. the market baseline")
ax.legend(fontsize=8, ncol=2, loc="lower left")
fig.tight_layout()
fig.savefig(FIGURES / "daily_2_walkforward_accuracy.png", dpi=150, bbox_inches="tight")
"""),
    md("## 4. Robustness (criterion B)"),
    code("""
def acc(p, idx):
    return (p.loc[idx] == y.loc[idx]).mean()

halves = {"2023–24": preds.index[preds.index < HALF], "2025–26": preds.index[preds.index >= HALF]}
q_preds, _ = walk_forward(data, freq="QS")
depth_preds = {d: walk_forward(data, depth=d)[0] for d in [2, 4]}
rob = []
for m in REDDIT_MODELS:
    row = {"model": m}
    for h, idx in halves.items():
        row[f"{h} model / baseline"] = f"{acc(preds[m], idx):.1%} / {acc(preds[M3], idx):.1%}"
        row[f"wins {h}"] = acc(preds[m], idx) > acc(preds[M3], idx)
    row["quarterly model / baseline"] = f"{(q_preds[m] == y).mean():.1%} / {(q_preds[M3] == y).mean():.1%}"
    row["wins quarterly"] = (q_preds[m] == y).mean() > (q_preds[M3] == y).mean()
    if m == M8:
        for d, p in depth_preds.items():
            row[f"wins depth {d}"] = (p[M8] == y).mean() > (p[M3] == y).mean()
    rob.append(row)
rob = pd.DataFrame(rob).set_index("model")
rob
"""),
    code("""
out = pd.read_csv("reports/model_results.csv")
DATASET = "daily 2022–26 walk-forward"
out = pd.concat([out[out["dataset"] != DATASET], results.assign(dataset=DATASET)[out.columns]], ignore_index=True)
out.round(4).to_csv("reports/model_results.csv", index=False)
"""),
    md("## 5. Verdict against the pre-set criteria"),
    code("""
verdicts = {}
for m in REDDIT_MODELS:
    oa, ob, pv = tests[m]
    a = ob > oa and pv < BONFERRONI
    b = bool(rob.loc[m].filter(like="wins").fillna(True).all())
    verdicts[m] = a and b
    print(f"{m:<50} A (p < 0.004, more right): {'PASS' if a else 'FAIL'} (p = {pv:.3f})   "
          f"B (robust): {'PASS' if b else 'FAIL'}")
print("\\nOVERALL:", "Reddit adds measurable information to the market baseline" if any(verdicts.values())
      else "over ~4 years of daily data, Reddit adds nothing measurable to what the market already shows")
"""),
]

# --------------------------------------------------------------------------- 07
NB07 = [
    md("""
# 07 · Bloomberg extras: professional news vs. Reddit, and a second gold price

Two side checks using the Bloomberg pull. Neither is a new model of Reddit; both describe or stress-test.

1. **Do Reddit and professional news move together?** Daily Reddit mood (VADER, unvalidated) against
   Bloomberg's own news sentiment for five gold miners (NEM, AEM, KGC, WPM, FNV; Bloomberg has **no**
   sentiment field for gold itself, pitfall #37), and against "real money" GLD+IAU ETF flows. Also weekly
   Reddit post counts against Bloomberg's weekly count of gold news stories (2024+). Correlations only,
   so nothing is learned from the data.
2. **Does the hourly result depend on the price source?** `04_model.ipynb` used Yahoo's GC=F futures.
   Here the same four models run on Bloomberg **spot** gold (XAU), with the same split.
"""),
    code(SETUP + """
from scipy.stats import spearmanr
daily = pd.read_parquet("data/bbg/daily_features.parquet")
"""),
    md("## 1. Reddit mood vs. Bloomberg miner news mood, daily 2022–2026"),
    code("""
import matplotlib.dates as mdates
news = daily["miner_news"]
reddit = daily["mean_sentiment"].where(daily["had_posts"])
fig, ax1 = plt.subplots(figsize=(13, 4.5))
ax1.plot(daily.index, reddit.rolling(20, min_periods=10).mean(), color="tab:blue", lw=1.3, label="Reddit mood, VADER (20-day mean)")
ax1.set_ylabel("Reddit mood (VADER)", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(daily.index, news.rolling(20, min_periods=10).mean(), color="tab:red", lw=1.3, label="Bloomberg miner news sentiment (20-day mean)")
ax2.set_ylabel("Bloomberg news sentiment", color="tab:red")
lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9)
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.set_title("Retail (Reddit) vs. professional news (Bloomberg, gold miners) mood")
fig.tight_layout()
fig.savefig(FIGURES / "extra_1_reddit_vs_bloomberg_news.png", dpi=150, bbox_inches="tight")

def spear(a, b):
    both = pd.concat([a, b], axis=1).dropna()
    r, p = spearmanr(both.iloc[:, 0], both.iloc[:, 1])
    return r, p, len(both)
for label, a, b in [("same day, daily values", reddit, news),
                    ("20-day means", reddit.rolling(20, min_periods=10).mean(), news.rolling(20, min_periods=10).mean())]:
    r, p, n = spear(a, b)
    print(f"Reddit mood vs. miner news, {label}: Spearman rho = {r:+.2f} (p = {p:.3g}, n = {n})")
print("(20-day means overlap from day to day, so their p-value overstates certainty; read the rho only.)")
"""),
    md("""
**Lead / lag.** Positive *k* = the second series *k* trading days **later** than Reddit. If Reddit led
professional news or ETF money, the correlation would peak at positive *k*.
"""),
    code("""
flow = daily["etf_flow_pct"]
posts = daily["log_posts_per_day"]
rows = []
for k in range(-5, 6):
    r1, p1, _ = spear(reddit, news.shift(-k))
    r2, p2, _ = spear(reddit, flow.shift(-k))
    r3, p3, _ = spear(posts, flow.shift(-k))
    rows.append({"k (days)": k, "mood vs news": r1, "p (news)": p1, "mood vs ETF flow": r2, "p (flow)": p2,
                 "chatter vs ETF flow": r3, "p (chatter)": p3})
leadlag = pd.DataFrame(rows).set_index("k (days)")
print("11 lags x 3 pairs = 33 tests; by Bonferroni only p < 0.0015 counts as more than chance.")
leadlag.round(3)
"""),
    md("## 2. Weekly attention: Reddit posts vs. Bloomberg gold news stories (2024+)"),
    code("""
weekly_news = pd.read_parquet("data/bbg/gold_news_weekly.parquet").set_index("week")["story_count"]
weekly_news = weekly_news.iloc[:-1]   # the last week was pulled mid-week (partial)
hist = pd.read_parquet("data/clean/history_posts.parquet")
hist["created_utc"] = pd.to_datetime(hist["created_utc"], utc=True, format="ISO8601")
weekly_reddit = hist.set_index("created_utc").tz_convert(None).resample("W-FRI").size()
both = pd.DataFrame({"reddit_posts": weekly_reddit, "gold_news_stories": weekly_news}).dropna()
print(f"{len(both)} weeks, {both.index.min():%Y-%m-%d} -> {both.index.max():%Y-%m-%d} (weeks ending Friday)")
for label, d in [("weekly levels", both), ("week-on-week changes", both.diff().dropna())]:
    r, p = spearmanr(d["reddit_posts"], d["gold_news_stories"])
    print(f"Reddit posts vs. Bloomberg gold stories, {label}: Spearman rho = {r:+.2f} (p = {p:.3g}, n = {len(d)})")
fig, ax1 = plt.subplots(figsize=(13, 3.8))
ax1.plot(both.index, both["reddit_posts"], color="slategray", label="Reddit posts / week")
ax2 = ax1.twinx()
ax2.plot(both.index, both["gold_news_stories"], color="tab:red", label="Bloomberg gold news stories / week")
lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9)
ax1.set_title("Attention: Reddit post volume vs. Bloomberg gold news volume")
fig.tight_layout()
fig.savefig(FIGURES / "extra_2_weekly_attention.png", dpi=150, bbox_inches="tight")
"""),
    md("""
## 3. Hourly models on Bloomberg spot gold instead of GC=F futures
`data/bbg/features_bbg.parquet` is built exactly like `features.parquet`, except prices come from Bloomberg
spot (`python -m processing.bucket --price-file data/bbg/xau_spot_hourly.parquet ...`). Spot trades in hours
when COMEX futures are shut, so there are more valid hours. Same split: train before Sept 28, test after.
"""),
    code("""
from scipy.stats import binomtest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from processing.features import PRICE_FEATURES, SENTIMENT_FEATURES

ALL = PRICE_FEATURES + SENTIMENT_FEATURES
SPLIT = pd.Timestamp("2026-09-28 00:00", tz="UTC")
M = ["1 majority class", "2 logistic, price-only", "3 logistic, price + sentiment",
     "4 decision tree (depth 3), price + sentiment"]

def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half

def run_models(features):
    # Spot trades in the grid's first two hours, before the 1-2h lags exist: drop those warm-up rows.
    data = features[features["target_valid"]].dropna(subset=ALL).copy()
    data["y"] = data["price_direction"].astype(int)
    train, test = data[data.index < SPLIT], data[data.index >= SPLIT]
    models = {M[0]: (DummyClassifier(strategy="most_frequent"), []),
              M[1]: (make_pipeline(StandardScaler(), LogisticRegression()), PRICE_FEATURES),
              M[2]: (make_pipeline(StandardScaler(), LogisticRegression()), ALL),
              M[3]: (DecisionTreeClassifier(max_depth=3, random_state=0), ALL)}
    preds, rows = {}, []
    for name, (model, cols) in models.items():
        X_tr = train[cols] if cols else np.zeros((len(train), 1))
        X_te = test[cols] if cols else np.zeros((len(test), 1))
        model.fit(X_tr, train["y"])
        preds[name] = model.predict(X_te)
        k = int((preds[name] == test["y"]).sum())
        lo, hi = wilson(k, len(test))
        rows.append({"model": name, "n_features": len(cols), "n_train": len(train), "n_test": len(test),
                     "train_acc": (model.predict(X_tr) == train["y"]).mean(), "test_acc": k / len(test),
                     "ci95_low": lo, "ci95_high": hi, "pred_up_share": preds[name].mean()})
    return pd.DataFrame(rows), preds, test["y"].to_numpy()

gcf, _, _ = run_models(pd.read_parquet("data/clean/features.parquet"))
prior = pd.read_csv("reports/model_results.csv")
prior = prior[prior["dataset"] == "VADER, all communities"].set_index("model")["test_acc"]
for name in M:   # the GC=F side must reproduce 04_model exactly
    assert round(gcf.set_index("model").loc[name, "test_acc"], 4) == round(prior[name], 4), name

spot, preds, y = run_models(pd.read_parquet("data/bbg/features_bbg.parquet"))
print(f"Test hours: GC=F {gcf['n_test'].iloc[0]}, Bloomberg spot {spot['n_test'].iloc[0]}")
for m in M[2:]:
    a, b = preds[M[1]], preds[m]
    oa, ob = int(((a == y) & (b != y)).sum()), int(((b == y) & (a != y)).sum())
    print(f"{m:<46} vs price-only on spot: McNemar p = {binomtest(ob, oa + ob, 0.5).pvalue if oa + ob else 1.0:.3f}")
pd.DataFrame({"GC=F futures (04_model)": gcf.set_index("model")["test_acc"],
              "Bloomberg spot": spot.set_index("model")["test_acc"]}).map("{:.1%}".format)
"""),
    code("""
out = pd.read_csv("reports/model_results.csv")
DATASET = "hourly, Bloomberg spot"
out = pd.concat([out[out["dataset"] != DATASET], spot.assign(dataset=DATASET)[out.columns]], ignore_index=True)
out.round(4).to_csv("reports/model_results.csv", index=False)
"""),
]

# --------------------------------------------------------------------------- 08
NB08 = [
    md("""
# 08 · Report exhibits: the GoldPulse Index, and does the crowd lead or follow gold?

Presentation follows Semenova et al. (2024), *Wisdom of the Crowds or Ignorance of the Masses? A Data-Driven
Guide to WallStreetBets*, Journal of Portfolio Management 50(4): numbered exhibits with notes, an event study
of returns around posts (their Exhibit 5), return statistics of trading on the crowd (Exhibit 6) and Granger
causality tests (Exhibit 7). Their topic-model and network clustering is **not** reproduced: it is unsupervised.
Nothing here trains a model; these are descriptive statistics and classical tests.

Data: `data/clean/goldpulse_index.parquet` (public index, Reddit + Yahoo) and `data/bbg/daily_features.parquet`
(Bloomberg XAU closes aligned to the 20:00 UTC Reddit day, git-ignored).

### What would count, fixed before running
* **Reactive crowd (event study):** gold's mean 7-day return *before* attention spikes is above its usual 7-day
  return (t-test p < 0.05), while the 7-day return *after* is not.
* **Leading crowd (Granger):** a Reddit series Granger-causes gold returns at p < 0.003 (Bonferroni, 16 tests:
  2 Reddit series × 2 directions × 4 lags). Gold → Reddit is tested at the same bar.
* **Tradable crowd (returns):** a follow-the-crowd rule earns a next-day mean return different from zero *and*
  from buy-and-hold at p < 0.0125 (Bonferroni, 4 rows).
"""),
    code(SETUP + """
from scipy import stats
index = pd.read_parquet("data/clean/goldpulse_index.parquet")
daily = pd.read_parquet("data/bbg/daily_features.parquet")
daily["r"] = np.log(daily["xau_close"]).diff()            # log return, close d-1 -> close d
daily["r_next"] = daily["r"].shift(-1)                     # close d -> close d+1 (the future)
EXHIBITS = Path("reports/exhibits"); EXHIBITS.mkdir(parents=True, exist_ok=True)
pub = index.dropna(subset=["attention_score"])
print(f"Index: {len(pub)} published days, {pub.index.min():%Y-%m-%d} -> {pub.index.max():%Y-%m-%d}")
print(pub["attention_band"].value_counts(normalize=True).round(3).to_string())
"""),
    md("## Exhibit 1 · The GoldPulse Index vs. the gold price, 2022–2026"),
    code("""
import matplotlib.dates as mdates
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.6), sharex=True, gridspec_kw={"height_ratios": [2, 1.4]})
gold = index["gold_close"].dropna()
ax1.plot(gold.index, gold, color="goldenrod", lw=1.5)
ax1.set_ylabel("gold, USD/oz (GC=F)")
frenzy = pub.index[pub["attention_band"] == "Frenzy"]
ax1.scatter(frenzy, index["gold_close"].ffill().reindex(frenzy), s=8, color="tab:red", zorder=3, label="Frenzy day (attention ≥ 95)")
ax1.legend(loc="upper left", fontsize=9)
ax1.set_title("GoldPulse Index vs. gold, Dec 2021 – Oct 2026 (scores published from Sept 2022)")
ax2.plot(pub.index, pub["attention_score"], color="tab:red", lw=1, label="Attention score")
ax2.plot(pub.index, pub["mood_score"], color="tab:blue", lw=1, alpha=0.7, label="Mood score (experimental)")
for y in (20, 80, 95):
    ax2.axhline(y, color="gray", lw=0.5, ls=":")
ax2.set_ylabel("score (0–100)")
ax2.legend(loc="lower left", fontsize=9, ncol=2)
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.tight_layout()
fig.savefig(FIGURES / "exhibit_1_index_vs_gold.png", dpi=150, bbox_inches="tight")
"""),
    md("""
## Exhibit 2 · Gold's path 14 trading days before and after a Reddit attention spike
Spike = a trading day whose chatter (`attention_spike`: posts per day vs. their trailing 20-day mean) is in the top 5%
of days. Spikes within 10 trading days of an earlier one are merged into it (first day kept), so one episode counts once.
The path is the cumulative log return relative to the spike day's close (0 on day 0).
"""),
    code("""
d = daily.dropna(subset=["attention_spike", "r"])
cut = d["attention_spike"].quantile(0.95)
events, last = [], None
for i, t in enumerate(d.index):
    if d.at[t, "attention_spike"] >= cut and (last is None or i - last > 10):
        events.append(i)
    if d.at[t, "attention_spike"] >= cut:
        last = i
W = 14
logp = np.log(d["xau_close"]).to_numpy()
paths = np.array([logp[i - W:i + W + 1] - logp[i] for i in events if i - W >= 0 and i + W < len(d)])
ev_days = [d.index[i] for i in events if i - W >= 0 and i + W < len(d)]
mood_med = d["mean_sentiment"].rolling(250, min_periods=60).median()
warm = np.array([d.at[t, "mean_sentiment"] > mood_med.at[t] for t in ev_days])
print(f"{len(paths)} spike episodes (threshold: attention_spike >= {cut:.2f}, i.e. ~{np.expm1(cut):.0%} above its 20-day norm)")

# Unconditional benchmark: the same paths around every day.
all_idx = np.arange(W, len(d) - W)
base = np.array([logp[i - W:i + W + 1] - logp[i] for i in all_idx])
k = np.arange(-W, W + 1)
fig, ax = plt.subplots(figsize=(10, 4.6))
for sel, lab, col in [(slice(None), f"all spikes (n={len(paths)})", "black"),
                      (warm, f"warm-mood spikes (n={warm.sum()})", "tab:orange"),
                      (~warm, f"cool-mood spikes (n={(~warm).sum()})", "tab:blue")]:
    m = paths[sel].mean(0); se = paths[sel].std(0, ddof=1) / np.sqrt(len(paths[sel]))
    ax.plot(k, m * 100, color=col, lw=1.8, label=lab)
    if lab.startswith("all"):
        ax.fill_between(k, (m - 1.96 * se) * 100, (m + 1.96 * se) * 100, color=col, alpha=0.12)
ax.plot(k, base.mean(0) * 100, color="gray", ls="--", lw=1, label="every day (benchmark)")
ax.axvline(0, color="red", lw=0.8); ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel("trading days from spike (0 = spike day's close)"); ax.set_ylabel("cumulative gold return, %")
ax.set_title("Gold around Reddit attention spikes, 2022–2026")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "exhibit_2_event_study.png", dpi=150, bbox_inches="tight")

before = -paths[:, W - 7]          # log(P0 / P-7)
after = paths[:, W + 7]            # log(P7 / P0)
ret7 = pd.Series(logp).diff(7).dropna().to_numpy()
rows = []
for name, x in [("7 days before spike", before), ("7 days after spike", after)]:
    t, p = stats.ttest_1samp(x, ret7.mean())
    rows.append({"window": name, "n": len(x), "mean %": x.mean() * 100, "usual 7-day mean %": ret7.mean() * 100,
                 "share up": (x > 0).mean(), "t": t, "p": p})
event_table = pd.DataFrame(rows).set_index("window")
event_table.round(3).to_csv(EXHIBITS / "exhibit_2_event_study.csv")
event_table.round(3)
"""),
    md("""
## Exhibit 3 · Distribution of next-day gold returns when following the crowd
Statistics of the next-day log return (close d → close d+1) for each rule, against buy-and-hold. "Warm mood" = the
day's mean VADER above its trailing 250-day median (no look-ahead). "Previous" = the return on the spike day itself,
which shows whether the crowd talks *after* moves (as in the paper's Exhibit 6).
"""),
    code("""
valid = d.dropna(subset=["r_next"]).copy()
valid["warm"] = valid["mean_sentiment"] > mood_med.reindex(valid.index)
valid["spike"] = valid["attention_spike"] >= cut
rules = {
    "buy and hold (every day)": valid["r_next"],
    "follow the mood: long if warm, short if cool": np.where(valid["warm"], 1, -1) * valid["r_next"],
    "long only on warm-mood days": valid.loc[valid["warm"], "r_next"],
    "after an attention spike (next day)": valid.loc[valid["spike"], "r_next"],
    "previous: the spike day itself": valid.loc[valid["spike"], "r"],
}
bh = valid["r_next"]
rows = []
for name, x in rules.items():
    x = pd.Series(x, index=valid.index if len(x) == len(valid) else None).dropna()
    t0, p0 = stats.ttest_1samp(x, 0)
    row = {"rule": name, "n days": len(x), "mean (bp)": x.mean() * 1e4, "std (bp)": x.std() * 1e4,
           "skew": stats.skew(x), "kurtosis": stats.kurtosis(x), "p (mean = 0)": p0,
           "annualised Sharpe": x.mean() / x.std() * np.sqrt(252)}
    if name.startswith("follow the mood"):
        row["p (vs buy and hold)"] = stats.ttest_rel(x, bh).pvalue
    elif name != "buy and hold (every day)" and not name.startswith("previous"):
        row["p (vs buy and hold)"] = stats.ttest_ind(x, bh, equal_var=False).pvalue
    rows.append(row)
ret_table = pd.DataFrame(rows).set_index("rule")
ret_table.round(4).to_csv(EXHIBITS / "exhibit_3_returns.csv")
ret_table.round(3)
"""),
    md("""
## Exhibit 4 · Granger causality: does Reddit help forecast gold, or gold forecast Reddit?
A series A Granger-causes B if A's past improves a forecast of B beyond B's own past. Reddit series are changes, so
they are stationary (checked with an augmented Dickey–Fuller test): Δ mood = day-to-day change in mean VADER, and
Δ attention = change in log posts per day. Cells: F statistic (p-value), at lags of 1, 2, 5 and 10 trading days.
"""),
    code("""
import warnings
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
g = pd.DataFrame({"gold return": d["r"], "Δ mood": d["mean_sentiment"].diff(),
                  "Δ attention": d["log_posts_per_day"].diff()}).dropna()
for c in g:
    print(f"ADF {c:<12} p = {adfuller(g[c], autolag='AIC')[1]:.2g}  ({'stationary' if adfuller(g[c], autolag='AIC')[1] < 0.05 else 'NOT stationary'})")
LAGS = [1, 2, 5, 10]
rows = []
for a, b in [("Δ mood", "gold return"), ("Δ attention", "gold return"),
             ("gold return", "Δ mood"), ("gold return", "Δ attention")]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = grangercausalitytests(g[[b, a]], maxlag=max(LAGS))
    row = {"test": f"{a} → {b}", "obs": len(g)}
    for L in LAGS:
        F, p = res[L][0]["ssr_ftest"][:2]
        row[f"lag {L}"] = f"{F:.2f} ({p:.3f})" + ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
        row[f"p{L}"] = p
    rows.append(row)
granger = pd.DataFrame(rows).set_index("test")
granger.to_csv(EXHIBITS / "exhibit_4_granger.csv")
print("Bonferroni bar: p < 0.003 (16 tests).  *** p<0.001  ** p<0.01  * p<0.05")
granger[[f"lag {L}" for L in LAGS] + ["obs"]]
"""),
    md("## Verdicts against the pre-set bars"),
    code("""
pb, pa = event_table.loc["7 days before spike", "p"], event_table.loc["7 days after spike", "p"]
mb = event_table.loc["7 days before spike", "mean %"]; usual = event_table.loc["7 days before spike", "usual 7-day mean %"]
print(f"Reactive crowd: before-spike return {mb:+.2f}% vs usual {usual:+.2f}% (p = {pb:.3f}); after p = {pa:.3f} -> "
      + ("YES" if (pb < 0.05 and mb > usual and pa >= 0.05) else "NO"))
lead = granger.loc[["Δ mood → gold return", "Δ attention → gold return"], [f"p{L}" for L in LAGS]].min().min()
follow = granger.loc[["gold return → Δ mood", "gold return → Δ attention"], [f"p{L}" for L in LAGS]].min().min()
print(f"Leading crowd (Reddit Granger-causes gold, p < 0.003): smallest p = {lead:.3f} -> {'YES' if lead < 0.003 else 'NO'}")
print(f"Gold Granger-causes Reddit (p < 0.003):                 smallest p = {follow:.3g} -> {'YES' if follow < 0.003 else 'NO'}")
trad = ret_table.drop(index=["buy and hold (every day)", "previous: the spike day itself"])
ok = trad[(trad["p (mean = 0)"] < 0.0125) & (trad["p (vs buy and hold)"] < 0.0125)]
print(f"Tradable crowd (p < 0.0125 vs 0 and vs buy-and-hold): {'YES: ' + ', '.join(ok.index) if len(ok) else 'NO'}")
"""),
]

# --------------------------------------------------------------------------- 09
NB09 = [
    md("""
# 09 · Course-method checks: descriptive statistics, outliers, encoding and precision/recall

Added after the main studies, to apply the remaining course techniques to the **same data and the same
models**. Nothing here changes a pre-set verdict: the models in `04_model.ipynb` and `06_daily.ipynb` are
refitted exactly as they were (the accuracies are asserted against `reports/model_results.csv`), and every
variant below is labelled as a robustness check run after the results were known.

1. **Descriptive statistics**: mean, median, standard deviation, range of the main variables.
2. **Outliers**: how many values sit beyond 3 standard deviations or outside the IQR fences,
   and what we decided to do about them.
3. **Precision, recall and F1** for every hourly and daily model, plus a daily confusion matrix.
4. **One-hot encoding** of `day_of_week` and **capping outliers** at the 1st/99th percentile:
   are the conclusions unchanged?
5. **Overfitting**: train vs. test accuracy of the decision tree as it gets deeper.
6. **The proposal's H2**, tested directly: is the mood before up moves different from the mood before down moves?

Bloomberg-dependent cells read the git-ignored `data/bbg/` tables, as in `06_daily.ipynb`.
"""),
    code(SETUP + """
from scipy.stats import binomtest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from processing.daily import (DAILY_CHATTER_FEATURES, DAILY_MARKET_FEATURES, DAILY_MOOD_FEATURES,
                              DAILY_PRICE_FEATURES)
from processing.features import PRICE_FEATURES, SENTIMENT_FEATURES

EXHIBITS = Path("reports/exhibits")
RESULTS = pd.read_csv("reports/model_results.csv").set_index(["dataset", "model"])

# Hourly study, exactly as 04_model.ipynb
features = pd.read_parquet("data/clean/features.parquet")
hourly = features[features["target_valid"]].copy()
hourly["y"] = hourly["price_direction"].astype(int)
H_ALL = PRICE_FEATURES + SENTIMENT_FEATURES
SPLIT = pd.Timestamp("2026-09-28 00:00", tz="UTC")
h_train, h_test = hourly[hourly.index < SPLIT], hourly[hourly.index >= SPLIT]
assert (len(h_train), len(h_test)) == (307, 124)

# Daily study, exactly as 06_daily.ipynb
daily_all = pd.read_parquet("data/bbg/daily_features.parquet")
REDDIT = DAILY_MARKET_FEATURES + DAILY_MOOD_FEATURES + DAILY_CHATTER_FEATURES
CHATTER = DAILY_MARKET_FEATURES + DAILY_CHATTER_FEATURES
daily = daily_all[daily_all["target_valid"]].dropna(subset=REDDIT).copy()
daily["y"] = daily["up"].astype(int)
TEST_START = pd.Timestamp("2023-01-01")
print(f"hourly: {len(h_train)} train / {len(h_test)} test hours;  daily: {len(daily)} usable days, "
      f"{(daily.index >= TEST_START).sum()} walk-forward test days")
"""),
    md("""
## 1. Descriptive statistics
Hourly variables cover the live window (Sept 8 – Oct 5 2026); daily variables cover 2022–2026. Returns are in
per cent. Engagement uses settled posts only.
"""),
    code("""
nxt = daily_all["xau_close"].shift(-1) / daily_all["xau_close"] - 1
VARIABLES = {
    "Next-hour gold return, % (hourly)": hourly["next_return"] * 100,
    "Posts per hour": features["post_count"],
    "Mean VADER mood, hours with posts": features.loc[features["had_posts"], "mean_sentiment"],
    "Mean engagement per post, settled (hourly)": features["mean_engagement"].dropna(),
    "Next-day gold return, % (daily, Bloomberg spot)": (nxt[daily_all["target_valid"]] * 100).dropna(),
    "Posts per day (daily)": daily_all["posts_per_day"].dropna(),
    "Mean VADER mood (daily)": daily_all["mean_sentiment"].dropna(),
}
rows = []
for name, s in VARIABLES.items():
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    z = (s - s.mean()) / s.std()
    rows.append({"variable": name, "n": len(s), "mean": s.mean(), "median": s.median(), "sd": s.std(),
                 "min": s.min(), "max": s.max(),
                 "z_gt_3": int((z.abs() > 3).sum()),
                 "outside_iqr": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())})
desc = pd.DataFrame(rows).set_index("variable")
desc["z_gt_3_share"] = desc["z_gt_3"] / desc["n"]
desc["outside_iqr_share"] = desc["outside_iqr"] / desc["n"]
desc.to_csv(EXHIBITS / "exhibit_5_descriptive_outliers.csv")
desc.round(3)
"""),
    md("""
## 2. Outliers
Two standard rules: a value is an outlier if it is more than **3 standard deviations** from the
mean, or outside the **IQR fences** (below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR). The table above counts both.
The largest values are listed below so they can be checked by eye.
"""),
    code("""
big_hours = hourly.assign(next_return_pct=hourly["next_return"] * 100).loc[
    lambda d: (d["next_return_pct"] - d["next_return_pct"].mean()).abs() > 3 * d["next_return_pct"].std(),
    ["next_return_pct", "post_count"]]
print("Hourly gold moves beyond 3 sd:")
print(big_hours.round(3).to_string())
dn = (nxt[daily_all["target_valid"]] * 100).dropna()
print("\\nLargest daily gold moves (%):")
print(pd.concat([dn.nsmallest(3), dn.nlargest(3)]).round(2).to_string())
print("\\nBusiest days (posts per day):")
print(daily_all["posts_per_day"].nlargest(5).round(0).to_string())
"""),
    md("""
**Are the extreme moves real or data errors?** Cross-check each flagged price move against an independent
source: the hourly ones (GC=F futures from Yahoo) against Bloomberg spot gold for the same hour, the daily ones
(Bloomberg spot) against Yahoo's daily GC=F closes. An error in one feed would not appear in the other.
"""),
    code("""
spot = pd.read_parquet("data/bbg/xau_spot_hourly.parquet").set_index("bucket_utc")["xau_usd"]
fut = features["xau_usd"]
check = []
for t in big_hours.index:
    a, b = t + pd.Timedelta(hours=1), t + pd.Timedelta(hours=2)   # the target move, (t+1):00 -> (t+2):00
    check.append({"hour t": t, "GC=F move %": (fut[b] / fut[a] - 1) * 100, "Bloomberg spot move %": (spot[b] / spot[a] - 1) * 100})
check = pd.DataFrame(check).set_index("hour t")
print(check.round(2).to_string())
assert (np.sign(check["GC=F move %"]) == np.sign(check["Bloomberg spot move %"])).all()
assert ((check["GC=F move %"] - check["Bloomberg spot move %"]).abs() < 0.2).all()

yahoo = pd.read_parquet("data/clean/gold_daily_yahoo.parquet").set_index("date")["close"]
yahoo_next = (yahoo.shift(-1) / yahoo - 1) * 100
extreme = pd.concat([dn.nsmallest(3), dn.nlargest(3)])
print(pd.DataFrame({"Bloomberg spot %": extreme, "Yahoo GC=F %": yahoo_next.reindex(extreme.index)}).round(2).to_string())
assert (np.sign(extreme) == np.sign(yahoo_next.reindex(extreme.index))).all()
print("Every flagged move appears in both sources with the same sign: real market moves, not feed errors.")
"""),
    md("""
**Decision: keep them.** The flagged values are real events, not errors: every extreme gold move appears in two
independent price feeds (above), and they *are* the thing we try to predict; the busiest posting days are the
late-January 2026 peak and sell-off, which is exactly what the index is meant to measure. Removing them would throw away the most informative days. Three things
already limit their influence: the target is only the **direction** of the move, so a 2% day counts the same as
a 0.1% day; counts and engagement enter the models as **log1p**; and logistic-regression inputs are
**standardised**. Section 4 below checks that **capping** the inputs at the 1st/99th percentile changes no
conclusion.
"""),
    md("""
## 3. Precision, recall and F1
"Positive" = gold goes **up**. Precision: of the hours/days we called "up", how many were up. Recall: of the
hours/days that were up, how many we caught. F1 combines the two. The models are refitted exactly as in
notebooks 04 and 06; the asserts check that the accuracies match `reports/model_results.csv`.
"""),
    code("""
def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half

def mcnemar(a, b, y):
    only_a, only_b = int(((a == y) & (b != y)).sum()), int(((b == y) & (a != y)).sum())
    return only_a, only_b, (binomtest(only_b, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0)

class Cap(BaseEstimator, TransformerMixin):
    \"\"\"Clips each column to its 1st-99th percentile, learned on the training rows only (winsorising).\"\"\"
    def __init__(self, lo=1, hi=99):
        self.lo, self.hi = lo, hi
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_, self.hi_ = np.percentile(X, self.lo, axis=0), np.percentile(X, self.hi, axis=0)
        return self
    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

NOT_CONTINUOUS = {"return_missing", "engagement_missing", "day_of_week", "hour_sin", "hour_cos"}

def prepare(cols, variant):
    \"\"\"Column handling for a variant: original (as pre-set), one-hot weekday, or capped continuous inputs.\"\"\"
    if variant == "original":
        return None
    other = [c for c in cols if c != "day_of_week"]
    if variant == "one-hot weekday":
        if "day_of_week" not in cols:
            return None
        return ColumnTransformer([("weekday", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["day_of_week"]),
                                  ("rest", "passthrough", other)])
    cont = [c for c in cols if c not in NOT_CONTINUOUS]
    return ColumnTransformer([("cap", Cap(), cont), ("rest", "passthrough", [c for c in cols if c not in cont])])

def logit(cols, variant, max_iter=100):
    pre = prepare(cols, variant)
    steps = ([pre] if pre is not None else []) + [StandardScaler(), LogisticRegression(max_iter=max_iter)]
    return make_pipeline(*steps)

def tree(cols, variant, depth=3):
    pre = prepare(cols, variant)
    return make_pipeline(*(([pre] if pre is not None else []) + [DecisionTreeClassifier(max_depth=depth, random_state=0)]))

H1, H2, H3, H4 = ("1 majority class", "2 logistic, price-only", "3 logistic, price + sentiment",
                  "4 decision tree (depth 3), price + sentiment")
def hourly_models(variant):
    return {H1: (DummyClassifier(strategy="most_frequent"), []),
            H2: (logit(PRICE_FEATURES, variant), PRICE_FEATURES),
            H3: (logit(H_ALL, variant), H_ALL),
            H4: (tree(H_ALL, variant), H_ALL)}

def fit_hourly(variant):
    preds, train_acc = {}, {}
    for name, (model, cols) in hourly_models(variant).items():
        X_tr = h_train[cols] if cols else np.zeros((len(h_train), 1))
        X_te = h_test[cols] if cols else np.zeros((len(h_test), 1))
        model.fit(X_tr, h_train["y"])
        preds[name] = pd.Series(model.predict(X_te), index=h_test.index)
        train_acc[name] = (model.predict(X_tr) == h_train["y"]).mean()
    return pd.DataFrame(preds), train_acc

D1, D2, D3 = "1 majority class", "2 logistic, gold returns only", "3 logistic, market baseline"
D7, D8, D9 = ("7 logistic, market + Reddit mood + chatter", "8 decision tree, market + Reddit mood + chatter",
              "9 logistic, market + chatter (no scorer)")
def daily_models(variant):
    return {D1: (DummyClassifier(strategy="most_frequent"), []),
            D2: (logit(DAILY_PRICE_FEATURES, variant, 1000), DAILY_PRICE_FEATURES),
            D3: (logit(DAILY_MARKET_FEATURES, variant, 1000), DAILY_MARKET_FEATURES),
            D7: (logit(REDDIT, variant, 1000), REDDIT),
            D8: (tree(REDDIT, variant), REDDIT),
            D9: (logit(CHATTER, variant, 1000), CHATTER)}

def walk_forward(variant, freq="MS"):
    \"\"\"Refit every model at the start of each month on all earlier days; predict that month (as notebook 06).\"\"\"
    starts = pd.date_range(TEST_START, daily.index.max(), freq=freq).union([TEST_START])
    preds = {m: [] for m in daily_models(variant)}
    for a, b in zip(starts, list(starts[1:]) + [daily.index.max() + pd.Timedelta(days=1)]):
        train, test = daily[daily.index < a], daily[(daily.index >= a) & (daily.index < b)]
        if test.empty:
            continue
        assert train.index.max() < test.index.min()
        for name, (model, cols) in daily_models(variant).items():
            X_tr = train[cols] if cols else np.zeros((len(train), 1))
            X_te = test[cols] if cols else np.zeros((len(test), 1))
            model.fit(X_tr, train["y"])
            preds[name].append(pd.Series(model.predict(X_te), index=test.index))
    return pd.DataFrame({m: pd.concat(p) for m, p in preds.items()})

h_pred, h_train_acc = fit_hourly("original")
d_pred = walk_forward("original")
y_h, y_d = h_test["y"], daily.loc[d_pred.index, "y"]
for m in h_pred:
    assert abs((h_pred[m] == y_h).mean() - RESULTS.loc[("VADER, all communities", m), "test_acc"]) < 1e-4, m
for m in d_pred:
    assert abs((d_pred[m] == y_d).mean() - RESULTS.loc[("daily 2022–26 walk-forward", m), "test_acc"]) < 1e-4, m
print("refitted models reproduce reports/model_results.csv exactly")

def prf_rows(study, preds, y):
    out = []
    for m in preds:
        p, r, f, _ = precision_recall_fscore_support(y, preds[m], labels=[1], zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, preds[m], labels=[0, 1]).ravel()
        out.append({"study": study, "model": m, "accuracy": (preds[m] == y).mean(), "precision_up": p[0],
                    "recall_up": r[0], "f1_up": f[0], "TP": tp, "FP": fp, "FN": fn, "TN": tn})
    return out

prf = pd.DataFrame(prf_rows("hourly", h_pred, y_h) + prf_rows("daily", d_pred, y_d))
prf.to_csv(EXHIBITS / "exhibit_6_precision_recall.csv", index=False)
prf.round(3)
"""),
    md("""
**Reading it.** The majority-class models show why accuracy alone can mislead: the hourly one never
says "up", so it has **recall 0 and F1 0** for "up" even with 49% accuracy; the daily one always says "up", so it
has **recall 1** but precision only equal to the share of up days. Accuracy is still a fair headline here because
the two classes are close to 50/50 (48% up hourly, 54% up daily), and with balanced classes accuracy is a fair
measure.
"""),
    code("""
labels_txt = ["down/flat", "up"]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
for ax, m in zip(axes, [D1, D3, D7, D8]):
    ConfusionMatrixDisplay(confusion_matrix(y_d, d_pred[m], labels=[0, 1]), display_labels=labels_txt
                           ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{m}\\naccuracy {(d_pred[m] == y_d).mean():.1%}", fontsize=9.5)
fig.suptitle(f"Next-day gold direction, walk-forward test days (n = {len(y_d)}). Rows = actual, columns = predicted",
             y=1.03)
fig.tight_layout()
fig.savefig(FIGURES / "model_cm_daily.png", dpi=150, bbox_inches="tight")
"""),
    md("""
## 4. One-hot weekday and capped outliers: do the conclusions change?
The pre-set models feed `day_of_week` (Monday = 0 … Sunday = 6) in as a number. A weekday is really a
**nominal** (category) feature, usually **one-hot encoded** (one 0/1 column per day), because Friday is not "four times"
Monday. The **capped** variant clips every continuous input to its 1st–99th percentile, with the limits learned on
training rows only. Both are robustness checks run after the pre-set verdicts, so they cannot change those verdicts;
they only tell us whether the verdicts depend on these two choices.
"""),
    code("""
rows = []
variant_preds = {"original": (h_pred, d_pred)}
for variant in ["one-hot weekday", "capped 1st–99th pct"]:
    variant_preds[variant] = (fit_hourly(variant)[0], walk_forward(variant))
for study, base, others, y in [("hourly", H2, [H3, H4], y_h), ("daily", D3, [D7, D8, D9], y_d)]:
    for m in [base] + others:
        row = {"study": study, "model": m}
        for variant, (hp, dp) in variant_preds.items():
            p = hp if study == "hourly" else dp
            row[f"{variant} acc"] = (p[m] == y).mean()
            row[f"{variant} p vs baseline"] = np.nan if m == base else mcnemar(p[base], p[m], y)[2]
        rows.append(row)
robust = pd.DataFrame(rows)
robust.to_csv(EXHIBITS / "exhibit_7_encoding_capping.csv", index=False)
robust.round(3)
"""),
    code("""
PASS_BAR = {"hourly": 0.00625, "daily": 0.004}   # the strictest pre-set bars in notebooks 04 and 06
for variant in variant_preds:
    wins = []
    for study, base in [("hourly", H2), ("daily", D3)]:
        part = robust[robust["study"] == study].set_index("model")
        for m, r in part.drop(index=base).iterrows():
            if r[f"{variant} p vs baseline"] < PASS_BAR[study] and r[f"{variant} acc"] > part.loc[base, f"{variant} acc"]:
                wins.append(f"{study}: {m}")
    print(f"{variant:<22} models beating their baseline at the pre-set bar: {', '.join(wins) if wins else 'none'}")
"""),
    md("""
## 5. Overfitting: the decision tree as it gets deeper
A deeper tree fits the training hours better and better, but past a few levels it is memorising noise: training
accuracy keeps rising while test accuracy does not. Our reported tree is fixed at depth 3, chosen before testing.
"""),
    code("""
rows = []
for depth in range(1, 9):
    t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(h_train[H_ALL], h_train["y"])
    rows.append({"depth": depth, "train_acc": (t.predict(h_train[H_ALL]) == h_train["y"]).mean(),
                 "test_acc": (t.predict(h_test[H_ALL]) == h_test["y"]).mean()})
overfit = pd.DataFrame(rows).set_index("depth")
overfit.to_csv(EXHIBITS / "exhibit_8_tree_depth.csv")
overfit.round(3)
"""),
    md("""
## 6. A direct test of the proposal's H2: does sentiment carry direction?
**H0:** mean sentiment is unrelated to the direction of the next move. **H1:** more bullish sentiment is followed by
a rising price. The models above ask this indirectly; here it is tested directly, with no model. For every hour
(or day) with posts and a valid target, compare the mood before an **up** move with the mood before a **down/flat**
move. Mann–Whitney U test (no normality assumption), two-sided; two tests (hourly, daily), so the bar is
p < 0.05 / 2 = 0.025. This test was added after the main results and is labelled as such in the report.
"""),
    code("""
from scipy.stats import mannwhitneyu, pointbiserialr
h2_rows = []
hourly_posts = hourly[hourly["had_posts"]]
daily_posts = daily_all[daily_all["target_valid"] & daily_all["had_posts"]].copy()
daily_posts["y"] = daily_posts["up"].astype(int)
for label, d, unit in [("hourly, Sept 8 – Oct 5 2026", hourly_posts, "hours"),
                       ("daily, 2022–2026", daily_posts, "days")]:
    up, down = d.loc[d["y"] == 1, "mean_sentiment"], d.loc[d["y"] == 0, "mean_sentiment"]
    h2_rows.append({"test": label, "unit": unit, "n_up": len(up), "n_down": len(down),
                    "mood_before_up": up.mean(), "mood_before_down": down.mean(),
                    "mann_whitney_p": mannwhitneyu(up, down, alternative="two-sided").pvalue,
                    "point_biserial_r": pointbiserialr(d["y"], d["mean_sentiment"]).statistic})
h2 = pd.DataFrame(h2_rows).set_index("test")
h2.to_csv(EXHIBITS / "exhibit_9_h2_direction.csv")
h2.round(4)
"""),
]

NOTEBOOKS = {"02": ("02_features.ipynb", NB02), "03": ("03_eda.ipynb", NB03), "04": ("04_model.ipynb", NB04),
             "05": ("05_macro.ipynb", NB05), "06": ("06_daily.ipynb", NB06),
             "07": ("07_bloomberg_extras.ipynb", NB07), "08": ("08_report_exhibits.ipynb", NB08),
             "09": ("09_course_checks.ipynb", NB09)}


def build(key):
    filename, cells = NOTEBOOKS[key]
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, kernel_name="python3", timeout=600,
                   resources={"metadata": {"path": str(HERE)}}).execute()
    nbformat.write(nb, HERE / filename)
    print(f"built and executed notebooks/{filename}")


if __name__ == "__main__":
    keys = sys.argv[1:] or ["all"]
    for key in (list(NOTEBOOKS) if keys == ["all"] else keys):
        build(key)
