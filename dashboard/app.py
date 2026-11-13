"""GoldPulse Index dashboard.

    streamlit run dashboard/app.py

Reads only committed, non-Bloomberg data: data/clean/goldpulse_index.parquet
(built by `python -m processing.index`) and data/goldpulse.sqlite3 (the live
collector). Files are re-read every 30 minutes, so new commits from the
collector show up without a redeploy.
"""
import sqlite3
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "data" / "clean" / "goldpulse_index.parquet"
DB_PATH = ROOT / "data" / "goldpulse.sqlite3"
NEUTRAL_BAND = 0.05  # VADER's usual band, as in processing/bucket.py

BAND_TEXT = {
    "Quiet": "quieter than usual",
    "Normal": "around its usual level",
    "Busy": "busier than usual",
    "Frenzy": "at a frenzy, among the biggest surges of the past year",
    "Gloomy": "gloomier than usual",
    "Typical": "in its usual range",
    "Upbeat": "more upbeat than usual",
}
DISCLAIMER = ("For information only. Not investment advice. The GoldPulse Index describes online "
              "discussion; it does not predict prices.")

st.set_page_config(page_title="GoldPulse Index", page_icon="🪙", layout="wide")

# Same look as the presentation: Poppins, warm neutrals, Reddit orange for Frenzy and attention.
ORANGE, GOLD, MOOD_GREY, BAR, MUTED, GRID = "#FF4500", "#C9962B", "#8C857D", "#FFB59A", "#7A756F", "#EEE9E3"
st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap");
html, body, h1, h2, h3, p, li, label, [data-testid="stMarkdownContainer"], [data-testid="stMetric"] {
  font-family: "Poppins", "Segoe UI", sans-serif; }
h1 { font-weight: 400 !important; letter-spacing: -0.5px; }
h2, h3 { font-weight: 500 !important; }
[data-testid="stMetricValue"] { font-weight: 600; color: #3D3B3F; }
[data-testid="stAlert"] > div { background: #F1ECE6; color: #3D3B3F; }
[data-testid="stCaptionContainer"] { color: #7A756F; }
</style>
""", unsafe_allow_html=True)


def style(chart):
    return (chart.configure(font="Poppins")
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor="#BBB7B3",
                            tickColor="#BBB7B3", labelFontSize=11, titleFontSize=11)
            .configure_title(color=MUTED, fontSize=12, fontWeight="normal", anchor="start")
            .configure_legend(labelColor=MUTED, titleColor=MUTED)
            .configure_view(stroke=None))


@st.cache_data(ttl=30 * 60)
def load_index() -> pd.DataFrame:
    return pd.read_parquet(INDEX_PATH)


@st.cache_data(ttl=30 * 60)
def load_recent_hourly(days: int = 7) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        posts = pd.read_sql("SELECT created_utc, sentiment FROM posts", conn)
        prices = pd.read_sql("SELECT bucket_utc, xau_usd FROM price_ticks", conn)
    posts["hour"] = pd.to_datetime(posts["created_utc"], utc=True, format="ISO8601").dt.floor("h")
    prices["hour"] = pd.to_datetime(prices["bucket_utc"], utc=True, format="ISO8601")
    end = posts["hour"].max()
    start = end - pd.Timedelta(days=days)
    hourly = posts[posts["hour"] > start].groupby("hour").agg(posts=("sentiment", "size"),
                                                              mood=("sentiment", "mean"))
    grid = pd.date_range(start + pd.Timedelta(hours=1), end, freq="h", tz="UTC", name="hour")
    hourly = hourly.reindex(grid)
    hourly["posts"] = hourly["posts"].fillna(0).astype(int)
    hourly["gold"] = prices.set_index("hour")["xau_usd"].reindex(grid)
    return hourly.reset_index()


index = load_index()
pub = index.dropna(subset=["attention_score"])
today = pub.iloc[-1]
week_ago = pub.iloc[-8]
last_day = pub.index[-1]
gold = index["gold_close"].dropna()

# --------------------------------------------------------------------------- header
st.title("GoldPulse Index")
st.markdown("**How much, and how warmly, is the public talking about gold?** A daily read of "
            "r/Gold, r/Silverbugs and gold posts in r/investing.")
st.caption(f"Data through {last_day:%A %d %B %Y} (UTC). {DISCLAIMER}")

c1, c2, c3 = st.columns(3)
c1.metric(f"Attention score: {today['attention_band']}", f"{today['attention_score']:.0f} / 100",
          f"{today['attention_score'] - week_ago['attention_score']:+.0f} vs. a week ago",
          help="This week's posts per day vs. the previous 90 days, ranked against the past year. "
               "0 = quietest week of the year, 100 = biggest surge.")
c2.metric(f"Mood score: {today['mood_band']} (experimental)", f"{today['mood_score']:.0f} / 100",
          f"{today['mood_score'] - week_ago['mood_score']:+.0f} vs. a week ago",
          help="This week's average VADER tone, ranked against the past year. Experimental: VADER "
               "agreed with our hand labels only 38% of the time.")
c3.metric("Gold (GC=F futures)", f"${gold.iloc[-1]:,.0f} / oz",
          f"{gold.iloc[-1] / gold.iloc[-6] - 1:+.1%} over 5 trading days",
          help=f"Last close {gold.index[-1]:%d %b %Y}, Yahoo Finance. Shown for context only.")

st.info(f"In plain words: gold chatter on Reddit is {BAND_TEXT[today['attention_band']]} "
        f"({today['attention_7d']:.0f} posts a day this week, {today['attention_ratio']:.0%} of the "
        f"90-day norm), and its tone is {BAND_TEXT[today['mood_band']]}.", icon=None)

# --------------------------------------------------------------------------- history
left, right = st.columns([2, 1], gap="large")
left.subheader("The index over time")
span = left.radio("Show", ["3 months", "1 year", "All (since 2022)"], index=2, horizontal=True)
start = {"3 months": last_day - pd.DateOffset(months=3), "1 year": last_day - pd.DateOffset(years=1),
         "All (since 2022)": index.index.min()}[span]
view = index.loc[start:].reset_index()
pview = pub.loc[start:].reset_index()

XAXIS = alt.X("date:T", title=None, axis=alt.Axis(format="%Y" if span.startswith("All") else "%b %Y",
                                                   tickCount="year" if span.startswith("All") else 6))
frenzy = view[view["attention_band"] == "Frenzy"].assign(gold_close=lambda d: index["gold_close"].ffill()
                                                         .reindex(d["date"]).to_numpy())
price = alt.Chart(view.dropna(subset=["gold_close"])).mark_line(color=GOLD, strokeWidth=1.4).encode(
    x=XAXIS, y=alt.Y("gold_close:Q", title="gold, USD/oz", scale=alt.Scale(zero=False),
                                          axis=alt.Axis(format="~s")),
    tooltip=[alt.Tooltip("date:T"), alt.Tooltip("gold_close:Q", format=",.0f", title="gold")])
dots = alt.Chart(frenzy).mark_circle(color=ORANGE, size=60, opacity=1).encode(
    x="date:T", y="gold_close:Q", tooltip=[alt.Tooltip("date:T", title="Frenzy day")])
left.altair_chart(style((price + dots).properties(height=230, title="gold price · red dots = Frenzy days")),
                  width="stretch")

scores = pview.melt(id_vars="date", value_vars=["attention_score", "mood_score"], var_name="score")
scores["score"] = scores["score"].map({"attention_score": "Attention", "mood_score": "Mood (experimental)"})
bands = pd.DataFrame({"y": [0, 20, 80, 95], "y2": [20, 80, 95, 100],
                      "band": ["Quiet", "Normal", "Busy", "Frenzy"]})
shade = alt.Chart(bands).mark_rect(opacity=0.6).encode(
    y="y:Q", y2="y2:Q", color=alt.Color("band:N", scale=alt.Scale(
        domain=["Quiet", "Normal", "Busy", "Frenzy"], range=["#ECEEF1", "#FFFFFF", "#FDF1E2", "#FBDDD2"]),
        legend=alt.Legend(title="Attention band", orient="bottom")))
lines = alt.Chart(scores).mark_line(strokeWidth=1.3).encode(
    x=XAXIS, y=alt.Y("value:Q", title="score (0–100)", scale=alt.Scale(domain=[0, 100])),
    color=alt.Color("score:N", scale=alt.Scale(range=[ORANGE, MOOD_GREY]),
                    legend=alt.Legend(orient="bottom", title="Score")),
    tooltip=[alt.Tooltip("date:T"), "score:N", alt.Tooltip("value:Q", format=".0f")])
left.altair_chart(style((shade + lines).resolve_scale(color="independent").properties(height=200)),
                  width="stretch")

right.subheader("Posts per day")
recent = index.loc[last_day - pd.Timedelta(days=89):].reset_index()
bars = alt.Chart(recent).mark_bar(color=BAR).encode(
    x=alt.X("date:T", title=None), y=alt.Y("posts:Q", title="posts"),
    tooltip=[alt.Tooltip("date:T"), "posts:Q"])
avg = alt.Chart(recent).mark_line(color=ORANGE, strokeWidth=1.5).encode(x="date:T", y="attention_7d:Q")
right.altair_chart(style((bars + avg).properties(height=430, title="last 90 days · red line = 7-day average")),
                   width="stretch")

# --------------------------------------------------------------------------- hourly
with st.expander("Last 7 days, hour by hour (live collector)"):
    try:
        hourly = load_recent_hourly()
        h1 = alt.Chart(hourly).mark_bar(color=BAR).encode(
            x=alt.X("hour:T", title=None), y=alt.Y("posts:Q", title="posts / hour"))
        st.altair_chart(style(h1.properties(height=150)), width="stretch")
        base = alt.Chart(hourly).encode(x=alt.X("hour:T", title=None))
        mood = base.transform_window(m="mean(mood)", frame=[-5, 0]).mark_line(color=MOOD_GREY).encode(
            y=alt.Y("m:Q", title="tone, VADER (6h average)"))
        g = base.mark_line(color=GOLD).encode(
            y=alt.Y("gold:Q", title="gold, USD/oz (hourly)", scale=alt.Scale(zero=False)))
        st.altair_chart(style(mood.properties(height=160, title="Reddit tone, 6-hour average")), width="stretch")
        st.altair_chart(style(g.properties(height=160, title="Gold price, hourly")), width="stretch")
        st.caption(f"Up to {hourly['hour'].max():%d %b %Y %H:00} UTC. Gold has gaps when the market is closed.")
    except Exception as exc:  # the DB is optional for the daily index
        st.warning(f"Hourly data unavailable: {exc}")

# --------------------------------------------------------------------------- findings
st.subheader("Does Reddit predict gold? We tested it")
st.markdown("""
| Test | Result |
|---|---|
| **Next hour** (124 test hours, Sept 28 – Oct 5 2026) | Adding Reddit sentiment to a price-only model: 48.4% → 56.5% correct, **not significant** (p = 0.19) |
| **Next day** (968 test days, 2023 – 2026, retrained monthly) | Reddit adds **no significant gain** over professional market data (best p = 0.12). No model beat simply guessing "up" |
| **Who leads?** (Granger causality, 2022 – 2026) | **Gold moves come first; Reddit chatter follows** (p < 0.001). Reddit does not help forecast gold (smallest p = 0.20) |
| **Trading on the mood** | Following Reddit's mood earned a Sharpe ratio of 0.07, vs. 0.93 for simply holding gold |

**So the index tells you how excited the crowd is, not where gold goes next.** A Frenzy reading usually
means gold *has already* moved.
""")

with st.expander("How the index is built"):
    st.markdown("""
* **Posts:** every post in r/Gold and r/Silverbugs, plus r/investing posts that mention gold, via the
  Arctic Shift archive. Removed, deleted and non-English posts are dropped, and cross-posts are de-duplicated.
  About 190,000 posts since Dec 2021.
* **Attention score:** this week's average posts per day ÷ the average of the 90 days before, ranked
  against the past 365 days (0–100). Bands: Quiet < 20 ≤ Normal < 80 ≤ Busy < 95 ≤ Frenzy.
* **Mood score:** this week's average VADER tone (−1 to +1, post-weighted), ranked the same way.
  *Experimental:* on 45 hand-labelled posts VADER agreed with people only 38% of the time.
* **Gold price:** COMEX gold futures (GC=F) from Yahoo Finance, for context.
* Every score uses only that day and earlier days. Code: `processing/index.py`; method and limits:
  `docs/PRODUCT.md`.
""")

st.caption(f"GoldPulse · research project · {DISCLAIMER}")
