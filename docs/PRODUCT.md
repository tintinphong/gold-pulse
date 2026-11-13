# GoldPulse Index: product outline

**One line:** a daily, public measure of how much, and how warmly, retail investors talk about gold,
built from 190,000+ Reddit posts since 2022. It is a thermometer, not a forecast.

## The problem
Gold buyers and the firms that serve them (bullion dealers, brokers, financial media) see retail
excitement build in waves, like the October 2025 break above $4,000 and the January 2026 run past
$5,000. They have no consistent, numeric way to tell whether today's crowd is calm or frenzied.
Professional sentiment products (Bloomberg's news sentiment, for example) don't even cover gold itself:
Bloomberg returns "Field Not Applicable" for spot gold (pitfall #37).

## The product
| Component | What it shows | How it's built | Status |
|---|---|---|---|
| **Attention score** (headline, 0–100) | How unusual this week's gold chatter is: 0 = the quietest in a year, 100 = the biggest surge | 7-day posts per day ÷ the previous 90 days' average, ranked against the past 365 days | Robust: a post count, no scorer needed |
| **Attention band** | Quiet (< 20) · Normal (20–80) · Busy (80–95) · Frenzy (≥ 95) | From the score | – |
| **Mood score** (0–100) | Whether posts read warmer or gloomier than usual this year | 7-day post-weighted VADER score, ranked against the past 365 days | **Experimental**: VADER failed our hand-label check |
| **History** | Daily from Sept 2022 (scores need 6 months of history), back data from Dec 2021 | `python -m processing.index` → `data/clean/goldpulse_index.parquet` | – |

Sources: r/Gold, r/Silverbugs and gold-keyword posts in r/investing (Arctic Shift archive), plus daily
GC=F gold futures from Yahoo for context. **No Bloomberg data** is in the product (licensing).

## Who pays, and for what
| Customer | Use | Model |
|---|---|---|
| Retail gold buyers | "Is everyone piling in right now?" Context before buying, to avoid buying in a frenzy | **Free** public dashboard |
| Bullion dealers and brokers | Stock and staffing planning for demand surges; customer newsletters | **Licence**: daily index feed + API |
| Financial media | "Retail gold fever hits a 1-year high" stories, with a chart | **Licence**: data and embeddable chart, with attribution |
| Researchers and students | Clean, de-duplicated historical series for their own studies | **Paid export** of the full daily history |

Precedents for this kind of product: the AAII Investor Sentiment Survey, CNN's Fear & Greed Index, Google Trends.

## What it does *not* do, and why we can say so
We tested whether Reddit predicts gold, at two horizons, with supervised models and pre-set pass/fail rules:
- **Next hour:** +8 points over a price-only model on 124 test hours, but not significant (p = 0.19).
- **Next day, over 4 years:** 968 walk-forward test days against a Bloomberg market baseline gave no
  significant gain (best p = 0.12), and no model beat simply guessing "up" (53.7%).

So the index is sold as **context, never as a buy/sell signal**. Every page carries: *"For information only.
Not investment advice. The GoldPulse Index describes online discussion; it does not predict prices."*

## Risks and limits
- **Data access:** we depend on Arctic Shift, a free one-person archive. Reddit rejected our API
  application, so a commercial version would need a licensed Reddit data feed.
- **Who's talking:** Reddit users are not all gold buyers. Silver fans dominate r/Silverbugs (57% of posts).
- **The mood score's accuracy:** VADER agreed with human labels only 37.8% of the time. A better gold-specific
  scorer, trained on more hand labels, is the first upgrade.
- **Regulation:** describing sentiment is information, not advice. Adding buy/sell calls would change that.
