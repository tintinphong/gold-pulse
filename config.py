from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# r/Gold and r/Silverbugs are on-topic by construction; r/investing is broad,
# so it only counts when a post also matches KEYWORDS.
FULL_COVERAGE_SUBREDDITS = ["Gold", "Silverbugs"]
KEYWORD_FILTERED_SUBREDDITS = ["investing"]

KEYWORDS = [
    "gold", "xau", "xauusd", "bullion", "spot price", "gold price",
    "troy ounce", "gold etf", "gld", "gold miner", "precious metal",
]

POLL_INTERVAL_SECONDS = 15 * 60          # Reddit collector cadence (Phase 2)
PRICE_POLL_INTERVAL_SECONDS = 60 * 60    # XAU/USD reference pull cadence
SUBMISSIONS_PER_POLL = 100               # posts requested per subreddit per poll

# Arctic Shift archives a post at creation and only backfills its real
# score/num_comments on a later crawl, so engagement isn't trustworthy until
# roughly this much time has passed. See collector/engagement_refresh.py.
ENGAGEMENT_SETTLE_SECONDS = 36 * 60 * 60
# Past this age a still-stale post is presumed gone for good (deleted inside
# the settle window) and stops being retried, so it can't clog the queue.
ENGAGEMENT_GIVE_UP_SECONDS = 7 * 24 * 60 * 60
ARCTIC_SHIFT_MIN_INTERVAL_SECONDS = 1.0  # be polite: it's a free, one-maintainer service
BACKFILL_PAGE_LIMIT = 100                # API max per page

PRICE_TICKER = "GC=F"                    # COMEX gold futures, USD/oz — closest free XAU/USD proxy on Yahoo Finance
# Each live price run re-syncs this many trailing days of hourly bars, so a
# run of skipped cron jobs (or a whole weekend) heals on the next one.
PRICE_LOOKBACK_DAYS = 5

DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "goldpulse.sqlite3"
LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "goldpulse.log"
