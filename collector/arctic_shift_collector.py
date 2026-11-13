"""Collects Reddit posts through the Arctic Shift archive API.

Replaces the original PRAW collector: Reddit turned down our API application,
and Arctic Shift (https://github.com/ArthurHeitmann/arctic_shift) serves the
same submissions with no credentials and no app approval. It also serves
history, which PRAW's .new() never could -- see backfill.py.

API docs: https://github.com/ArthurHeitmann/arctic_shift/blob/master/api/README.md
"""

import logging
import random
import threading
import time
from datetime import datetime, timedelta, timezone

import requests

import config
from sentiment.scorer import clean_text, detect_language, score_sentiment
from storage.db import insert_post

logger = logging.getLogger("goldpulse.collector.arctic_shift")

_BASE = "https://arctic-shift.photon-reddit.com/api"
_SEARCH_URL = f"{_BASE}/posts/search"
_IDS_URL = f"{_BASE}/posts/ids"

_HEADERS = {"User-Agent": "GoldPulse/1.0 (university coursework project)"}

_REMOVED_MARKERS = {"[removed]", "[deleted]", ""}
_DELETED_AUTHORS = {None, "[deleted]", "[removed]"}

# A rate-limit "seconds from now" larger than this is really a unix timestamp.
_EPOCH_THRESHOLD = 10_000_000
# However long the server claims, never park a poll for more than this.
_MAX_BACKOFF_SECONDS = 300

_last_request_at = 0.0
_rate_lock = threading.Lock()


def _unwrap(payload) -> list[dict]:
    """Pulls the list of items out of a response body.

    The API README documents the per-post fields but not the envelope, so this
    tolerates both a bare list and the {"data": [...]} shape that Arctic
    Shift inherits from Pushshift. If a live response turns out to nest items
    somewhere else, this is the only function that needs to change.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("data") or []
    return []


def _throttle():
    global _last_request_at
    with _rate_lock:
        wait = config.ARCTIC_SHIFT_MIN_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def _retry_after_seconds(resp) -> float | None:
    reset = resp.headers.get("X-RateLimit-Reset")
    if reset:
        try:
            value = float(reset)
            # Some APIs send a delta, others an absolute epoch. A "delta" past
            # this threshold is really a timestamp -- sleeping on it verbatim
            # would park the collector for decades.
            if value > _EPOCH_THRESHOLD:
                value -= datetime.now(tz=timezone.utc).timestamp()
            return max(0.0, value)
        except ValueError:
            pass
    reset_at = resp.headers.get("X-RateLimit-Reset-At")
    if reset_at:
        try:
            target = datetime.fromisoformat(reset_at.replace("Z", "+00:00"))
            return max(0.0, (target - datetime.now(tz=timezone.utc)).total_seconds())
        except ValueError:
            pass
    return None


def _is_query_timeout(resp) -> bool:
    """Arctic Shift reports an overloaded query as HTTP 422 with
    {"error": "Timeout. Maybe slow down a bit"}. That's transient, so it gets
    retried; any other 422 is a real request error and still raises."""
    if resp.status_code != 422:
        return False
    try:
        error = resp.json().get("error") or ""
    except (ValueError, AttributeError):
        return False
    return "timeout" in str(error).lower()


def _get(url: str, params: dict, *, max_retries: int = 5) -> list[dict]:
    """GETs one page, honouring Arctic Shift's dynamic rate limits.

    It's a free service run by one person, so requests are spaced out and 429s
    wait for the reset the server asks for rather than hammering it.
    """
    for attempt in range(max_retries):
        _throttle()
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=30)

        if resp.status_code == 429 or resp.status_code >= 500 or _is_query_timeout(resp):
            if attempt == max_retries - 1:
                resp.raise_for_status()
            backoff = _retry_after_seconds(resp) if resp.status_code == 429 else None
            if backoff is None:
                backoff = 2 ** (attempt + 1)
            backoff = min(backoff, _MAX_BACKOFF_SECONDS) + random.uniform(0, 1)
            logger.warning(
                "arctic shift %s on %s, sleeping %.1fs (attempt %d/%d)",
                resp.status_code, url, backoff, attempt + 1, max_retries,
            )
            time.sleep(backoff)
            continue

        resp.raise_for_status()
        return _unwrap(resp.json())

    return []


def fetch_posts(
    subreddit: str,
    *,
    limit: int = 100,
    before: int | None = None,
    after: int | None = None,
    sort: str = "desc",
) -> list[dict]:
    """One page of submissions from a subreddit. `before`/`after` are unix seconds."""
    params = {"subreddit": subreddit, "limit": limit, "sort": sort}
    if before is not None:
        params["before"] = before
    if after is not None:
        params["after"] = after
    return _get(_SEARCH_URL, params)


def fetch_posts_by_ids(ids: list[str]) -> list[dict]:
    """Looks up specific submissions by id (max 500 per the API docs)."""
    if not ids:
        return []
    return _get(_IDS_URL, {"ids": ",".join(ids)})


def compute_engagement(score: int, num_comments: int) -> float:
    """Comments mean back-and-forth discussion, so they're weighted above a
    single upvote when approximating how much a post is "engaged with"."""
    return score + 2 * num_comments


def matches_keywords(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in config.KEYWORDS)


def transform(
    item: dict,
    *,
    needs_keyword_match: bool = False,
    now: datetime | None = None,
) -> tuple[dict | None, str]:
    """Turns one raw API item into a `posts` row.

    Pure: no network, no database, no clock beyond the injectable `now`, so the
    whole filtering and scoring contract is testable offline. Returns
    (row, "ok") or (None, reason) where reason is one of
    "removed" / "no_keyword" / "lang".
    """
    now = now or datetime.now(tz=timezone.utc)

    title = item.get("title") or ""
    selftext = item.get("selftext") or ""
    if item.get("author") in _DELETED_AUTHORS or (
        selftext in _REMOVED_MARKERS and not title
    ):
        return None, "removed"

    raw_text = f"{title} {selftext}".strip()
    if needs_keyword_match and not matches_keywords(raw_text):
        return None, "no_keyword"

    cleaned = clean_text(raw_text)
    lang = detect_language(cleaned)
    if lang != "en":
        return None, "lang"

    score = item.get("score") or 0
    num_comments = item.get("num_comments") or 0
    created_utc = item["created_utc"]
    created = datetime.fromtimestamp(created_utc, tz=timezone.utc)

    # Arctic Shift archives a post the moment it appears and only backfills
    # real score/num_comments on a later crawl (~36h), so anything younger
    # than that carries placeholder engagement. Flagged rather than dropped;
    # collector/engagement_refresh.py settles these later.
    stale = (now - created) < timedelta(seconds=config.ENGAGEMENT_SETTLE_SECONDS)

    return {
        "post_id": item["id"],
        "created_utc": created.isoformat(),
        "subreddit": item.get("subreddit", ""),
        "text": cleaned,
        "score": score,
        "num_comments": num_comments,
        "lang": lang,
        "sentiment": score_sentiment(cleaned),
        "engagement": compute_engagement(score, num_comments),
        "engagement_stale": int(stale),
        "collected_at": now.isoformat(),
    }, "ok"


def new_stats() -> dict:
    return {"seen": 0, "inserted": 0, "stale": 0, "skipped_lang": 0,
            "skipped_removed": 0, "skipped_keyword": 0, "skipped_malformed": 0,
            "failed_subreddits": 0}


def ingest(items: list[dict], *, needs_keyword_match: bool, stats: dict,
           now: datetime | None = None, store: bool = True) -> list[dict]:
    """Scores and stores a page of raw items. Returns the rows it accepted.

    Live collection and backfill both go through here, so a backfilled row and
    a freshly polled row are identical by construction. `store=False` runs the
    same filtering and scoring but writes nothing, which is what backfill's
    --dry-run uses -- so a dry run exercises the real code path.
    """
    accepted = []
    for item in items:
        stats["seen"] += 1
        try:
            row, reason = transform(item, needs_keyword_match=needs_keyword_match, now=now)
        except (KeyError, TypeError, ValueError, OSError):
            # One malformed item shouldn't cost us the rest of the page.
            logger.exception("skipping unparseable item: %r", item.get("id", "<no id>"))
            stats["skipped_malformed"] += 1
            continue
        if row is None:
            stats[{"removed": "skipped_removed",
                   "lang": "skipped_lang",
                   "no_keyword": "skipped_keyword"}[reason]] += 1
            continue
        if not store:
            accepted.append(row)
            stats["inserted"] += 1
            stats["stale"] += row["engagement_stale"]
        elif insert_post(row):
            accepted.append(row)
            stats["inserted"] += 1
            stats["stale"] += row["engagement_stale"]
    return accepted


class CollectionFailed(RuntimeError):
    """Every subreddit failed -- the run collected nothing at all."""


def collect_once() -> dict:
    """Polls every configured subreddit once. Returns a summary of the run.

    Raises CollectionFailed if no subreddit could be reached. A silent
    zero-row run is exactly how the PRAW outage hid behind green cron runs
    for two days, so a total failure has to be loud.
    """
    stats = new_stats()

    subreddits = config.FULL_COVERAGE_SUBREDDITS + config.KEYWORD_FILTERED_SUBREDDITS
    for subreddit_name in subreddits:
        needs_keyword_match = subreddit_name in config.KEYWORD_FILTERED_SUBREDDITS
        try:
            items = fetch_posts(subreddit_name, limit=config.SUBMISSIONS_PER_POLL)
        except requests.RequestException:
            logger.exception("arctic shift fetch failed for r/%s", subreddit_name)
            stats["failed_subreddits"] += 1
            continue
        ingest(items, needs_keyword_match=needs_keyword_match, stats=stats)

    logger.info(
        "poll complete: seen=%d inserted=%d (stale=%d) skipped_lang=%d "
        "skipped_removed=%d skipped_keyword=%d skipped_malformed=%d failed_subreddits=%d",
        stats["seen"], stats["inserted"], stats["stale"], stats["skipped_lang"],
        stats["skipped_removed"], stats["skipped_keyword"],
        stats["skipped_malformed"], stats["failed_subreddits"],
    )

    if stats["failed_subreddits"] == len(subreddits):
        raise CollectionFailed(
            f"all {len(subreddits)} subreddits failed -- Arctic Shift unreachable?"
        )
    return stats
