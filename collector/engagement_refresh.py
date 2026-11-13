"""Settles engagement numbers on posts that were stored too early.

Arctic Shift archives a submission when it appears and only backfills its real
score/num_comments on a later crawl (~36h). Rows collected before that carry
placeholder values flagged with engagement_stale=1; this pass re-fetches them
once they're old enough and writes the settled numbers back.
"""

import logging
from datetime import datetime, timedelta, timezone

import config
from collector.arctic_shift_collector import compute_engagement, fetch_posts_by_ids
from storage.db import select_stale_post_ids, update_post_engagement

logger = logging.getLogger("goldpulse.collector.engagement_refresh")

_ID_CHUNK = 100


def refresh_stale_engagement(*, max_posts: int = 200) -> dict:
    """Re-fetches stale posts past the settle window and updates them."""
    stats = {"candidates": 0, "fetched": 0, "updated": 0}

    now = datetime.now(tz=timezone.utc)
    settled_by = now - timedelta(seconds=config.ENGAGEMENT_SETTLE_SECONDS)
    give_up_at = now - timedelta(seconds=config.ENGAGEMENT_GIVE_UP_SECONDS)
    post_ids = select_stale_post_ids(
        settled_by.isoformat(), give_up_at.isoformat(), max_posts
    )
    stats["candidates"] = len(post_ids)
    if not post_ids:
        return stats

    for start in range(0, len(post_ids), _ID_CHUNK):
        chunk = post_ids[start:start + _ID_CHUNK]
        for item in fetch_posts_by_ids(chunk):
            stats["fetched"] += 1
            score = item.get("score") or 0
            num_comments = item.get("num_comments") or 0
            if update_post_engagement(
                item["id"], score, num_comments, compute_engagement(score, num_comments)
            ):
                stats["updated"] += 1

    logger.info(
        "engagement refresh: candidates=%d fetched=%d updated=%d",
        stats["candidates"], stats["fetched"], stats["updated"],
    )
    return stats
