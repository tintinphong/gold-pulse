"""Offline tests for the raw-post -> database-row transform.

transform() is deliberately pure, so the filtering, scoring and staleness
rules can be checked without touching the network or the database.
"""
import unittest
from datetime import datetime, timedelta, timezone

from collector.arctic_shift_collector import (
    _is_query_timeout,
    _retry_after_seconds,
    compute_engagement,
    ingest,
    new_stats,
    transform,
)

NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


def raw(**overrides) -> dict:
    """A settled, on-topic r/Gold post; override fields per test."""
    item = {
        "id": "abc123",
        "subreddit": "Gold",
        "author": "some_user",
        "title": "Gold just broke $3900 an ounce",
        "selftext": "Bullion premiums are climbing and I am staying long.",
        "created_utc": (NOW - timedelta(days=5)).timestamp(),
        "score": 120,
        "num_comments": 34,
    }
    item.update(overrides)
    return item


class TransformTests(unittest.TestCase):
    def test_happy_path_builds_a_row(self):
        row, reason = transform(raw(), now=NOW)
        self.assertEqual(reason, "ok")
        self.assertEqual(row["post_id"], "abc123")
        self.assertEqual(row["subreddit"], "Gold")
        self.assertEqual(row["lang"], "en")
        self.assertIn("Gold just broke", row["text"])
        self.assertEqual(row["engagement"], compute_engagement(120, 34))

    def test_deleted_author_is_skipped(self):
        _, reason = transform(raw(author="[deleted]"), now=NOW)
        self.assertEqual(reason, "removed")

    def test_removed_body_with_no_title_is_skipped(self):
        _, reason = transform(raw(title="", selftext="[removed]"), now=NOW)
        self.assertEqual(reason, "removed")

    def test_off_topic_post_is_skipped_when_keyword_filtering(self):
        off_topic = raw(
            subreddit="investing",
            title="Thoughts on rebalancing my bond ladder?",
            selftext="Nothing here is about the metal in question.",
        )
        _, reason = transform(off_topic, needs_keyword_match=True, now=NOW)
        self.assertEqual(reason, "no_keyword")

        # ...but the same filter lets a gold post through.
        _, reason = transform(raw(subreddit="investing"), needs_keyword_match=True, now=NOW)
        self.assertEqual(reason, "ok")

    def test_engagement_is_flagged_stale_only_inside_the_settle_window(self):
        fresh = raw(created_utc=(NOW - timedelta(hours=2)).timestamp(), score=1, num_comments=0)
        row, _ = transform(fresh, now=NOW)
        self.assertEqual(row["engagement_stale"], 1)

        settled = raw(created_utc=(NOW - timedelta(hours=48)).timestamp())
        row, _ = transform(settled, now=NOW)
        self.assertEqual(row["engagement_stale"], 0)


class IngestTests(unittest.TestCase):
    """store=False keeps these off the database while still running the real path."""

    def test_one_malformed_item_does_not_lose_the_rest_of_the_page(self):
        stats = new_stats()
        # The middle item survives every filter, then dies on a missing
        # created_utc -- i.e. it fails deep in transform, not at the gate.
        broken = {"id": "broken", "author": "u", "subreddit": "Gold",
                  "title": "Gold is holding above 3900", "selftext": "Still long."}
        page = [raw(id="ok1"), broken, raw(id="ok2")]
        accepted = ingest(page, needs_keyword_match=False, stats=stats, store=False)

        self.assertEqual([r["post_id"] for r in accepted], ["ok1", "ok2"])
        self.assertEqual(stats["skipped_malformed"], 1)
        self.assertEqual(stats["inserted"], 2)


class RateLimitHeaderTests(unittest.TestCase):
    class FakeResponse:
        def __init__(self, headers):
            self.headers = headers

    def test_reset_is_read_as_a_delta(self):
        resp = self.FakeResponse({"X-RateLimit-Reset": "30"})
        self.assertAlmostEqual(_retry_after_seconds(resp), 30, delta=1)

    def test_reset_sent_as_an_epoch_is_converted_not_slept_on_verbatim(self):
        soon = datetime.now(tz=timezone.utc).timestamp() + 45
        resp = self.FakeResponse({"X-RateLimit-Reset": str(soon)})
        self.assertAlmostEqual(_retry_after_seconds(resp), 45, delta=2)

    def test_missing_headers_give_no_opinion(self):
        self.assertIsNone(_retry_after_seconds(self.FakeResponse({})))


class QueryTimeoutTests(unittest.TestCase):
    """Arctic Shift's overload reply is a 422, not a 429/5xx. Not retrying it
    is how r/Gold went silently uncollected for ~10h on Oct 5."""

    class FakeResponse:
        def __init__(self, status_code, body):
            self.status_code = status_code
            self._body = body

        def json(self):
            if isinstance(self._body, Exception):
                raise self._body
            return self._body

    def test_422_timeout_is_retryable(self):
        resp = self.FakeResponse(422, {"error": "Timeout. Maybe slow down a bit", "data": None})
        self.assertTrue(_is_query_timeout(resp))

    def test_other_422_still_raises(self):
        self.assertFalse(_is_query_timeout(self.FakeResponse(422, {"error": "Invalid subreddit"})))
        self.assertFalse(_is_query_timeout(self.FakeResponse(422, ValueError("not json"))))

    def test_success_is_not_a_timeout(self):
        self.assertFalse(_is_query_timeout(self.FakeResponse(200, {"data": []})))


if __name__ == "__main__":
    unittest.main()
