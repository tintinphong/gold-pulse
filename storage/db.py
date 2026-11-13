import sqlite3
from contextlib import contextmanager

from config import DATA_DIR, DB_PATH

POSTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    post_id       TEXT PRIMARY KEY,
    created_utc   TEXT NOT NULL,
    subreddit     TEXT NOT NULL,
    text_hash     TEXT NOT NULL,
    score         INTEGER NOT NULL,
    num_comments  INTEGER NOT NULL,
    lang          TEXT NOT NULL,
    sentiment     REAL NOT NULL,
    engagement    REAL NOT NULL,
    engagement_stale INTEGER NOT NULL DEFAULT 0,
    collected_at  TEXT NOT NULL
)
"""

PRICE_SCHEMA = """
CREATE TABLE IF NOT EXISTS price_ticks (
    bucket_utc    TEXT PRIMARY KEY,
    xau_usd       REAL NOT NULL,
    collected_at  TEXT NOT NULL
)
"""


@contextmanager
def get_conn():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_columns(conn):
    # CREATE TABLE IF NOT EXISTS won't add a column to a posts table that
    # already exists in the committed .sqlite3 file, so new columns land here.
    existing = {row[1] for row in conn.execute("PRAGMA table_info(posts)")}
    if "engagement_stale" not in existing:
        conn.execute(
            "ALTER TABLE posts ADD COLUMN engagement_stale INTEGER NOT NULL DEFAULT 0"
        )


def init_db():
    with get_conn() as conn:
        conn.execute(POSTS_SCHEMA)
        conn.execute(PRICE_SCHEMA)
        _ensure_columns(conn)


def insert_post(row: dict) -> bool:
    """Returns True if a new row was inserted, False if post_id already existed.
    Only a hash of the post text is stored (processing.bucket.text_hash)."""
    from processing.bucket import text_hash
    row = {**row, "text_hash": text_hash(row["text"])}
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO posts
                (post_id, created_utc, subreddit, text_hash, score,
                 num_comments, lang, sentiment, engagement, engagement_stale,
                 collected_at)
            VALUES (:post_id, :created_utc, :subreddit, :text_hash, :score,
                    :num_comments, :lang, :sentiment, :engagement,
                    :engagement_stale, :collected_at)
            """,
            row,
        )
        return cur.rowcount > 0


def update_post_engagement(
    post_id: str, score: int, num_comments: int, engagement: float
) -> bool:
    """Writes settled score/num_comments over a row that was stored while
    Arctic Shift still had them frozen. Returns True if a row was updated."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            UPDATE posts
               SET score = ?, num_comments = ?, engagement = ?, engagement_stale = 0
             WHERE post_id = ?
            """,
            (score, num_comments, engagement, post_id),
        )
        return cur.rowcount > 0


def select_stale_post_ids(older_than_utc: str, newer_than_utc: str, limit: int) -> list[str]:
    """Post ids still flagged stale and worth re-checking.

    Bounded on both sides: old enough that Arctic Shift should have re-crawled
    real numbers by now, but not so old that we keep retrying forever. Posts
    the API never returns (deleted inside the settle window) would otherwise
    sit at the head of this queue permanently and starve newer ones.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT post_id FROM posts
             WHERE engagement_stale = 1
               AND created_utc < ?
               AND created_utc > ?
             ORDER BY created_utc
             LIMIT ?
            """,
            (older_than_utc, newer_than_utc, limit),
        ).fetchall()
        return [row[0] for row in rows]


def upsert_price_ticks(rows: list[dict]) -> tuple[int, int]:
    """Inserts new hour buckets and corrects ones whose price differs.

    Unchanged rows are left alone (collected_at included), so re-syncing an
    overlapping window doesn't touch the file and the bot has nothing to
    commit. Returns (inserted, updated).
    """
    with get_conn() as conn:
        existing = dict(conn.execute("SELECT bucket_utc, xau_usd FROM price_ticks"))
        inserted = updated = 0
        for row in rows:
            old = existing.get(row["bucket_utc"])
            if old is None:
                inserted += 1
            elif abs(old - row["xau_usd"]) > 1e-6:
                updated += 1
            else:
                continue
            conn.execute(
                """
                INSERT OR REPLACE INTO price_ticks (bucket_utc, xau_usd, collected_at)
                VALUES (:bucket_utc, :xau_usd, :collected_at)
                """,
                row,
            )
        return inserted, updated


def select_price_buckets(first_utc: str, last_utc: str) -> list[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT bucket_utc FROM price_ticks WHERE bucket_utc BETWEEN ? AND ?",
            (first_utc, last_utc),
        ).fetchall()
        return [row[0] for row in rows]


def delete_price_ticks(buckets: list[str]) -> int:
    with get_conn() as conn:
        cur = conn.executemany(
            "DELETE FROM price_ticks WHERE bucket_utc = ?", [(b,) for b in buckets]
        )
        return cur.rowcount
