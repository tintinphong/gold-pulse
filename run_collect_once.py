"""Entrypoint for the GitHub Actions collect.yml cron job: one poll, then exit."""
import logging

from collector.arctic_shift_collector import collect_once
from collector.engagement_refresh import refresh_stale_engagement
from storage.db import init_db


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("goldpulse.run_collect_once")

    init_db()
    collect_once()

    # Settling old rows is a bonus pass, not the point of this run -- a failure
    # here must not fail the collect job or we lose the poll for this cycle.
    try:
        refresh_stale_engagement()
    except Exception:
        logger.exception("engagement refresh failed, continuing")


if __name__ == "__main__":
    main()
