"""Standalone engagement refresh, for running the pass on its own.

The collect cron already runs this after each poll (see run_collect_once.py);
this entrypoint exists for catching up by hand after an outage.
"""
import logging

from collector.engagement_refresh import refresh_stale_engagement
from storage.db import init_db


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_db()
    refresh_stale_engagement()


if __name__ == "__main__":
    main()
