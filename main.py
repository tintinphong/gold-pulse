import logging
import sys
import time

import config
from collector.arctic_shift_collector import collect_once as collect_reddit_once
from collector.engagement_refresh import refresh_stale_engagement
from price.price_feed import collect_once as collect_price_once
from storage.db import init_db


def setup_logging():
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_PATH),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    setup_logging()
    logger = logging.getLogger("goldpulse.main")

    init_db()

    logger.info(
        "GoldPulse collector starting: subreddits=%s poll=%ss price_poll=%ss",
        config.FULL_COVERAGE_SUBREDDITS + config.KEYWORD_FILTERED_SUBREDDITS,
        config.POLL_INTERVAL_SECONDS,
        config.PRICE_POLL_INTERVAL_SECONDS,
    )

    ticks_per_price_pull = config.PRICE_POLL_INTERVAL_SECONDS // config.POLL_INTERVAL_SECONDS
    tick = 0
    while True:
        try:
            collect_reddit_once()
        except Exception:
            logger.exception("reddit poll failed, will retry next cycle")

        try:
            refresh_stale_engagement()
        except Exception:
            logger.exception("engagement refresh failed, will retry next cycle")

        if tick % ticks_per_price_pull == 0:
            try:
                collect_price_once()
            except Exception:
                logger.exception("price poll failed, will retry next cycle")

        tick += 1
        time.sleep(config.POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
