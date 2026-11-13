"""One XAU/USD sync (trailing PRICE_LOOKBACK_DAYS of hourly bars), then exit.

Run by collect.yml every 15 minutes, and by price.yml on manual dispatch.
"""
import logging

from price.price_feed import collect_once
from storage.db import init_db


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_db()
    collect_once()


if __name__ == "__main__":
    main()
