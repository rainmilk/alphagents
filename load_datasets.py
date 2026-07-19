# -*- coding: utf-8 -*-
"""
load_datasets.py — Pre-fetch A-share market data into the local dataset store.

This is the ONLY entry point that downloads data. It fetches the FULL
(universe, start_date, end_date) span from the configured source (westock /
tushare / akshare), preprocesses it, and persists the result to:

    datasets/{universe}_{start_date}_{end_date}.pkl

start_date / end_date define the FULL pre-fetch window (e.g. the entire
history you intend to train/test on). The train/test split itself happens
later, at retrieval time: DataLoader.load_data() / retrieve_dataset() slice
(train_start, train_end) and (test_start, test_end) out of this single
archive, so every experiment shares one pre-fetched file and one split point.

After running this once, every other entry point (MASE main.py, the 9
baselines) retrieves train/test slices from that local store via
DataLoader.load_data() — no network access at experiment time.

Examples:
    # Use config.yaml defaults (data.universe.* = the FULL span)
    python load_datasets.py

    # Explicit FULL span
    python load_datasets.py --universe hs300 \
        --start-date 2019-01-01 --end-date 2025-12-31 --source westock

    # Re-download even if the archive already exists
    python load_datasets.py --force-refresh
"""

import os
import argparse
import yaml


def _resolve_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(
        description="Pre-fetch A-share data into the local dataset store (datasets/)."
    )
    parser.add_argument(
        "--universe", type=str, default=None,
        help="Stock universe (hs300/zz500/all_a). Default: config data.universe.index",
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Start date YYYY-MM-DD. Default: config data.universe.start_date",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="End date YYYY-MM-DD. Default: config data.universe.end_date",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Data source (westock/tushare/akshare/auto). Default: config data.source",
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Re-download even if the dataset archive already exists.",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    args = parser.parse_args()

    # Resolve defaults from config so the archive key matches what
    # DataLoader.load_data() will build when other methods retrieve it.
    cfg = _resolve_config(args.config)
    univ_cfg = cfg.get("data", {}).get("universe", {})
    source_default = cfg.get("data", {}).get("source", "auto")

    universe = args.universe or univ_cfg.get("index", "hs300")
    start_date = args.start_date or univ_cfg.get("start_date")
    end_date = args.end_date or univ_cfg.get("end_date")
    source = args.source or source_default

    if not (start_date and end_date):
        parser.error(
            "start_date/end_date are required: pass --start-date/--end-date "
            "or set data.universe.start_date/end_date in the config."
        )

    # Import here so --help works without pulling heavy deps.
    from dataloader.loader import DataLoader, dataset_path

    print("=" * 60)
    print("[load_datasets] Pre-fetching local dataset")
    print(f"  universe   : {universe}")
    print(f"  start_date : {start_date}")
    print(f"  end_date   : {end_date}")
    print(f"  source     : {source}")
    print(f"  archive    : {dataset_path(universe, start_date, end_date)}")
    print("=" * 60)

    loader = DataLoader(config_path=args.config)
    loader.fetch_and_store_dataset(
        start_date=start_date,
        end_date=end_date,
        universe=universe,
        source=source,
        force_refresh=args.force_refresh,
    )
    print("\n[load_datasets] Done. Other methods can now retrieve this dataset locally.")


if __name__ == "__main__":
    main()
