"""Offline guard for legacy standalone data-fetch scripts.

The MASE workflow fetches market data exactly ONCE via the standalone
`load_datasets.py` CLI and persists it to the local `datasets/` store.
Every other entry point (main.py, the 9 baselines) then RETRIEVES from
that store and never touches the network.

Some vendored baseline sub-packages (AlphaForge, AlphaAgent, AlphaFAMA,
mcts_llm_alpha) still ship their *original* live-fetch scripts
(baostock / Stooq / qlib.init / QlibDataLoader / tushare / akshare).
Those are NOT part of the MASE flow. To prevent accidental network
access and data drift, they call `assert_offline_or_optin` and are
blocked by default.

Opt in to a legacy live fetch intentionally with:
    export MASE_ALLOW_LEGACY_FETCH=1
"""
import os
import warnings


def assert_offline_or_optin(where: str) -> None:
    """Block a live data-source connection unless explicitly opted in.

    Args:
        where: human-readable identifier of the caller (e.g. ``__name__`` or a
            script path) used in the error / warning message.
    """
    if os.environ.get("MASE_ALLOW_LEGACY_FETCH") == "1":
        warnings.warn(
            f"[legacy-fetch] {where}: live data access enabled via "
            f"MASE_ALLOW_LEGACY_FETCH=1. This bypasses the unified "
            f"datasets/ store and may produce data inconsistent with MASE runs.",
            stacklevel=2,
        )
        return
    raise RuntimeError(
        f"[offline-guard] {where} would connect to a live data source, but the "
        f"MASE workflow forbids network access outside `load_datasets.py`.\n"
        f"  -> To use the unified local store, pre-fetch once:\n"
        f"      python load_datasets.py --universe <u> --start-date <s> --end-date <e>\n"
        f"  -> To intentionally run this legacy live-fetch script, set:\n"
        f"      export MASE_ALLOW_LEGACY_FETCH=1\n"
    )
