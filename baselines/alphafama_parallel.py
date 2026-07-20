# -*- coding: utf-8 -*-
"""Out-of-__main__ worker for parallel Alpha101 factor computation.

This lives in its own module on purpose: on Windows (spawn start method),
ProcessPoolExecutor children unpickle the work function by *import name*.
If the worker lived inside ``run_alphafama.py`` (the __main__ script), every
child would re-import and re-run that script — re-executing the whole
pipeline (data load, etc.) and recursing the pool until it fails. By keeping
the worker here, children import ONLY this lightweight module (pandas +
AlphaFactory), avoiding the re-execution / freeze_support recursion.
"""
import pandas as pd
from baselines.AlphaFAMA.src.alpha_functions import AlphaFactory


def _compute_factors_chunk(ticker_groups):
    """Compute Alpha101 exposures + forward returns for a *chunk* of tickers.

    Runs inside a worker process under ProcessPoolExecutor. Returns
    ``(ex_list, ret_list)`` with the same element structure as the original
    serial loop. The per-ticker math is identical to the serial path — only
    the scheduling is distributed, so numerical results are unchanged.
    """
    ex_list, ret_list = [], []
    for ticker, grp in ticker_groups:
        alphas = AlphaFactory.all_alphas(grp)
        ex_list.append(
            pd.DataFrame(alphas, index=grp.index).assign(ticker=ticker)
        )
        # IC TARGET must be the forward-period return so AlphaFAMA's Rank-IC is
        # comparable to the other baselines. We keep the daily `returns` column
        # intact for the Alpha101 feature inputs and use `forward_return` here.
        # Rename to 'returns' so compute_ic_matrix (which reads ['returns']) works.
        ret_list.append(
            grp[["forward_return"]]
            .rename(columns={"forward_return": "returns"})
            .assign(ticker=ticker)
        )
    return ex_list, ret_list
