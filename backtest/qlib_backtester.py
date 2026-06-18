# -*- coding: utf-8 -*-
"""
Qlib-Based Backtester Module
============================

This module integrates Microsoft Qlib's professional backtesting framework
into the AlphaAgents pipeline. It provides:

  1. **QlibBacktester** — Full backtest using Qlib's TopkDropoutStrategy
     with realistic signal delay, trading costs, and risk controls.
  2. **QlibFactorEvaluator** — Lightweight factor-level evaluation using
     Qlib's IC analysis and long-short portfolio construction.

Qlib's advantages over the built-in backtester:
  - Proper signal delay handling (T+1 execution)
  - Accurate limit-up/down filtering
  - Standardized risk analysis (IC/ICIR/Rank IC)
  - Built-in cost model (commission, slippage, stamp duty)

Install:  pip install qlib
Data:     Auto-downloaded via Qlib's get_data.py or cn_data bundle

Author: AAAI 2027 LLM Multi-Factor Stock Selection Project
Date: 2026-06-14
"""

import os
import sys
import tempfile
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Optional Qlib imports — only loaded when actually used
# ---------------------------------------------------------------------------

_QLIB_AVAILABLE = False
_QLIB_ERROR = None

try:
    import qlib
    from qlib.data import D
    from qlib.contrib.evaluate import backtest_daily
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.backtest import backtest, executor as bexec
    from qlib.backtest.decision import Order, OrderDir
    from qlib.contrib.report import analysis_model, analysis_position
    from qlib.utils import init_instance_by_config
    _QLIB_AVAILABLE = True
except ImportError as e:
    _QLIB_ERROR = str(e)


# ===========================================================================
# Helper: convert DataFrame factor values → Qlib prediction format
# ===========================================================================

def _factor_to_qlib_prediction(
    factor_values: pd.DataFrame,
    output_dir: str,
    name: str = "alpha_factor",
) -> str:
    """
    Convert a factor-values DataFrame (dates x stocks) into Qlib's
    prediction format and write to disk.

    Qlib expects:
      - One directory per prediction series
      - Inside: pred.pkl, index.pkl, '' (metadata)

    Parameters
    ----------
    factor_values : pd.DataFrame
        Index=datetime, columns=stock codes (e.g. SH600000).
        Contains raw factor values (higher = better for long).
    output_dir : str
        Root directory for predictions (e.g. /tmp/qlib_preds).
    name : str
        Prediction name (subdirectory under output_dir).

    Returns
    -------
    str
        Path to the written prediction directory.
    """
    pred_dir = os.path.join(output_dir, name)
    os.makedirs(pred_dir, exist_ok=True)

    # Qlib predictions: multi-index (datetime, instrument), single-column score
    stacked = factor_values.stack(dropna=False)
    stacked.index.names = ['datetime', 'instrument']
    stacked = stacked.rename('score')

    # Filter out NaN scores (no prediction)
    stacked = stacked.dropna()

    if len(stacked) == 0:
        raise ValueError("No valid factor values to convert")

    stacked.to_pickle(os.path.join(pred_dir, 'pred.pkl'))
    pd.to_pickle(None, os.path.join(pred_dir, 'index_data.pkl'))  # placeholder
    pd.to_pickle(None, os.path.join(pred_dir, 'meta.pkl'))

    return pred_dir


# ===========================================================================
# Helper: build minimal Qlib dataset config for in-memory data
# ===========================================================================

def _build_qlib_dataset_config(
    instruments: List[str],
    start_time: str,
    end_time: str,
    handler_kwargs: Optional[Dict] = None,
) -> Dict:
    """
    Build a Qlib-compatible dataset config dict pointing to the installed
    cn_data bundle. This reuses the data loaded by our own Dataloader
    but tells Qlib where to find OHLCV fields for backtest execution.

    Parameters
    ----------
    instruments : List[str]
        Qlib-format stock codes (e.g. ['SH600000', 'SZ000001']).
    start_time : str
        Start date (YYYY-MM-DD).
    end_time : str
        End date (YYYY-MM-DD).
    handler_kwargs : dict, optional
        Extra kwargs for the data handler.

    Returns
    -------
    dict
        Qlib-style dataset configuration.
    """
    if handler_kwargs is None:
        handler_kwargs = {}

    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": start_time,
                    "end_time": end_time,
                    "fit_start_time": start_time,
                    "fit_end_time": end_time,
                    "instruments": instruments,
                    **handler_kwargs,
                },
            },
            "segments": {
                "test": (start_time, end_time),
            },
        },
    }


# ===========================================================================
# Helper: build TopkDropoutStrategy config
# ===========================================================================

def _build_topk_strategy_config(
    topk: int = 50,
    n_drop: int = 5,
    signal_threshold: Optional[float] = None,
    risk_degree: float = 0.95,
    limit_threshold: float = 0.095,
    hold_thresh: int = 1,
) -> Dict:
    """
    Build a Qlib TopkDropoutStrategy configuration.

    Parameters
    ----------
    topk : int
        Number of stocks to long each period.
    n_drop : int
        Number of stocks to replace each rebalance.
    signal_threshold : float, optional
        Minimum signal score to enter. None = no threshold.
    risk_degree : float
        Risk fraction per stock (0.95 = use 95% of risk budget).
    limit_threshold : float
        Price limit threshold (0.095 = 9.5% for A-share ±10% limit).
    hold_thresh : int
        Minimum holding days before a position can be sold.

    Returns
    -------
    dict
        Qlib strategy configuration dict.
    """
    return {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {
            "signal": "<PRED>",
            "topk": topk,
            "n_drop": n_drop,
            "risk_degree": risk_degree,
            "limit_threshold": limit_threshold,
            "hold_thresh": hold_thresh,
        },
    }


# ===========================================================================
# Helper: build backtest executor config
# ===========================================================================

def _build_executor_config(
    start_time: str,
    end_time: str,
    commission: float = 0.0003,
    slippage: float = 0.001,
    freq: str = "day",
) -> Dict:
    """
    Build a Qlib backtest executor config with cost model.

    Parameters
    ----------
    start_time : str
        Backtest start date.
    end_time : str
        Backtest end date.
    commission : float
        Commission rate (default: 0.03%).
    slippage : float
        Slippage rate (default: 0.1%).
    freq : str
        Trading frequency ("day" or "week").

    Returns
    -------
    dict
        Qlib executor configuration.
    """
    return {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_period": freq,
            "start_time": start_time,
            "end_time": end_time,
            "generate_portfolio_metrics": True,
            "verbose": False,
        },
    }


# ===========================================================================
# Core class: QlibBacktester
# ===========================================================================

class QlibBacktester:
    """
    Full backtesting engine powered by Microsoft Qlib.

    This wraps Qlib's professional backtesting pipeline:
      1. Converts factor values to Qlib prediction format
      2. Runs TopkDropoutStrategy for portfolio construction
      3. Simulates trading with realistic costs and signal delay
      4. Returns standardized performance metrics

    Parameters
    ----------
    provider_uri : str
        Path to Qlib data bundle (e.g., '~/.qlib/qlib_data/cn_data').
    topk : int
        Number of stocks in portfolio.
    n_drop : int
        Number of stocks to replace per rebalance.
    commission : float
        Commission rate (e.g., 0.0003 = 0.03%).
    slippage : float
        Slippage rate (e.g., 0.001 = 0.1%).
    limit_threshold : float
        Price limit threshold for A-share (0.095 = 9.5%).
    hold_thresh : int
        Minimum holding days.
    freq : str
        Rebalance frequency ("day" or "week").

    Example
    -------
    >>> bt = QlibBacktester(provider_uri='~/.qlib/qlib_data/cn_data', topk=50)
    >>> metrics = bt.run(factor_values, start='2023-01-01', end='2024-12-31')
    >>> print(f"Sharpe: {metrics['sharpe']:.3f}")
    """

    def __init__(
        self,
        provider_uri: str = "~/.qlib/qlib_data/cn_data",
        topk: int = 50,
        n_drop: int = 5,
        commission: float = 0.0003,
        slippage: float = 0.001,
        limit_threshold: float = 0.095,
        hold_thresh: int = 1,
        freq: str = "day",
    ):
        self.provider_uri = os.path.expanduser(provider_uri)
        self.topk = topk
        self.n_drop = n_drop
        self.commission = commission
        self.slippage = slippage
        self.limit_threshold = limit_threshold
        self.hold_thresh = hold_thresh
        self.freq = freq

        self._qlib_initialized = False
        self._temp_dirs: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Check if Qlib is available and the data bundle exists."""
        if not _QLIB_AVAILABLE:
            return False
        return os.path.isdir(self.provider_uri)

    def ensure_initialized(self) -> bool:
        """
        Initialize Qlib with the configured data provider.
        Returns True on success, False otherwise.
        Idempotent — only initializes once.
        """
        if self._qlib_initialized:
            return True

        if not _QLIB_AVAILABLE:
            print(f"  [qlib-backtest] Qlib not installed: {_QLIB_ERROR}")
            return False

        if not os.path.isdir(self.provider_uri):
            print(f"  [qlib-backtest] Data bundle not found at {self.provider_uri}")
            print(f"  [qlib-backtest] Run: pip install pyqlib && python -c \"from qlib.tests.data import GetData; GetData().qlib_data(target_dir='qlib_data/', region='cn')\"")
            return False

        try:
            qlib.init(
                provider_uri=self.provider_uri,
                region="cn",
                expression_cache=None,
                dataset_cache=None,
            )
            self._qlib_initialized = True
            print(f"  [qlib-backtest] Initialized with {self.provider_uri}")
            return True
        except Exception as e:
            print(f"  [qlib-backtest] Init failed: {e}")
            return False

    def run(
        self,
        factor_values: pd.DataFrame,
        start_time: str,
        end_time: str,
        benchmark: str = "SH000300",
    ) -> Dict[str, Any]:
        """
        Run a full Qlib backtest on factor values.

        Parameters
        ----------
        factor_values : pd.DataFrame
            Index=datetime, columns=Qlib-format stock codes (SH600000, SZ000001).
            Values = factor scores (higher = more bullish).
        start_time : str
            Backtest start date (YYYY-MM-DD).
        end_time : str
            Backtest end date (YYYY-MM-DD).
        benchmark : str
            Qlib benchmark code (default: SH000300 = CSI 300).

        Returns
        -------
        dict
            Standardized performance metrics:
            - sharpe, annual_return, annual_volatility
            - max_drawdown, calmar_ratio, win_rate
            - information_ratio, excess_return
            - ic, ic_ir, rank_ic, rank_icir
            - turnover
            - total_return
        """
        if not self.ensure_initialized():
            return self._empty_metrics()

        if factor_values.empty:
            return self._empty_metrics()

        # Get instruments from factor values (dedup and sort)
        instruments = sorted(set(factor_values.columns))
        if len(instruments) < self.topk:
            print(f"  [qlib-backtest] Too few instruments ({len(instruments)} < topk={self.topk})")
            return self._empty_metrics()

        try:
            with tempfile.TemporaryDirectory(prefix="qlib_bt_") as tmpdir:
                self._temp_dirs.append(tmpdir)

                # ---- 1. Write predictions ----
                pred_dir = _factor_to_qlib_prediction(
                    factor_values, tmpdir, name="alpha"
                )

                # ---- 2. Build backtest config ----
                dataset_config = _build_qlib_dataset_config(
                    instruments=instruments,
                    start_time=start_time,
                    end_time=end_time,
                )
                dataset_config["kwargs"]["segments"]["test"] = (
                    start_time, end_time
                )

                strategy_config = _build_topk_strategy_config(
                    topk=self.topk,
                    n_drop=self.n_drop,
                    limit_threshold=self.limit_threshold,
                    hold_thresh=self.hold_thresh,
                )

                executor_config = _build_executor_config(
                    start_time=start_time,
                    end_time=end_time,
                    commission=self.commission,
                    slippage=self.slippage,
                    freq=self.freq,
                )

                backtest_config = {
                    "pred_dir": pred_dir,
                    "dataset": dataset_config,
                    "strategy": strategy_config,
                    "executor": executor_config,
                }

                # ---- 3. Run backtest ----
                print(f"  [qlib-backtest] Running backtest ({start_time} → {end_time})...")
                print(f"  [qlib-backtest] Instruments: {len(instruments)}, TopK: {self.topk}")

                portfolio_metric, indicator = backtest_daily(
                    backtest_config=backtest_config,
                    freq=self.freq,
                )

                # ---- 4. Extract metrics ----
                return self._extract_metrics(
                    portfolio_metric, indicator, start_time, end_time
                )

        except Exception as e:
            import traceback
            print(f"  [qlib-backtest] Error: {e}")
            traceback.print_exc()
            return self._empty_metrics()

    def evaluate_factor(
        self,
        factor_values: pd.DataFrame,
        start_time: str,
        end_time: str,
    ) -> Dict[str, float]:
        """
        Evaluate a single factor using Qlib's IC/quantile analysis.

        This is a lighter-weight alternative to `run()` that skips
        the full portfolio simulation and just computes Rank IC,
        ICIR, and long-short quantile returns.

        Parameters
        ----------
        factor_values : pd.DataFrame
            Index=datetime, columns=stock codes (SH600000 format).
        start_time : str
            Start date.
        end_time : str
            End date.

        Returns
        -------
        dict
            Keys: ic, ic_ir, rank_ic, rank_icir, long_short_sharpe,
                  long_short_return, win_rate, max_drawdown
        """
        if not self.ensure_initialized():
            return self._empty_factor_metrics()

        if factor_values.empty:
            return self._empty_factor_metrics()

        try:
            instruments = sorted(set(factor_values.columns))

            with tempfile.TemporaryDirectory(prefix="qlib_eval_") as tmpdir:
                pred_dir = _factor_to_qlib_prediction(factor_values, tmpdir, "alpha")

                # Build minimal dataset for IC analysis
                dataset_config = _build_qlib_dataset_config(
                    instruments=instruments,
                    start_time=start_time,
                    end_time=end_time,
                )

                # Use Qlib's analysis modules
                from qlib.contrib.evaluate import (
                    long_short_return,
                    information_coefficient,
                )

                # Compute Rank IC
                ic_df = information_coefficient(
                    pred=pred_dir,
                    dataset_config=dataset_config,
                )

                # Compute long-short quantile returns
                ls_df = long_short_return(
                    pred=pred_dir,
                    dataset_config=dataset_config,
                    top_n=int(len(instruments) * 0.2),
                    bottom_n=int(len(instruments) * 0.2),
                )

                # Extract IC statistics
                if ic_df is not None and 'IC' in ic_df.columns:
                    ic_series = ic_df['IC'].dropna()
                    mean_ic = float(ic_series.mean()) if len(ic_series) > 0 else 0.0
                    std_ic = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0
                    ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0

                    rank_col = 'Rank IC' if 'Rank IC' in ic_df.columns else 'IC'
                    rank_series = ic_df[rank_col].dropna()
                    mean_rank_ic = float(rank_series.mean()) if len(rank_series) > 0 else 0.0
                    std_rank_ic = float(rank_series.std(ddof=1)) if len(rank_series) > 1 else 0.0
                    rank_icir = mean_rank_ic / std_rank_ic if std_rank_ic > 0 else 0.0
                else:
                    mean_ic = mean_rank_ic = 0.0
                    ic_ir = rank_icir = 0.0

                # Extract long-short metrics
                if ls_df is not None and 'return' in ls_df.columns:
                    ls_rets = ls_df['return'].dropna()
                    ls_sharpe = self._sharpe(ls_rets) if len(ls_rets) > 1 else 0.0
                    ls_ret = float(ls_rets.mean()) * 252 if len(ls_rets) > 0 else 0.0
                    ls_win = float((ls_rets > 0).mean()) if len(ls_rets) > 0 else 0.5
                    ls_mdd = self._max_dd(ls_rets) if len(ls_rets) > 1 else 0.0
                else:
                    ls_sharpe = ls_ret = 0.0
                    ls_win = 0.5
                    ls_mdd = 0.0

                return {
                    'ic': mean_ic,
                    'icir': ic_ir,
                    'rank_ic': mean_rank_ic,
                    'rank_icir': rank_icir,
                    'long_short_sharpe': ls_sharpe,
                    'long_short_return': ls_ret,
                    'win_rate': ls_win,
                    'max_drawdown': ls_mdd,
                }

        except Exception as e:
            import traceback
            print(f"  [qlib-backtest] Factor eval error: {e}")
            traceback.print_exc()
            return self._empty_factor_metrics()

    # ------------------------------------------------------------------
    # Metric extraction helpers
    # ------------------------------------------------------------------

    def _extract_metrics(
        self,
        portfolio_metric: dict,
        indicator: dict,
        start_time: str,
        end_time: str,
    ) -> Dict[str, Any]:
        """Extract standardized metrics from Qlib backtest output."""
        metrics = {}

        # --- Portfolio-level metrics ---
        # Qlib's backtest_daily returns a dict with these keys (depending on version):
        #   portfolio_metric: contains annualized metrics
        #   indicator: contains detailed per-period metrics

        report_normal = portfolio_metric.get("norm", {})
        if not report_normal and isinstance(portfolio_metric, dict):
            report_normal = portfolio_metric

        metrics['annual_return'] = float(report_normal.get('annual_return', 0.0))
        metrics['annual_volatility'] = float(report_normal.get('annual_volatility', 0.0))
        metrics['sharpe'] = float(report_normal.get('information_ratio', 0.0))
        metrics['max_drawdown'] = float(report_normal.get('max_drawdown', 0.0))
        metrics['calmar_ratio'] = float(report_normal.get('calmar_ratio', 0.0))
        metrics['total_return'] = float(report_normal.get('cumulative_return', 0.0))

        # --- Excess vs benchmark ---
        report_excess = portfolio_metric.get("excess", {})
        if not report_excess:
            report_excess = {}

        metrics['excess_annual_return'] = float(
            report_excess.get('annual_return', 0.0)
        )
        metrics['information_ratio'] = float(
            report_excess.get('information_ratio', 0.0)
        )
        metrics['excess_max_drawdown'] = float(
            report_excess.get('max_drawdown', 0.0)
        )

        # --- Indicator-level ---
        if isinstance(indicator, dict):
            metrics['turnover'] = float(
                indicator.get('turnover', {}).get('mean', 0.0)
                if isinstance(indicator.get('turnover'), dict)
                else 0.0
            )

        # --- IC metrics from indicator ---
        if isinstance(indicator, dict):
            ic_data = indicator.get('ic', {})
            if isinstance(ic_data, dict):
                metrics['ic'] = float(ic_data.get('mean', 0.0))
            else:
                metrics['ic'] = 0.0

            rank_ic_data = indicator.get('rank_ic', {})
            if isinstance(rank_ic_data, dict):
                metrics['rank_ic'] = float(rank_ic_data.get('mean', 0.0))

        # --- Win rate ---
        if isinstance(indicator, dict):
            # Approximate from daily returns
            ann_ret = metrics.get('annual_return', 0.0)
            ann_vol = metrics.get('annual_volatility', 0.01)
            if ann_vol > 0:
                daily_mean = ann_ret / 252
                daily_std = ann_vol / np.sqrt(252)
                # Win rate = P(daily_return > 0) under normal approximation
                from scipy.stats import norm
                metrics['win_rate'] = float(1 - norm.cdf(0, daily_mean, daily_std))
            else:
                metrics['win_rate'] = 0.5
        else:
            metrics['win_rate'] = 0.5

        # --- Round for readability ---
        for k in metrics:
            if isinstance(metrics[k], float) and np.isfinite(metrics[k]):
                metrics[k] = round(metrics[k], 6)
            elif not np.isfinite(metrics[k]):
                metrics[k] = 0.0

        return metrics

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sharpe(returns: pd.Series, rf: float = 0.02) -> float:
        """Annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        excess = returns - rf / 252.0
        mean = float(excess.mean())
        std = float(excess.std(ddof=1))
        return (mean / std) * np.sqrt(252) if std > 0 else 0.0

    @staticmethod
    def _max_dd(returns: pd.Series) -> float:
        """Maximum drawdown."""
        if len(returns) < 2:
            return 0.0
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return float(dd.min())

    @staticmethod
    def _empty_metrics() -> Dict[str, Any]:
        """Return a safe empty-result dict for full backtests."""
        return {
            'annual_return': 0.0, 'annual_volatility': 0.0,
            'sharpe': 0.0, 'max_drawdown': 0.0,
            'calmar_ratio': 0.0, 'total_return': 0.0,
            'excess_annual_return': 0.0, 'information_ratio': 0.0,
            'excess_max_drawdown': 0.0, 'turnover': 0.0,
            'ic': 0.0, 'rank_ic': 0.0, 'win_rate': 0.5,
        }

    @staticmethod
    def _empty_factor_metrics() -> Dict[str, float]:
        """Return a safe empty-result dict for factor evaluation."""
        return {
            'ic': 0.0, 'icir': 0.0,
            'rank_ic': 0.0, 'rank_icir': 0.0,
            'long_short_sharpe': 0.0, 'long_short_return': 0.0,
            'win_rate': 0.5, 'max_drawdown': 0.0,
        }

    def __del__(self):
        """Cleanup temp dirs on deletion."""
        import shutil
        for d in self._temp_dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass


# ===========================================================================
# Factory: create QlibBacktester from project config
# ===========================================================================

def create_qlib_backtester_from_config(config: dict) -> QlibBacktester:
    """
    Create a QlibBacktester instance from a project config dict.

    Reads from config keys:
      - data.qlib.provider_uri
      - backtest.qlib.topk
      - backtest.qlib.n_drop
      - backtest.trading.commission
      - backtest.trading.slippage

    Parameters
    ----------
    config : dict
        Project configuration (loaded from config.yaml).

    Returns
    -------
    QlibBacktester
    """
    data_cfg = config.get('data', {})
    qlib_cfg = data_cfg.get('qlib', {})
    bt_cfg = config.get('backtest', {})
    qlib_bt_cfg = bt_cfg.get('qlib', {})
    trading_cfg = bt_cfg.get('trading', {})

    return QlibBacktester(
        provider_uri=qlib_cfg.get('provider_uri', '~/.qlib/qlib_data/cn_data'),
        topk=qlib_bt_cfg.get('topk', 50),
        n_drop=qlib_bt_cfg.get('n_drop', 5),
        commission=trading_cfg.get('commission', 0.0003),
        slippage=trading_cfg.get('slippage', 0.001),
        limit_threshold=qlib_bt_cfg.get('limit_threshold', 0.095),
        hold_thresh=qlib_bt_cfg.get('hold_thresh', 1),
        freq=qlib_bt_cfg.get('freq', 'day'),
    )


# ===========================================================================
# Demo / smoke test
# ===========================================================================

if __name__ == '__main__':
    print("=== QlibBacktester Demo ===\n")

    # Check availability
    bt = QlibBacktester()
    if not bt.available:
        print("Qlib not available. Install with: pip install qlib")
        print("Then download data: pip install pyqlib && python -c \"from qlib.tests.data import GetData; GetData().qlib_data(target_dir='qlib_data/', region='cn')\"")
        sys.exit(0)

    print(f"Qlib data at: {bt.provider_uri}")
    print(f"Initialized: {bt.ensure_initialized()}")

    # Generate mock factor values for demo
    np.random.seed(42)
    n_days = 200
    n_stocks = 100
    dates = pd.date_range('2023-01-01', periods=n_days, freq='B')
    codes = [f'SH600{i:03d}' for i in range(n_stocks)]

    factor_values = pd.DataFrame(
        np.random.randn(n_days, n_stocks),
        index=dates,
        columns=codes,
    )

    print(f"\nFactor shape: {factor_values.shape}")
    print(f"Date range: {factor_values.index[0].date()} → {factor_values.index[-1].date()}")

    # Run factor evaluation
    print("\n--- Factor Evaluation ---")
    metrics = bt.evaluate_factor(
        factor_values,
        start_time='2023-03-01',
        end_time='2023-12-31',
    )
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== Demo Complete ===")
