"""
Plot cumulative returns (or cumulative excess returns) from baseline
portfolio_value.csv files.

Typical usage
-------------
# Single experiment directory: one subplot with all baselines
python visual/plot_cumulative_returns.py \
    experiments/zz500_2022-07-01_2024-12-31_forward-5_holding-1 \
    -o visual/cumret_zz500.png

# Two markets side-by-side (like the paper Figure 3)
python visual/plot_cumulative_returns.py \
    experiments/zz500_2022-07-01_2024-12-31_forward-5_holding-1 \
    experiments/hs300_2022-07-01_2024-12-31_forward-5_holding-1 \
    --labels "(a) CSI 500" "(b) CSI 300" \
    -o visual/cumret_two_panels.png

# Cumulative EXCESS return: subtract a buy-and-hold benchmark curve
python visual/plot_cumulative_returns.py \
    experiments/zz500_2022-07-01_2024-12-31_forward-5_holding-1 \
    --benchmark data/benchmark/zz500_portfolio_values.csv \
    -o visual/excess_ret_zz500.png

# Use automatic benchmark discovery (experiments/<...>/benchmark/portfolio_values.csv)
python visual/plot_cumulative_returns.py \
    experiments/zz500_2022-07-01_2024-12-31_forward-5_holding-1 \
    --auto-benchmark \
    -o visual/excess_ret_zz500.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Use a non-interactive backend unless we explicitly request display; this
# keeps the script safe on headless servers while still allowing `--show`.
if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")


def _find_portfolio_files(experiment_dir: str) -> Dict[str, Path]:
    """Find all portfolio_values.csv files under an experiment directory.

    The directory layout is assumed to be:
        experiments/<experiment_name>/<baseline_name>/portfolio_values.csv

    We only look at direct subdirectories of ``experiment_dir`` and skip
    common non-baseline folders (results, __pycache__, .git, benchmark when
    auto-benchmark is used, etc.).
    """
    root = Path(experiment_dir)
    files: Dict[str, Path] = {}
    if not root.is_dir():
        print(f"[warn] Not a directory: {root}", file=sys.stderr)
        return files

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        # Skip helper/artifact folders that are not baselines.
        if name.startswith((".", "_")) or name in {
            "__pycache__", "results", "benchmark", "logs", "plots", "figures"
        }:
            continue
        csv_path = subdir / "portfolio_values.csv"
        if csv_path.exists():
            files[name] = csv_path
        else:
            print(f"[warn] {subdir}: portfolio_values.csv not found", file=sys.stderr)
    return files


def _read_portfolio_series(csv_path: Path) -> Optional[pd.Series]:
    """Read a portfolio_values.csv file and return a date-indexed Series.

    Expected format:
        ,portfolio_value
        2024-07-02,1.0
        2024-07-03,1.0043
        ...
    """
    try:
        df = pd.read_csv(
            csv_path,
            index_col=0,
            parse_dates=True,
            dayfirst=False,
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] Failed to read {csv_path}: {exc}", file=sys.stderr)
        return None

    if df.empty:
        print(f"[warn] Empty file: {csv_path}", file=sys.stderr)
        return None

    # Accept either an explicit 'portfolio_value' column or the second column.
    col = None
    for candidate in ("portfolio_value", "value", "nav", "net_value"):
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        col = df.columns[0]

    series = df[col].astype(float).squeeze()
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    series.name = csv_path.parent.name
    return series


def _to_cumulative_return(series: pd.Series) -> pd.Series:
    """Convert a NAV series to cumulative return starting from 0."""
    start_value = series.dropna().iloc[0]
    if start_value == 0:
        raise ValueError(f"Starting portfolio value is zero for {series.name}")
    return (series / start_value) - 1.0


def _subtract_benchmark(
    strategy_cumret: pd.Series,
    benchmark_series: pd.Series,
) -> pd.Series:
    """Compute cumulative excess return by subtracting aligned benchmark."""
    bench_cumret = _to_cumulative_return(benchmark_series)
    # Align to the strategy's date range and forward-fill missing benchmark
    # values (e.g. holidays). Drop dates where both are not available.
    aligned = pd.concat([strategy_cumret, bench_cumret], axis=1)
    aligned.columns = ["strategy", "benchmark"]
    aligned["benchmark"] = aligned["benchmark"].ffill().bfill()
    aligned = aligned.dropna()
    excess = aligned["strategy"] - aligned["benchmark"]
    excess.name = strategy_cumret.name
    return excess


def _resolve_benchmark(
    experiment_dir: str,
    explicit_benchmark: Optional[str],
    auto_benchmark: bool,
) -> Optional[pd.Series]:
    """Resolve the benchmark series for one experiment directory."""
    if explicit_benchmark is not None:
        path = Path(explicit_benchmark)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")
        return _read_portfolio_series(path)

    if auto_benchmark:
        candidate = Path(experiment_dir) / "benchmark" / "portfolio_values.csv"
        if candidate.exists():
            return _read_portfolio_series(candidate)
        print(
            f"[warn] Auto-benchmark not found: {candidate}; drawing cumulative return instead",
            file=sys.stderr,
        )
    return None


def _make_palette(n: int) -> List[Tuple[float, float, float, float]]:
    """Return a color palette with enough distinct colors."""
    cmap = plt.cm.tab10
    if n <= 10:
        return [cmap(i / 10) for i in range(n)]
    # Fallback for many baselines.
    cmap = plt.cm.tab20
    return [cmap(i / 20) for i in range(n)]


def _clean_label(raw: str) -> str:
    """Convert a directory name to a human-readable legend label."""
    mapping = {
        "alphafama": "AlphaFAMA",
        "alphaforge": "AlphaForge",
        "alphagent": "AlphaAgent",
        "alphaagent": "AlphaAgent",
        "alphagrail": "AlphaGrail",
        "alphagen": "AlphaGen",
        "alphamcts": "AlphaMCTS",
        "mcts_llm": "MCTS-LLM",
        "mcts_llm_alpha": "MCTS-LLM-Alpha",
        "xgboost_simple": "XGBoost",
        "lightgbm_simple": "LightGBM",
        "rd_agent": "RD-Agent",
        "mase": "MASE",
    }
    key = raw.lower()
    return mapping.get(key, raw.replace("_", " ").title())


def plot_experiment(
    ax: plt.Axes,
    experiment_dir: str,
    panel_label: Optional[str] = None,
    benchmark: Optional[pd.Series] = None,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> None:
    """Plot all baseline curves for a single experiment directory on ``ax``."""
    files = _find_portfolio_files(experiment_dir)
    if not files:
        ax.set_title(panel_label or Path(experiment_dir).name)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    include_set = set(include) if include else None
    exclude_set = set(exclude) if exclude else set()

    series_list: List[pd.Series] = []
    labels: List[str] = []
    for name, csv_path in files.items():
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue
        series = _read_portfolio_series(csv_path)
        if series is None:
            continue
        series_list.append(series)
        labels.append(_clean_label(name))

    if not series_list:
        ax.set_title(panel_label or Path(experiment_dir).name)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    colors = _make_palette(len(series_list))
    for series, label, color in zip(series_list, labels, colors):
        cumret = _to_cumulative_return(series)
        if benchmark is not None:
            cumret = _subtract_benchmark(cumret, benchmark)
        ax.plot(cumret.index, cumret.values, label=label, color=color, linewidth=1.5)

    title = panel_label or Path(experiment_dir).name
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Date", fontsize=10)
    ylabel = "Cumulative Excess Return" if benchmark is not None else "Cumulative Return"
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    # When subplots share an axis, matplotlib hides y-tick labels on the
    # non-leftmost panels. Re-enable them so every panel is readable.
    ax.tick_params(labelleft=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot cumulative returns from baseline portfolio_value.csv files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dirs",
        nargs="+",
        help="Experiment directories containing baseline subdirectories.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="visual/cumulative_returns.png",
        help="Output image path (default: visual/cumulative_returns.png).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Panel labels, one per experiment directory (e.g. '(a) CSI 500').",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Single benchmark portfolio_values.csv to subtract from every panel.",
    )
    parser.add_argument(
        "--auto-benchmark",
        action="store_true",
        help="Look for <experiment_dir>/benchmark/portfolio_values.csv per panel.",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=None,
        help="Only include these baseline subdirectory names.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        help="Exclude these baseline subdirectory names.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Overall figure title (optional).",
    )
    parser.add_argument(
        "--figsize",
        default=None,
        help="Figure size in inches, e.g. '14,5'. Default scales with panel count.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively (default: only save to file).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output image DPI (default: 300).",
    )
    args = parser.parse_args(argv)

    if args.labels and len(args.labels) != len(args.dirs):
        parser.error("--labels must have the same number of entries as experiment dirs")

    # Prepare figure size.
    n_panels = len(args.dirs)
    if args.figsize:
        figsize = tuple(float(x) for x in args.figsize.split(","))
    else:
        width = 7 * n_panels
        height = 5
        figsize = (width, height)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=True)
    if n_panels == 1:
        axes = [axes]

    for idx, exp_dir in enumerate(args.dirs):
        benchmark = _resolve_benchmark(exp_dir, args.benchmark, args.auto_benchmark)
        panel_label = args.labels[idx] if args.labels else None
        plot_experiment(
            ax=axes[idx],
            experiment_dir=exp_dir,
            panel_label=panel_label,
            benchmark=benchmark,
            include=args.include,
            exclude=args.exclude,
        )

    if args.title:
        fig.suptitle(args.title, fontsize=14, y=1.02)

    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure: {out_path}")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
