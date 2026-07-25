# AAAI 2027 LLM Multi-Factor Stock Selection Project

## Overview

This is the complete codebase for the AAAI 2027 paper:  
**"LLM-Driven Self-Evolving Multi-Factor Stock Selection with State-Aware Memory"**  
(MASE: Memory-Augmented Self-Evolving Framework)

## Project Structure

```
AAAI2027_LLM_MultiFactor/
├── config/                 # Configuration files
│   └── config.yaml        # Main configuration ([ACTIVE]/[UNUSED] annotated)
├── data/                  # Data loading and output
│   ├── loader.py          # Data loader (sample / real A-share data)
│   ├── train/             # Train split CSVs (price/fundamental/industry)
│   ├── test/              # Test split CSVs (price/fundamental/industry)
│   └── memory_bank.json/  # Factor memory bank snapshots
├── methods/               # Core methods (4 modules)
│   ├── debate.py         # Multi-agent debate evaluator + Chair synthesis
│   ├── evolve.py         # Self-evolving factor generator (config-driven)
│   ├── memory.py         # State-aware factor memory bank (FAISS)
│   └── fusion.py         # Factor fusion + normalization + portfolio construction
├── backtest/              # Backtest engine
│   └── engine.py         # Backtest simulation (holding_period, turnover-aware)
├── metrics/               # Evaluation metrics
│   └── evaluator.py      # Comprehensive metrics
├── viz/                   # Visualization tools
│   └── plotter.py        # Plotting functions (standalone runnable)
├── experiments/           # Experiment results (parameter-tagged root)
│   └── {universe}_{start}_{end}_forward-{fp}_holding-{hp}/   # e.g. hs300_2019-01-01_2025-12-31_forward-10_holding-1
│       ├── {yyyymmdd}/    # MASE per-run outputs (date-stamped)
│       │   └── results/   # performance_metrics.csv, portfolios.csv, evolution_history.json, ...
│       │       ├── self_evolve/round_0/generated_factors.json   # seed factors
│       │       ├── debate/debate_factors_result.json            # expert opinions + Chair synthesis
│       │       └── fusion/final_factors.json                    # final fused factors
│       ├── alphaagent/    # Baseline outputs (one subdir per method)
│       ├── alphaforge/    #   each contains daily_returns.csv, portfolio_values.csv, ...
│       ├── lstm/          #   (9 baselines total, see "Baselines" section below)
│       └── ...            #   mcts_llm_alpha, alphagen, alphafama, alphagrail, xgboost, xgboost_simple
├── paper/                 # Paper-related files
│   └── figures/          # Generated figures
├── main.py                # Main pipeline (argparse CLI + --test mode)
├── run_experiments.py     # Experiment runner
├── test_train_test_split.py # Train/test split verification script
├── sync_evolve_to_feishu.py # Feishu sync helper
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Installation

### 1. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Install LLM APIs

For full LLM-driven functionality, you need API access to:
- OpenAI GPT-4o (for factor evaluation)
- DeepSeek (for factor generation)
- Sentence Transformers (for factor embedding)

## Usage

### Data Modes

| Mode | Flag | Data | Use Case |
|---|---|---|---|
| **Sample** (default) | no flag or `--sample` | Synthetic random walk data (100 stocks x 500 days) | Fast testing, CI, no API needed |
| **Real** | `--real` | A-share data via westock / AkShare / Tushare | Production research |

**Real data source chain** (automatic fallback):
1. **westock** — WorkBuddy built-in A-share data (preferred, no extra setup)
2. **AkShare** — Open-source Python library (`pip install akshare`)
3. **Tushare** — Requires token (`pip install tushare`, set `TUSHARE_TOKEN`)

### Local Dataset Store (pre-fetch once, retrieve many)

Real data is **fetched exactly once** by the standalone `load_datasets.py` CLI, which
downloads the **FULL** `(universe, start_date, end_date)` span, preprocesses it, and
persists it to `datasets/{universe}_{start_date}_{end_date}.pkl`. **Every other entry point**
(`main.py --real`, all 9 baselines running `--real`) then only **retrieves** that archive
from disk — no network access at experiment time. This makes repeated experiments
(different forward/holding periods, different factor strategies) instant and offline.

`start_date` / `end_date` in `load_datasets.py` define the **full pre-fetch window** (the
entire history you intend to train/test on). The train/test **split itself happens at
retrieval time**: `DataLoader.load_data()` (and the module-level `retrieve_dataset()`)
read the full archive once and slice out the `(train_start, train_end)` and
`(test_start, test_end)` windows on demand — so every experiment shares one pre-fetched
file and a single split location (`config.yaml` `train_start_date` / `train_end_date` /
`test_start_date` / `test_end_date`).

```bash
# Pre-fetch the FULL span using config.yaml defaults (data.universe.*)
python load_datasets.py

# Explicit FULL span + source
python load_datasets.py --universe hs300 \
    --start-date 2019-01-01 --end-date 2025-12-31 --source westock

# Re-download even if the archive already exists
python load_datasets.py --force-refresh
```

Both `load_data()` and `retrieve_dataset()` return a **`DatasetBundle`** with three
slices, each a `(price_data, fundamental_data, industry_data)` triple:

| Attribute | Contents |
|---|---|
| `bundle.full`  | the entire pre-fetched span |
| `bundle.train` | rows in `[train_start, train_end]` |
| `bundle.test`  | rows in `[test_start, test_end]` |

So a baseline no longer re-implements date masking — it just does:
```python
bundle = loader.load_data()                      # split read from config
train_price, train_fund, train_ind = bundle.train
test_price,  test_fund,  test_ind  = bundle.test
# full span (if needed, e.g. for forward-return lookback): bundle.full[0]
```

> If you run `main.py --real` (or a baseline with real data) **before** pre-fetching,
> `retrieve_dataset()` raises a `FileNotFoundError` telling you the exact
> `load_datasets.py` command to run first. Sample mode (`--sample`, the default) is
> unaffected and needs no pre-fetch.

### Offline Guard (single live-fetch path)

To keep the "fetch once, retrieve many" contract enforceable, **`load_datasets.py` is the
only sanctioned live-data entry point.** The vendored baseline sub-packages (AlphaForge,
AlphaAgent, AlphaFAMA, mcts_llm_alpha) still ship their *original* live-fetch scripts
(baostock / Stooq / `qlib.init` / `QlibDataLoader`). Those are **not** part of the MASE
flow and are blocked by default via `dataloader/_offline_guard.py` — running one raises:

```text
[offline-guard] <script> would connect to a live data source, but the MASE workflow
forbids network access outside `load_datasets.py`.
  → To use the unified local store, pre-fetch once:
      python load_datasets.py --universe <u> --start-date <s> --end-date <e>
  → To intentionally run this legacy live-fetch script, set:
      export MASE_ALLOW_LEGACY_FETCH=1
```

Set `MASE_ALLOW_LEGACY_FETCH=1` only when you deliberately want to reproduce an original
baseline's data pipeline for reference — doing so bypasses the unified `datasets/` store
and may produce data inconsistent with MASE runs.

### Quick Demo (No LLM Required)

```bash
python main.py
```

This runs the pipeline with sample data and random portfolios.

### Full Pipeline (Sample Data)

```bash
python main.py --full
```

### Full Pipeline (Real A-Share Data)

> **First, pre-fetch the data once** (see Local Dataset Store above). The commands
> below then retrieve it locally — no re-download.

```bash
# Auto-detect best available source (data already pre-fetched)
python main.py --full --real

# Specify data source
python main.py --full --real --source akshare

# Custom date range + universe (must match the pre-fetched archive)
python main.py --full --real --start 2023-01-01 --end 2024-12-31 --universe hs300

# Custom forward period and holding period
python main.py --full --real --forward-period 20 --holding-period 5
```

### Test Pipeline (Skip Evolution, Run Backtest Only)

Load a previously saved `final_factors.json` and run step7+step8 directly on test data.
This reuses trained factor weights without re-running the full pipeline.

```bash
# Specify factor file explicitly (path is now under the parameter-tagged root)
python main.py --test --factor-path experiments/hs300_2019-01-01_2025-12-31_forward-10_holding-1/20260601/results/final_factors.json

# Auto-detect latest final_factors.json under experiments/{param_dir}/*/results/
python main.py --test

# With real data and custom date range
python main.py --test --factor-path PATH --real --start 2023-01-01 --end 2024-12-31

# Custom holding period (1=daily, 5=weekly, 20=monthly)
python main.py --test --factor-path PATH --holding-period 5

# Custom context days for rolling-window factors
python main.py --test --factor-path PATH --context-days 60
```

### Pipeline Steps (--full mode)

1. **Load data** — sample or real A-share data with preprocessing; saves train/test splits as CSVs to `data/train/` and `data/test/`; prepends `context_days` of training data to test split for rolling-window factor lookback
2. **Initialize memory** — FAISS-backed factor memory bank
3. **Generate factors** — LLM-driven seed factor generation
4. **Evolve factors** — Self-improving iterative evolution (backtester coarse filter, patience-based early stopping)
5. **4b. Retrieve from memory** — State-aware top-k factor retrieval from memory bank _(before debate so memory factors also get debate scores)_
6. **5. Evaluate factors** — Multi-agent debate with 5 experts (final quality gate)
7. **5b. Chair synthesis** — Chair Agent cross-factor ranking with selection/rejection reasons (chair_synthesis.json)
8. **6. Fuse factors** — ICIR-weighted fusion with Bayesian SNR shrinkage + IPR correlation penalty + industry neutralization
9. **7. Construct portfolio** — Top-N portfolio with risk constraints (weights reindexed to full stock universe)
10. **8. Backtest** — Out-of-sample performance metrics on test period
11. **9. Save results** — Metrics + portfolios + evolution history to disk

### CLI Reference

```
python main.py [OPTIONS]

Options:
  --full                   Run full end-to-end pipeline (default: quick demo)
  --test                   Run test pipeline: load saved factors from JSON → step7+step8
  --real                   Use real A-share data (default: sample data)
  --sample                 Use sample/synthetic data (default)
  --source {auto,westock,akshare,tushare}
                           Real data source (default: auto)
  --force-refresh          Skip cache, re-download real data
  --start DATE             Start date for real data (default: 2022-01-01)
  --end DATE               End date for real data (default: 2024-12-31)
  --universe {hs300,zz500,all_a}
                           Stock universe for real data (default: hs300)
  --n-seeds N              [legacy] Convenience: set n_seeds_alpha101_stage (alpha101 seed count). Prefer the three explicit flags below.
  --n-seeds-hypothesis N   Number of hypothesis-driven seed factors (default: config)
  --n-seeds-alpha101-stage N  Number of plain alpha101 seed factors (default: config)
  --n-seeds-memory-augment N Number of memory-augmented seed factors (default: config; 0 = off)
  --n-evolution-rounds N   Override evolution.max_rounds from config (default: 5)
  --n-best-factors N       Override evolution.n_best_factors from config
  --forward-period N       Forward return horizon in trading days (None → config or 20)
  --holding-period N       Backtest holding period: 1=daily, 5=weekly, 20=monthly
  --factor-path PATH       Path to final_factors.json for --test mode
                           (default: auto-detect from experiments/{param_dir}/*/results/)
  --context-days N         Context days prepended before test_start_date for
                           rolling-window factors (None → auto-detect from data or config)
  --config PATH            Path to config file (default: config/config.yaml)
  --output-dir PATH        Output directory (default: experiments/{universe}_{start}_{end}_forward-{fp}_holding-{hp}/{YYYYMMDD}/results/)
```

### Run Experiments

```bash
python run_experiments.py
```

This runs:
- Main experiment (5 runs)
- Ablation studies (remove each component)
- Baseline comparisons
- Robustness tests

### Generate Figures

```bash
python viz/plotter.py
```

## Core Modules

### 1. Multi-Agent Debate Evaluator (`methods/debate.py`)

Implements structured debate among 5 expert agents:
- Momentum Expert
- Value Expert
- Quality Expert
- Volatility Expert
- Growth Expert

Each agent evaluates factors independently (Phase 1), then engages in structured debate rounds (Phase 2) with full reproducibility. A Chair Agent (step5b) then performs cross-factor ranking with explicit selection/rejection reasons. All expert opinions — scores, reasoning, concerns, strengths, and per-round consensus — are saved to `experiments/{universe}_{start}_{end}_forward-{fp}_holding-{hp}/{yyyymmdd}/results/debate/debate_factors_result.json`.

### 2. Self-Evolving Factor Generator (`methods/evolve.py`)

Generates factors through iterative evolution with config-driven hyperparameters:

- Generate seed factors (LLM or rule-based fallback) — count controlled by `evolution.n_seed_factors`
- Backtest and evaluate via `FactorBacktester`
- Parse errors (`ValueError`) → factor marked `is_valid=False` with `parse_error` stored, automatically excluded from ranking and JSON output
- Evolution loop: improve via LLM (`n_improve`) + mutate via rules (`n_mutate`), patience-based early stopping
- Select top-k factors per round — controlled by `evolution.n_best_factors`
- Convergence check using `convergence_delta` over `convergence_window`
- Quality filter on final output: `min_ic`, `min_sharpe`, `max_drawdown`
- Returns `EvolutionResult` (dataclass with `EvolutionRound` history)

**Key config keys** (all active): `n_seed_factors`, `n_best_factors`, `max_rounds`, `forward_period`, `n_improve`, `n_mutate`, `convergence_delta`, `convergence_window`, `patience`, `min_ic`, `min_sharpe`, `max_drawdown`.

### 3. State-Aware Factor Memory Bank (`methods/memory.py`)

Stores and retrieves historical high-quality factors:
- Encode market state (4 dimensions: vix, trend, dispersion, turnover)
- Embed factors (Sentence-BERT)
- Retrieve similar factors (state-aware, FAISS `IndexFlatIP`)
- Update quality scores (online learning)
- Runs after evolution (step4b) so retrieved memory factors participate in debate

### 4. Factor Fusion & Portfolio Construction (`methods/fusion.py`)

Fuses multiple factors into a composite score with a principled multi-stage pipeline:

**Weight Computation** (`_compute_weights`):
- **ICIR² weighting**: Base weights proportional to `|ICIR|²` (SNR-optimal)
- **Bayesian SNR shrinkage**: Shrinkage toward equal weight when ICIR estimates are noisy. `δ = 1/(1 + max(0, var(ICIR)·T - 1))`, where T is the number of periods used to estimate IC/ICIR. More principled than the old heuristic `1/(1 + cv·√n)` — properly accounts for estimation precision via sample size.
- **Debate score blend**: `weights = κ·norm(debate_score) + (1-κ)·base_weights`
- **Market state tilt**: Weights multiplied by market-regime multipliers (trend/correlation-based)

**Correlation Penalty** (IPR — Inter-factor Penalty for Redundancy):
- Compute pairwise factor correlation matrix
- Exposure `E[i] = Σ|corr(i,j)|·w_j / (1-w_i)` = weighted average correlation of factor i with all others
- Penalty `p[i] = 1/(1 + α·E[i])` — continuous sigmoid-type compression
- Redundant factors (high correlation with high-weight peers) get down-weighted; diverse factors get relatively boosted

**Two-Pass Normalization**:
1. First pass: each factor individually → zscore + optional industry neutralization (unifies scale, removes sector bias)
2. Second pass: weighted sum composite → zscore + industry neutralization (restores std=1 after weight averaging shrinks variance)

**Sign Extraction**: Factor sign determined by `sign(IC)`. When `|IC| ≈ 0` (direction unreliable), defaults to +1 (no flip) — conservative choice since near-zero-IC factors already have near-zero weight.

**Portfolio Construction** (`PortfolioConstructor`):
- `build()` returns `list[Portfolio]` dataclass with weights only for top-N stocks
- **Step7 post-processing**: Weights are reindexed to the full stock universe (300 stocks) with zero-fill for unselected stocks, ensuring correct alignment with price data in backtest

### 5. Backtest Engine (`backtest/engine.py`)

- Supports configurable `holding_period` (1=daily, 5=weekly, 20=monthly)
- Turnover calculation: `Σ|Δw| / 2` — standard single-sided turnover metric (buy + sell both contribute to `Σ|Δw|`, division by 2 recovers the actual capital traded)
- Multi-period weight carry-forward for `holding_period > 1`

## Configuration

Edit `config/config.yaml` to customize:

All config keys are annotated with `[ACTIVE]` (wired to code) or `[UNUSED]` (defined as reference/roadmap, no runtime effect). This makes it clear which knobs actually change behavior.

Key active sections:
- **data**: Source, universe, train/test split dates, `context_days` (rolling-window lookback)
- **evolution**: All 12 hyperparameters now wired (was previously partially hardcoded)
- **memory**: State encoding, retrieval settings
- **fusion**: Weighting strategy, normalization method, debate blend, IPR alpha, market tilt
- **backtest**: Holding period, transaction costs, position limits

## Results

All results are saved under a **parameter-tagged root** `experiments/{universe}_{start}_{end}_forward-{fp}_holding-{hp}/`
(e.g. `experiments/hs300_2019-01-01_2025-12-31_forward-10_holding-1/`). The universe/start/end are taken from
`config.yaml`'s `data.universe` (or CLI `--train-start`/`--test-end`/`--universe` overrides); `forward`/`holding` come from
`evolution.forward_period` / `backtest.trading.holding_period` (or `--forward-period` / `--holding-period`).
MASE and all 9 baselines share this same root, so results are directly comparable in one place.

**`experiments/{universe}_{start}_{end}_forward-{fp}_holding-{hp}/{yyyymmdd}/results/`** — MASE per-run outputs:
- `performance_metrics.csv`: Main results. Includes backtest metrics **plus `mean_rank_ic` and `icir`** (test-period Rank-IC of the fused composite factor vs N-day forward returns — added for fair comparison with baselines)
- `portfolios.csv`: Portfolio weights
- `evolution_history.json`: Evolution round history (serialized from dataclass)
- `ablation_studies.json`: Ablation study results
- `baseline_comparisons.json`: Baseline comparison results
- `self_evolve/round_0/generated_factors.json`: Seed factors (LLM Phase 1)
- `self_evolve/round_1/generated_factors.json` … `round_N/`: Evolved factors per round (invalid factors with parse errors are excluded)
- `debate/debate_factors_result.json`: All expert opinions + Chair Agent cross-factor synthesis
- `fusion/final_factors.json`: Final fused factors with weights, signs, and `n_periods` metadata (invalid factors excluded)

**`experiments/{universe}_{start}_{end}_forward-{fp}_holding-{hp}/{method}/`** — Baseline per-method outputs
(one subdir per baseline, see **Baselines** section below). Common files: `daily_returns.csv`, `portfolio_values.csv`,
and a `results.json`/`*_results.json` with metrics (incl. `mean_rank_ic`, `icir`, `sharpe_ratio`, etc.).

Train/test data persists as CSVs under `data/train/` and `data/test/` with `data/split_info.json` metadata.

## Baselines

The paper compares MASE against 9 formulaic-alpha LLM baselines. Each baseline is a **standalone runner** under
`baselines/` (not orchestrated by `main.py`); run them directly.

| # | Method | Runner script | Output subdir |
|---|--------|---------------|---------------|
| 1 | AlphaAgent | `baselines/run_alphaagent.py` | `alphaagent/` |
| 2 | AlphaForge | `baselines/run_alphaforge.py` | `alphaforge/` |
| 3 | MCTS-LLM | `baselines/run_mcts_llm_alpha.py` | `mcts_llm/` |
| 4 | AlphaGen | `baselines/run_alphagen.py` | `alphagen/` |
| 5 | AlphaFAMA | `baselines/run_alphafama.py` | `alphafama/` |
| 6 | AlphaGrail | `baselines/run_alphagrail.py` | `alphagrail/` |
| 7 | LSTM | `baselines/run_lstm_baseline.py` | `lstm/` |
| 8 | XGBoost | `baselines/run_alpha_xgboost.py` | `xgboost/` |
| 9 | XGBoost (simple) | `baselines/run_xgboost_simple.py` | `xgboost_simple/` |

### Running a baseline

```bash
# Each baseline reads config.yaml for universe/date/forward/holding defaults
python baselines/run_alphaagent.py --config-path config/config.yaml

# Override key parameters (consistent across all baselines)
python baselines/run_alphaforge.py \
    --config-path config/config.yaml \
    --train-start 2022-01-01 --test-end 2024-06-30 \
    --forward-period 10 --holding-period 5
```

Common CLI flags (all baselines): `--config-path`, `--train-start`, `--test-end`, `--forward-period`, `--holding-period`,
`--output-dir`. `holding_period` and `forward_period` default to `config.yaml`
(`backtest.trading.holding_period` / `evolution.forward_period`) so runs stay aligned with the MASE pipeline.

### Output layout

Every baseline writes its results under the **shared parameter-tagged root** used by MASE:

```
experiments/{universe}_{start}_{end}_forward-{fp}_holding-{hp}/{method}/
├── daily_returns.csv      # strategy daily returns (from unified BacktestEngine)
├── portfolio_values.csv   # portfolio net-value curve
├── results.json           # metrics (sharpe_ratio, max_drawdown, mean_rank_ic, icir, ...); filename varies per method
└── <method-specific>      # e.g. alphaforge: zoo_factors.json, predictions.npy, weights.npy, alphaforge_results.json
```

> Note: AlphaForge's output root was moved from `results/` to `experiments/` so all 9 baselines (and MASE) share
> the same `experiments/{param_dir}/` tree — results are directly comparable without manual regrouping.

## Troubleshooting

### Issue: "Module not found"

Make sure you're in the project root directory and have installed all dependencies.

### Issue: "API key not found"

Set your API keys as environment variables:
```bash
export OPENAI_API_KEY="your_key"
export DEEPSEEK_API_KEY="your_key"
```

### Issue: "Out of memory"

Reduce the number of stocks or factors in `config.yaml`.

### Issue: "No real data source available"

The pipeline falls back to synthetic data automatically. To use real data:
1. **westock**: Available inside WorkBuddy sessions — no setup needed
2. **AkShare**: `pip install akshare`
3. **Tushare**: Register at tushare.pro, set `TUSHARE_TOKEN` env var

### Issue: "UnicodeDecodeError: 'gbk' codec can't decode"

All source files now include `# -*- coding: utf-8 -*-` declarations and all `open()` calls specify `encoding='utf-8'`. This error should no longer occur.

### Issue: "TypeError: Object of type XXX is not JSON serializable"

Evolution history uses `dataclasses.asdict()` for serialization. All dataclass objects are properly handled. NaN/Inf values are sanitized to `null` before JSON serialization.

### Issue: "ValueError: Parse error in expression ..."

Factor expressions with syntax errors are caught during evaluation, marked `is_valid=False`, and automatically excluded from ranking and all JSON output. Check `parse_error` on the factor for the specific syntax issue.

### Issue: "Error: --factor-path not specified and no final_factors.json found"

Run `python main.py --full` first to generate a `final_factors.json`, then use `--test` mode. Or specify the path explicitly with `--factor-path`.

## Citation

If you use this code, please cite our paper (once published):

```bibtex
@inproceedings{aaai2027_llm_factor,
  title={LLM-Driven Self-Evolving Multi-Factor Stock Selection with State-Aware Memory},
  author={Your Name},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2027}
}
```

## Contact

For questions, please contact: your.email@example.com

## License

MIT License
