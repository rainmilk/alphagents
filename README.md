# AAAI 2027 LLM Multi-Factor Stock Selection Project

## Overview

This is the complete codebase for the AAAI 2027 paper:  
**"LLM-Driven Self-Evolving Multi-Factor Stock Selection with State-Aware Memory"**  
(MASE: Memory-Augmented Self-Evolving Framework)

## Project Structure

```
AAAI2027_LLM_MultiFactor/
├── config/                 # Configuration files
│   └── config.yaml        # Main configuration
├── data/                  # Data loading and output
│   ├── loader.py          # Data loader (sample / real A-share data)
│   ├── raw/               # Raw data (westock/AkShare/Tushare/Qlib kline CSVs)
│   ├── train/             # Train split CSVs (price/fundamental/industry)
│   ├── test/              # Test split CSVs (price/fundamental/industry)
│   └── memory_bank.json/  # Factor memory bank snapshots
├── methods/               # Core methods (4 modules)
│   ├── debate.py         # Multi-agent debate evaluator
│   ├── evolve.py         # Self-evolving factor generator
│   ├── memory.py         # State-aware factor memory bank (FAISS)
│   └── fusion.py         # ICIR-weighted factor fusion + portfolio construction
├── backtest/              # Backtest engine
│   └── engine.py         # Backtest simulation
├── metrics/               # Evaluation metrics
│   └── evaluator.py      # Comprehensive metrics
├── viz/                   # Visualization tools
│   └── plotter.py        # Plotting functions (standalone runnable)
├── experiments/           # Experiment results
│   ├── results/          # Fixed output directory (portfolios, baselines, etc.)
│   └── {yyyymmdd}/       # Date-stamped run outputs
│       ├── self_evolve/  # LLM-generated factors per round
│       │   ├── round_0/  # Seed factors (generated_factors.json + backtest csv)
│       │   └── round_N/  # Evolved factors (improved_factors.json + backtest csv)
│       ├── debate/       # Debate expert opinions (debate_factors_result.json)
│       └── fusion/       # Final fused factors (final_factors.json)
├── paper/                 # Paper-related files
│   └── figures/          # Generated figures
├── main.py                # Main pipeline (argparse CLI)
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

### Data Source Selection

The pipeline supports two data modes controlled by a single switch:

| Mode | Flag | Data | Use Case |
|---|---|---|---|
| **Sample** (default) | `--sample` or no flag | Synthetic random walk data (100 stocks × 500 days) | Fast testing, CI, no external API needed |
| **Real** | `--real` | A-share data via westock → AkShare → Tushare → Qlib → synthetic fallback | Production, paper experiments |

**Real data source chain** (automatic fallback):
1. **westock** — WorkBuddy built-in A-share data (preferred, no extra setup)
2. **AkShare** — Open-source Python library (`pip install akshare`)
3. **Tushare** — Requires token (`pip install tushare`, set `TUSHARE_TOKEN`)
4. **Qlib** — High-performance `.bin` format (`pip install qlib`, auto-downloads cn_data)
5. **Synthetic** — Final fallback (same shape, random data)

Real data is cached to `data/cache_*.pkl` after first load. Use `--force-refresh` to skip cache.

### Quick Demo (No LLM Required)

Run a quick demo without LLM calls:

```bash
python main.py
```

This runs the pipeline with sample data and random portfolios.

### Full Pipeline (Sample Data)

Run the complete end-to-end pipeline with synthetic data:

```bash
python main.py --full
```

### Full Pipeline (Real A-Share Data)

```bash
# Auto-detect best available source
python main.py --full --real --n-seeds=1 --n-best-factors=2

# Specify data source
python main.py --full --real --source qlib
python main.py --full --real --source akshare

# Custom date range + universe
python main.py --full --real --start 2023-01-01 --end 2024-12-31 --universe hs300

# Skip cache, force re-download
python main.py --full --real --force-refresh
```

### Pipeline Steps (--full mode)

1. **Load data** — sample or real A-share data with preprocessing; saves train/test splits as CSVs to `data/train/` and `data/test/`
2. **Initialize memory** — FAISS-backed factor memory bank
3. **Generate factors** — LLM-driven seed factor generation
4. **Evolve factors** — Self-improving iterative evolution (backtester coarse filter)
5. **Evaluate factors** — Multi-agent debate with 5 experts (final quality gate)
5c. **Chair synthesis** — Chair Agent cross-factor ranking with selection/rejection reasons (chair_synthesis.json)
6. **Retrieve from memory** — State-aware top-k factor retrieval
7. **Fuse factors** — ICIR-weighted fusion with correlation penalty
8. **Construct portfolio** — Top-N portfolio with risk constraints
9. **Backtest & save** — Performance metrics + evolution history

### CLI Reference

```
python main.py [OPTIONS]

Options:
  --full                   Run full end-to-end pipeline (default: quick demo)
  --real                   Use real A-share data (default: sample data)
  --sample                 Use sample/synthetic data (default)
  --source {auto,westock,akshare,tushare,qlib}
                           Real data source (default: auto)
  --force-refresh          Skip cache, re-download real data
  --start DATE             Start date for real data (default: 2022-01-01)
  --end DATE               End date for real data (default: 2024-12-31)
  --universe {hs300,zz500,all_a}
                           Stock universe for real data (default: hs300)
  --n-seeds N             Override llm.generator.n_seeds from config
  --n-evolution-rounds N   Override evolution.max_rounds from config (default: 5)
  --n-best-factors N       Override evolution.n_best_factors from config (controls n_best_factors in SelfEvolvingGenerator)
  --forward-period N       Forward return horizon in trading days (None → config or 20)
  --config PATH            Path to config file (default: config/config.yaml)
  --output-dir PATH        Output directory (default: experiments/YYYYMMDD/results/)
```

### Run Experiments

Run all experiments for the paper:

```bash
python run_experiments.py
```

This runs:
- Main experiment (5 runs)
- Ablation studies (remove each component)
- Baseline comparisons
- Robustness tests

### Generate Figures

Generate all figures for the paper:

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

Each agent evaluates factors independently (Phase 1), then engages in structured debate rounds (Phase 2) with full reproducibility. All expert opinions — scores, reasoning, concerns, strengths, and per-round consensus — are saved to `experiments/{yyyymmdd}/debate/debate_factors_result.json` for post-hoc analysis.

### 2. Self-Evolving Factor Generator (`methods/evolve.py`)

Generates factors through iterative evolution:
- Generate seed factors (LLM or rule-based fallback) — count controlled by `evolution.n_seed_factors`
- Backtest and evaluate via `FactorBacktester`
- Select top-k factors per round — controlled by `evolution.n_best_factors` (maps to `SelfEvolvingGenerator.n_best_factors`)
- Reflect on failures
- Generate improvements
- Repeat until convergence (max rounds: `evolution.max_rounds`)
- Returns `EvolutionResult` (dataclass with `EvolutionRound` history)

### 3. State-Aware Factor Memory Bank (`methods/memory.py`)

Stores and retrieves historical high-quality factors:
- Encode market state (4 dimensions: vix, trend, dispersion, turnover)
- Embed factors (Sentence-BERT)
- Retrieve similar factors (state-aware, FAISS `IndexFlatIP`)
- Update quality scores (online learning)

### 4. Factor Fusion & Portfolio Construction (`methods/fusion.py`)

Fuses multiple factors into a composite score:
- Normalize factors (Z-score/Rank)
- Weight by ICIR (dynamic)
- Penalize correlations
- Construct portfolio via `PortfolioConstructor` (Top-N, returns `list[Portfolio]` dataclass)
- Apply risk constraints

## Configuration

Edit `config/config.yaml` to customize:
- Data source and universe
- LLM models and parameters
- Evolution hyperparameters (`n_seed_factors`, `top_k`, `max_rounds`, etc.)
- Memory bank settings
- Fusion strategy
- Backtest parameters
- Experiment settings (cross-validation, baselines, metrics)

Key evolution config keys:
- `evolution.forward_period`: Forward return horizon in trading days (default: 20)
- `evolution.n_seed_factors`: Number of seed factors to generate
- `evolution.n_best_factors`: Number of top factors to select per round (passed as `n_best_factors` to `SelfEvolvingGenerator`)
- `evolution.max_rounds`: Maximum evolution rounds before convergence

## Results

All results are saved under `experiments/`:

**`experiments/results/`** — fixed outputs:
- `performance_metrics.csv`: Main results
- `portfolios.csv`: Portfolio weights
- `evolution_history.json`: Evolution round history (serialized from dataclass)
- `ablation_studies.json`: Ablation study results
- `baseline_comparisons.json`: Baseline comparison results

**`experiments/{yyyymmdd}/`** — date-stamped per-run outputs:
- `self_evolve/round_0/generated_factors.json`: Seed factors (LLM Phase 1)
- `self_evolve/round_1/generated_factors.json` … `round_N/`: Evolved factors per round
- `debate/debate_factors_result.json`: All expert opinions (scores, reasoning, concerns, strengths) across independent evaluation and debate rounds, plus Chair Agent cross-factor synthesis record
- `fusion/final_factors.json`: Final fused factors after ICIR-weighted fusion

Train/test data persists as CSVs under `data/train/` and `data/test/` with `data/split_info.json` metadata.

## Paper Writing

The paper is structured as follows:
1. **Introduction**: Motivation and contributions
2. **Related Work**: LLM for finance, factor selection
3. **Methodology**: 4 core modules (MASE framework)
4. **Experiments**: Main results, ablation, baselines
5. **Conclusion**: Summary and future work

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
4. **Qlib**: `pip install qlib` — auto-downloads cn_data bundle (~1 GB on first use)

### Issue: "UnicodeDecodeError: 'gbk' codec can't decode"

All source files now include `# -*- coding: utf-8 -*-` declarations and all `open()` calls specify `encoding='utf-8'`. This error should no longer occur.

### Issue: "TypeError: Object of type XXX is not JSON serializable"

Evolution history uses `dataclasses.asdict()` for serialization. All dataclass objects (`EvolutionRound`, `CandidateFactor`, `Portfolio`) are now properly handled.

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
