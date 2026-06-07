# AAAI 2027 LLM Multi-Factor Stock Selection Project

## Overview

This is the complete codebase for the AAAI 2027 paper:  
**"LLM-Driven Self-Evolving Multi-Factor Stock Selection with State-Aware Memory"**

## Project Structure

```
AAAI2027_LLM_MultiFactor/
├── config/                 # Configuration files
│   └── config.yaml        # Main configuration
├── data/                  # Data loading and preprocessing
│   └── loader.py          # Data loader module
├── methods/               # Core methods (4 modules)
│   ├── debate.py         # Multi-agent debate evaluator
│   ├── evolve.py         # Self-evolving factor generator
│   ├── memory.py         # Factor memory bank
│   └── fusion.py         # Factor fusion and portfolio construction
├── backtest/              # Backtest engine
│   └── engine.py         # Backtest simulation
├── metrics/               # Evaluation metrics
│   └── evaluator.py      # Comprehensive metrics
├── viz/                   # Visualization tools
│   └── plotter.py        # Plotting functions
├── experiments/           # Experiment results
│   └── results/          # Output directory
├── paper/                 # Paper-related files
│   └── figures/          # Generated figures
├── main.py                # Main pipeline
├── run_experiments.py     # Experiment runner
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

For full functionality, you need API access to:
- OpenAI GPT-4o (for factor evaluation)
- DeepSeek (for factor generation)
- Sentence Transformers (for factor embedding)

## Usage

### Quick Demo (No LLM Required)

Run a quick demo without LLM calls:

```bash
python main.py
```

This will run the pipeline with sample data and random portfolios.

### Full Pipeline

Run the complete end-to-end pipeline:

```bash
python main.py --full
```

This will:
1. Load data
2. Generate factors using LLM
3. Evaluate factors (multi-agent debate)
4. Evolve factors (self-improving)
5. Retrieve from memory (state-aware)
6. Fuse factors (ICIR-weighted)
7. Construct portfolio
8. Backtest and evaluate

### Run Experiments

Run all experiments for the paper:

```bash
python run_experiments.py
```

This will run:
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

Each agent evaluates factors independently, then engages in structured debate.

### 2. Self-Evolving Factor Generator (`methods/evolve.py`)

Generates factors through iterative evolution:
- Generate seed factors (LLM)
- Backtest and evaluate
- Reflect on failures
- Generate improvements
- Repeat until convergence

### 3. Factor Memory Bank (`methods/memory.py`)

Stores and retrieves historical high-quality factors:
- Encode market state (4 dimensions)
- Embed factors (Sentence-BERT)
- Retrieve similar factors (state-aware)
- Update quality scores (online learning)

### 4. Factor Fusion & Portfolio Construction (`methods/fusion.py`)

Fuses multiple factors into a composite score:
- Normalize factors (Z-score/Rank)
- Weight by ICIR (dynamic)
- Penalize correlations
- Construct portfolio (Top-N)
- Apply risk constraints

## Configuration

Edit `config/config.yaml` to customize:
- Data source and universe
- LLM models and parameters
- Evolution hyperparameters
- Memory bank settings
- Fusion strategy
- Backtest parameters

## Results

All results are saved to `experiments/results/`:
- `performance_metrics.csv`: Main results
- `ablation_studies.json`: Ablation study results
- `baseline_comparisons.json`: Baseline comparison results
- `evolution_history.json`: Evolution history

## Paper Writing

The paper is structured as follows:
1. **Introduction**: Motivation and contributions
2. **Related Work**: LLM for finance, factor selection
3. **Methodology**: 4 core modules
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
