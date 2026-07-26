# Quick Start Guide

## Prerequisites

Before starting, ensure you have:
- Python 3.8+ installed
- OpenAI API key with GPT-4 access
- At least 16GB RAM
- 10GB free disk space

## 1. Installation (5 minutes)

### Clone and Install
```bash
# Clone the repository
git clone https://github.com/dtbtc/mcts-llm-alpha.git
cd mcts-llm-alpha

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
cd mcts-llm-alpha
pip install -e ".[dev]"
```

### Set Up Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-actual-key-here
```

## 2. Download Market Data (10 minutes)

```bash
# Install Qlib
pip install qlib

# Download CSI300 data (Chinese market)
python -m qlib.run.get_data qlib_data --target_dir ./qlib_data --region cn

# Set data path in .env
# QLIB_PROVIDER_URI=/absolute/path/to/qlib_data
```

## 3. Your First Alpha Search (5 minutes)

### Basic Search
```bash
# Run a quick 10-iteration search
python run_search.py --iterations 10
```

### What You'll See
```
============================================================
MCTS-LLM Alpha挖掘系统
============================================================

[1] 加载配置...
  - 最大迭代次数: 10
  - 评估模式: Qlib真实评估 + 相对排名
  - LLM模型: deepseek-v4-flash

[2] 初始化数据提供者...
  - Qlib初始化成功

[3] 初始化LLM客户端...
  - 找到API密钥，使用真实LLM

[4] 创建MCTS搜索实例...
  - MCTS搜索实例创建完成

[5] 开始MCTS搜索...

【第1步】LLM生成Alpha画像...
生成的Alpha画像:
----------------------------------------
### Alpha Factor Portrait

**Alpha Name:** Momentum Volatility Adjusted

**Description:** Captures price momentum normalized by recent volatility

**Formula Logic:**
```
momentum = (close - close[20]) / close[20]
volatility = std(returns, 20)
signal = momentum / volatility
alpha = rank(signal)
```
----------------------------------------

【第2步】将画像转换为符号公式...
符号公式: Rank((($close - Ref($close, w1)) / Ref($close, w1)) / Std(Delta($close, 1), w2), w3)

【第3步】评估候选参数组...
评估参数组1:
  参数: {'w1': 20, 'w2': 20, 'w3': 10}
  评分详情:
    - Effectiveness: 6.82
    - Stability: 7.45
    - Turnover: 8.23
    - Diversity: 9.50
    - Overfitting: 5.00
  平均分: 7.40
```

## 4. Understanding the Output

### Key Metrics
- **Effectiveness (有效性)**: Signal strength (IC, ICIR)
- **Stability (稳定性)**: Consistency over time
- **Turnover (换手率)**: Trading frequency (higher score = lower turnover)
- **Diversity (多样性)**: Uniqueness from existing factors
- **Overfitting (过拟合)**: Generalization ability

### Success Criteria
- Overall score > 5.0
- Effectiveness > 3.0
- Diversity > 2.0

## 5. Common Use Cases

### A. Exploratory Search
```bash
# High exploration, quick iterations
python run_search.py --iterations 20 --config config/exploration.yaml
```

### B. Production Search
```bash
# Thorough search with checkpointing
python run_search.py --iterations 100 --checkpoint-freq 10
```

### C. Resume from Checkpoint
```bash
# Continue a previous search
python -m mcts_llm_alpha resume --checkpoint checkpoints/mcts_checkpoint_20240101_120000.pkl
```

### D. With Seed Formula
```bash
# Start with a known good formula
python run_search.py --seed-formula "Rank(($close - Mean($close, 20)) / Std($close, 20), 10)"
```

## 6. Monitoring Progress

### Log Files
```bash
# Real-time log monitoring
tail -f logs/mcts_llm_alpha_latest.log

# Search for successful discoveries
grep "入库Alpha" logs/mcts_llm_alpha_latest.log
```

### Checkpoints
```bash
# List checkpoints
ls -la checkpoints/

# Analyze checkpoint
python -m mcts_llm_alpha analyze --checkpoint checkpoints/latest.pkl
```

## 7. Quick Tips

### Memory Issues?
```yaml
# Reduce batch size in config
evaluation:
  batch_size: 500  # From 1000
  cache_size: 5000  # From 10000
```

### Slow Evaluation?
```yaml
# Increase parallelism
evaluation:
  n_jobs: -1  # Use all CPU cores
```

### Want Better Quality?
```yaml
# Use better model and stricter thresholds
llm:
  model: "gpt-4"
  
mcts:
  effectiveness_threshold: 4.0
  diversity_threshold: 3.0
```

## 8. Next Steps

1. **Read the Docs**
   - [Configuration Guide](configuration.md) - Customize your search
   - [Evaluation System](evaluation_system.md) - Understand scoring
   - [Architecture](architecture.md) - Deep dive into design

2. **Experiment**
   - Try different seed formulas
   - Adjust exploration parameters
   - Create custom configurations

3. **Analyze Results**
   - Review discovered factors in `results/`
   - Backtest top factors in your trading system
   - Compare with existing factors

## 9. Troubleshooting

### "No module named 'qlib'"
```bash
pip install qlib
```

### "OPENAI_API_KEY not found"
```bash
# Linux/Mac
export OPENAI_API_KEY=sk-your-key

# Windows
set OPENAI_API_KEY=sk-your-key
```

### "Qlib data not found"
```bash
# Download data first
python -m qlib.run.get_data qlib_data --target_dir ./qlib_data --region cn

# Update .env
QLIB_PROVIDER_URI=/absolute/path/to/qlib_data
```

### Getting Help
- Check [Common Issues](common_issues.md)
- Open an [issue on GitHub](https://github.com/dtbtc/mcts-llm-alpha/issues)
- Review the [FAQ](faq.md)

## 10. Example Results

After a successful run, you'll see:
```
搜索完成！
============================================================

📊 初始公式信息:
公式: Rank((($close - Ref($close, 20)) / Std($close, 20)), 10)
整体分数: 6.54

🎯 最佳发现公式:
公式: Rank((Mean($volume, 5) / Mean($volume, 20)) * Corr($close, $volume, 15), 10)
最佳分数: 7.842

📚 Alpha仓库统计:
入库因子数: 12
仓库平均分: 6.73
```

Congratulations! You've discovered your first alpha factors! 🎉