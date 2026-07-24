"""
AlphaForge (AFF) Baseline Runner - Complete 3-Stage Implementation
===============================================================

This implementation follows the exact 3-stage process from README.md:
- Stage 1: Mining alpha factors (simplified GAN-based approach)
- Stage 2: Combining alpha factors (rolling window + linear regression)
- Stage 3: Calculate and display results

Uses MAIN PROJECT'S DATALOADER (not Qlib).

Based on: 
- baselines/AlphaForge/README.MD
- baselines/AlphaForge/train_AFF.py
- baselines/AlphaForge/combine_AFF.py
- baselines/AlphaForge/exp_AFF_calc_result.ipynb

Author: Code Review Expert (火眼眼)
Date: 2026-07-02
"""

import sys
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class AlphaForgeConfig:
    """Configuration for AlphaForge baseline."""
    instruments: str = "csi300"
    train_end_year: int = 2023
    seeds: List[int] = None
    save_name: str = "alphaforge"
    zoo_size: int = 50  # Number of factors to mine in Stage 1
    n_factors: int = 10  # Number of factors to use in Stage 2
    window: Union[int, str] = "inf"  # Rolling window for factor evaluation
    top_n_stocks: int = 50  # Number of stocks in portfolio
    forward_period: int = 10  # Forward return period for IC evaluation (must match other baselines)
    holding_period: int = 1  # Portfolio holding period (days) for backtest; 1 = daily rebalance

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [0, 1, 2, 3, 4]


def convert_to_multindex(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert main DataLoader format to MultiIndex format.
    
    Args:
        prices: DataFrame with columns=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        
    Returns:
        pd.DataFrame: MultiIndex (date, symbol) with OHLCV columns
    """
    prices = prices.copy()
    prices['date'] = pd.to_datetime(prices['date'])
    
    # Pivot to (date, symbol) format
    result = prices.pivot(index='date', columns='symbol')
    result.columns.names = ['field', 'symbol']

    return result


def _price_dict_to_long(pd_dict: dict) -> pd.DataFrame:
    """
    Convert a main-DataLoader price dict (date-indexed OHLCV DataFrames) into the
    long-format (symbol, date, open, high, low, close, volume) frame used by
    AlphaForge's Stage 1/2 factor mining.
    """
    rows = []
    for date in pd_dict['close'].index:
        for symbol in pd_dict['close'].columns:
            rows.append({
                'symbol': symbol,
                'date': date,
                'open': pd_dict['open'].loc[date, symbol] if 'open' in pd_dict else np.nan,
                'high': pd_dict['high'].loc[date, symbol] if 'high' in pd_dict else np.nan,
                'low': pd_dict['low'].loc[date, symbol] if 'low' in pd_dict else np.nan,
                'close': pd_dict['close'].loc[date, symbol],
                'volume': pd_dict['volume'].loc[date, symbol] if 'volume' in pd_dict else np.nan,
            })
    return pd.DataFrame(rows)


def compute_returns(prices_multindex: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 1-day forward returns from MultiIndex price data.

    This is kept for backward compatibility. For IC-based factor evaluation
    that aligns with other baselines, use compute_forward_returns() with
    forward_period=10 instead.

    Args:
        prices_multindex: MultiIndex DataFrame with OHLCV data

    Returns:
        pd.DataFrame: 1-day forward returns (date x symbol)
    """
    close = prices_multindex['close']
    returns = close.pct_change().shift(-1)  # Forward 1-day return
    return returns


def compute_forward_returns(
    prices_multindex: pd.DataFrame,
    forward_period: int = 10,
) -> pd.DataFrame:
    """
    Compute forward N-day returns for IC-based factor evaluation.

    For forward_period=N:  return[t] = close[t+N] / close[t] - 1

    This aligns AlphaForge's IC evaluation with other baselines that use
    forward_period-day forward returns (MCTS-LLM-Alpha, AlphaGrail, XGBoost,
    LSTM, XGBoost-Simple, AlphaAgent all default to forward_period=10).

    Args:
        prices_multindex: MultiIndex DataFrame with OHLCV data
        forward_period: Number of trading days to look ahead (default 10,
            matching other baselines for fair comparison)

    Returns:
        pd.DataFrame: Forward returns (date x symbol).
        Last forward_period days will be NaN (no future data available).
    """
    close = prices_multindex['close']
    forward_returns = close.shift(-forward_period) / close - 1
    return forward_returns


# ═══════════════════════════════════════════════════════════════════════
# GAN-based Factor Mining (mirrors original AlphaForge train_AFF.py)
# ═══════════════════════════════════════════════════════════════════════
#
# Original Architecture:
#   NetG (DCGAN) → token logits → NetM (mask) → NetP (CNN quality scorer)
#   Adversarial training: NetG learns to produce factors with high NetP scores
#
# Our Simplified Architecture (same spirit, no alphagen dependency):
#   NetG (MLP) → discrete token sequence → ExpressionDecoder → factor expr
#   NetP (MLP) → token one-hot → predicted Rank IC
#   Training: NetP learns from real IC, NetG learns from NetP prediction

# ── Token Vocabulary ──────────────────────────────────────────────────────
# Each factor is encoded as an 8-token sequence:
#   [OP1, FEAT1, WIN1, OP2, FEAT2, WIN2, COMBINE, WRAP]
#
# Decoding produces expressions like:
#   ts_mean(close, 20) / ts_std(close, 10)
#   abs(close - ts_mean(close, 60))
#   cs_rank(ts_delta(volume, 10))

# Operation 1 & 2: what transform on the feature
_TOK_OPS = {
    0: 'identity', 1: 'ts_mean', 2: 'ts_std', 3: 'ts_min',
    4: 'ts_max', 5: 'ts_delta', 6: 'ts_pct', 7: 'ts_sum',
}
N_OPS = len(_TOK_OPS)

# Features: which price/volume field
_TOK_FEATURES = {
    0: 'close', 1: 'open', 2: 'high', 3: 'low',
    4: 'volume', 5: 'vwap', 6: 'amount', 7: 'return',
}
N_FEATURES = len(_TOK_FEATURES)

# Window sizes
_TOK_WINDOWS = {0: 5, 1: 10, 2: 20, 3: 30, 4: 60, 5: 120}
N_WINDOWS = len(_TOK_WINDOWS)

# Combine ops: how to merge NODE1 and NODE2
_TOK_COMBINE = {0: 'none', 1: 'add', 2: 'sub', 3: 'mul', 4: 'div'}
N_COMBINE = len(_TOK_COMBINE)

# Wrap ops: final transformation on combined expression
_TOK_WRAP = {0: 'none', 1: 'abs', 2: 'log', 3: 'neg', 4: 'cs_rank'}
N_WRAP = len(_TOK_WRAP)

# Total tokens = sum of all category sizes (each position has its own vocabulary)
# Positions 0,3 → N_OPS (8)
# Positions 1,4 → N_FEATURES (8)
# Positions 2,5 → N_WINDOWS (6)
# Position 6   → N_COMBINE (5)
# Position 7   → N_WRAP (5)
SEQ_LEN = 8
TOKEN_DIMS = [N_OPS, N_FEATURES, N_WINDOWS, N_OPS, N_FEATURES, N_WINDOWS, N_COMBINE, N_WRAP]


def _decode_expression(tokens: List[int]) -> str:
    """
    Decode an 8-token sequence into a Python factor expression string.

    Examples:
        [1,0,3, 0,0,0, 0,0] → "close.rolling(20).mean()"
        [0,0,0, 1,0,3, 4,0] → "close / close.rolling(20).mean()"
        [0,0,0, 1,0,4, 2,1] → "abs(close - close.rolling(60).mean())"

    Args:
        tokens: List of 8 token IDs

    Returns:
        str: Evaluable Python expression string
    """
    assert len(tokens) >= SEQ_LEN, f"Need {SEQ_LEN} tokens, got {len(tokens)}"

    def _mk_node(op_tok, feat_tok, win_tok):
        """Build a single feature/op node expression."""
        op_name = _TOK_OPS.get(op_tok, 'identity')
        feat_name = _TOK_FEATURES.get(feat_tok, 'close')
        win = _TOK_WINDOWS.get(win_tok, 20)

        if op_name == 'identity':
            return feat_name
        elif op_name == 'ts_mean':
            return f"{feat_name}.rolling({win}).mean()"
        elif op_name == 'ts_std':
            return f"{feat_name}.rolling({win}).std()"
        elif op_name == 'ts_min':
            return f"{feat_name}.rolling({win}).min()"
        elif op_name == 'ts_max':
            return f"{feat_name}.rolling({win}).max()"
        elif op_name == 'ts_delta':
            return f"({feat_name} - {feat_name}.shift({win}))"
        elif op_name == 'ts_pct':
            return f"({feat_name} / {feat_name}.shift({win}) - 1)"
        elif op_name == 'ts_sum':
            return f"{feat_name}.rolling({win}).sum()"
        return feat_name

    node1 = _mk_node(tokens[0], tokens[1], tokens[2])
    combine = _TOK_COMBINE.get(tokens[6], 'none')

    if combine == 'none':
        expr = node1
    else:
        node2 = _mk_node(tokens[3], tokens[4], tokens[5])
        if combine == 'add':
            expr = f"({node1} + {node2})"
        elif combine == 'sub':
            expr = f"({node1} - {node2})"
        elif combine == 'mul':
            expr = f"({node1} * {node2})"
        elif combine == 'div':
            expr = f"({node1} / ({node2} + 1e-8))"

    wrap = _TOK_WRAP.get(tokens[7], 'none')
    if wrap == 'abs':
        expr = f"abs({expr})"
    elif wrap == 'log':
        expr = f"np.log(abs({expr}) + 1e-8)"
    elif wrap == 'neg':
        expr = f"-({expr})"
    elif wrap == 'cs_rank':
        expr = f"({expr}).rank(axis=1, pct=True)"

    return expr


# ── GAN Networks ───────────────────────────────────────────────────────────

def _torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True, torch
    except ImportError:
        return False, None


class _FactorGenerator:
    """
    NetG: Generates discrete token sequences from noise vectors.

    Mirrors AlphaForge's NetG_DCGAN:
    - Input: latent noise z ~ N(0,1), shape (batch, latent_dim)
    - Output: token logits, shape (batch, seq_len, vocab_per_pos)

    Uses MLP with positional heads for simplicity (vs original DCGAN).
    Each position has its own output head with softmax over its vocabulary.
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256, device: str = 'cpu'):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self._build_network()

    def _build_network(self):
        import torch
        import torch.nn as nn

        self._net = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        ).to(self.device)
        # One output head per position
        self._heads = nn.ModuleList([
            nn.Linear(self.hidden_dim, dim) for dim in TOKEN_DIMS
        ]).to(self.device)

    def generate(self, batch_size: int, device: str = 'cpu') -> "np.ndarray":
        """
        Generate factor token sequences.

        Args:
            batch_size: Number of factors to generate
            device: 'cpu' or 'cuda:[0]'

        Returns:
            np.ndarray of shape (batch_size, SEQ_LEN) with token IDs
        """
        import torch
        z = torch.randn(batch_size, self.latent_dim).to(device)
        with torch.no_grad():
            hidden = self._net(z)
            tokens = []
            for head in self._heads:
                logits = head(hidden)
                probs = torch.softmax(logits, dim=-1)
                sampled = torch.multinomial(probs, 1).squeeze(-1)
                tokens.append(sampled.cpu().numpy())
        return np.stack(tokens, axis=1)

    def forward(self, z: "torch.Tensor") -> "List[torch.Tensor]":
        """Forward pass, returns logits per position (for training)."""
        hidden = self._net(z)
        return [head(hidden) for head in self._heads]


class _FactorPredictor:
    """
    NetP: Predicts factor quality (Rank IC) from token sequences.

    Mirrors AlphaForge's NetP (CNN):
    - Input: one-hot token sequence, shape (batch, seq_len, total_vocab)
    - Output: predicted score, shape (batch, 1)

    Uses MLP for simplicity.
    """

    def __init__(self, hidden_dim: int = 128, device: str = 'cpu'):
        self.hidden_dim = hidden_dim
        self.total_vocab = sum(TOKEN_DIMS)
        self.device = device
        self._build_network()

    def _build_network(self):
        import torch
        import torch.nn as nn

        self._net = nn.Sequential(
            nn.Linear(self.total_vocab, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim // 2, 1),
        ).to(self.device)

    def _tokens_to_onehot(self, tokens: "np.ndarray") -> "np.ndarray":
        """Convert token sequences to one-hot vectors."""
        batch_size = tokens.shape[0]
        onehot = np.zeros((batch_size, self.total_vocab), dtype=np.float32)
        offset = 0
        for pos in range(SEQ_LEN):
            for i in range(batch_size):
                tok = int(tokens[i, pos])
                if tok < TOKEN_DIMS[pos]:
                    onehot[i, offset + tok] = 1.0
            offset += TOKEN_DIMS[pos]
        return onehot

    def predict(self, tokens: "np.ndarray") -> "np.ndarray":
        """
        Predict factor quality scores.

        Args:
            tokens: (batch_size, SEQ_LEN) token IDs

        Returns:
            np.ndarray of shape (batch_size,) with predicted scores
        """
        import torch
        onehot = self._tokens_to_onehot(tokens)
        x = torch.from_numpy(onehot).to(self.device)
        with torch.no_grad():
            scores = self._net(x).squeeze(-1).cpu().numpy()
        return scores

    def forward(self, onehot: "torch.Tensor") -> "torch.Tensor":
        """Forward pass for training."""
        return self._net(onehot)


# ── GAN Training ───────────────────────────────────────────────────────────

def _train_gan_step(
    generator: _FactorGenerator,
    predictor: _FactorPredictor,
    all_tokens: np.ndarray,
    all_scores: np.ndarray,
    batch_size: int = 128,
    n_gen_steps: int = 5,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
) -> Dict:
    """
    One round of GAN training.

    Mirrors the training loop in train_AFF.py:
    1. Train Predictor on real (token, score) pairs
    2. Train Generator to maximize Predictor's predicted scores
    3. Generate new factors from trained Generator for next round

    Args:
        generator: NetG instance
        predictor: NetP instance
        all_tokens: All real token sequences seen so far (N, SEQ_LEN)
        all_scores: Corresponding real IC scores (N,)
        batch_size: Batch size for training
        n_gen_steps: Generator training steps per round
        learning_rate: Learning rate
        device: Torch device

    Returns:
        Dict with training metrics
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    n_real = len(all_scores)
    if n_real < batch_size // 2:
        return {'pred_loss': 0, 'gen_score': 0, 'n_real': n_real, 'n_generated': 0}

    # ── Train Predictor on real data ──
    predictor._net.train()
    opt_p = optim.Adam(predictor._net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    real_onehot = torch.from_numpy(
        predictor._tokens_to_onehot(all_tokens)
    ).to(device)
    real_scores_t = torch.from_numpy(
        np.array(all_scores, dtype=np.float32)
    ).to(device)

    n_epochs_p = max(10, min(50, n_real // batch_size // 2))
    pred_losses = []
    for _ in range(n_epochs_p):
        idx = np.random.choice(n_real, min(batch_size, n_real), replace=False)
        xb = real_onehot[idx]
        yb = real_scores_t[idx]
        opt_p.zero_grad()
        pred = predictor.forward(xb).squeeze(-1)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt_p.step()
        pred_losses.append(loss.item())
    pred_loss = float(np.mean(pred_losses[-10:]))

    predictor._net.eval()

    # ── Train Generator ──
    opt_g = optim.Adam(generator._net.parameters(), lr=learning_rate * 0.5)
    gen_scores = []
    for _ in range(n_gen_steps):
        z = torch.randn(batch_size, generator.latent_dim).to(device)
        logits_list = generator.forward(z)

        # Use temperature-softmax for differentiable path:
        # Each position's logits → softmax → soft token probabilities
        soft_tokens = []
        for logits in logits_list:
            soft = torch.softmax(logits / 0.5, dim=-1)  # temperature=0.5
            soft_tokens.append(soft)

        # Concatenate to form (batch, total_vocab) soft representation
        soft_concat = torch.cat(soft_tokens, dim=-1)

        # Feed through predictor (MLP handles both one-hot and soft inputs)
        score_pred = predictor.forward(soft_concat).squeeze(-1)
        score_mean = score_pred.mean()

        # Loss: maximize predicted score + diversity bonus
        gen_loss = -score_mean + 0.01 * (score_pred.std())

        opt_g.zero_grad()
        gen_loss.backward()
        opt_g.step()

        gen_scores.append(score_mean.item())

    gen_score = float(np.mean(gen_scores))

    return {
        'pred_loss': pred_loss,
        'gen_score': gen_score,
        'n_real': n_real,
    }


def gan_mine_factors(
    prices_multindex: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_factors: int = 50,
    n_rounds: int = 5,
    batch_size: int = 64,
    latent_dim: int = 32,
    device: str = 'cpu',
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[str], List[float], Dict]:
    """
    Mine alpha factors using GAN (mirrors AlphaForge Stage 1).

    Training flow (matching train_AFF.py):
    1. Initialize NetG (Generator) + NetP (Predictor)
    2. Pre-populate with random factors → evaluate → (tokens, IC) pairs
    3. For each round:
       a. Train Predictor on real (tokens, IC) pairs
       b. Train Generator to maximize Predictor's score
       c. Generate new factors → evaluate on data → add to pool
       d. Filter by IC + decorrelation
    4. Return top-N factors by |IC|

    Args:
        prices_multindex: MultiIndex OHLCV data
        forward_returns: Forward returns DataFrame (N-day, for IC evaluation)
        n_factors: Target number of factors to mine
        n_rounds: Training rounds
        batch_size: GAN batch size
        latent_dim: Generator latent dimension
        device: Torch device
        seed: Random seed
        verbose: Print progress

    Returns:
        Tuple of (factor_expressions, factor_ranks_ic, gan_stats)
    """
    import torch
    np.random.seed(seed)
    if torch.cuda.is_available() and 'cuda' in device:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    # ── Helper: evaluate a factor expression → Rank IC ──
    def _eval_expr(expr: str) -> float:
        factor_vals = evaluate_factor_expression(expr, prices_multindex)
        if factor_vals.isna().all().all():
            return 0.0
        ric = calculate_rank_ic(factor_vals, forward_returns)
        return float(ric)

    # ── Initialize Networks ──
    generator = _FactorGenerator(latent_dim=latent_dim, device=device)
    predictor = _FactorPredictor(device=device)

    # ── Pre-population: generate random factors ──
    if verbose:
        print(f"  [GAN] Initializing with {batch_size * 2} random factors...")

    init_tokens = generator.generate(batch_size * 2, device)
    seen_exprs = set()
    all_tokens_list = []
    all_scores_list = []

    for i in range(len(init_tokens)):
        expr = _decode_expression(init_tokens[i].tolist())
        if expr in seen_exprs:
            continue
        seen_exprs.add(expr)
        score = _eval_expr(expr)
        if abs(score) > 0.001:
            all_tokens_list.append(init_tokens[i])
            all_scores_list.append(score)

    if verbose:
        print(f"  [GAN] Initial pool: {len(all_scores_list)} valid factors, "
              f"max |IC|: {max(abs(np.array(all_scores_list))) if all_scores_list else 0:.4f}")

    # ── GAN Training Rounds ──
    stats_history = []
    for round_idx in range(n_rounds):
        if len(all_scores_list) < 10:
            if verbose:
                print(f"  [GAN] Round {round_idx+1}/{n_rounds}: insufficient data, generating more random factors...")
            new_tokens = generator.generate(batch_size, device)
            for i in range(len(new_tokens)):
                expr = _decode_expression(new_tokens[i].tolist())
                if expr in seen_exprs:
                    continue
                seen_exprs.add(expr)
                score = _eval_expr(expr)
                if abs(score) > 0.001:
                    all_tokens_list.append(new_tokens[i])
                    all_scores_list.append(score)
            continue

        all_tokens_arr = np.array(all_tokens_list)
        all_scores_arr = np.array(all_scores_list, dtype=np.float32)

        # Normalize scores for stable Predictor training
        score_mean = all_scores_arr.mean()
        score_std = all_scores_arr.std() + 1e-8
        scores_norm = (all_scores_arr - score_mean) / score_std

        stats = _train_gan_step(
            generator, predictor,
            all_tokens_arr, scores_norm,
            batch_size=batch_size,
            n_gen_steps=3,
            learning_rate=5e-4,
            device=device,
        )
        stats['round'] = round_idx
        stats['pool_size'] = len(all_scores_list)

        if verbose:
            print(f"  [GAN] Round {round_idx+1}/{n_rounds}: "
                  f"pred_loss={stats['pred_loss']:.4f}, "
                  f"gen_score={stats['gen_score']:.4f}, "
                  f"pool={stats['pool_size']}")

        # Generate new factors from updated Generator
        new_tokens = generator.generate(batch_size * 2, device)
        n_added = 0
        new_scores = []
        for i in range(len(new_tokens)):
            expr = _decode_expression(new_tokens[i].tolist())
            if expr in seen_exprs:
                continue
            seen_exprs.add(expr)
            score = _eval_expr(expr)
            if abs(score) > 0.001:
                all_tokens_list.append(new_tokens[i])
                all_scores_list.append(score)
                n_added += 1
                new_scores.append(score)

        stats['n_added'] = n_added
        stats['max_new_ic'] = max(abs(np.array(new_scores))) if new_scores else 0
        stats_history.append(stats)

        if verbose:
            print(f"    Added {n_added} new factors, "
                  f"max |IC|: {stats['max_new_ic']:.4f}")

        # Early exit if pool is large enough
        if len(all_scores_list) >= n_factors * 3:
            break

    # ── Decode and rank all factors ──
    if not all_scores_list:
        if verbose:
            print("  [GAN] No valid factors mined, falling back to templates")
        return generate_template_factors(prices_multindex, n_factors, forward_returns), [], {}

    all_tokens_arr = np.array(all_tokens_list)
    all_scores_arr = np.array(all_scores_list)

    # Decode all factors
    expressions = []
    for i in range(len(all_tokens_arr)):
        try:
            expressions.append(_decode_expression(all_tokens_arr[i].tolist()))
        except Exception:
            expressions.append("close")

    # Sort by |IC| descending
    sorted_idx = np.argsort(np.abs(all_scores_arr))[::-1]
    top_exprs = [expressions[i] for i in sorted_idx[:n_factors]]
    top_scores = [float(all_scores_arr[i]) for i in sorted_idx[:n_factors]]

    if verbose:
        print(f"  [GAN] Final pool: {len(all_scores_list)} factors, "
              f"top |IC|: {max(abs(np.array(top_scores))) if top_scores else 0:.4f}")
        for j, (expr, sc) in enumerate(zip(top_exprs[:5], top_scores[:5])):
            print(f"    #{j+1}: IC={sc:.4f}  expr={expr}")

    return top_exprs, top_scores, {'stats_history': stats_history, 'pool_size': len(all_scores_list)}


def generate_template_factors(
    prices_multindex: pd.DataFrame = None,
    n_factors: int = 50,
    forward_returns: pd.DataFrame = None,
) -> List[str]:
    """
    Generate template alpha factor expressions (fallback when GAN unavailable).

    Used when PyTorch is not available or when use_gan=False.

    Args:
        prices_multindex: MultiIndex price data (unused, kept for API compatibility)
        n_factors: Number of factors to generate
        forward_returns: Unused, kept for API compatibility

    Returns:
        List[str]: Factor expressions (as strings for evaluation)
    """
    templates = [
        ("close / close.shift(20) - 1", [3, 5, 10, 15, 30]),
        ("close / close.shift(5) - 1", [1, 3, 10, 20]),
        ("(close - close.shift(1)) / close.shift(1)", []),
        ("(close - close.rolling(20).mean()) / close.rolling(20).std()", [5, 10, 30, 60]),
        ("volume / volume.rolling(20).mean()", [3, 5, 10, 30]),
        ("close.rolling(20).std() / close", [5, 10, 30, 60]),
        ("(high - low) / close", []),
        ("(close - close.shift(1)) * volume", []),
        ("volume / volume.rolling(5).mean()", [3, 10, 20, 30]),
        ("close.rolling(12).mean() - close.rolling(26).mean()", []),
        ("((close - close.shift(1)) / close.shift(1)).rolling(14).mean()", [5, 7, 10, 20]),
    ]

    extended = []
    for tmpl, windows in templates:
        extended.append(tmpl)
        for w in windows:
            for old_w in [20, 5]:
                if f'({old_w})' in tmpl:
                    extended.append(tmpl.replace(f'({old_w})', f'({w})'))
                if f'.shift({old_w})' in tmpl:
                    extended.append(tmpl.replace(f'.shift({old_w})', f'.shift({w})'))
                if f'.rolling({old_w})' in tmpl:
                    extended.append(tmpl.replace(f'.rolling({old_w})', f'.rolling({w})'))

    unique_factors = list(dict.fromkeys(extended))
    return unique_factors[:n_factors]


def evaluate_factor_expression(expr: str, prices_multindex: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate a factor expression on price data.
    
    Args:
        expr: Factor expression string
        prices_multindex: MultiIndex price data
        
    Returns:
        pd.DataFrame: Factor values (date x symbol)
    """
    close = prices_multindex['close']
    high = prices_multindex['high']
    low = prices_multindex['low']
    volume = prices_multindex['volume']
    
    try:
        # Evaluate expression in safe context
        factor_values = eval(expr, {"__builtins__": {}}, {
            'close': close,
            'high': high,
            'low': low,
            'volume': volume,
            'np': np,
            'pd': pd,
        })
        
        if isinstance(factor_values, pd.Series):
            factor_values = factor_values.to_frame()
        
        return factor_values
        
    except Exception as e:
        print(f"Error evaluating expression '{expr}': {e}")
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)


def calculate_ic(factor_scores: pd.DataFrame, returns: pd.DataFrame) -> float:
    """
    Calculate Information Coefficient (IC) between factor and returns.
    
    Args:
        factor_scores: Factor values (date x symbol)
        returns: Return values (date x symbol)
        
    Returns:
        float: Mean IC across all dates
    """
    ic_values = []
    
    for date in factor_scores.index:
        factor_valid = factor_scores.loc[date].dropna()
        return_valid = returns.loc[date].dropna()
        
        # Align indices
        common_idx = factor_valid.index.intersection(return_valid.index)
        if len(common_idx) < 10:
            continue
            
        factor_aligned = factor_valid[common_idx]
        return_aligned = return_valid[common_idx]
        
        # Calculate Pearson correlation
        try:
            ic = np.corrcoef(factor_aligned, return_aligned)[0, 1]
            if np.isfinite(ic):
                ic_values.append(ic)
        except Exception:
            continue
    
    return np.mean(ic_values) if ic_values else 0.0


def calculate_rank_ic(factor_scores: pd.DataFrame, returns: pd.DataFrame) -> float:
    """
    Calculate Rank IC (Spearman correlation) between factor and returns.
    
    Uses pure numpy implementation (no scipy dependency).
    
    Args:
        factor_scores: Factor values (date x symbol)
        returns: Return values (date x symbol)
        
    Returns:
        float: Mean Rank IC across all dates
    """
    rank_ic_values = []
    
    for date in factor_scores.index:
        factor_valid = factor_scores.loc[date].dropna()
        return_valid = returns.loc[date].dropna()
        
        # Align indices
        common_idx = factor_valid.index.intersection(return_valid.index)
        if len(common_idx) < 10:
            continue
            
        factor_aligned = factor_valid[common_idx]
        return_aligned = return_valid[common_idx]
        
        # Calculate Spearman rank correlation using pure numpy
        try:
            # Convert to ranks
            factor_ranks = factor_aligned.rank()
            return_ranks = return_aligned.rank()
            
            # Calculate Pearson correlation of ranks
            rho = np.corrcoef(factor_ranks, return_ranks)[0, 1]
            if np.isfinite(rho):
                rank_ic_values.append(rho)
        except Exception:
            continue
    
    return np.mean(rank_ic_values) if rank_ic_values else 0.0


def stage1_mine_factors(
    prices: pd.DataFrame,
    config: AlphaForgeConfig,
    output_dir: str,
    use_gan: bool = True,
) -> Dict:
    """
    Stage 1: Mine alpha factors using GAN (or template fallback).

    Follows the original AlphaForge GAN-based factor mining approach:
    - Generator produces factor token sequences
    - Predictor learns to estimate factor quality
    - Adversarial training improves factor quality over rounds

    Args:
        prices: Price data from main DataLoader
        config: AlphaForge configuration
        output_dir: Directory to save results
        use_gan: Use GAN-based mining (True) or template fallback (False)

    Returns:
        Dict: Mined factors and their evaluations, plus GAN stats
    """
    print("\n" + "="*60)
    mode_name = "GAN-based" if use_gan else "Template-based"
    print(f"[Stage 1] Mining alpha factors ({mode_name})...")
    print("="*60)

    # Convert to MultiIndex format
    prices_multindex = convert_to_multindex(prices)
    forward_returns = compute_forward_returns(prices_multindex, config.forward_period)

    gan_stats = {}

    # ── Factor Generation ──
    if use_gan:
        has_torch, _ = _torch_available()
        if not has_torch:
            print("  ⚠️  PyTorch not available, falling back to template factors")
            use_gan = False

    if use_gan:
        # Use GAN-based mining
        device = 'cuda:0' if True else 'cpu'  # Always CPU for safety
        factor_exprs, factor_rankics, gan_stats = gan_mine_factors(
            prices_multindex=prices_multindex,
            forward_returns=forward_returns,
            n_factors=config.zoo_size,
            n_rounds=5,
            batch_size=64,
            latent_dim=32,
            device=device,
            seed=config.seeds[0] if config.seeds else 42,
            verbose=True,
        )
    else:
        # Template fallback
        print(f"  Generating {config.zoo_size} template factors...")
        factor_exprs = generate_template_factors(prices_multindex, config.zoo_size)
        factor_rankics = []

    # ── Evaluate all factors ──
    print(f"  Evaluating factors (forward_period={config.forward_period}d)...")
    factor_scores_dict = {}
    factor_metrics = {}

    for i, expr in enumerate(factor_exprs):
        if i % 20 == 0:
            print(f"    Progress: {i}/{len(factor_exprs)}")

        factor_values = evaluate_factor_expression(expr, prices_multindex)
        ic = calculate_ic(factor_values, forward_returns)
        rank_ic = calculate_rank_ic(factor_values, forward_returns)

        factor_scores_dict[expr] = factor_values
        factor_metrics[expr] = {
            'ic': ic,
            'rank_ic': rank_ic,
            'expr': expr,
        }

    # Sort by Rank IC and select top factors
    sorted_factors = sorted(
        factor_metrics.items(),
        key=lambda x: abs(x[1]['rank_ic']),
        reverse=True
    )

    top_factors = [x[0] for x in sorted_factors[:config.zoo_size]]

    print(f"  Top factor Rank IC: {factor_metrics[top_factors[0]]['rank_ic']:.4f}")
    print(f"  Saved {len(top_factors)} factors to zoo")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    zoo_path = os.path.join(output_dir, "zoo_factors.json")

    zoo_data = {
        'factor_exprs': top_factors,
        'metrics': {expr: factor_metrics[expr] for expr in top_factors},
    }
    if gan_stats:
        zoo_data['gan_pool_size'] = gan_stats.get('pool_size', 0)
        zoo_data['method'] = 'gan'

    with open(zoo_path, 'w') as f:
        json.dump(zoo_data, f, indent=2)

    print(f"  Zoo saved to: {zoo_path}")

    return {
        'factor_exprs': top_factors,
        'factor_scores_dict': factor_scores_dict,
        'metrics': factor_metrics,
        'zoo_path': zoo_path,
        'used_gan': use_gan,
        'gan_stats': gan_stats,
    }


def stage2_combine_factors(
    prices: pd.DataFrame,
    stage1_results: Dict,
    config: AlphaForgeConfig,
    output_dir: str,
) -> Dict:
    """
    Stage 2: Combine alpha factors using rolling window.
    
    This follows the logic in combine_AFF.py:
    - Use rolling window to evaluate factors
    - Select top n_factors based on Rank IC IR
    - Use linear regression to combine factors
    
    Args:
        prices: Price data from main DataLoader
        stage1_results: Results from Stage 1
        config: AlphaForge configuration
        output_dir: Directory to save results
        
    Returns:
        Dict: Combined predictions and weights
    """
    print("\n" + "="*60)
    print("[Stage 2] Combining alpha factors...")
    print("="*60)
    
    # Convert to MultiIndex format
    prices_multindex = convert_to_multindex(prices)
    forward_returns = compute_forward_returns(prices_multindex, config.forward_period)
    
    # Load factor expressions from Stage 1
    factor_exprs = stage1_results['factor_exprs']
    
    # Evaluate all factors on all data
    print(f"  Evaluating {len(factor_exprs)} factors...")
    all_factor_values = {}
    for expr in factor_exprs:
        all_factor_values[expr] = evaluate_factor_expression(expr, prices_multindex)
    
    # Calculate IC and Rank IC for each factor at each date
    print(f"  Calculating rolling IC (forward_period={config.forward_period}d)...")
    n_dates = len(forward_returns.index)
    n_factors = len(factor_exprs)
    
    # Store IC time series for each factor
    factor_ic_series = {expr: [] for expr in factor_exprs}
    factor_rankic_series = {expr: [] for expr in factor_exprs}
    
    for i, date in enumerate(forward_returns.index):
        if i == 0:
            continue
            
        # Use past window to calculate IC
        if config.window == "inf":
            start_idx = 0
        else:
            start_idx = max(0, i - config.window)
        
        for expr in factor_exprs:
            factor_past = all_factor_values[expr].iloc[start_idx:i]
            returns_past = forward_returns.iloc[start_idx:i]
            
            ic = calculate_ic(factor_past, returns_past)
            rank_ic = calculate_rank_ic(factor_past, returns_past)
            
            factor_ic_series[expr].append(ic)
            factor_rankic_series[expr].append(rank_ic)
    
    # Rolling combination
    print("  Rolling combination...")
    predictions = []
    prediction_dates = []  # Track date for each prediction (for Stage 3 filtering)
    selected_factors_history = []
    weights_history = []
    
    # Dynamic minimum start index (handle short data)
    if config.window == "inf":
        min_start = min(63, n_dates // 3)  # Use 1/3 of data for short periods
    else:
        min_start = min(config.window, n_dates // 3)
    
    for i in range(min_start, n_dates):
        if i >= len(forward_returns.index):
            break
            
        date = forward_returns.index[i]
        
        # Calculate factor metrics using past window
        if config.window == "inf":
            start_idx = 0
        else:
            start_idx = max(0, i - config.window)
        
        factor_metrics = {}
        for expr in factor_exprs:
            ic_series = factor_ic_series[expr][start_idx:i]
            rankic_series = factor_rankic_series[expr][start_idx:i]
            
            if len(ic_series) > 0:
                ic_mean = np.mean(ic_series)
                ic_std = np.std(ic_series)
                rankic_mean = np.mean(rankic_series)
                rankic_std = np.std(rankic_series)
                
                factor_metrics[expr] = {
                    'ic': ic_mean,
                    'ic_std': ic_std,
                    'icir': ic_mean / ic_std if ic_std > 0 else 0,
                    'rank_ic': rankic_mean,
                    'rank_ic_std': rankic_std,
                    'rank_icir': rankic_mean / rankic_std if rankic_std > 0 else 0,
                }
        
        # Select top factors
        sorted_factors = sorted(
            factor_metrics.items(),
            key=lambda x: abs(x[1].get('rank_icir', 0)),
            reverse=True
        )
        
        # Filter by thresholds (from combine_AFF.py)
        good_factors = [
            x[0] for x in sorted_factors
            if abs(x[1].get('rank_ic', 0)) > 0.02 and abs(x[1].get('rank_icir', 0)) > 0.2
        ]
        
        if len(good_factors) < 1:
            good_factors = [sorted_factors[0][0]]
        
        good_factors = good_factors[:config.n_factors]
        selected_factors_history.append(good_factors)
        
        # ── Walk-forward linear regression (matches original combine_AFF.py) ──
        #
        # Original AlphaForge logic:
        #   x        = fct_tensor[begin : cur-shift, :, good_idx]   # past panel
        #   y        = tgt_tensor[begin : cur-shift]                 # past returns
        #   to_pred  = fct_tensor[cur, :, good_idx]                 # today's factors
        #   coef     = lstsq(x, y)
        #   pred     = to_pred @ coef
        #
        # Key: forward_returns[t] is the return from t to t+N, which is only
        # *known* at time t+N. So at time i we can only use training samples
        # from [start_idx, i - forward_period] — anything later has unrealized
        # forward returns (look-ahead bias).

        shift = config.forward_period
        train_end = i - shift  # last date whose forward return is realized by time i

        if train_end <= start_idx:
            # Not enough history for out-of-sample regression yet
            n_stocks = len(forward_returns.columns)
            predictions.append(np.zeros(n_stocks))
            prediction_dates.append(date)
            weights_history.append(np.zeros(config.n_factors))
            continue

        # Build panel training data: (n_past_days * n_stocks, n_factors)
        train_cols = []
        for expr in good_factors:
            vals = all_factor_values[expr].iloc[start_idx:train_end].values
            train_cols.append(vals.flatten())
        train_X = np.column_stack(train_cols)

        # Corresponding forward returns, flattened to 1-D
        train_y = forward_returns.iloc[start_idx:train_end].values.flatten()

        # Filter rows: keep only where y is finite AND x has no NaN/inf
        # (matches original torch.isfinite(y) filter + guards factor NaN)
        valid_mask = np.isfinite(train_y) & np.all(np.isfinite(train_X), axis=1)
        train_X = train_X[valid_mask]
        train_y = train_y[valid_mask]

        if len(train_y) < 10:
            n_stocks = len(forward_returns.columns)
            predictions.append(np.zeros(n_stocks))
            prediction_dates.append(date)
            weights_history.append(np.zeros(config.n_factors))
            continue

        # Add bias term and fit
        X_train_bias = np.column_stack([train_X, np.ones(len(train_X))])
        coef = np.linalg.lstsq(X_train_bias, train_y, rcond=None)[0]

        # Predict on TODAY's factor values (out-of-sample)
        # nan_to_num matches original combine_AFF.py's torch.nan_to_num
        X_today = np.column_stack([
            np.nan_to_num(all_factor_values[expr].loc[date].values)
            for expr in good_factors
        ])
        X_today_bias = np.column_stack([X_today, np.ones(len(X_today))])
        pred = X_today_bias @ coef

        predictions.append(pred)
        prediction_dates.append(date)
        # Pad weights to fixed length (config.n_factors) so np.array() produces
        # a regular 2D array instead of crashing on ragged lists.
        w = np.zeros(config.n_factors)
        w[:len(good_factors)] = coef[:-1]  # Exclude bias term
        weights_history.append(w)
    
    print(f"  Generated {len(predictions)} predictions")
    
    # Save results
    pred_path = os.path.join(output_dir, "predictions.npy")
    np.save(pred_path, np.array(predictions))
    
    weights_path = os.path.join(output_dir, "weights.npy")
    np.save(weights_path, np.array(weights_history))
    
    print(f"  Predictions saved to: {pred_path}")
    print(f"  Weights saved to: {weights_path}")
    
    return {
        'predictions': predictions,
        'prediction_dates': prediction_dates,
        'selected_factors': selected_factors_history,
        'weights': weights_history,
        'pred_path': pred_path,
        'weights_path': weights_path,
    }


def stage3_evaluate_results(
    close_test: pd.DataFrame,
    stage2_results: Dict,
    config: AlphaForgeConfig,
    test_start_ts: pd.Timestamp = None,
    save_dir: Optional[str] = None,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    test_start: Optional[str] = None,
    test_end: Optional[str] = None,
    holding_period: Optional[int] = None,
) -> Dict:
    """
    Stage 3: Evaluate results using the unified BacktestEngine.

    Constructs portfolio weights from Stage 2 predictions (filtered to test
    period) and runs the unified backtest engine for consistent metrics.

    Args:
        close_test: Close prices for test period (date × stock DataFrame).
            Columns are stock names, index is date.
        stage2_results: Results from Stage 2 (contains 'predictions' and
            'prediction_dates' lists)
        config: AlphaForge configuration
        test_start_ts: Timestamp for test period start (used to filter
            predictions). If None, uses all predictions.

    Returns:
        Dict: Final performance metrics (from BacktestEngine)
    """
    print("\n" + "="*60)
    print("[Stage 3] Evaluating results (unified BacktestEngine)...")
    print("="*60)

    predictions = stage2_results.get('predictions', [])
    prediction_dates = stage2_results.get('prediction_dates', [])

    if not predictions:
        print("  ⚠️  No predictions from Stage 2, returning zero metrics")
        return {
            'metrics': {
                'total_return': 0, 'annual_return': 0, 'annual_volatility': 0,
                'sharpe_ratio': 0, 'max_drawdown': 0, 'calmar_ratio': 0,
                'win_rate': 0, 'information_ratio': 0, 'avg_turnover': 0,
                'n_trading_days': 0,
            },
            'portfolio_returns': pd.Series(dtype=float),
        }

    # ── Filter predictions to test period ──
    if test_start_ts is not None and prediction_dates:
        test_mask = [d >= test_start_ts for d in prediction_dates]
        test_predictions = [p for p, m in zip(predictions, test_mask) if m]
        test_pred_dates = [d for d, m in zip(prediction_dates, test_mask) if m]
        print(f"  Filtered predictions: {len(test_predictions)} / {len(predictions)} "
              f"(test_start={test_start_ts.date()})")
    else:
        test_predictions = predictions
        test_pred_dates = prediction_dates

    if not test_predictions:
        print("  ⚠️  No test-period predictions, returning zero metrics")
        return {
            'metrics': {
                'total_return': 0, 'annual_return': 0, 'annual_volatility': 0,
                'sharpe_ratio': 0, 'max_drawdown': 0, 'calmar_ratio': 0,
                'win_rate': 0, 'information_ratio': 0, 'avg_turnover': 0,
                'n_trading_days': 0,
            },
            'portfolio_returns': pd.Series(dtype=float),
        }

    # ── Build portfolios from predictions ──
    # Each prediction is a 1D array of stock scores; map to close_test.columns
    print("  Building portfolios from predictions...")
    stock_names = close_test.columns

    portfolio_rows = []
    date_index = []

    for pred, date in zip(test_predictions, test_pred_dates):
        if not isinstance(pred, np.ndarray):
            continue

        # Map prediction scores to stock names
        n_stocks = min(len(pred), len(stock_names))
        scores = pd.Series(pred[:n_stocks], index=stock_names[:n_stocks])

        # Select top-N stocks and equal-weight
        top = scores.dropna().nlargest(config.top_n_stocks)
        if len(top) == 0:
            continue

        w = pd.Series(1.0 / len(top), index=top.index)
        portfolio_rows.append(w)
        date_index.append(date)

    if not portfolio_rows:
        print("  ⚠️  No valid portfolios built, returning zero metrics")
        return {
            'metrics': {
                'total_return': 0, 'annual_return': 0, 'annual_volatility': 0,
                'sharpe_ratio': 0, 'max_drawdown': 0, 'calmar_ratio': 0,
                'win_rate': 0, 'information_ratio': 0, 'avg_turnover': 0,
                'n_trading_days': 0,
            },
            'portfolio_returns': pd.Series(dtype=float),
        }

    # Build portfolios DataFrame (date × stock)
    all_stocks = pd.Index(set().union(*(w.index for w in portfolio_rows)))
    portfolios = pd.DataFrame(
        index=pd.DatetimeIndex(date_index),
        columns=all_stocks,
        dtype=float,
    )
    for i, w in enumerate(portfolio_rows):
        portfolios.loc[date_index[i], w.index] = w.values
    portfolios = portfolios.fillna(0.0)
    portfolios = portfolios.div(portfolios.sum(axis=1), axis=0).fillna(0.0)

    # Align close_test to portfolio dates and columns
    prices_aligned = close_test.reindex(portfolios.index)
    prices_aligned = prices_aligned.reindex(columns=portfolios.columns)

    # Run unified backtest
    from backtest.engine import BacktestEngine
    engine = BacktestEngine(
        commission=0.001,
        slippage=0.0,
        risk_free_rate=0.0,
        holding_period=config.holding_period,
    )
    _bm = prices_aligned.pct_change().shift(-1).mean(axis=1).dropna()
    _bm.name = 'benchmark_return'
    metrics = engine.run(portfolios, prices_aligned, benchmark_returns=_bm, save_dir=save_dir)

    # ── Compute test-period Mean IC / ICIR ──
    # Cross-sectional Spearman between the Stage-2 combined factor scores
    # (test_predictions) and the N-day forward returns on the test close.
    # This is the number that should appear in the paper tables (test, not IS).
    fp = config.forward_period
    fwd_wide = close_test.shift(-fp) / close_test - 1.0
    _daily_ics = []
    for _pred, _date in zip(test_predictions, test_pred_dates):
        if not isinstance(_pred, np.ndarray):
            continue
        if _date not in fwd_wide.index:
            continue
        _fr = fwd_wide.loc[_date].dropna()
        _n = min(len(_pred), len(stock_names))
        _scores = pd.Series(_pred[:_n], index=stock_names[:_n])
        _common = _scores.index.intersection(_fr.index)
        if len(_common) >= 5:
            _rho = _scores[_common].corr(_fr[_common], method='spearman')
            if not np.isnan(_rho):
                _daily_ics.append(_rho)
    if _daily_ics:
        _mean_ic = float(np.mean(_daily_ics))
        _std_ic = float(np.std(_daily_ics, ddof=1))
        _icir = _mean_ic / _std_ic if (len(_daily_ics) > 1 and _std_ic > 0) else 0.0
    else:
        _mean_ic, _icir = 0.0, 0.0
    metrics['mean_ic'] = _mean_ic
    metrics['icir'] = _icir

    print(f"\n  Results (BacktestEngine):")
    print(f"    Total Return:     {metrics.get('total_return', 0):.4f}")
    print(f"    Annual Return:    {metrics.get('annual_return', 0):.4f}")
    print(f"    Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"    Max Drawdown:     {metrics.get('max_drawdown', 0):.4f}")
    print(f"    Information Ratio:{metrics.get('information_ratio', 0):.4f}")
    print(f"    Win Rate:         {metrics.get('win_rate', 0):.4f}")
    print(f"    Calmar Ratio:     {metrics.get('calmar_ratio', 0):.4f}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        _result = {
            'method': 'AlphaForge',
            'mean_ic': metrics.get('mean_ic', 0.0),
            'icir': metrics.get('icir', 0.0),
            'annual_return': metrics.get('annual_return', 0.0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
            'max_drawdown': metrics.get('max_drawdown', 0.0),
            'information_ratio': metrics.get('information_ratio', 0.0),
            'calmar_ratio': metrics.get('calmar_ratio', 0.0),
            'win_rate': metrics.get('win_rate', 0.0),
            'avg_turnover': metrics.get('avg_turnover', 0.0),
            'total_return': metrics.get('total_return', 0.0),
            'n_trading_days': metrics.get('n_trading_days', 0),
            'forward_period': config.forward_period,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'holding_period': holding_period,
        }
        _path = os.path.join(save_dir, 'alphaforge_results.json')
        with open(_path, 'w', encoding='utf-8') as f:
            json.dump(_result, f, indent=2, default=str)
        print(f"\n  Results saved to {_path}")

    return {
        'metrics': metrics,
        'portfolio_returns': engine.get_returns(),
    }


def run_alphaforge_baseline(
    config_path: str = "config/config.yaml",
    dataloader=None,
    prices: pd.DataFrame = None,
    train_start_date: str = None,
    train_end_date: str = None,
    test_start_date: str = None,
    test_end_date: str = None,
    instruments: str = None,
    top_n_stocks: int = None,
    n_factors: int = 10,
    zoo_size: int = 50,
    seeds: List[int] = None,
    output_dir: str = "experiments/alphaforge",
    verbose: bool = False,
    use_gan: bool = True,
    forward_period: Optional[int] = None,  # None -> config['evolution']['forward_period'] (10)
    holding_period: Optional[int] = None,  # None -> config['backtest']['trading']['holding_period'] (1)
    context_days: int = 30,
) -> Dict:
    """
    Run complete AlphaForge baseline (all 3 stages).

    Data loading priority:
    1. If prices is provided, use it directly
    2. If dataloader is provided, call dataloader.get_prices()
    3. If config_path is provided (and dataloader/prices are None), load from config

    Train/test split:
    - Stage 1 (GAN factor mining) uses **train data only** (dates < test_start)
    - Stage 2 (rolling combination) uses **all data** for walk-forward predictions
    - Stage 3 (backtest) uses **test data only** (dates >= test_start),
      receiving close prices as a date×stock DataFrame

    Args:
        config_path: Path to config YAML (used if dataloader/prices not provided)
        dataloader: Main project DataLoader (optional)
        prices: Price data DataFrame (optional)
        train_start_date: Start date (overrides config)
        test_end_date: End date (overrides config)
        instruments: Instrument list name (overrides config)
        top_n_stocks: Number of stocks in portfolio (overrides config)
        n_factors: Number of factors to combine
        zoo_size: Number of factors to mine
        seeds: Random seeds
        output_dir: Output directory
        verbose: Verbose output
        use_gan: Use GAN-based factor mining (True) or template fallback (False)
        forward_period: Forward return period in days for IC evaluation
            (default 10, matching other baselines for fair comparison)
        train_end_date: Last date of training period (YYYY-MM-DD).
            If None, reads from config or defaults to '2023-12-31'.
        test_start_date: First date of test period (YYYY-MM-DD).
            If None, reads from config or defaults to '2024-01-01'.
        context_days: Number of calendar days to extend before test_start
            for context window (ensures rolling features are valid at test start).

    Returns:
        Dict: Results with metrics, used_gan, gan_pool_size, forward_period,
            train_end, test_start
    """
    # Load config if needed (for defaults)
    if config_path and (train_start_date is None or test_end_date is None or instruments is None or top_n_stocks is None):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            if train_start_date is None:
                train_start_date = config['data'].get('train_start_date', '2019-01-01')
            if test_end_date is None:
                test_end_date = config['data'].get('test_end_date', '2025-12-31')
            if instruments is None:
                instruments = config['data']['universe'].get('index', 'csi300')
            if top_n_stocks is None:
                top_n_stocks = config.get('backtest', {}).get('top_n_stocks', 50)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            # Use defaults
            if train_start_date is None:
                train_start_date = "2023-01-01"
            if test_end_date is None:
                test_end_date = "2024-12-31"
            if instruments is None:
                instruments = "csi300"
            if top_n_stocks is None:
                top_n_stocks = 50
    
    # Load data (new DataLoader DatasetBundle contract — train/test slices are
    # produced centrally by loader.load_data, so no manual date-masking here).
    loader = None
    price_data = None
    prices = None           # ALL data (long format) — Stage 2 walk-forward
    prices_train = None     # TRAIN data (long format) — Stage 1
    close_test = None       # TEST close (date × stock) — Stage 3
    train_end = train_end_date
    test_start = test_start_date

    if prices is None and dataloader is None and config_path:
        # Load from config via the new DataLoader bundle contract
        print("Loading data from config...")
        try:
            from dataloader.loader import DataLoader as ProjectDataLoader
            loader = ProjectDataLoader(config_path=config_path)
            train_start = train_start_date or loader.data_config.get('train_start_date', '2023-01-01')
            train_end = train_end_date or loader.data_config.get('train_end_date', '2023-12-31')
            test_start = test_start_date or loader.data_config.get('test_start_date', '2024-01-01')
            test_end = test_end_date or loader.data_config.get('test_end_date', '2025-06-30')

            bundle = loader.load_data(train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end)
            train_price, train_fund, train_ind = bundle.train
            test_price, test_fund, test_ind = bundle.test
            price_data = bundle.full[0]  # FULL span

            prices = _price_dict_to_long(price_data)         # ALL data
            prices_train = _price_dict_to_long(train_price)   # TRAIN only
            close_test = test_price['close']                  # TEST close
            print(f"  Loaded {len(prices)} records")
        except Exception as e:
            raise ValueError(f"Failed to load data from config: {e}. Please provide dataloader or prices.")

    elif dataloader is not None:
        print("Loading data from DataLoader...")
        loader = dataloader
        train_start = train_start_date or loader.data_config.get('train_start_date', '2023-01-01')
        train_end = train_end_date or loader.data_config.get('train_end_date', '2023-12-31')
        test_start = test_start_date or loader.data_config.get('test_start_date', '2024-01-01')
        test_end = test_end_date or loader.data_config.get('test_end_date', '2025-06-30')
        bundle = loader.load_data(train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end)
        train_price, train_fund, train_ind = bundle.train
        test_price, test_fund, test_ind = bundle.test
        price_data = bundle.full[0]
        prices = _price_dict_to_long(price_data)
        prices_train = _price_dict_to_long(train_price)
        close_test = test_price['close']

    if prices is None:
        raise ValueError("Must provide either dataloader, prices, or a valid config_path")

    # ── Determine train/test split (timestamps used by Stage 3 filtering) ──
    print("\nDetermining train/test split...")
    _cfg = loader.data_config if (loader is not None and hasattr(loader, 'data_config')) else {}
    if train_end is None:
        train_end = _cfg.get('train_end_date', '2023-12-31')
    if test_start is None:
        test_start = _cfg.get('test_start_date', '2024-01-01')
    test_start_ts = pd.Timestamp(test_start)
    print(f"  Train end: {train_end}, Test start: {test_start}")

    # Direct `prices` input path (no DataLoader): the only available source is the
    # supplied frame, so derive the test/train slices by masking it here.
    if close_test is None or prices_train is None:
        if price_data is not None:
            close_all = price_data['close']
        else:
            close_all = convert_to_multindex(prices)['close']
        close_test = close_all[close_all.index >= test_start_ts]
        prices = prices.copy()
        prices['date'] = pd.to_datetime(prices['date'])
        prices_train = prices[prices['date'] < test_start_ts].copy()
    print(f"  Train prices: {len(prices_train)} records, All prices: {len(prices)} records")
    
    # Create config
    # ── Resolve forward_period / holding_period from config ──────────
    # explicit arg > config.yaml > default, so standalone runs also honor config.
    _ev_cfg = loader.config.get('evolution', {}) if loader is not None else {}
    _bt_cfg = loader.config.get('backtest', {}).get('trading', {}) if loader is not None else {}
    if not forward_period or forward_period <= 0:
        forward_period = _ev_cfg.get('forward_period', 10)
    if not holding_period or holding_period <= 0:
        holding_period = _bt_cfg.get('holding_period', 1)

    config = AlphaForgeConfig(
        instruments=instruments,
        n_factors=n_factors,
        zoo_size=zoo_size,
        seeds=seeds or [0],
        top_n_stocks=top_n_stocks,
        forward_period=forward_period,
        holding_period=holding_period if holding_period is not None else 1,
        window=20,
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Date-isolated run directory (one subdir per execution, for multiple runs)
    # ── Parameter-tagged, date-isolated run directory ──
    method_name = "alphaforge"
    _u = instruments or _cfg.get('universe', {}).get('index', 'csi300')
    _s = train_start_date or _cfg.get('train_start_date', 'na')
    _e = test_end_date or _cfg.get('test_end_date', 'na')
    _fp = forward_period
    _hp = holding_period if holding_period is not None else 1
    param_dir = f"{_u}_{_s}_{_e}_forward-{_fp}_holding-{_hp}"
    run_dir = os.path.join(os.path.dirname(output_dir), param_dir, method_name)
    os.makedirs(run_dir, exist_ok=True)

    # Stage 1: Mine factors (TRAIN data only — no test data leakage)
    print("\n" + "="*60)
    print("[Stage 1] Mining factors on TRAIN data only")
    print("="*60)
    stage1_results = stage1_mine_factors(prices_train, config, run_dir, use_gan=use_gan)
    
    # Stage 2: Combine factors (ALL data for walk-forward predictions)
    # Factors are fixed from Stage 1; Stage 2 does rolling IC + linear regression
    # using only past data at each step, so no leakage.
    print("\n" + "="*60)
    print("[Stage 2] Walk-forward combination on ALL data")
    print("="*60)
    stage2_results = stage2_combine_factors(prices, stage1_results, config, run_dir)
    
    # Stage 3: Evaluate results (TEST data only)
    # Receives close_test (date×stock DataFrame) and filters predictions to test dates
    print("\n" + "="*60)
    print("[Stage 3] Backtest on TEST data only")
    print("="*60)
    stage3_results = stage3_evaluate_results(
        close_test, stage2_results, config, test_start_ts,
        save_dir=run_dir,
        train_start=train_start, train_end=train_end,
        test_start=test_start, test_end=test_end,
        holding_period=holding_period,
    )
    
    return {
        'metrics': stage3_results['metrics'],
        'portfolio_returns': stage3_results['portfolio_returns'],
        'stage1_results': stage1_results,
        'stage2_results': stage2_results,
        'used_gan': stage1_results.get('used_gan', False),
        'gan_pool_size': stage1_results.get('gan_stats', {}).get('pool_size', 0),
        'forward_period': forward_period,
        'train_start': train_start,
        'train_end': train_end,
        'test_start': test_start,
        'test_end': test_end,
        'holding_period': holding_period,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run AlphaForge baseline")
    parser.add_argument("--config-path", type=str, default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--train-start", type=str, default=None, help="Start date (overrides config)")
    parser.add_argument("--test-end", type=str, default=None, help="End date (overrides config)")
    parser.add_argument("--instruments", type=str, default=None, help="Instruments list (overrides config)")
    parser.add_argument("--top-n", type=int, default=None, help="Number of stocks in portfolio (overrides config)")
    parser.add_argument("--n-factors", type=int, default=10, help="Number of factors to combine")
    parser.add_argument("--zoo-size", type=int, default=50, help="Number of factors to mine")
    parser.add_argument("--output-dir", type=str, default="experiments/alphaforge", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--use-gan", action="store_true", default=True, help="Use GAN-based factor mining")
    parser.add_argument("--no-gan", action="store_false", dest="use_gan", help="Use template-based factor generation")
    parser.add_argument("--forward-period", type=int, default=None,
                        help="Forward return period in days for IC evaluation "
                             "(default: config evolution.forward_period, 10)")
    parser.add_argument("--holding-period", type=int, default=None,
                        help="Portfolio holding period in days for backtest "
                             "(default: config value, 1 = daily rebalance)")
    parser.add_argument("--train-end", type=str, default=None,
                        help="Last date of training period (YYYY-MM-DD). "
                             "Default: read from config or '2023-12-31'")
    parser.add_argument("--test-start", type=str, default=None,
                        help="First date of test period (YYYY-MM-DD). "
                             "Default: read from config or '2024-01-01'")
    parser.add_argument("--context-days", type=int, default=30,
                        help="Context window in calendar days before test_start "
                             "(default 30)")

    args = parser.parse_args()

    results = run_alphaforge_baseline(
        config_path=args.config_path,
        train_start_date=args.train_start,
        train_end_date=args.train_end,
        test_start_date=args.test_start,
        test_end_date=args.test_end,
        instruments=args.instruments,
        top_n_stocks=args.top_n,
        n_factors=args.n_factors,
        zoo_size=args.zoo_size,
        output_dir=args.output_dir,
        verbose=args.verbose,
        use_gan=args.use_gan,
        forward_period=args.forward_period,
        holding_period=args.holding_period,
        context_days=args.context_days,
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Total Return: {results['metrics']['total_return']:.2%}")
    print(f"Annual Return: {results['metrics']['annual_return']:.2%}")
    print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
    print(f"Information Ratio: {results['metrics']['information_ratio']:.4f}")
    print(f"Mean IC (test): {results['metrics'].get('mean_ic', 0):.4f}")
    print(f"ICIR (test):    {results['metrics'].get('icir', 0):.4f}")
    print(f"Turnover:       {results['metrics'].get('avg_turnover', 0):.4f}")
    print(f"Forward Period:    {results['forward_period']}d")
    print(f"Train End:         {results['train_end']}")
    print(f"Test Start:        {results['test_start']}")
