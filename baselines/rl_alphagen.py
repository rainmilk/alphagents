# -*- coding: utf-8 -*-
"""
RL AlphaGen — Reinforcement Learning Factor Generation
=======================================================

This module implements two RL strategies for factor generation, replacing the
random sampling approach in run_alphagen.py:

  Option B (default): REINFORCE + Simple MLP Policy
    - Lightweight policy gradient, no GPU needed
    - Token embedding → MLP → action logits with masking
    - Trains in minutes on CPU

  Option C (optional): MaskablePPO + LSTMSharedNet
    - Faithful reproduction of original AlphaGen RL pipeline
    - sb3-contrib MaskablePPO + 2-layer LSTM feature extractor
    - Requires torch + sb3-contrib + stable-baselines3

Both options share the same Gym environment (RLAlphaEnv) which wraps the
existing ExpressionBuilder + AlphaPool infrastructure.

Architecture:
  ┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
  │  RL Policy  │────▶│  RLAlphaEnv      │────▶│  AlphaPool   │
  │  (MLP/LSTM) │     │  (Gym wrapper)   │     │  (numpy)     │
  └─────────────┘     │                  │     └──────────────┘
                      │  ExpressionBuilder│           │
   action ──────────▶│  + action masking │──────▶ reward = ensemble IC
                      └──────────────────┘

Author: Code Review Expert
Date: 2026-07-03
"""

import math
import time
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Lazy imports for torch / sb3 (loaded only when needed) ─────────────

def _import_torch():
    """Import torch lazily; raise informative error if missing."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.distributions import Categorical
        return torch, nn, optim, Categorical
    except ImportError:
        raise ImportError(
            "PyTorch is required for RL training. Install with:\n"
            "  pip install torch  (CPU: --index-url https://download.pytorch.org/whl/cpu)\n"
            "Or use --rl-method random for the no-dependency fallback."
        )

def _import_sb3():
    """Import sb3-contrib + stable-baselines3 lazily."""
    try:
        from sb3_contrib.ppo_mask import MaskablePPO
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        return MaskablePPO, BaseFeaturesExtractor
    except ImportError:
        raise ImportError(
            "sb3-contrib and stable-baselines3 are required for PPO training. Install with:\n"
            "  pip install sb3-contrib stable-baselines3\n"
            "Or use --rl-method reinforce (default) which only needs torch."
        )


# ═══════════════════════════════════════════════════════════════════════
#  Section 1: Action Space — Token-to-Integer Mapping
# ═══════════════════════════════════════════════════════════════════════

# Import token vocabulary from run_alphagen
from run_alphagen import (
    FEATURES,
    OPERATOR_CLASSES,
    CONSTANTS,
    DELTA_TIMES,
    MAX_EXPR_LENGTH,
    ExpressionBuilder,
    ExprNode,
    Feature,
    Constant,
    DeltaTime,
    TokenType,
    AlphaPool,
    evaluate_factor,
    _normalize_cross_section,
)

# ── Build action vocabulary ────────────────────────────────────────────

# Layout: [FEATURES | UNARY_OPS | BINARY_OPS | ROLLING_OPS | PAIR_ROLLING_OPS |
#          CONSTANTS | DELTA_TIMES | SEP]

_UNARY_OPS = [name for name, (_, n, t) in OPERATOR_CLASSES.items() if t == 'unary']
_BINARY_OPS = [name for name, (_, n, t) in OPERATOR_CLASSES.items() if t == 'binary']
_ROLLING_OPS = [name for name, (_, n, t) in OPERATOR_CLASSES.items() if t == 'rolling']
_PAIR_ROLLING_OPS = [name for name, (_, n, t) in OPERATOR_CLASSES.items() if t == 'pair_rolling']

# Action layout offsets
OFFSET_FEATURE = 0
OFFSET_UNARY = OFFSET_FEATURE + len(FEATURES)
OFFSET_BINARY = OFFSET_UNARY + len(_UNARY_OPS)
OFFSET_ROLLING = OFFSET_BINARY + len(_BINARY_OPS)
OFFSET_PAIR_ROLLING = OFFSET_ROLLING + len(_ROLLING_OPS)
OFFSET_CONSTANT = OFFSET_PAIR_ROLLING + len(_PAIR_ROLLING_OPS)
OFFSET_DELTA_TIME = OFFSET_CONSTANT + len(CONSTANTS)
OFFSET_SEP = OFFSET_DELTA_TIME + len(DELTA_TIMES)

SIZE_FEATURE = len(FEATURES)
SIZE_UNARY = len(_UNARY_OPS)
SIZE_BINARY = len(_BINARY_OPS)
SIZE_ROLLING = len(_ROLLING_OPS)
SIZE_PAIR_ROLLING = len(_PAIR_ROLLING_OPS)
SIZE_CONSTANT = len(CONSTANTS)
SIZE_DELTA_TIME = len(DELTA_TIMES)
SIZE_SEP = 1

SIZE_ACTION = OFFSET_SEP + SIZE_SEP  # Total: 63

# Reverse lookup: action_id → (token_type, value)
def _build_action_table():
    table = []
    for i, name in enumerate(FEATURES):
        table.append(('FEATURE', name))
    for i, name in enumerate(_UNARY_OPS):
        table.append(('OPERATOR', name))
    for i, name in enumerate(_BINARY_OPS):
        table.append(('OPERATOR', name))
    for i, name in enumerate(_ROLLING_OPS):
        table.append(('OPERATOR', name))
    for i, name in enumerate(_PAIR_ROLLING_OPS):
        table.append(('OPERATOR', name))
    for i, val in enumerate(CONSTANTS):
        table.append(('CONSTANT', val))
    for i, d in enumerate(DELTA_TIMES):
        table.append(('DELTA_TIME', d))
    table.append(('SEP', None))
    return table

_ACTION_TABLE = _build_action_table()


def action_to_token(action_id: int) -> Tuple[str, object]:
    """Convert integer action ID to (token_type, value)."""
    return _ACTION_TABLE[action_id]


def get_action_mask(builder: ExpressionBuilder) -> np.ndarray:
    """
    Compute boolean action mask based on current builder state.
    
    Returns: np.ndarray of shape (SIZE_ACTION,) — True for valid actions.
    """
    mask = np.zeros(SIZE_ACTION, dtype=bool)
    valid_types = builder.valid_action_types()
    
    # Features
    if valid_types.get('FEATURE', False):
        mask[OFFSET_FEATURE:OFFSET_UNARY] = True
    
    # Operators — need to check which specific ops are valid
    if valid_types.get('OPERATOR', False):
        valid_ops = set(builder.get_valid_ops())
        for i, name in enumerate(_UNARY_OPS):
            if name in valid_ops:
                mask[OFFSET_UNARY + i] = True
        for i, name in enumerate(_BINARY_OPS):
            if name in valid_ops:
                mask[OFFSET_BINARY + i] = True
        for i, name in enumerate(_ROLLING_OPS):
            if name in valid_ops:
                mask[OFFSET_ROLLING + i] = True
        for i, name in enumerate(_PAIR_ROLLING_OPS):
            if name in valid_ops:
                mask[OFFSET_PAIR_ROLLING + i] = True
    
    # Constants
    if valid_types.get('CONSTANT', False):
        mask[OFFSET_CONSTANT:OFFSET_DELTA_TIME] = True
    
    # Delta times
    if valid_types.get('DELTA_TIME', False):
        mask[OFFSET_DELTA_TIME:OFFSET_SEP] = True
    
    # SEP
    if valid_types.get('SEP', False):
        mask[OFFSET_SEP] = True
    
    # Safety: if no action is valid (shouldn't happen), allow SEP to terminate
    if not mask.any():
        mask[OFFSET_SEP] = True
    
    return mask


def apply_action(builder: ExpressionBuilder, action_id: int) -> bool:
    """
    Apply an integer action to the ExpressionBuilder.
    
    Returns: True if the action was successfully applied.
    """
    token_type, value = action_to_token(action_id)
    
    if token_type == 'FEATURE':
        return builder.add_feature(value)
    elif token_type == 'OPERATOR':
        return builder.add_operator(value)
    elif token_type == 'CONSTANT':
        return builder.add_constant(value)
    elif token_type == 'DELTA_TIME':
        return builder.add_delta_time(value)
    elif token_type == 'SEP':
        return True  # SEP is handled by caller
    return False


# ═══════════════════════════════════════════════════════════════════════
#  Section 2: RLAlphaEnv — Gym-compatible Environment
# ═══════════════════════════════════════════════════════════════════════

class RLAlphaEnv:
    """
    Gym-compatible environment for RL-based factor generation.
    
    Wraps ExpressionBuilder + AlphaPool into a step/reset interface
    compatible with both custom REINFORCE and sb3-contrib MaskablePPO.
    
    Observation: (MAX_EXPR_LENGTH,) uint8 array of action IDs (0-padded)
    Action: Discrete(SIZE_ACTION)
    Reward: Ensemble IC when factor is accepted into pool, 0 per step
    
    Episode:
      1. reset() → empty builder, state = zeros
      2. step(action) → apply token to builder, state updated
      3. When SEP action → evaluate factor, try_add to pool, return reward
      4. Episode ends on SEP or MAX_EXPR_LENGTH reached
    """
    
    def __init__(
        self,
        train_data: Dict[str, np.ndarray],
        train_fwd_ret: np.ndarray,
        pool: AlphaPool,
        min_tokens: int = 3,
        reward_per_step: float = 0.0,
    ):
        """
        Args:
            train_data: dict of {feature: (n_dates, n_stocks) ndarray}
            train_fwd_ret: (n_dates, n_stocks) forward returns
            pool: AlphaPool to add factors to
            min_tokens: minimum tokens before SEP is allowed (encourages complexity)
            reward_per_step: small per-step reward (default 0, same as original)
        """
        self.train_data = train_data
        self.train_fwd_ret = train_fwd_ret
        self.pool = pool
        self.min_tokens = min_tokens
        self.reward_per_step = reward_per_step
        
        self._builder = ExpressionBuilder()
        self._state = np.zeros(MAX_EXPR_LENGTH, dtype=np.uint8)
        self._counter = 0
        self._step_count = 0
        
        # Stats
        self.n_episodes = 0
        self.n_factors_added = 0
        self.best_reward = 0.0
    
    @property
    def action_space_size(self) -> int:
        return SIZE_ACTION
    
    @property
    def observation_space_shape(self) -> Tuple[int]:
        return (MAX_EXPR_LENGTH,)
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode."""
        self._builder.reset()
        self._state = np.zeros(MAX_EXPR_LENGTH, dtype=np.uint8)
        self._counter = 0
        self._step_count = 0
        self.n_episodes += 1
        return self._state.copy()
    
    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step.
        
        Returns: (observation, reward, done, info)
        """
        self._step_count += 1
        info = {}
        
        token_type, value = action_to_token(action_id)
        
        # Check if SEP
        if token_type == 'SEP':
            tree = self._builder.get_tree()
            if tree is not None:
                reward = self._evaluate_and_add(tree)
            else:
                reward = 0.0
                info['invalid'] = True
            return self._state.copy(), reward, True, info
        
        # Apply token to builder
        success = apply_action(self._builder, action_id)
        
        if not success:
            # Invalid action (shouldn't happen with proper masking)
            # Try to finish if we have a valid tree
            tree = self._builder.get_tree()
            if tree is not None:
                reward = self._evaluate_and_add(tree)
            else:
                reward = 0.0
            return self._state.copy(), reward, True, info
        
        # Update state
        if self._counter < MAX_EXPR_LENGTH:
            self._state[self._counter] = action_id
            self._counter += 1
        
        # Check if we've reached max length
        if self._step_count >= MAX_EXPR_LENGTH or self._builder.tokens_used >= MAX_EXPR_LENGTH:
            tree = self._builder.get_tree()
            if tree is not None:
                reward = self._evaluate_and_add(tree)
            else:
                reward = 0.0
            return self._state.copy(), reward, True, info
        
        # Normal step — small per-step reward
        return self._state.copy(), self.reward_per_step, False, info
    
    def _evaluate_and_add(self, tree: ExprNode) -> float:
        """Evaluate factor expression and try to add to pool. Returns reward."""
        try:
            raw_value = tree.evaluate(self.train_data)
            normalized = _normalize_cross_section(raw_value)
            if isinstance(normalized, np.ndarray):
                val = normalized.astype(np.float64)
            else:
                val = np.array(normalized, dtype=np.float64)
            
            metrics = evaluate_factor(val, self.train_fwd_ret)
            accepted, ensemble_ic = self.pool.try_add_factor(tree, val, metrics)
            
            if accepted:
                self.n_factors_added += 1
                self.best_reward = max(self.best_reward, ensemble_ic)
            
            return ensemble_ic
        except Exception:
            return 0.0
    
    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid actions (for MaskablePPO).

        Enforces min_tokens: SEP is hidden until the builder has used
        at least min_tokens tokens, encouraging the policy to generate
        complex expressions rather than trivial single-token ones.
        """
        mask = get_action_mask(self._builder)
        # Hide SEP until min_tokens reached (prevents trivial expressions)
        if self._builder.tokens_used < self.min_tokens:
            mask[OFFSET_SEP] = False
            # Safety: if hiding SEP leaves no valid actions, restore it
            # (can happen at MAX_EXPR_LENGTH with an unfinished expression)
            if not mask.any():
                mask[OFFSET_SEP] = True
        return mask
    
    def get_valid_actions(self) -> np.ndarray:
        """Return array of valid action IDs."""
        return np.where(self.action_masks())[0]


# ═══════════════════════════════════════════════════════════════════════
#  Section 3: Option B — REINFORCE + Simple MLP Policy
# ═══════════════════════════════════════════════════════════════════════

def reinforce_train(
    env: RLAlphaEnv,
    n_episodes: int = 2000,
    lr: float = 1e-3,
    gamma: float = 1.0,
    d_model: int = 64,
    hidden_dim: int = 128,
    n_layers: int = 2,
    ent_coef: float = 0.01,
    log_interval: int = 100,
    verbose: bool = True,
    device: str = 'cpu',
) -> Dict:
    """
    Train a factor generation policy using REINFORCE algorithm.
    
    This is Option B: lightweight RL with a simple MLP policy network.
    No LSTM, no PPO — just vanilla policy gradient with action masking.
    
    Algorithm:
      1. Collect episode: policy samples actions, env applies them
      2. Compute discounted returns: G_t = Σ γ^k * r_{t+k}
      3. Update: ∇J = E[∇log π(a|s) * G_t]  (+ entropy bonus)
    
    Args:
        env: RLAlphaEnv instance
        n_episodes: Number of training episodes
        lr: Learning rate
        gamma: Discount factor (1.0 = no discount, same as original AlphaGen)
        d_model: Token embedding dimension
        hidden_dim: MLP hidden layer dimension
        n_layers: Number of MLP hidden layers
        ent_coef: Entropy coefficient for exploration
        log_interval: Print stats every N episodes
        verbose: Print training progress
        device: 'cpu' or 'cuda'
    
    Returns:
        Dict with training stats and final pool state
    """
    torch, nn, optim, Categorical = _import_torch()
    device_t = torch.device(device)
    
    # ── Build policy network ───────────────────────────────────────────
    class SimpleMLPPolicy(nn.Module):
        def __init__(self, n_actions, max_len, d_model, hidden_dim, n_layers):
            super().__init__()
            self.n_actions = n_actions
            self.token_emb = nn.Embedding(n_actions + 1, d_model, padding_idx=0)
            
            layers = []
            in_dim = d_model * max_len
            for _ in range(n_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
                in_dim = hidden_dim
            self.mlp = nn.Sequential(*layers)
            self.action_head = nn.Linear(hidden_dim, n_actions)
        
        def forward(self, obs, mask):
            """
            Args:
                obs: (batch, seq_len) long tensor of action IDs
                mask: (batch, n_actions) bool tensor of valid actions
            Returns:
                Categorical distribution over actions
            """
            emb = self.token_emb(obs)  # (batch, seq, d_model)
            flat = emb.flatten(1)      # (batch, seq * d_model)
            features = self.mlp(flat)
            logits = self.action_head(features)
            # Apply action mask
            logits = logits.masked_fill(~mask, -1e9)
            return Categorical(logits=logits)
    
    policy = SimpleMLPPolicy(
        n_actions=SIZE_ACTION,
        max_len=MAX_EXPR_LENGTH,
        d_model=d_model,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    ).to(device_t)
    
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    # ── Training loop ──────────────────────────────────────────────────
    if verbose:
        print(f"\n  [REINFORCE] Training {n_episodes} episodes, lr={lr}, γ={gamma}")
        print(f"  [REINFORCE] Policy: MLP({d_model}×{MAX_EXPR_LENGTH} → {hidden_dim}×{n_layers} → {SIZE_ACTION})")
        print(f"  [REINFORCE] Device: {device_t}")
    
    episode_rewards = []
    episode_lengths = []
    pool_sizes = []
    best_ic_history = []
    
    t0 = time.time()
    
    for ep in range(n_episodes):
        # ── Collect episode ────────────────────────────────────────────
        obs = env.reset()
        log_probs = []
        rewards = []
        entropies = []
        done = False
        
        while not done:
            obs_t = torch.LongTensor(obs).unsqueeze(0).to(device_t)
            mask = env.action_masks()
            mask_t = torch.BoolTensor(mask).unsqueeze(0).to(device_t)
            
            dist = policy(obs_t, mask_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            obs, reward, done, info = env.step(action.item())
            
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
        
        # ── Compute returns ────────────────────────────────────────────
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device_t)
        
        # Normalize returns (reduce variance)
        if returns.numel() > 1 and returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # ── Policy gradient update ─────────────────────────────────────
        log_probs_t = torch.stack(log_probs).squeeze()
        entropies_t = torch.stack(entropies).squeeze()
        
        if log_probs_t.dim() == 0:
            log_probs_t = log_probs_t.unsqueeze(0)
            entropies_t = entropies_t.unsqueeze(0)
        
        policy_loss = -(log_probs_t * returns).mean()
        entropy_loss = -entropies_t.mean()
        loss = policy_loss + ent_coef * entropy_loss
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        
        # ── Logging ────────────────────────────────────────────────────
        ep_reward = sum(rewards)
        episode_rewards.append(ep_reward)
        episode_lengths.append(len(rewards))
        pool_sizes.append(env.pool.size)
        best_ic_history.append(env.pool.best_ic_ret)
        
        if verbose and (ep + 1) % log_interval == 0:
            recent_rewards = episode_rewards[-log_interval:]
            recent_lengths = episode_lengths[-log_interval:]
            elapsed = time.time() - t0
            print(f"  [REINFORCE] Ep {ep+1:5d}/{n_episodes} | "
                  f"Avg reward: {np.mean(recent_rewards):+.4f} | "
                  f"Avg len: {np.mean(recent_lengths):.1f} | "
                  f"Pool: {env.pool.size}/{env.pool.capacity} | "
                  f"Best IC: {env.pool.best_ic_ret:.4f} | "
                  f"Factors added: {env.n_factors_added} | "
                  f"{elapsed:.0f}s")
    
    elapsed = time.time() - t0
    if verbose:
        print(f"\n  [REINFORCE] Training complete in {elapsed:.1f}s")
        print(f"  [REINFORCE] Final pool: {env.pool.size}/{env.pool.capacity}, "
              f"Best IC: {env.pool.best_ic_ret:.4f}, "
              f"Total factors added: {env.n_factors_added}")
    
    return {
        'method': 'reinforce',
        'n_episodes': n_episodes,
        'final_pool_size': env.pool.size,
        'best_ic': env.pool.best_ic_ret,
        'n_factors_added': env.n_factors_added,
        'n_episodes_total': env.n_episodes,
        'training_time': elapsed,
        'episode_rewards': episode_rewards,
        'pool_sizes': pool_sizes,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Section 4: Option C — MaskablePPO + LSTMSharedNet
# ═══════════════════════════════════════════════════════════════════════

def _build_lstm_policy_class():
    """
    Build LSTMSharedNet class adapted from AlphaForge/alphagen/rl/policy.py.
    
    This is a faithful reproduction of the original AlphaGen policy network:
      Token Embedding → Positional Encoding → LSTM → Mean Pooling → Features
    
    Returns the class definition (created lazily to defer torch import).
    """
    torch, nn, _, _ = _import_torch()
    MaskablePPO, BaseFeaturesExtractor = _import_sb3()
    
    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding (same as original AlphaGen)."""
        def __init__(self, d_model: int, max_len: int = 5000):
            super().__init__()
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('_pe', pe)
        
        def forward(self, x):
            seq_len = x.size(0) if x.dim() == 2 else x.size(1)
            return x + self._pe[:seq_len]
    
    class LSTMSharedNet(BaseFeaturesExtractor):
        """
        LSTM feature extractor for MaskablePPO.
        
        Faithful reproduction of AlphaForge/alphagen/rl/policy.py LSTMSharedNet.
        Processes token ID sequences through:
          1. Token embedding (n_actions+1 → d_model, +1 for BEG token)
          2. Positional encoding
          3. Multi-layer LSTM
          4. Mean pooling over sequence
        
        The output feature vector is consumed by PPO's MlpPolicy head.
        """
        def __init__(
            self,
            observation_space,
            n_layers: int = 2,
            d_model: int = 128,
            dropout: float = 0.1,
            device: str = 'cpu',
        ):
            super().__init__(observation_space, d_model)
            
            # Derive n_actions from observation space
            # observation_space is Box(low=0, high=n_actions-1, shape=(MAX_EXPR_LENGTH,))
            n_actions = int(observation_space.high.max()) + 1
            
            self._device = torch.device(device)
            self._d_model = d_model
            self._n_actions = n_actions
            
            self._token_emb = nn.Embedding(n_actions + 1, d_model, padding_idx=0)
            self._pos_enc = PositionalEncoding(d_model)
            
            self._lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
        
        def forward(self, obs):
            """
            Args:
                obs: (batch, seq_len) float tensor (will be cast to long)
            Returns:
                (batch, d_model) feature vector
            """
            bs, seqlen = obs.shape
            # Prepend BEG token (ID = n_actions)
            beg = torch.full((bs, 1), fill_value=self._n_actions,
                           dtype=torch.long, device=obs.device)
            obs_long = torch.cat((beg, obs.long()), dim=1)
            # Compute real length (non-padding)
            real_len = (obs_long != 0).sum(1).max().item()
            real_len = max(real_len, 1)
            
            src = self._pos_enc(self._token_emb(obs_long))
            res = self._lstm(src[:, :real_len])[0]  # (bs, real_len, d_model)
            return res.mean(dim=1)  # (bs, d_model)
    
    return LSTMSharedNet


def _build_gym_env(env: RLAlphaEnv):
    """
    Wrap RLAlphaEnv into a proper Gymnasium environment for sb3-contrib.
    
    sb3-contrib 2.9+ requires gymnasium API:
      - reset() → (obs, info)
      - step(action) → (obs, reward, terminated, truncated, info)
      - action_masks() → np.ndarray[bool]
    """
    import gymnasium as gym
    import gymnasium.spaces
    
    class GymAlphaEnv(gym.Env):
        """Gymnasium wrapper around RLAlphaEnv for sb3-contrib compatibility."""
        
        def __init__(self, rl_env: RLAlphaEnv):
            super().__init__()
            self.rl_env = rl_env
            self.action_space = gym.spaces.Discrete(SIZE_ACTION)
            self.observation_space = gym.spaces.Box(
                low=0, high=SIZE_ACTION,
                shape=(MAX_EXPR_LENGTH,),
                dtype=np.uint8,
            )
        
        def reset(self, *, seed=None, options=None):
            obs = self.rl_env.reset()
            return obs, {}
        
        def step(self, action):
            obs, reward, done, info = self.rl_env.step(int(action))
            # gymnasium: (obs, reward, terminated, truncated, info)
            return obs, reward, done, False, info
        
        def action_masks(self):
            return self.rl_env.action_masks()
    
    return GymAlphaEnv(env)


def ppo_train(
    env: RLAlphaEnv,
    n_timesteps: int = 100_000,
    n_layers: int = 2,
    d_model: int = 128,
    dropout: float = 0.1,
    lr: float = 3e-4,
    batch_size: int = 128,
    ent_coef: float = 0.01,
    gamma: float = 1.0,
    n_steps: int = 2048,
    verbose: bool = True,
    device: str = 'auto',
    tensorboard_log: Optional[str] = None,
) -> Dict:
    """
    Train a factor generation policy using MaskablePPO + LSTMSharedNet.
    
    This is Option C: faithful reproduction of original AlphaGen RL pipeline.
    Uses sb3-contrib's MaskablePPO with action masking + LSTM feature extractor.
    
    The training closely follows train_RL.py from AlphaForge:
      - MaskablePPO('MlpPolicy', env, ...)
      - policy_kwargs: LSTMSharedNet(n_layers=2, d_model=128, dropout=0.1)
      - gamma=1.0, ent_coef=0.01, batch_size=128
    
    Args:
        env: RLAlphaEnv instance
        n_timesteps: Total training timesteps (original: 200K-400K)
        n_layers: LSTM layers (original: 2)
        d_model: Model dimension (original: 128)
        dropout: Dropout rate (original: 0.1)
        lr: Learning rate (sb3 default: 3e-4)
        batch_size: PPO batch size (original: 128)
        ent_coef: Entropy coefficient (original: 0.01)
        gamma: Discount factor (original: 1.0)
        n_steps: Rollout buffer size per env
        verbose: Print training progress
        device: 'cpu', 'cuda', or 'auto'
        tensorboard_log: Path for tensorboard logs (optional)
    
    Returns:
        Dict with training stats and final pool state
    """
    torch, _, _, _ = _import_torch()
    MaskablePPO, _ = _import_sb3()
    
    # Resolve device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Build LSTMSharedNet class
    LSTMSharedNet = _build_lstm_policy_class()
    
    # Wrap env in Gym interface
    gym_env = _build_gym_env(env)
    
    if verbose:
        print(f"\n  [PPO] Training {n_timesteps:,} timesteps")
        print(f"  [PPO] Policy: LSTMSharedNet(n_layers={n_layers}, d_model={d_model}, dropout={dropout})")
        print(f"  [PPO] γ={gamma}, ent_coef={ent_coef}, batch_size={batch_size}, lr={lr}")
        print(f"  [PPO] Device: {device}")
    
    # Build MaskablePPO model (faithful to train_RL.py)
    model = MaskablePPO(
        'MlpPolicy',
        gym_env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=n_layers,
                d_model=d_model,
                dropout=dropout,
                device=device,
            ),
        ),
        gamma=gamma,
        ent_coef=ent_coef,
        batch_size=batch_size,
        learning_rate=lr,
        n_steps=n_steps,
        verbose=1 if verbose else 0,
        device=device,
        tensorboard_log=tensorboard_log,
    )
    
    # Train
    t0 = time.time()
    model.learn(total_timesteps=n_timesteps)
    elapsed = time.time() - t0
    
    if verbose:
        print(f"\n  [PPO] Training complete in {elapsed:.1f}s")
        print(f"  [PPO] Final pool: {env.pool.size}/{env.pool.capacity}, "
              f"Best IC: {env.pool.best_ic_ret:.4f}, "
              f"Total factors added: {env.n_factors_added}")
    
    return {
        'method': 'ppo',
        'n_timesteps': n_timesteps,
        'final_pool_size': env.pool.size,
        'best_ic': env.pool.best_ic_ret,
        'n_factors_added': env.n_factors_added,
        'n_episodes_total': env.n_episodes,
        'training_time': elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Section 5: Unified Entry Point
# ═══════════════════════════════════════════════════════════════════════

def train_rl_factors(
    method: str,
    train_data: Dict[str, np.ndarray],
    train_fwd_ret: np.ndarray,
    pool: AlphaPool,
    min_tokens: int = 3,
    # REINFORCE params
    n_episodes: int = 2000,
    reinforce_lr: float = 1e-3,
    reinforce_d_model: int = 64,
    # PPO params
    n_timesteps: int = 100_000,
    ppo_d_model: int = 128,
    ppo_n_layers: int = 2,
    # Common
    gamma: float = 1.0,
    ent_coef: float = 0.01,
    device: str = 'cpu',
    verbose: bool = True,
) -> Tuple[AlphaPool, Dict]:
    """
    Train RL policy to generate factors and populate AlphaPool.
    
    This is the unified entry point called by run_alphagen_baseline()
    when --rl-method is 'reinforce' or 'ppo'.
    
    Args:
        method: 'reinforce' (Option B) or 'ppo' (Option C)
        train_data: dict of {feature: (n_dates, n_stocks) ndarray}
        train_fwd_ret: (n_dates, n_stocks) forward returns
        pool: AlphaPool to populate during training
        min_tokens: Minimum tokens before SEP allowed
        n_episodes: REINFORCE episodes (Option B)
        reinforce_lr: REINFORCE learning rate
        reinforce_d_model: REINFORCE embedding dimension
        n_timesteps: PPO timesteps (Option C)
        ppo_d_model: PPO LSTM dimension
        ppo_n_layers: PPO LSTM layers
        gamma: Discount factor
        ent_coef: Entropy coefficient
        device: 'cpu' or 'cuda'
        verbose: Print progress
    
    Returns:
        (pool, training_stats) — the populated AlphaPool and training metrics
    """
    # Create RL environment
    env = RLAlphaEnv(
        train_data=train_data,
        train_fwd_ret=train_fwd_ret,
        pool=pool,
        min_tokens=min_tokens,
    )
    
    if method == 'reinforce':
        stats = reinforce_train(
            env=env,
            n_episodes=n_episodes,
            lr=reinforce_lr,
            gamma=gamma,
            d_model=reinforce_d_model,
            ent_coef=ent_coef,
            verbose=verbose,
            device=device,
        )
    elif method == 'ppo':
        stats = ppo_train(
            env=env,
            n_timesteps=n_timesteps,
            n_layers=ppo_n_layers,
            d_model=ppo_d_model,
            gamma=gamma,
            ent_coef=ent_coef,
            verbose=verbose,
            device=device,
        )
    else:
        raise ValueError(f"Unknown RL method: {method}. Use 'reinforce' or 'ppo'.")
    
    return pool, stats
