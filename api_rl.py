from __future__ import annotations

"""
RL API Layer for dashboard integration.

Provides:
- load_rl(model_path="models/demo_rl.pth") -> (env, agent)
- run_rl_step(agent, state) -> {action, reward, avg_wait_time}
- simulate_episode(agent, env, max_steps) -> pandas.DataFrame
- make_dummy_episode(max_steps) -> pandas.DataFrame

This module performs no heavy work at import time.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random
import numpy as np

import pandas as pd

try:
    import torch
except ImportError:
    torch = None

# Print reminder after import
print("⚠️ Reminder: Place your demo RL weights at Traffic-simulation-rl/models/demo_rl.pth or pass a custom path to load_rl().")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _models_dir() -> Path:
    return _repo_root() / "models"


class _DummyEnv:
    """Lightweight fallback environment for demo without SUMO.

    Exposes reset() and step(action) with a simple synthetic process.
    """

    def __init__(self, action_size: int = 4):
        self.t = 0
        self.max_steps = 200
        self.action_size = action_size

    def reset(self):
        self.t = 0
        # 12-dim zero state (matches DQNAgent default)
        return [0.0] * 12

    def step(self, action: int):
        self.t += 1
        # synthetic metrics
        reward = 1.0 - 0.01 * abs((action or 0) - (self.t % self.action_size))
        avg_wait_time = max(0.0, 10.0 - 0.05 * self.t)
        queue_length = max(0, 50 - self.t)
        info = {
            "avg_wait_time": avg_wait_time,
            "queue_length": queue_length,
            "junction_id": "demo_junction",
            "waiting_time": avg_wait_time,
            "throughput": None,
        }
        done = self.t >= self.max_steps
        next_state = [0.0] * 12
        return next_state, reward, done, info


def load_rl(model_path: str = "models/demo_rl.pth") -> Tuple[Any, Any]:
    """
    Loads environment and agent with pretrained weights.
    Returns (env, agent). NEVER starts training here.

    If SUMO/env creation fails, returns a lightweight fallback env to keep the
    dashboard demo working. If weights are missing, raises FileNotFoundError
    with clear instructions.
    """
    repo_root = _repo_root()
    weights_path = (repo_root / model_path).resolve()

    # Import agent lazily
    try:
        from agents.dqn_agent import DQNAgent  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Could not import DQNAgent from agents.dqn_agent. Adjust api_rl.py imports to your layout."
        ) from exc

    # Try to create the real environment; fall back to Dummy if SUMO not available
    env: Any
    try:
        from envs.traffic_env import TrafficEnv  # type: ignore
        try:
            env = TrafficEnv(max_steps=200)
        except Exception:
            env = _DummyEnv(action_size=4)
    except Exception:
        env = _DummyEnv(action_size=4)

    # Determine action size if available
    action_size: Optional[int] = None
    try:
        action_space = getattr(env, "action_space", None)
        action_size = getattr(action_space, "n", None)
    except Exception:
        action_size = None

    agent = DQNAgent(action_size=action_size or 4)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Pretrained weights not found at {model_path}. "
            "Place demo_rl.pth here or pass a custom path, or enable 'Use dummy agent' in the dashboard."
        )

    if not hasattr(agent, "load"):
        raise AttributeError(
            "Agent is missing a .load(path) method. Implement DQNAgent.load(path) to restore weights."
        )

    # Force CPU loading by temporarily changing device
    if torch is not None:
        original_device = agent.device
        agent.device = torch.device("cpu")
        agent.load(str(weights_path))
        agent.device = original_device
    else:
        agent.load(str(weights_path))
    return env, agent


def run_rl_step(agent: Any, state: List[float]) -> Dict[str, Optional[float]]:
    """
    Returns a small dict for the dashboard:
    { "action": int, "reward": float, "avg_wait_time": float }
    """
    if not hasattr(agent, "act"):
        raise AttributeError("Agent must implement act(state) -> action (int).")
    action = agent.act(state)
    return {"action": int(action), "reward": None, "avg_wait_time": None}


def simulate_episode(agent: Any, env: Any, max_steps: int = 100) -> pd.DataFrame:
    """
    Returns a pandas DataFrame with schema:
    time, action, reward, avg_wait_time, queue_length, phase, junction_id, waiting_time, throughput
    """
    records: List[Dict[str, Any]] = []
    # reset supports Gym and Gymnasium signatures
    reset_out = env.reset()
    state = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    for t in range(max_steps):
        action = agent.act(state) if hasattr(agent, "act") else 0
        step_out = env.step(action)

        # Normalize step outputs
        if not isinstance(step_out, tuple):
            raise ValueError("env.step(action) must return a tuple.")
        if len(step_out) == 4:
            next_state, reward, done, info = step_out
        elif len(step_out) == 5:
            next_state, reward, terminated, truncated, info = step_out
            done = bool(terminated) or bool(truncated)
        else:
            raise ValueError("Unsupported env.step signature.")

        avg_wait_time = None
        queue_length = None
        junction_id = None
        waiting_time = None
        throughput = None

        if isinstance(info, dict):
            avg_wait_time = info.get("avg_wait_time")
            queue_length = info.get("queue_length")
            junction_id = info.get("junction_id")
            waiting_time = info.get("waiting_time")
            throughput = info.get("throughput")

        records.append(
            {
                "time": t,
                "action": int(action),
                "phase": int(action),
                "reward": float(reward) if reward is not None else None,
                "avg_wait_time": float(avg_wait_time) if avg_wait_time is not None else None,
                "queue_length": int(queue_length) if queue_length is not None else None,
                "junction_id": junction_id,
                "waiting_time": float(waiting_time) if waiting_time is not None else None,
                "throughput": float(throughput) if throughput is not None else None,
            }
        )

        state = next_state
        if done:
            break

    df = pd.DataFrame.from_records(records)
    required_cols = [
        "time",
        "action",
        "reward",
        "avg_wait_time",
        "queue_length",
        "phase",
        "junction_id",
        "waiting_time",
        "throughput",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    return df[required_cols]


def make_dummy_episode(max_steps: int = 100) -> pd.DataFrame:
    """
    Generate a dummy episode with realistic-looking data for dashboard demo.
    
    Returns DataFrame with EXACTLY these columns:
    time, action, reward, avg_wait_time, queue_length
    
    Args:
        max_steps: Number of simulation steps
        
    Returns:
        pd.DataFrame with dummy traffic simulation data
    """
    records: List[Dict[str, Any]] = []
    
    # Initialize with some randomness for realistic variation
    np.random.seed(42)  # For reproducible dummy data
    
    for t in range(max_steps):
        # Generate realistic action pattern (0-3)
        action = random.randint(0, 3)
        
        # Generate reward with some trend and noise
        base_reward = 0.5 + 0.3 * np.sin(t * 0.1)  # Oscillating base
        reward = base_reward + np.random.normal(0, 0.1)
        
        # Generate wait time with realistic traffic patterns
        # Start high, decrease over time, with some noise
        base_wait = max(0.1, 15.0 - t * 0.1 + 2 * np.sin(t * 0.05))
        avg_wait_time = base_wait + np.random.normal(0, 1.0)
        avg_wait_time = max(0.0, avg_wait_time)
        
        # Generate queue length with realistic patterns
        # Peak early, then decrease
        base_queue = max(0, 40 - t * 0.3 + 5 * np.sin(t * 0.08))
        queue_length = int(base_queue + np.random.normal(0, 2))
        queue_length = max(0, queue_length)
        
        records.append({
            "time": t,
            "action": action,
            "reward": round(reward, 3),
            "avg_wait_time": round(avg_wait_time, 2),
            "queue_length": queue_length,
        })
    
    df = pd.DataFrame.from_records(records)
    
    # Ensure exact column order and types
    required_cols = ["time", "action", "reward", "avg_wait_time", "queue_length"]
    return df[required_cols]


