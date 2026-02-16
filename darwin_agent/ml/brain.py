"""ML Brain — Q-Learning with linear function approximation + Prioritized Replay."""

import numpy as np
import json
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from darwin_agent.ml.features import N_FEATURES


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: Optional[np.ndarray]
    done: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    td_error: float = 1.0  # For prioritized replay


@dataclass
class Action:
    strategy: str
    sizing: str
    direction_bias: str
    confidence: float


class QLearningBrain:
    STRATEGIES = ["momentum", "mean_reversion", "scalping", "breakout", "hold"]
    SIZINGS = ["conservative", "normal", "aggressive"]

    def __init__(self, n_features: int = N_FEATURES, learning_rate: float = 0.01,
                 gamma: float = 0.95, epsilon: float = 0.3,
                 epsilon_decay: float = 0.9995, epsilon_min: float = 0.05):
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.n_actions = len(self.STRATEGIES) * len(self.SIZINGS)  # 15
        self.weights = np.random.randn(self.n_actions, n_features) * 0.01
        self.bias = np.zeros(self.n_actions)

        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.total_decisions = 0
        self.exploration_count = 0
        self.exploitation_count = 0
        self.regime_bonuses: Dict[str, np.ndarray] = {}

        # Tracking for meta-learning: which actions performed well recently
        self._action_rewards: Dict[int, deque] = {i: deque(maxlen=50) for i in range(self.n_actions)}

    def predict_q(self, state: np.ndarray, regime: str = "unknown") -> np.ndarray:
        q = self.weights @ state + self.bias
        if regime in self.regime_bonuses:
            q += self.regime_bonuses[regime]
        return q

    def choose_action(self, state: np.ndarray, regime: str = "unknown",
                      health_pct: float = 1.0) -> Tuple[int, Action]:
        self.total_decisions += 1

        # FIX: When dying, explore LESS (stick to safe strategies), not more
        if health_pct < 0.3:
            eff_eps = max(self.epsilon_min, self.epsilon * 0.5)
        elif health_pct < 0.5:
            eff_eps = max(self.epsilon_min, self.epsilon * 0.75)
        else:
            eff_eps = self.epsilon

        if np.random.random() < eff_eps:
            self.exploration_count += 1
            if health_pct < 0.4:
                safe = self._safe_actions()
                idx = np.random.choice(safe)
            else:
                idx = np.random.randint(self.n_actions)
        else:
            self.exploitation_count += 1
            q = self.predict_q(state, regime)
            if health_pct < 0.5:
                # Penalize aggressive, BOOST conservative
                for i in self._aggressive_actions():
                    q[i] *= health_pct
                for i in self._conservative_actions():
                    q[i] *= (1.0 + (1.0 - health_pct) * 0.5)  # Boost safe actions
            idx = int(np.argmax(q))

        return idx, self._decode(idx)

    def learn(self, state, action, reward, next_state, done, regime="unknown"):
        # Calculate TD error for prioritized replay
        cur_q = self.weights[action] @ state + self.bias[action]
        if done or next_state is None:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.predict_q(next_state, regime))
        td_error = abs(target - cur_q)

        exp = Experience(state=state, action=action, reward=reward,
                         next_state=next_state, done=done,
                         metadata={"regime": regime}, td_error=td_error)
        self.memory.append(exp)

        # Track action performance
        self._action_rewards[action].append(reward)

        self._update(state, action, reward, next_state, done, regime)
        if len(self.memory) >= self.batch_size:
            self._prioritized_replay()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _update(self, state, action, reward, next_state, done, regime):
        cur_q = self.weights[action] @ state + self.bias[action]
        target = reward if (done or next_state is None) else reward + self.gamma * np.max(self.predict_q(next_state, regime))
        td = target - cur_q
        self.weights[action] += self.lr * td * state
        self.bias[action] += self.lr * td
        self.weights = np.clip(self.weights, -10, 10)
        self.bias = np.clip(self.bias, -5, 5)
        if regime != "unknown":
            if regime not in self.regime_bonuses:
                self.regime_bonuses[regime] = np.zeros(self.n_actions)
            # Exponential moving average for regime bonuses (with decay)
            decay = 0.99
            self.regime_bonuses[regime] *= decay
            self.regime_bonuses[regime][action] += self.lr * 0.1 * td
            self.regime_bonuses[regime] = np.clip(self.regime_bonuses[regime], -3, 3)

    def _prioritized_replay(self):
        """Prioritized Experience Replay: higher TD-error = more likely to be sampled."""
        n = len(self.memory)
        batch_n = min(self.batch_size, n)

        # Compute priorities (|TD| + small epsilon for stability)
        priorities = np.array([abs(e.td_error) + 0.01 for e in self.memory])
        probs = priorities / priorities.sum()

        idx = np.random.choice(n, batch_n, replace=False, p=probs)
        for i in idx:
            e = self.memory[i]
            old_q = self.weights[e.action] @ e.state + self.bias[e.action]
            regime = e.metadata.get("regime", "unknown")
            if e.done or e.next_state is None:
                target = e.reward
            else:
                target = e.reward + self.gamma * np.max(self.predict_q(e.next_state, regime))
            # Update TD error for future sampling
            e.td_error = abs(target - old_q)
            self._update(e.state, e.action, e.reward, e.next_state, e.done, regime)

    def calculate_reward(self, pnl_pct, health_change, trade_duration_minutes, strategy_used):
        """Improved reward function with normalized units."""
        # Normalize PnL: clip to reasonable range, asymmetric penalty for losses
        r = pnl_pct * (1.0 if pnl_pct > 0 else 1.5)

        # Health change normalized (health is 0-100, reward contribution scaled)
        r += health_change * 0.05

        # Speed bonus: reward efficient wins, no penalty for slow ones
        if pnl_pct > 0 and trade_duration_minutes < 60:
            r += 0.3
        elif pnl_pct > 0 and trade_duration_minutes < 180:
            r += 0.1

        # Hold gets small positive reward (avoiding bad trades is good)
        if strategy_used == "hold":
            r += 0.05

        # Survival bonus (agent lived another cycle)
        r += 0.05

        # Risk-adjusted: big wins from aggressive sizing should be tempered
        # (this will be modulated externally by the selector)
        return float(np.clip(r, -10, 10))

    def get_action_performance(self) -> Dict[str, float]:
        """Return average reward per action for meta-learning."""
        perf = {}
        for action_idx, rewards in self._action_rewards.items():
            if len(rewards) >= 3:
                action = self._decode(action_idx)
                key = f"{action.strategy}/{action.sizing}"
                perf[key] = float(np.mean(list(rewards)))
        return perf

    def _decode(self, idx: int) -> Action:
        ns = len(self.SIZINGS)
        si = idx // ns
        zi = idx % ns
        strat = self.STRATEGIES[min(si, len(self.STRATEGIES) - 1)]
        sizing = self.SIZINGS[zi]
        conf = 1.0 / (1.0 + np.exp(-self.bias[idx]))
        return Action(strategy=strat, sizing=sizing, direction_bias="neutral", confidence=float(conf))

    def _safe_actions(self):
        ns = len(self.SIZINGS)
        safe = [i for i in range(self.n_actions)
                if self.STRATEGIES[i // ns] == "hold" or self.SIZINGS[i % ns] == "conservative"]
        return safe or list(range(self.n_actions))

    def _aggressive_actions(self):
        ns = len(self.SIZINGS)
        return [i for i in range(self.n_actions) if self.SIZINGS[i % ns] == "aggressive"]

    def _conservative_actions(self):
        ns = len(self.SIZINGS)
        return [i for i in range(self.n_actions) if self.SIZINGS[i % ns] == "conservative"]

    def export_brain(self):
        return {
            "weights": self.weights.tolist(), "bias": self.bias.tolist(),
            "epsilon": self.epsilon,
            "regime_bonuses": {k: v.tolist() for k, v in self.regime_bonuses.items()},
            "total_decisions": self.total_decisions,
            "exploration_count": self.exploration_count,
            "exploitation_count": self.exploitation_count,
            "n_features": self.n_features, "n_actions": self.n_actions,
            "action_rewards": {
                str(k): list(v) for k, v in self._action_rewards.items() if len(v) > 0
            },
        }

    def import_brain(self, data, mutation_rate=0.05):
        if "weights" in data:
            w = np.array(data["weights"])
            b = np.array(data["bias"])
            if w.shape == self.weights.shape:
                self.weights = w + np.random.randn(*w.shape) * mutation_rate
                self.bias = b + np.random.randn(*b.shape) * mutation_rate
            else:
                ma = min(w.shape[0], self.weights.shape[0])
                mf = min(w.shape[1], self.weights.shape[1])
                self.weights[:ma, :mf] = w[:ma, :mf]
        if "regime_bonuses" in data:
            for r, b in data["regime_bonuses"].items():
                arr = np.array(b)
                if len(arr) == self.n_actions:
                    self.regime_bonuses[r] = arr
        if "epsilon" in data:
            self.epsilon = min(0.3, data["epsilon"] * 1.5)
        if "action_rewards" in data:
            for k, v in data["action_rewards"].items():
                idx = int(k)
                if 0 <= idx < self.n_actions:
                    self._action_rewards[idx] = deque(v[-20:], maxlen=50)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.export_brain(), f)

    def load(self, path, as_inheritance=False):
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.import_brain(data, mutation_rate=0.05 if as_inheritance else 0.0)

    def get_stats(self):
        return {
            "total_decisions": self.total_decisions,
            "exploration_rate": round(self.exploration_count / max(1, self.total_decisions), 4),
            "exploitation_rate": round(self.exploitation_count / max(1, self.total_decisions), 4),
            "current_epsilon": round(self.epsilon, 4),
            "memory_size": len(self.memory),
            "regimes_learned": list(self.regime_bonuses.keys()),
            "weight_magnitude": round(float(np.mean(np.abs(self.weights))), 4),
        }
