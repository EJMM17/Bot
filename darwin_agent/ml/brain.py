"""ML Brain — Q-Learning with linear function approximation."""

import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from darwin_agent.ml.features import FEATURE_NAMES, N_FEATURES


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: Optional[np.ndarray]
    done: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


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

        self.n_actions = len(self.STRATEGIES) * len(self.SIZINGS)
        self.weights = np.random.randn(self.n_actions, n_features) * 0.01
        self.bias = np.zeros(self.n_actions)

        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.total_decisions = 0
        self.exploration_count = 0
        self.exploitation_count = 0
        self.regime_bonuses: Dict[str, np.ndarray] = {}

        # LLM-like retrieval memory (online prototypes + quality stats).
        self.pattern_memory: Dict[str, Dict[str, Any]] = {}
        self.pattern_lr = 0.08
        self.pattern_memory_max_regimes = 16

    def _sanitize_state(self, state: np.ndarray) -> np.ndarray:
        arr = np.array(state, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.n_features:
            fixed = np.zeros(self.n_features, dtype=np.float32)
            n = min(self.n_features, arr.shape[0])
            if n > 0:
                fixed[:n] = arr[:n]
            arr = fixed
        arr = np.nan_to_num(arr, nan=0.0, posinf=5.0, neginf=-5.0)
        return np.clip(arr, -5.0, 5.0)

    def predict_q(self, state: np.ndarray, regime: str = "unknown") -> np.ndarray:
        state = self._sanitize_state(state)
        q = self.weights @ state + self.bias
        if regime in self.regime_bonuses:
            q += self.regime_bonuses[regime]
        q += self._pattern_bias(state, regime)
        return q

    def _pattern_bias(self, state: np.ndarray, regime: str) -> np.ndarray:
        """Return additive Q-bias using learned success prototypes per action."""
        bias = np.zeros(self.n_actions)
        keys = [regime] if regime in self.pattern_memory else []
        if "global" in self.pattern_memory:
            keys.append("global")

        for key in keys:
            mem = self.pattern_memory.get(key, {})
            for action_str, stats in mem.items():
                try:
                    action_idx = int(action_str)
                except (TypeError, ValueError):
                    continue
                if not (0 <= action_idx < self.n_actions):
                    continue

                centroid = self._sanitize_state(np.array(stats.get("centroid", []), dtype=np.float32))
                dist = float(np.linalg.norm(state - centroid))
                similarity = 1.0 / (1.0 + dist)
                win_rate = float(stats.get("win_rate", 0.5))
                reward_avg = float(stats.get("avg_reward", 0.0))
                count = max(0.0, float(stats.get("count", 0)))
                support = min(1.0, count / 25.0)
                memory_signal = ((win_rate - 0.5) * 2.0 + reward_avg * 0.08) * similarity * support
                bias[action_idx] += memory_signal
        return np.clip(bias, -2.5, 2.5)

    def _prune_pattern_memory(self):
        if len(self.pattern_memory) <= self.pattern_memory_max_regimes:
            return
        removable = [k for k in self.pattern_memory.keys() if k != "global"]
        if not removable:
            return

        def score(regime_name: str) -> float:
            entries = self.pattern_memory.get(regime_name, {})
            return float(sum(float(v.get("count", 0)) for v in entries.values()))

        removable.sort(key=score)
        to_remove = len(self.pattern_memory) - self.pattern_memory_max_regimes
        for regime_name in removable[:to_remove]:
            self.pattern_memory.pop(regime_name, None)

    def _remember_pattern(self, state, action, reward, regime):
        """Online pattern learner (lightweight memory inspired by LLM-style retrieval)."""
        state = self._sanitize_state(state)
        keys = [regime, "global"] if regime and regime != "unknown" else ["global"]
        for key in keys:
            bucket = self.pattern_memory.setdefault(key, {})
            entry = bucket.setdefault(str(action), {
                "count": 0,
                "wins": 0,
                "losses": 0,
                "avg_reward": 0.0,
                "centroid": state.tolist(),
            })

            count = int(entry["count"]) + 1
            wins = int(entry.get("wins", 0)) + (1 if reward > 0 else 0)
            losses = int(entry.get("losses", 0)) + (1 if reward <= 0 else 0)
            avg_reward = float(entry["avg_reward"]) + (float(reward) - float(entry["avg_reward"])) / count

            old_centroid = self._sanitize_state(np.array(entry.get("centroid", state.tolist()), dtype=np.float32))
            centroid = (1 - self.pattern_lr) * old_centroid + self.pattern_lr * state

            entry["count"] = count
            entry["wins"] = wins
            entry["losses"] = losses
            entry["avg_reward"] = avg_reward
            entry["win_rate"] = wins / max(1, count)
            entry["centroid"] = centroid.tolist()

        self._prune_pattern_memory()

    def choose_action(self, state: np.ndarray, regime: str = "unknown",
                      health_pct: float = 1.0) -> Tuple[int, Action]:
        state = self._sanitize_state(state)
        self.total_decisions += 1
        eff_eps = min(0.5, self.epsilon * 2) if health_pct < 0.3 else self.epsilon

        if np.random.random() < eff_eps:
            self.exploration_count += 1
            if health_pct < 0.4:
                safe = self._safe_actions()
                idx = int(np.random.choice(safe))
            else:
                idx = int(np.random.randint(self.n_actions))
        else:
            self.exploitation_count += 1
            q = self.predict_q(state, regime)
            if health_pct < 0.5:
                for i in self._aggressive_actions():
                    q[i] *= health_pct
            idx = int(np.argmax(q))

        return idx, self._decode(idx)

    def learn(self, state, action, reward, next_state, done, regime="unknown"):
        state = self._sanitize_state(state)
        next_state = self._sanitize_state(next_state) if next_state is not None else None
        self.memory.append(Experience(state=state, action=action, reward=reward,
                                      next_state=next_state, done=done,
                                      metadata={"regime": regime}))
        self._update(state, action, reward, next_state, done, regime)
        self._remember_pattern(state, action, reward, regime)
        if len(self.memory) >= self.batch_size:
            self._replay()
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
            self.regime_bonuses[regime][action] += self.lr * 0.1 * td
            self.regime_bonuses[regime] = np.clip(self.regime_bonuses[regime], -3, 3)

    def _replay(self):
        idx = np.random.choice(len(self.memory), min(self.batch_size, len(self.memory)), replace=False)
        for i in idx:
            e = self.memory[i]
            self._update(e.state, e.action, e.reward, e.next_state, e.done, e.metadata.get("regime", "unknown"))

    def calculate_reward(self, pnl_pct, health_change, trade_duration_minutes, strategy_used):
        r = pnl_pct * (1.0 if pnl_pct > 0 else 1.5)
        r += health_change * 0.1
        if pnl_pct > 0 and trade_duration_minutes < 60:
            r += 0.5
        if strategy_used == "hold":
            r += 0.05
        r += 0.1
        return float(np.clip(r, -10, 10))

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

    def export_brain(self):
        return {
            "weights": self.weights.tolist(), "bias": self.bias.tolist(),
            "epsilon": self.epsilon,
            "regime_bonuses": {k: v.tolist() for k, v in self.regime_bonuses.items()},
            "pattern_memory": self.pattern_memory,
            "total_decisions": self.total_decisions,
            "exploration_count": self.exploration_count,
            "exploitation_count": self.exploitation_count,
            "n_features": self.n_features, "n_actions": self.n_actions,
        }

    def _sanitize_pattern_memory(self, data: Any) -> Dict[str, Dict[str, Any]]:
        if not isinstance(data, dict):
            return {}

        clean: Dict[str, Dict[str, Any]] = {}
        for regime_name, entries in data.items():
            if not isinstance(regime_name, str) or not isinstance(entries, dict):
                continue
            clean_entries: Dict[str, Any] = {}
            for action_key, stats in entries.items():
                try:
                    action_idx = int(action_key)
                except (TypeError, ValueError):
                    continue
                if not (0 <= action_idx < self.n_actions) or not isinstance(stats, dict):
                    continue
                centroid = self._sanitize_state(np.array(stats.get("centroid", []), dtype=np.float32)).tolist()
                count = max(0, int(stats.get("count", 0)))
                wins = max(0, int(stats.get("wins", 0)))
                losses = max(0, int(stats.get("losses", 0)))
                avg_reward = float(stats.get("avg_reward", 0.0))
                win_rate = float(stats.get("win_rate", wins / max(1, count)))
                clean_entries[str(action_idx)] = {
                    "count": count,
                    "wins": wins,
                    "losses": losses,
                    "avg_reward": avg_reward,
                    "win_rate": max(0.0, min(1.0, win_rate)),
                    "centroid": centroid,
                }
            if clean_entries:
                clean[regime_name] = clean_entries
        return clean

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
                mb = min(b.shape[0], self.bias.shape[0])
                self.bias[:mb] = b[:mb]
        if "regime_bonuses" in data:
            for r, b in data["regime_bonuses"].items():
                arr = np.array(b)
                if len(arr) == self.n_actions:
                    self.regime_bonuses[r] = arr
        if "pattern_memory" in data:
            self.pattern_memory = self._sanitize_pattern_memory(data["pattern_memory"])
        if "epsilon" in data:
            try:
                self.epsilon = min(0.3, float(data["epsilon"]) * 1.5)
            except (TypeError, ValueError):
                pass

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
            "pattern_memory_keys": list(self.pattern_memory.keys()),
            "weight_magnitude": round(float(np.mean(np.abs(self.weights))), 4),
        }

    def explain_action(self, state: np.ndarray, action_idx: int, regime: str = "unknown") -> str:
        """Human-readable explanation of why an action is preferred."""
        state = self._sanitize_state(state)
        if not (0 <= action_idx < self.n_actions):
            return f"Q-explain [{regime}] invalid-action={action_idx}"

        weights = self.weights[action_idx]
        contrib = weights * state
        top_idx = np.argsort(np.abs(contrib))[-3:][::-1]
        top_features = ", ".join([
            f"{FEATURE_NAMES[int(i)]}:{contrib[i]:+.3f}"
            if int(i) < len(FEATURE_NAMES) else f"f{int(i)}:{contrib[i]:+.3f}"
            for i in top_idx
        ])

        q_raw = float(self.weights[action_idx] @ state + self.bias[action_idx])
        q_mem = float(self._pattern_bias(state, regime)[action_idx])

        mem_txt = "no-memory"
        mem = self.pattern_memory.get(regime, {}).get(str(action_idx))
        if mem:
            mem_txt = (
                f"memory wr={mem.get('win_rate', 0.0):.2f} "
                f"r={mem.get('avg_reward', 0.0):+.2f} n={mem.get('count', 0)}"
            )
        return f"Q-explain[{regime}] a={action_idx} q={q_raw:+.3f} mem={q_mem:+.3f} top={top_features} | {mem_txt}"
