"""Adaptive Strategy Selector — Thompson Sampling + learned confidence weights."""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from darwin_agent.ml.brain import QLearningBrain, Action
from darwin_agent.ml.features import FeatureEngineer, MarketFeatures
from darwin_agent.markets.base import Candle, MarketSignal, TimeFrame
from darwin_agent.strategies.base import STRATEGY_REGISTRY


@dataclass
class TradeDecision:
    should_trade: bool
    action: Optional[Action] = None
    signal: Optional[MarketSignal] = None
    features: Optional[MarketFeatures] = None
    brain_action_idx: int = 0
    reason: str = ""


@dataclass
class RegimeStats:
    trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    best_strategy: str = "unknown"
    strategy_results: Dict[str, Dict] = field(default_factory=dict)

    @property
    def win_rate(self):
        return self.wins / self.trades if self.trades > 0 else 0.0


@dataclass
class ThompsonArm:
    """Beta distribution parameters for Thompson Sampling."""
    alpha: float = 1.0  # success count (prior=1)
    beta: float = 1.0   # failure count (prior=1)
    total_reward: float = 0.0
    n_pulls: int = 0

    def sample(self) -> float:
        """Sample from Beta(alpha, beta) distribution."""
        return float(np.random.beta(self.alpha, self.beta))

    def update(self, reward: float):
        """Update with binary-ish outcome: reward > 0 = success."""
        self.n_pulls += 1
        self.total_reward += reward
        if reward > 0:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        # Decay old observations (prevents lock-in to stale data)
        decay = 0.995
        self.alpha = max(1.0, self.alpha * decay)
        self.beta = max(1.0, self.beta * decay)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta,
                "total_reward": self.total_reward, "n_pulls": self.n_pulls}

    @classmethod
    def from_dict(cls, data: dict) -> 'ThompsonArm':
        return cls(alpha=data.get("alpha", 1.0), beta=data.get("beta", 1.0),
                   total_reward=data.get("total_reward", 0.0),
                   n_pulls=data.get("n_pulls", 0))


class AdaptiveSelector:
    def __init__(self, brain: QLearningBrain):
        self.brain = brain
        self.feature_engineer = FeatureEngineer()
        self.strategies = dict(STRATEGY_REGISTRY)
        self.sizing_multipliers = {"conservative": 0.5, "normal": 1.0, "aggressive": 1.5}
        self.regime_stats: Dict[str, RegimeStats] = {}
        self.last_state: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self.last_regime: str = "unknown"
        self.pending_trade: Optional[Dict] = None

        # Thompson Sampling arms: regime -> strategy -> arm
        self.thompson_arms: Dict[str, Dict[str, ThompsonArm]] = {}

        # Learned confidence weights (meta-learning via gradient descent)
        # confidence = w_signal * signal.conf + w_brain * brain.conf + w_regime * regime_fit
        self._conf_weights = np.array([0.5, 0.25, 0.25])  # [signal, brain, regime]
        self._conf_lr = 0.01  # Learning rate for weight updates
        self._last_conf_components: Optional[np.ndarray] = None
        self._last_combined_conf: float = 0.0

        # Track pending trades with outcome binding (supports multiple)
        self._open_trades: Dict[str, Dict] = {}  # symbol -> trade info

    def decide(self, candles: List[Candle], symbol: str,
               timeframe: TimeFrame, health_pct: float) -> TradeDecision:
        features = self.feature_engineer.extract(candles, symbol)
        if features is None:
            return TradeDecision(should_trade=False, reason="Not enough data")

        action_idx, action = self.brain.choose_action(
            features.features, features.regime, health_pct)

        self.last_state = features.features.copy()
        self.last_action = action_idx
        self.last_regime = features.regime

        if action.strategy == "hold":
            return TradeDecision(
                should_trade=False, action=action, features=features,
                brain_action_idx=action_idx,
                reason=f"HOLD | {features.regime} | eps={self.brain.epsilon:.3f}")

        strategy = self.strategies.get(action.strategy)
        if strategy is None:
            return TradeDecision(should_trade=False, reason=f"Strategy '{action.strategy}' not found")

        signal = strategy.analyze(candles, symbol, timeframe)
        if signal is None:
            return TradeDecision(
                should_trade=False, action=action, features=features,
                brain_action_idx=action_idx,
                reason=f"{action.strategy}: no signal | {features.regime}")

        # Thompson Sampling for regime-strategy fitness (replaces hardcoded fit_map)
        fit = self._thompson_regime_fit(features.regime, action.strategy)

        # Learned confidence weighting (replaces hardcoded 0.6/0.2/0.2)
        components = np.array([signal.confidence, action.confidence, fit])
        combined = float(np.dot(self._conf_weights, components))
        signal.confidence = max(0.0, min(1.0, combined))

        # Store for meta-learning gradient update
        self._last_conf_components = components
        self._last_combined_conf = combined

        self.pending_trade = {
            "strategy": action.strategy, "sizing": action.sizing,
            "regime": features.regime, "entry_time": datetime.now(timezone.utc),
            "state": features.features.copy(), "action_idx": action_idx,
            "symbol": symbol,
        }
        # Outcome binding: track by symbol
        self._open_trades[symbol] = dict(self.pending_trade)

        return TradeDecision(
            should_trade=True, action=action, signal=signal, features=features,
            brain_action_idx=action_idx,
            reason=f"Brain: {action.strategy}/{action.sizing} | {features.regime} | conf:{signal.confidence:.2f} | fit:{fit:.2f}")

    def report_result(self, pnl_pct: float, health_change: float,
                      new_candles=None, symbol=""):
        if self.last_state is None or self.last_action is None:
            return

        next_state = None
        if new_candles and symbol:
            nf = self.feature_engineer.extract(new_candles, symbol)
            if nf:
                next_state = nf.features

        dur = 0
        strat = "unknown"
        regime = self.last_regime

        # Outcome binding: use trade info from the specific symbol
        trade_info = self._open_trades.pop(symbol, None) or self.pending_trade
        if trade_info:
            dur = (datetime.now(timezone.utc) - trade_info["entry_time"]).total_seconds() / 60
            strat = trade_info["strategy"]
            regime = trade_info.get("regime", self.last_regime)

        reward = self.brain.calculate_reward(pnl_pct, health_change, dur, strat)
        self.brain.learn(self.last_state, self.last_action, reward, next_state, False, regime)

        # Update Thompson Sampling arm
        self._update_thompson(regime, strat, pnl_pct)

        # Update learned confidence weights via gradient descent
        self._update_conf_weights(pnl_pct)

        self._update_regime(regime, strat, pnl_pct)
        self.pending_trade = None

    def report_hold_result(self, candles, symbol):
        if self.last_state is None or self.last_action is None:
            return
        nf = self.feature_engineer.extract(candles, symbol)
        ns = nf.features if nf else None
        reward = self.brain.calculate_reward(0, 0, 0, "hold")
        self.brain.learn(self.last_state, self.last_action, reward, ns, False, self.last_regime)

    def _thompson_regime_fit(self, regime: str, strategy: str) -> float:
        """Thompson Sampling: sample from learned Beta distribution for regime-strategy pair."""
        if regime not in self.thompson_arms:
            self.thompson_arms[regime] = {}

        arms = self.thompson_arms[regime]
        if strategy not in arms:
            # Initialize with weak prior based on common sense (but learnable)
            prior = self._initial_prior(regime, strategy)
            arms[strategy] = ThompsonArm(alpha=prior, beta=max(1.0, 2.0 - prior))

        return arms[strategy].sample()

    def _initial_prior(self, regime: str, strategy: str) -> float:
        """Weak prior for Thompson Sampling initialization. Will be quickly overridden by data."""
        priors = {
            "trending_volatile": {"momentum": 1.5, "mean_reversion": 0.7, "scalping": 1.0, "breakout": 1.4},
            "trending_calm": {"momentum": 1.4, "mean_reversion": 0.8, "scalping": 0.9, "breakout": 1.3},
            "choppy": {"momentum": 0.6, "mean_reversion": 0.8, "scalping": 0.8, "breakout": 0.6},
            "ranging_tight": {"momentum": 0.7, "mean_reversion": 1.5, "scalping": 1.3, "breakout": 0.8},
            "ranging_normal": {"momentum": 0.9, "mean_reversion": 1.3, "scalping": 1.1, "breakout": 1.0},
        }
        return priors.get(regime, {}).get(strategy, 1.0)

    def _update_thompson(self, regime: str, strategy: str, pnl_pct: float):
        """Update Thompson Sampling arm with trade result."""
        if regime not in self.thompson_arms:
            self.thompson_arms[regime] = {}
        if strategy not in self.thompson_arms[regime]:
            prior = self._initial_prior(regime, strategy)
            self.thompson_arms[regime][strategy] = ThompsonArm(alpha=prior, beta=max(1.0, 2.0 - prior))
        self.thompson_arms[regime][strategy].update(pnl_pct)

    def _update_conf_weights(self, pnl_pct: float):
        """Meta-learning: adjust confidence weights based on outcome."""
        if self._last_conf_components is None:
            return
        # Gradient: if trade was profitable, increase weight on components that were high
        # if trade lost, decrease weight on components that were high
        gradient = self._last_conf_components * np.sign(pnl_pct) * abs(pnl_pct) * 0.01
        self._conf_weights += self._conf_lr * gradient
        # Keep weights positive and normalized (sum to 1)
        self._conf_weights = np.clip(self._conf_weights, 0.05, 0.9)
        self._conf_weights /= self._conf_weights.sum()
        self._last_conf_components = None

    def _update_regime(self, regime, strategy, pnl_pct):
        if regime not in self.regime_stats:
            self.regime_stats[regime] = RegimeStats()
        s = self.regime_stats[regime]
        s.trades += 1
        s.total_pnl += pnl_pct
        if pnl_pct > 0:
            s.wins += 1
        if strategy not in s.strategy_results:
            s.strategy_results[strategy] = {"trades": 0, "wins": 0, "total_pnl": 0.0}
        sr = s.strategy_results[strategy]
        sr["trades"] += 1
        sr["total_pnl"] += pnl_pct
        if pnl_pct > 0:
            sr["wins"] += 1
        sr["win_rate"] = sr["wins"] / sr["trades"] if sr["trades"] > 0 else 0
        best = max(s.strategy_results.items(), key=lambda x: x[1].get("total_pnl", 0))
        s.best_strategy = best[0]

    def get_playbook(self):
        pb = {}
        for regime, s in self.regime_stats.items():
            thompson_info = {}
            if regime in self.thompson_arms:
                for strat, arm in self.thompson_arms[regime].items():
                    thompson_info[strat] = {"mean": round(arm.mean, 3), "pulls": arm.n_pulls}

            pb[regime] = {
                "best_strategy": s.best_strategy,
                "trades": s.trades, "win_rate": round(s.win_rate, 3),
                "total_pnl": round(s.total_pnl, 2),
                "confidence_weights": {
                    "signal": round(float(self._conf_weights[0]), 3),
                    "brain": round(float(self._conf_weights[1]), 3),
                    "regime": round(float(self._conf_weights[2]), 3),
                },
                "thompson_sampling": thompson_info,
                "strategy_breakdown": {
                    k: {"trades": v["trades"], "win_rate": round(v.get("win_rate", 0), 3),
                         "pnl": round(v["total_pnl"], 2)}
                    for k, v in s.strategy_results.items()
                }
            }
        return pb

    def export_for_dna(self):
        thompson_data = {}
        for regime, arms in self.thompson_arms.items():
            thompson_data[regime] = {strat: arm.to_dict() for strat, arm in arms.items()}

        return {
            "brain": self.brain.export_brain(),
            "regime_stats": {
                r: {"trades": s.trades, "wins": s.wins, "total_pnl": s.total_pnl,
                     "best_strategy": s.best_strategy, "strategy_results": s.strategy_results}
                for r, s in self.regime_stats.items()
            },
            "thompson_arms": thompson_data,
            "conf_weights": self._conf_weights.tolist(),
        }

    def import_from_dna(self, data, mutation_rate=0.05):
        if "brain" in data:
            self.brain.import_brain(data["brain"], mutation_rate)
        if "regime_stats" in data:
            for r, d in data["regime_stats"].items():
                self.regime_stats[r] = RegimeStats(
                    trades=d.get("trades", 0), wins=d.get("wins", 0),
                    total_pnl=d.get("total_pnl", 0),
                    best_strategy=d.get("best_strategy", "unknown"),
                    strategy_results=d.get("strategy_results", {}),
                )
        if "thompson_arms" in data:
            for regime, arms_data in data["thompson_arms"].items():
                self.thompson_arms[regime] = {
                    strat: ThompsonArm.from_dict(arm_data)
                    for strat, arm_data in arms_data.items()
                }
        if "conf_weights" in data:
            w = np.array(data["conf_weights"])
            if len(w) == 3:
                # Inherit with small mutation
                self._conf_weights = w + np.random.randn(3) * mutation_rate * 0.1
                self._conf_weights = np.clip(self._conf_weights, 0.05, 0.9)
                self._conf_weights /= self._conf_weights.sum()
