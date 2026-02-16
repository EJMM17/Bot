"""Adaptive Strategy Selector — ML Brain decides which strategy to use."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from darwin_agent.markets.base import Candle, MarketSignal, TimeFrame
from darwin_agent.ml.brain import Action, QLearningBrain
from darwin_agent.ml.features import FeatureEngineer, MarketFeatures
from darwin_agent.ml.outcome_model import OnlineTradeOutcomeModel
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

        # Extra trading ML layer: online per-(regime,strategy) win-prob model.
        self.outcome_models: Dict[str, OnlineTradeOutcomeModel] = {}

    def _model_key(self, regime: str, strategy: str) -> str:
        return f"{regime}::{strategy}"

    def _get_outcome_model(self, regime: str, strategy: str) -> OnlineTradeOutcomeModel:
        key = self._model_key(regime, strategy)
        if key not in self.outcome_models:
            self.outcome_models[key] = OnlineTradeOutcomeModel(n_features=self.brain.n_features)
        return self.outcome_models[key]

    def _build_ai_reason(self, features: MarketFeatures, action_idx: int,
                         action: Action, signal: MarketSignal, ml_win_prob: float) -> str:
        explanation = self.brain.explain_action(features.features, action_idx, features.regime)
        trend = features.features[0] if len(features.features) > 0 else 0.0
        momentum = features.features[6] if len(features.features) > 6 else 0.0
        volatility = features.features[13] if len(features.features) > 13 else 0.0
        regime_hint = f"regime={features.regime} trend={trend:+.2f} mom={momentum:+.2f} vol={volatility:+.2f}"
        return (
            f"AI decision: {action.strategy}/{action.sizing} conf={signal.confidence:.2f} "
            f"ml_win={ml_win_prob:.2f} | {regime_hint} | {explanation}"
        )

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

        fit = self._regime_fit(features.regime, action.strategy)
        outcome_model = self._get_outcome_model(features.regime, action.strategy)
        ml_win_prob = outcome_model.predict_proba(features.features)

        signal.confidence = (
            signal.confidence * 0.45
            + action.confidence * 0.15
            + fit * 0.15
            + ml_win_prob * 0.25
        )
        signal.confidence = float(np.clip(np.nan_to_num(signal.confidence, nan=0.0), 0.0, 1.0))

        self.pending_trade = {
            "strategy": action.strategy,
            "sizing": action.sizing,
            "regime": features.regime,
            "entry_time": datetime.now(timezone.utc),
            "state": features.features.copy(),
            "action_idx": action_idx,
            "model_key": self._model_key(features.regime, action.strategy),
            "ml_win_prob": ml_win_prob,
        }

        reason = self._build_ai_reason(features, action_idx, action, signal, ml_win_prob)

        return TradeDecision(
            should_trade=True, action=action, signal=signal, features=features,
            brain_action_idx=action_idx, reason=reason)

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
        pending_state = self.last_state
        if self.pending_trade:
            dur = (datetime.now(timezone.utc) - self.pending_trade["entry_time"]).total_seconds() / 60
            strat = self.pending_trade["strategy"]
            regime = self.pending_trade.get("regime", regime)
            pending_state = self.pending_trade.get("state", pending_state)

        reward = self.brain.calculate_reward(pnl_pct, health_change, dur, strat)
        done = strat == "unknown"
        self.brain.learn(self.last_state, self.last_action, reward, next_state, done, self.last_regime)

        if strat != "unknown":
            self._update_regime(regime, strat, pnl_pct)
            try:
                model = self._get_outcome_model(regime, strat)
                label = 1 if pnl_pct > 0 else 0
                weight = min(2.0, max(0.25, abs(pnl_pct) + 0.25))
                model.update(pending_state, label, sample_weight=weight)
            except Exception:
                # Keep selector resilient even if ML model update fails.
                pass

        self.pending_trade = None

    def report_hold_result(self, candles, symbol):
        if self.last_state is None or self.last_action is None:
            return
        nf = self.feature_engineer.extract(candles, symbol)
        ns = nf.features if nf else None
        reward = self.brain.calculate_reward(0, 0, 0, "hold")
        self.brain.learn(self.last_state, self.last_action, reward, ns, False, self.last_regime)

    def _regime_fit(self, regime, strategy):
        fit_map = {
            "trending_volatile": {"momentum": 0.9, "mean_reversion": 0.2, "scalping": 0.5, "breakout": 0.8},
            "trending_calm": {"momentum": 0.8, "mean_reversion": 0.3, "scalping": 0.4, "breakout": 0.7},
            "choppy": {"momentum": 0.1, "mean_reversion": 0.3, "scalping": 0.3, "breakout": 0.2},
            "ranging_tight": {"momentum": 0.2, "mean_reversion": 0.9, "scalping": 0.7, "breakout": 0.3},
            "ranging_normal": {"momentum": 0.4, "mean_reversion": 0.7, "scalping": 0.6, "breakout": 0.5},
        }
        base = fit_map.get(regime, {}).get(strategy, 0.5)
        if regime in self.regime_stats:
            s = self.regime_stats[regime]
            if strategy in s.strategy_results and s.strategy_results[strategy].get("trades", 0) >= 5:
                learned = s.strategy_results[strategy].get("win_rate", 0.5)
                return base * 0.4 + learned * 0.6
        return base

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
            pb[regime] = {
                "best_strategy": s.best_strategy,
                "trades": s.trades,
                "win_rate": round(s.win_rate, 3),
                "total_pnl": round(s.total_pnl, 2),
                "strategy_breakdown": {
                    k: {
                        "trades": v["trades"],
                        "win_rate": round(v.get("win_rate", 0), 3),
                        "pnl": round(v["total_pnl"], 2),
                    }
                    for k, v in s.strategy_results.items()
                }
            }
        return pb

    def export_for_dna(self):
        return {
            "brain": self.brain.export_brain(),
            "regime_stats": {
                r: {
                    "trades": s.trades,
                    "wins": s.wins,
                    "total_pnl": s.total_pnl,
                    "best_strategy": s.best_strategy,
                    "strategy_results": s.strategy_results,
                }
                for r, s in self.regime_stats.items()
            },
            "outcome_models": {
                k: v.export() for k, v in self.outcome_models.items()
            },
        }

    def import_from_dna(self, data, mutation_rate=0.05):
        if "brain" in data:
            self.brain.import_brain(data["brain"], mutation_rate)
        if "regime_stats" in data:
            for r, d in data["regime_stats"].items():
                self.regime_stats[r] = RegimeStats(
                    trades=d.get("trades", 0),
                    wins=d.get("wins", 0),
                    total_pnl=d.get("total_pnl", 0),
                    best_strategy=d.get("best_strategy", "unknown"),
                    strategy_results=d.get("strategy_results", {}),
                )
        if "outcome_models" in data and isinstance(data["outcome_models"], dict):
            for key, payload in data["outcome_models"].items():
                if not isinstance(key, str) or not isinstance(payload, dict):
                    continue
                self.outcome_models[key] = OnlineTradeOutcomeModel.from_dict(
                    payload, n_features=self.brain.n_features
                )
