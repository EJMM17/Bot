"""Risk Manager — Every trade must pass through here. No exceptions."""

from datetime import datetime, timezone
from typing import Dict
from darwin_agent.markets.base import MarketSignal, OrderSide


class DailyStats:
    def __init__(self, date: str, peak_capital: float = 0):
        self.date = date
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.pnl = 0.0
        self.peak_capital = peak_capital


class RiskManager:
    def __init__(self, config):
        self.max_position_pct = config.max_position_pct
        self.max_open_positions = config.max_open_positions
        self.max_daily_trades = config.max_daily_trades
        self.max_daily_loss_pct = config.max_daily_loss_pct
        self.default_stop_loss_pct = config.default_stop_loss_pct
        self.default_take_profit_pct = config.default_take_profit_pct
        self.min_risk_reward_ratio = config.min_risk_reward_ratio
        self.daily_stats: Dict[str, DailyStats] = {}

    def _today(self, capital: float) -> DailyStats:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today not in self.daily_stats:
            self.daily_stats[today] = DailyStats(today, capital)
            # Clean old daily stats (keep last 7 days only)
            if len(self.daily_stats) > 7:
                oldest = sorted(self.daily_stats.keys())[:-7]
                for k in oldest:
                    del self.daily_stats[k]
        return self.daily_stats[today]

    def approve_trade(self, signal: MarketSignal, capital: float,
                      open_positions: int) -> tuple:
        today = self._today(capital)

        if today.trades >= self.max_daily_trades:
            return False, f"Daily limit reached ({self.max_daily_trades})"

        if open_positions >= self.max_open_positions:
            return False, f"Max positions ({self.max_open_positions})"

        if today.pnl < 0 and today.peak_capital > 0:
            loss_pct = abs(today.pnl) / today.peak_capital * 100
            if loss_pct >= self.max_daily_loss_pct:
                return False, f"Daily loss limit ({loss_pct:.1f}%)"

        if signal.entry_price > 0 and signal.stop_loss > 0 and signal.take_profit > 0:
            if signal.side == OrderSide.BUY:
                risk = signal.entry_price - signal.stop_loss
                reward = signal.take_profit - signal.entry_price
            else:
                risk = signal.stop_loss - signal.entry_price
                reward = signal.entry_price - signal.take_profit
            if risk > 0:
                rr = reward / risk
                if rr < self.min_risk_reward_ratio:
                    return False, f"R:R too low ({rr:.2f})"

        if signal.confidence < 0.55:
            return False, f"Confidence too low ({signal.confidence:.2f})"

        if capital < 15 and signal.confidence < 0.7:
            return False, f"Low capital: need conf>=0.70"

        return True, "Approved"

    def calculate_position_size(self, capital, entry_price, stop_loss):
        max_risk = capital * (self.max_position_pct / 100)
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            risk_per_unit = entry_price * (self.default_stop_loss_pct / 100)
        size = max_risk / risk_per_unit if risk_per_unit > 0 else 0
        max_affordable = (capital * 0.9) / entry_price if entry_price > 0 else 0
        return max(0, min(size, max_affordable))

    def record_trade_result(self, pnl, capital):
        today = self._today(capital)
        today.trades += 1
        today.pnl += pnl
        if pnl > 0:
            today.wins += 1
        else:
            today.losses += 1
        today.peak_capital = max(today.peak_capital, capital)

    def get_risk_report(self, capital):
        today = self._today(capital)
        return {
            "daily_trades": today.trades,
            "daily_limit": self.max_daily_trades,
            "daily_pnl": round(today.pnl, 2),
            "daily_loss_pct": round(abs(today.pnl) / today.peak_capital * 100, 2) if today.peak_capital > 0 and today.pnl < 0 else 0,
            "open_positions": 0,
            "max_positions": self.max_open_positions,
            "max_risk_per_trade": f"{self.max_position_pct}%",
            "capital_status": "CRITICAL" if capital < 15 else "LOW" if capital < 25 else "NORMAL",
        }
