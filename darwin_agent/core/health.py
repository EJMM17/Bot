"""Health System — HP tracking. When HP hits 0, the agent dies permanently."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
from enum import Enum


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    WOUNDED = "wounded"
    CRITICAL = "critical"
    DEAD = "dead"


@dataclass
class HealthEvent:
    timestamp: datetime
    hp_before: float
    hp_after: float
    change: float
    reason: str
    event_type: str


@dataclass
class HealthSystem:
    max_hp: float = 100.0
    current_hp: float = 100.0
    is_alive: bool = True
    cause_of_death: Optional[str] = None
    death_time: Optional[datetime] = None

    peak_capital: float = 50.0
    current_capital: float = 50.0
    win_streak: int = 0
    loss_streak: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    history: List[HealthEvent] = field(default_factory=list)

    instant_death_capital: float = 10.0
    critical_capital: float = 25.0
    critical_hp_penalty: float = 50.0
    max_drawdown_pct: float = 20.0
    drawdown_hp_penalty: float = 30.0

    def get_status(self) -> HealthStatus:
        if not self.is_alive or self.current_hp <= 0:
            return HealthStatus.DEAD
        if self.current_hp > 70:
            return HealthStatus.HEALTHY
        if self.current_hp > 40:
            return HealthStatus.WOUNDED
        return HealthStatus.CRITICAL

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def current_drawdown_pct(self) -> float:
        if self.peak_capital <= 0:
            return 0.0
        return ((self.peak_capital - self.current_capital) / self.peak_capital) * 100

    def _apply_change(self, change: float, reason: str, event_type: str) -> float:
        if not self.is_alive:
            return 0.0
        hp_before = self.current_hp
        self.current_hp = max(0.0, min(self.max_hp, self.current_hp + change))
        actual = self.current_hp - hp_before
        self.history.append(HealthEvent(
            timestamp=_utcnow(), hp_before=hp_before, hp_after=self.current_hp,
            change=actual, reason=reason, event_type=event_type
        ))
        if self.current_hp <= 0:
            self._die(reason)
        return actual

    def _die(self, cause: str):
        self.is_alive = False
        self.current_hp = 0.0
        self.cause_of_death = cause
        self.death_time = _utcnow()

    def record_trade(self, pnl: float, pnl_pct: float):
        if not self.is_alive:
            return
        self.total_trades += 1

        if pnl > 0:
            self.winning_trades += 1
            self.win_streak += 1
            self.loss_streak = 0
            hp_gain = min(10.0, max(2.0, pnl_pct * 2))
            self._apply_change(hp_gain, f"Win ${pnl:.2f} ({pnl_pct:+.2f}%)", "trade_win")
            if self.win_streak >= 5:
                self._apply_change(15.0, f"Win streak x{self.win_streak}!", "streak")
        else:
            self.loss_streak += 1
            self.win_streak = 0
            hp_loss = min(20.0, max(5.0, abs(pnl_pct) * 3))
            self._apply_change(-hp_loss, f"Loss ${pnl:.2f} ({pnl_pct:+.2f}%)", "trade_loss")
            if self.loss_streak >= 3:
                self._apply_change(-25.0, f"Loss streak x{self.loss_streak}!", "streak")

    def update_capital(self, new_capital: float):
        if not self.is_alive:
            return
        self.current_capital = new_capital
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
        if new_capital <= self.instant_death_capital:
            self._apply_change(-999, f"Capital ${new_capital:.2f} below death line", "instant_death")
            return
        if new_capital <= self.critical_capital:
            self._apply_change(-self.critical_hp_penalty, f"Capital critical: ${new_capital:.2f}", "capital_warning")
        dd = self.current_drawdown_pct
        if dd >= self.max_drawdown_pct:
            self._apply_change(-self.drawdown_hp_penalty, f"Drawdown {dd:.1f}% exceeded limit", "drawdown")

    def get_vitals(self) -> dict:
        return {
            "hp": round(self.current_hp, 1),
            "max_hp": self.max_hp,
            "status": self.get_status().value,
            "is_alive": self.is_alive,
            "capital": round(self.current_capital, 2),
            "peak_capital": round(self.peak_capital, 2),
            "drawdown_pct": round(self.current_drawdown_pct, 2),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "win_streak": self.win_streak,
            "loss_streak": self.loss_streak,
            "cause_of_death": self.cause_of_death,
        }
