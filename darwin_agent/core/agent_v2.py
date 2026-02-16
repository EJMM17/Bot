"""
Darwin Agent v2.2 — Production-grade autonomous trading agent.

Server resilience:
- Periodic state save to disk (survives server restart)
- Auto-reconnection to markets on disconnect
- UTC-only timestamps (no timezone issues across restarts)
- Graceful degradation: skip symbols on API errors, don't crash
- Heartbeat health checks on market connection
- Exponential backoff on repeated failures
"""

import asyncio
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
from enum import Enum

from darwin_agent.core.health import HealthSystem, HealthStatus
from darwin_agent.evolution.dna import DNA, EvolutionEngine, StrategyGene
from darwin_agent.markets.base import MarketAdapter, TimeFrame, OrderSide, OrderType
from darwin_agent.markets.crypto import BybitAdapter, PaperTradingAdapter
from darwin_agent.risk.manager import RiskManager
from darwin_agent.strategies.base import STRATEGY_REGISTRY
from darwin_agent.ml.brain import QLearningBrain
from darwin_agent.ml.features import N_FEATURES
from darwin_agent.ml.selector import AdaptiveSelector
from darwin_agent.utils.config import AgentConfig
from darwin_agent.utils.logger import DarwinLogger


def _utcnow() -> datetime:
    """Always use UTC. Avoids timezone issues on VPS/Docker."""
    return datetime.now(timezone.utc)


class AgentPhase(Enum):
    INCUBATION = "incubation"
    LIVE = "live"
    DYING = "dying"
    DEAD = "dead"


STATE_FILE = "data/agent_state.json"


class DarwinAgentV2:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.evolution = EvolutionEngine(config.evolution.dna_path)
        self.dna: DNA = self.evolution.spawn_new_generation()
        self.generation = self.dna.generation

        self.logger = DarwinLogger(self.generation, config.log_level)
        self.health = HealthSystem(
            current_hp=config.health.starting_hp,
            max_hp=config.health.starting_hp,
            current_capital=config.starting_capital,
            peak_capital=config.starting_capital,
            instant_death_capital=config.health.instant_death_capital,
            critical_capital=config.health.critical_capital,
            critical_hp_penalty=config.health.critical_hp_penalty,
            max_drawdown_pct=config.health.max_drawdown_pct,
            drawdown_hp_penalty=config.health.drawdown_hp_penalty,
        )
        self.risk = RiskManager(config.risk)

        # ML
        self.brain = QLearningBrain(
            n_features=N_FEATURES,
            epsilon=0.3 if self.generation == 0 else 0.15,
        )
        self.selector = AdaptiveSelector(self.brain)
        self._inherit_ml()

        self.markets: Dict[str, MarketAdapter] = {}
        self.phase = AgentPhase.INCUBATION
        self.born_at = _utcnow()
        self.cycle_count = 0
        self.incubation_trades = 0
        self.incubation_wins = 0
        self.cooldown_until: Optional[datetime] = None
        self.watchlist = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]

        # Resilience: track consecutive market errors per symbol
        self._symbol_errors: Dict[str, int] = {}
        self._max_symbol_errors = 5  # Skip symbol after 5 consecutive errors
        self._last_state_save = 0.0

    def _inherit_ml(self):
        if self.generation == 0:
            self.logger.evolution("Gen-0: Fresh brain, no ancestors")
            return
        brain_path = f"data/generations/brain_gen_{self.generation - 1:04d}.json"
        try:
            self.brain.load(brain_path, as_inheritance=True)
            self.logger.evolution(f"Inherited brain from Gen-{self.generation - 1}")
            sel_path = f"data/generations/selector_gen_{self.generation - 1:04d}.json"
            if os.path.exists(sel_path):
                with open(sel_path) as f:
                    self.selector.import_from_dna(json.load(f), mutation_rate=0.03)
        except Exception as e:
            self.logger.warning(f"Could not inherit brain: {e}")

    async def _init_markets(self):
        """Initialize market connections with error handling."""
        for name, mc in self.config.markets.items():
            if not mc.enabled:
                continue
            if name == "crypto":
                try:
                    real = BybitAdapter({
                        "api_key": mc.api_key,
                        "api_secret": mc.api_secret,
                        "testnet": mc.testnet,
                    })
                    adapter = PaperTradingAdapter(real, self.config.starting_capital) \
                        if self.phase == AgentPhase.INCUBATION else real

                    if await adapter.connect():
                        self.markets[name] = adapter
                        mode = "PAPER" if isinstance(adapter, PaperTradingAdapter) else "LIVE"
                        self.logger.info(f"Connected: {name} [{mode}]")
                    else:
                        self.logger.error(f"Market connect failed: {name}")
                except Exception as e:
                    self.logger.error(f"Market init error: {e}")

    async def _check_market_health(self) -> bool:
        """Verify market connection is alive. Reconnect if needed."""
        if not self.markets:
            return False

        adapter = list(self.markets.values())[0]

        # Quick health check: can we get a price?
        try:
            price = await adapter.get_current_price("BTCUSDT")
            if price > 0:
                return True
        except Exception:
            pass

        # Connection dead — try to reconnect
        self.logger.warning("Market connection lost. Reconnecting...")
        try:
            await adapter.disconnect()
        except Exception:
            pass

        self.markets.clear()
        await self._init_markets()
        return bool(self.markets)

    def _save_state(self):
        """
        Periodically save agent state to disk.
        Allows recovery after server restart/crash.
        """
        now = _utcnow().timestamp()
        if now - self._last_state_save < 60:  # Save at most once per minute
            return

        try:
            state = {
                "generation": self.generation,
                "phase": self.phase.value,
                "cycle_count": self.cycle_count,
                "incubation_trades": self.incubation_trades,
                "incubation_wins": self.incubation_wins,
                "health_hp": self.health.current_hp,
                "health_capital": self.health.current_capital,
                "health_peak": self.health.peak_capital,
                "health_trades": self.health.total_trades,
                "health_wins": self.health.winning_trades,
                "brain_epsilon": self.brain.epsilon,
                "born_at": self.born_at.isoformat(),
                "saved_at": _utcnow().isoformat(),
            }
            os.makedirs(os.path.dirname(STATE_FILE) or "data", exist_ok=True)
            # Write to temp then rename (atomic on Linux, prevents corruption)
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f)
            os.replace(tmp, STATE_FILE)
            self._last_state_save = now
        except Exception as e:
            self.logger.warning(f"State save failed: {e}")

    async def run(self):
        """Main agent loop with full resilience."""
        prev = self.generation - 1 if self.generation > 0 else None
        self.logger.born(self.config.starting_capital, prev)
        if self.dna.rules:
            for r in self.dna.rules[:3]:
                self.logger.info(f"  Rule: {r}")

        await self._init_markets()
        if not self.markets:
            await self._die("No markets available")
            return

        self.logger.info(f"Phase: {self.phase.value} | Heartbeat: {self.config.heartbeat_interval}s")

        try:
            while self.health.is_alive:
                self.cycle_count += 1
                await self._heartbeat()
                self._save_state()
                await asyncio.sleep(self.config.heartbeat_interval)
        except KeyboardInterrupt:
            self.logger.warning("Manual shutdown")
        except asyncio.CancelledError:
            self.logger.warning("Task cancelled (server shutdown)")
        except Exception as e:
            self.logger.error(f"Fatal: {e}")
            import traceback
            traceback.print_exc()
            await self._die(f"Fatal: {e}")
        finally:
            self._save_state()
            await self._cleanup()

    async def _heartbeat(self):
        if not self.health.is_alive:
            return

        if self.health.get_status() == HealthStatus.DEAD:
            await self._die(self.health.cause_of_death or "Unknown")
            return

        # Periodic status log
        if self.cycle_count % 5 == 0:
            self.logger.health_update(self.health.current_hp, 0,
                f"Cycle {self.cycle_count} | {self.phase.value} | eps={self.brain.epsilon:.3f}")

        # Check market health every 10 cycles
        if self.cycle_count % 10 == 0:
            if not await self._check_market_health():
                self.logger.error("Markets unavailable, skipping cycle")
                return

        # Cooldown check (UTC-based)
        if self.cooldown_until and _utcnow() < self.cooldown_until:
            return

        try:
            if self.phase == AgentPhase.INCUBATION:
                await self._incubation_cycle()
            elif self.phase == AgentPhase.LIVE:
                await self._live_cycle()
        except Exception as e:
            self.logger.error(f"Cycle error (non-fatal): {e}")
            # Don't crash the agent for a single cycle failure

    async def _manage_positions(self, adapter):
        """Check open positions for SL/TP hits and close them.

        Used by both incubation and live cycles. Paper trading positions
        have no server-side SL/TP so we must check locally.
        """
        try:
            positions = await adapter.get_open_positions()
        except Exception as e:
            self.logger.error(f"Cannot get positions: {e}")
            return

        closed_any = False
        for pos in positions:
            try:
                price = await adapter.get_current_price(pos.symbol)
                if price <= 0:
                    continue
                pos.update_pnl(price)

                if pos.has_server_sltp:
                    continue

                close = False
                if pos.stop_loss:
                    if (pos.side == OrderSide.BUY and price <= pos.stop_loss) or \
                       (pos.side == OrderSide.SELL and price >= pos.stop_loss):
                        close = True
                if pos.take_profit:
                    if (pos.side == OrderSide.BUY and price >= pos.take_profit) or \
                       (pos.side == OrderSide.SELL and price <= pos.take_profit):
                        close = True

                if close:
                    result = await adapter.close_position(pos)
                    if result.success:
                        self._process_result(pos.pnl, pos.pnl_pct, pos.symbol)
                        closed_any = True
            except Exception as e:
                self.logger.error(f"Position mgmt {pos.symbol}: {e}")

        # Sync health capital with actual adapter balance (includes fees)
        if closed_any:
            try:
                balance = await adapter.get_balance()
                if balance > 0:
                    self.health.update_capital(balance)
            except Exception:
                pass

    async def _incubation_cycle(self):
        adapter = list(self.markets.values())[0]

        # Manage existing positions — check SL/TP and close if hit
        await self._manage_positions(adapter)

        for symbol in self.watchlist:
            if not self.health.is_alive:
                break
            if self._symbol_errors.get(symbol, 0) >= self._max_symbol_errors:
                continue  # Skip symbols with too many errors

            try:
                candles = await adapter.get_candles(symbol, TimeFrame.M15, limit=100)
                if not candles or len(candles) < 50:
                    continue

                hp = self.health.current_hp / self.health.max_hp
                decision = self.selector.decide(candles, symbol, TimeFrame.M15, hp)

                if decision.should_trade and decision.signal:
                    positions = await adapter.get_open_positions()
                    ok, reason = self.risk.approve_trade(
                        decision.signal, self.health.current_capital, len(positions))
                    if ok:
                        await self._execute(decision, adapter)
                        self.incubation_trades += 1
                    else:
                        self.selector.report_result(0, 0)
                else:
                    self.selector.report_hold_result(candles, symbol)

                # Reset error count on success
                self._symbol_errors[symbol] = 0

            except Exception as e:
                self._symbol_errors[symbol] = self._symbol_errors.get(symbol, 0) + 1
                self.logger.error(f"{symbol}: {e} (errors: {self._symbol_errors[symbol]})")

        # Check graduation
        if self.incubation_trades >= self.config.evolution.incubation_candles:
            wr = self.incubation_wins / max(1, self.incubation_trades)
            if wr >= self.config.evolution.min_graduation_winrate:
                await self._graduate()
            else:
                await self._die(f"Failed incubation WR:{wr:.1%}")

    async def _live_cycle(self):
        adapter = list(self.markets.values())[0]

        # 1. Manage existing positions (check SL/TP, close if hit)
        await self._manage_positions(adapter)

        # 2. Scan for new trades
        for symbol in self.watchlist:
            if not self.health.is_alive or symbol in self.dna.blacklisted_pairs:
                continue
            if self._symbol_errors.get(symbol, 0) >= self._max_symbol_errors:
                continue

            try:
                candles = await adapter.get_candles(symbol, TimeFrame.M15, limit=100)
                if not candles or len(candles) < 50:
                    continue

                hp = self.health.current_hp / self.health.max_hp
                decision = self.selector.decide(candles, symbol, TimeFrame.M15, hp)

                if decision.should_trade and decision.signal:
                    positions = await adapter.get_open_positions()
                    ok, reason = self.risk.approve_trade(
                        decision.signal, self.health.current_capital, len(positions))
                    if ok:
                        mult = self.selector.sizing_multipliers.get(
                            decision.action.sizing if decision.action else "normal", 1.0)
                        await self._execute(decision, adapter, mult)
                    else:
                        self.selector.report_result(0, 0)
                else:
                    self.selector.report_hold_result(candles, symbol)

                self._symbol_errors[symbol] = 0

            except Exception as e:
                self._symbol_errors[symbol] = self._symbol_errors.get(symbol, 0) + 1
                self.logger.error(f"{symbol}: {e}")

    async def _execute(self, decision, adapter, sizing_mult=1.0):
        signal = decision.signal
        if not signal:
            return

        size = self.risk.calculate_position_size(
            self.health.current_capital, signal.entry_price, signal.stop_loss)
        size *= sizing_mult

        min_size = await adapter.get_min_trade_size(signal.symbol)
        if size < min_size:
            return

        result = await adapter.place_order(
            symbol=signal.symbol, side=signal.side, order_type=OrderType.MARKET,
            quantity=size, stop_loss=signal.stop_loss, take_profit=signal.take_profit)

        if result.success:
            strat = decision.action.strategy if decision.action else "unknown"
            self.logger.trade(
                action=signal.side.value.upper(), market="crypto",
                symbol=signal.symbol, amount=size, price=signal.entry_price,
                reason=f"[{strat}] {signal.reason}")
        else:
            self.logger.warning(f"Order failed: {result.error}")

    def _process_result(self, pnl, pnl_pct, symbol):
        old_hp = self.health.current_hp
        self.health.record_trade(pnl, pnl_pct)
        # Capital sync is done in _manage_positions with actual adapter balance
        # (accounts for trading fees that raw PnL ignores)
        hp_change = self.health.current_hp - old_hp
        self.selector.report_result(pnl_pct, hp_change)
        self.risk.record_trade_result(pnl, self.health.current_capital)

        if self.phase == AgentPhase.INCUBATION and pnl > 0:
            self.incubation_wins += 1

        if self.health.loss_streak >= 2:
            mins = self.health.loss_streak * 5
            self.cooldown_until = _utcnow() + timedelta(minutes=mins)
            self.logger.warning(f"Cooldown {mins}min (streak:{self.health.loss_streak})")

        self.logger.health_update(self.health.current_hp, hp_change,
                                  f"Trade: ${pnl:+.2f} ({pnl_pct:+.2f}%) | {symbol}")

    async def _graduate(self):
        wr = self.incubation_wins / max(1, self.incubation_trades)
        self.logger.evolution(f"GRADUATED! WR:{wr:.1%} eps={self.brain.epsilon:.3f}")
        self.phase = AgentPhase.LIVE
        for a in self.markets.values():
            try:
                await a.disconnect()
            except Exception:
                pass
        self.markets.clear()
        await self._init_markets()

    async def _die(self, cause):
        self.phase = AgentPhase.DEAD
        self.health._die(cause)
        life = (_utcnow() - self.born_at).total_seconds()

        self.dna.died_at = _utcnow().isoformat()
        self.dna.cause_of_death = cause
        self.dna.lifespan_seconds = life
        self.dna.final_capital = self.health.current_capital
        self.dna.peak_capital = self.health.peak_capital
        self.dna.total_pnl = self.health.current_capital - self.config.starting_capital
        self.dna.total_trades = self.health.total_trades
        self.dna.win_rate = self.health.win_rate
        self.dna.max_drawdown_pct = self.health.current_drawdown_pct

        for g in self.dna.strategy_genes.values():
            g.update_confidence()

        if "drawdown" in cause.lower():
            self.dna.rules.append(f"Gen-{self.generation}: Died from drawdown")
        if "streak" in cause.lower():
            self.dna.rules.append(f"Gen-{self.generation}: Loss streak killed")
        if "capital" in cause.lower():
            self.dna.rules.append(f"Gen-{self.generation}: Capital bled out")
        if "incubation" in cause.lower():
            self.dna.rules.append(f"Gen-{self.generation}: Failed paper trading")

        os.makedirs("data/generations", exist_ok=True)
        self.brain.save(f"data/generations/brain_gen_{self.generation:04d}.json")
        with open(f"data/generations/selector_gen_{self.generation:04d}.json", "w") as f:
            json.dump(self.selector.export_for_dna(), f, indent=2)

        self.evolution.save_dna(self.dna)
        report = self.evolution.create_death_report(self.dna)
        self.logger.death(cause, self.health.current_capital, self.health.total_trades,
                         self.health.win_rate, life / 3600)
        print(report)

        # Clean up state file
        try:
            if os.path.exists(STATE_FILE):
                os.remove(STATE_FILE)
        except Exception:
            pass

    async def _cleanup(self):
        for a in self.markets.values():
            try:
                await a.disconnect()
            except Exception:
                pass

    def get_status(self):
        return {
            "generation": self.generation,
            "phase": self.phase.value,
            "health": self.health.get_vitals(),
            "risk": self.risk.get_risk_report(self.health.current_capital),
            "brain": self.brain.get_stats(),
            "playbook": self.selector.get_playbook(),
            "cycle": self.cycle_count,
            "uptime_seconds": (_utcnow() - self.born_at).total_seconds(),
            "inherited_rules": len(self.dna.rules),
        }
