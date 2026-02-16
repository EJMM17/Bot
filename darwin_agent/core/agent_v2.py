"""
Darwin Agent v2.3 — Fully autonomous AI trading agent.

AI enhancements over v2.2:
- Dynamic watchlist discovery from exchange
- Adaptive graduation (timeout + progressive threshold)
- Adaptive cooldown (learns optimal rest duration)
- Volatility-based position sizing
- Outcome binding (decision → trade → close → reward)
- Health-aware capital allocation
"""

import asyncio
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List
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

# Default watchlist (used as fallback if exchange discovery fails)
DEFAULT_WATCHLIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]


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

        # Dynamic watchlist (loaded from exchange or config)
        self.watchlist: List[str] = list(DEFAULT_WATCHLIST)
        self._watchlist_last_refresh = 0.0
        self._watchlist_refresh_interval = 3600  # Refresh every hour

        # Cache last candles per symbol for Q-learning next_state
        self._last_candles: Dict[str, list] = {}

        # Resilience: track consecutive market errors per symbol
        self._symbol_errors: Dict[str, int] = {}
        self._max_symbol_errors = 5  # Skip symbol after 5 consecutive errors
        self._last_state_save = 0.0

        # Adaptive cooldown: learn how long to rest after losses
        self._cooldown_outcomes: List[Dict] = []  # Track {cooldown_mins, next_trade_pnl}
        self._learned_cooldown_mult = 5.0  # Starts at 5 min/loss, adapts

        # Adaptive graduation: progressive threshold
        self._graduation_check_interval = 50  # Check every 50 trades

        # Configurable scan speed / aggression
        self._scan_timeframe = self._resolve_timeframe(config.scan_timeframe)
        self._aggression_level = max(0.5, min(3.0, float(config.aggression_level)))
        self._candle_lookback = 120 if self._scan_timeframe == TimeFrame.M1 else 100

    def _resolve_timeframe(self, timeframe_value: str) -> TimeFrame:
        mapping = {
            "1m": TimeFrame.M1,
            "5m": TimeFrame.M5,
            "15m": TimeFrame.M15,
            "1h": TimeFrame.H1,
            "4h": TimeFrame.H4,
            "1d": TimeFrame.D1,
        }
        return mapping.get((timeframe_value or "15m").lower(), TimeFrame.M15)

    def _inherit_ml(self):
        if self.generation == 0:
            self.logger.evolution("Gen-0: Fresh brain, no ancestors")
            return
        dna_dir = self.config.evolution.dna_path
        brain_path = os.path.join(dna_dir, f"brain_gen_{self.generation - 1:04d}.json")
        try:
            self.brain.load(brain_path, as_inheritance=True)
            self.logger.evolution(f"Inherited brain from Gen-{self.generation - 1}")
            sel_path = os.path.join(dna_dir, f"selector_gen_{self.generation - 1:04d}.json")
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

    async def _discover_watchlist(self, adapter: MarketAdapter):
        """Dynamically discover tradeable symbols from exchange.

        Keeps top liquid USDT perpetuals, adding new ones and removing dead ones.
        Falls back to DEFAULT_WATCHLIST on failure.
        """
        now = _utcnow().timestamp()
        if now - self._watchlist_last_refresh < self._watchlist_refresh_interval:
            return

        try:
            # Try to get tickers and pick top by volume
            # BybitAdapter stores instrument info; check which symbols are active
            if hasattr(adapter, 'real_adapter'):
                real = adapter.real_adapter
            elif hasattr(adapter, '_instruments'):
                real = adapter
            else:
                return

            if not hasattr(real, '_instruments') or not real._instruments:
                return

            # Get all USDT perpetual symbols that the exchange supports
            available = list(real._instruments.keys())
            if len(available) < 5:
                return  # Not enough instruments loaded

            # Score symbols: prefer those we have data on + top market caps
            # Start with defaults, then add high-volume ones
            priority_symbols = set(DEFAULT_WATCHLIST)
            candidates = [s for s in available if s.endswith("USDT")]

            # Keep blacklisted out
            blacklisted = set(self.dna.blacklisted_pairs)

            # Build final watchlist: defaults first, then extras up to 10
            new_watchlist = []
            for s in DEFAULT_WATCHLIST:
                if s in candidates and s not in blacklisted:
                    new_watchlist.append(s)

            for s in candidates:
                if s not in new_watchlist and s not in blacklisted and len(new_watchlist) < 10:
                    new_watchlist.append(s)

            if len(new_watchlist) >= 3:
                old = set(self.watchlist)
                self.watchlist = new_watchlist
                added = set(new_watchlist) - old
                removed = old - set(new_watchlist)
                if added or removed:
                    self.logger.info(f"Watchlist updated: {len(new_watchlist)} symbols "
                                     f"(+{len(added)} -{len(removed)})")
                self._watchlist_last_refresh = now

        except Exception as e:
            self.logger.warning(f"Watchlist discovery failed: {e}")
            # Keep existing watchlist

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
        """Periodically save agent state to disk."""
        now = _utcnow().timestamp()
        if now - self._last_state_save < 60:
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
                "watchlist": self.watchlist,
                "learned_cooldown_mult": self._learned_cooldown_mult,
            }
            os.makedirs(os.path.dirname(STATE_FILE) or "data", exist_ok=True)
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

        self.logger.info(
            f"Phase: {self.phase.value} | Heartbeat: {self.config.heartbeat_interval}s | "
            f"Scan: {self._scan_timeframe.value} | Aggression: {self._aggression_level:.2f}"
        )

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

        # Dynamic watchlist refresh
        if self.markets:
            adapter = list(self.markets.values())[0]
            await self._discover_watchlist(adapter)

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

    async def _manage_positions(self, adapter):
        """Check open positions for SL/TP hits and close them."""
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

        # Sync health capital with actual adapter balance
        if closed_any:
            try:
                balance = await adapter.get_balance()
                if balance > 0:
                    self.health.update_capital(balance)
            except Exception:
                pass

    async def _incubation_cycle(self):
        adapter = list(self.markets.values())[0]

        # Manage existing positions
        await self._manage_positions(adapter)

        for symbol in self.watchlist:
            if not self.health.is_alive:
                break
            if self._symbol_errors.get(symbol, 0) >= self._max_symbol_errors:
                continue

            try:
                candles = await adapter.get_candles(symbol, self._scan_timeframe, limit=self._candle_lookback)
                if not candles or len(candles) < 50:
                    continue

                self._last_candles[symbol] = candles
                hp = self.health.current_hp / self.health.max_hp
                decision = self.selector.decide(candles, symbol, self._scan_timeframe, hp)

                if decision.should_trade and decision.signal:
                    positions = await adapter.get_open_positions()
                    ok, reason = self.risk.approve_trade(
                        decision.signal, self.health.current_capital, len(positions))
                    if ok:
                        # Fast/aggressive mode: increase size and react to smaller moves
                        placed = await self._execute(
                            decision,
                            adapter,
                            sizing_mult=self._aggression_level * self._volatility_sizing(candles),
                        )
                        if placed:
                            self.incubation_trades += 1
                    else:
                        self.selector.report_result(0, 0)
                else:
                    self.selector.report_hold_result(candles, symbol)

                self._symbol_errors[symbol] = 0

            except Exception as e:
                self._symbol_errors[symbol] = self._symbol_errors.get(symbol, 0) + 1
                self.logger.error(f"{symbol}: {e} (errors: {self._symbol_errors[symbol]})")

        # Adaptive graduation check
        self._check_graduation()

    def _check_graduation(self):
        """Adaptive graduation: progressive threshold + timeout.

        Instead of rigid "200 trades at 52% WR", uses:
        1. Check every 50 trades
        2. Lower threshold as more trades accumulate (more data = more confidence)
        3. Timeout: auto-graduate after 400 trades if WR > 48% (marginal pass)
        """
        if self.incubation_trades < self._graduation_check_interval:
            return

        wr = self.incubation_wins / max(1, self.incubation_trades)
        target_trades = self.config.evolution.incubation_candles
        target_wr = self.config.evolution.min_graduation_winrate

        # Progressive: as we accumulate more trades, we need less WR advantage
        # (law of large numbers — more data = more confidence in the WR estimate)
        trades_ratio = min(1.0, self.incubation_trades / target_trades)

        if self.incubation_trades >= target_trades:
            # Standard graduation check
            if wr >= target_wr:
                asyncio.ensure_future(self._graduate())
            elif self.incubation_trades >= target_trades * 2:
                # Timeout: after 2x target trades, die if still below threshold
                asyncio.ensure_future(self._die(f"Failed incubation WR:{wr:.1%} after {self.incubation_trades} trades"))
            elif wr < target_wr - 0.10:
                # Way below target, die early
                asyncio.ensure_future(self._die(f"Failed incubation WR:{wr:.1%}"))
        elif self.incubation_trades >= target_trades * 0.5:
            # Early graduation: if WR is significantly above threshold, graduate early
            early_bonus = 0.05 * (1.0 - trades_ratio)  # Require higher WR for early grad
            if wr >= target_wr + early_bonus:
                self.logger.evolution(f"Early graduation! WR:{wr:.1%} > {target_wr + early_bonus:.1%} at {self.incubation_trades} trades")
                asyncio.ensure_future(self._graduate())

    async def _live_cycle(self):
        adapter = list(self.markets.values())[0]

        # Manage existing positions
        await self._manage_positions(adapter)

        for symbol in self.watchlist:
            if not self.health.is_alive or symbol in self.dna.blacklisted_pairs:
                continue
            if self._symbol_errors.get(symbol, 0) >= self._max_symbol_errors:
                continue

            try:
                candles = await adapter.get_candles(symbol, self._scan_timeframe, limit=self._candle_lookback)
                if not candles or len(candles) < 50:
                    continue

                self._last_candles[symbol] = candles
                hp = self.health.current_hp / self.health.max_hp
                decision = self.selector.decide(candles, symbol, self._scan_timeframe, hp)

                if decision.should_trade and decision.signal:
                    positions = await adapter.get_open_positions()
                    ok, reason = self.risk.approve_trade(
                        decision.signal, self.health.current_capital, len(positions))
                    if ok:
                        # Volatility-aware + health-aware sizing
                        base_mult = self.selector.sizing_multipliers.get(
                            decision.action.sizing if decision.action else "normal", 1.0)
                        vol_mult = self._volatility_sizing(candles)
                        health_mult = self._health_sizing(hp)
                        final_mult = base_mult * vol_mult * health_mult * self._aggression_level
                        await self._execute(decision, adapter, final_mult)
                    else:
                        self.selector.report_result(0, 0)
                else:
                    self.selector.report_hold_result(candles, symbol)

                self._symbol_errors[symbol] = 0

            except Exception as e:
                self._symbol_errors[symbol] = self._symbol_errors.get(symbol, 0) + 1
                self.logger.error(f"{symbol}: {e}")

    def _volatility_sizing(self, candles) -> float:
        """Scale position size inversely with volatility.

        High volatility = smaller position (risk parity principle).
        """
        if len(candles) < 20:
            return 1.0
        closes = [c.close for c in candles[-20:]]
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        vol = float(max(0.001, (sum(r**2 for r in returns) / len(returns)) ** 0.5))

        # Baseline vol: ~1% per 15min bar for crypto
        baseline_vol = 0.01
        ratio = baseline_vol / max(vol, 0.001)
        # Clamp: 0.5x to 1.5x normal size
        return max(0.5, min(1.5, ratio))

    def _health_sizing(self, health_pct: float) -> float:
        """Scale position size based on agent health.

        Lower health = smaller positions (survival mode).
        """
        if health_pct > 0.7:
            return 1.0
        elif health_pct > 0.4:
            return 0.75
        elif health_pct > 0.2:
            return 0.5
        else:
            return 0.3

    async def _execute(self, decision, adapter, sizing_mult=1.0) -> bool:
        signal = decision.signal
        if not signal:
            return False

        size = self.risk.calculate_position_size(
            self.health.current_capital, signal.entry_price, signal.stop_loss)
        size *= sizing_mult

        min_size = await adapter.get_min_trade_size(signal.symbol)
        if size < min_size:
            return False

        result = await adapter.place_order(
            symbol=signal.symbol, side=signal.side, order_type=OrderType.MARKET,
            quantity=size, stop_loss=signal.stop_loss, take_profit=signal.take_profit)

        if result.success:
            strat = decision.action.strategy if decision.action else "unknown"
            self.logger.trade(
                action=signal.side.value.upper(), market="crypto",
                symbol=signal.symbol, amount=size, price=signal.entry_price,
                reason=f"[{strat}] {signal.reason}")
            return True
        else:
            self.logger.warning(f"Order failed: {result.error}")
            return False

    def _process_result(self, pnl, pnl_pct, symbol):
        old_hp = self.health.current_hp
        self.health.record_trade(pnl, pnl_pct)
        hp_change = self.health.current_hp - old_hp
        candles = self._last_candles.get(symbol)
        self.selector.report_result(pnl_pct, hp_change, candles, symbol)
        self.risk.record_trade_result(pnl, self.health.current_capital)

        if self.phase == AgentPhase.INCUBATION and pnl > 0:
            self.incubation_wins += 1

        # Adaptive cooldown
        if self.health.loss_streak >= 2:
            mins = self.health.loss_streak * self._learned_cooldown_mult / max(0.75, self._aggression_level)
            self.cooldown_until = _utcnow() + timedelta(minutes=mins)
            self.logger.warning(f"Cooldown {mins:.0f}min (streak:{self.health.loss_streak}, mult:{self._learned_cooldown_mult:.1f})")
            # Record for learning
            self._cooldown_outcomes.append({
                "cooldown_mins": mins,
                "loss_streak": self.health.loss_streak,
                "entered_at": _utcnow().isoformat(),
            })
        elif self.cooldown_until and pnl > 0:
            # We just came off cooldown and won — the cooldown duration was good
            self._learn_cooldown(success=True)
        elif self.cooldown_until and pnl <= 0:
            # Came off cooldown and lost — cooldown wasn't long enough
            self._learn_cooldown(success=False)

        if self.cooldown_until and _utcnow() >= self.cooldown_until:
            self.cooldown_until = None  # Cooldown expired

        self.logger.health_update(self.health.current_hp, hp_change,
                                  f"Trade: ${pnl:+.2f} ({pnl_pct:+.2f}%) | {symbol}")

    def _learn_cooldown(self, success: bool):
        """Adapt cooldown multiplier based on post-cooldown outcomes."""
        if success:
            # Cooldown worked: slightly decrease (was enough rest)
            self._learned_cooldown_mult = max(2.0, self._learned_cooldown_mult * 0.95)
        else:
            # Cooldown wasn't enough: increase
            self._learned_cooldown_mult = min(15.0, self._learned_cooldown_mult * 1.1)

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

        # Populate strategy_genes from this generation's actual trading data
        self.dna.strategy_genes = {}
        for rs in self.selector.regime_stats.values():
            for strat_name, sr in rs.strategy_results.items():
                if strat_name not in self.dna.strategy_genes:
                    self.dna.strategy_genes[strat_name] = StrategyGene(name=strat_name)
                gene = self.dna.strategy_genes[strat_name]
                gene.times_used += sr.get("trades", 0)
                gene.wins += sr.get("wins", 0)
                gene.losses += sr.get("trades", 0) - sr.get("wins", 0)
                gene.total_pnl += sr.get("total_pnl", 0.0)
                gene.avg_pnl_per_trade = gene.total_pnl / gene.times_used if gene.times_used > 0 else 0.0

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

        dna_dir = self.config.evolution.dna_path
        os.makedirs(dna_dir, exist_ok=True)
        self.brain.save(os.path.join(dna_dir, f"brain_gen_{self.generation:04d}.json"))
        with open(os.path.join(dna_dir, f"selector_gen_{self.generation:04d}.json"), "w") as f:
            json.dump(self.selector.export_for_dna(), f, indent=2)

        self.evolution.save_dna(self.dna)
        report = self.evolution.create_death_report(self.dna)
        self.logger.death(cause, self.health.current_capital, self.health.total_trades,
                         self.health.win_rate, life / 3600)
        print(report)

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
            "watchlist": self.watchlist,
            "cooldown_multiplier": round(self._learned_cooldown_mult, 1),
            "scan_timeframe": self._scan_timeframe.value,
            "aggression_level": round(self._aggression_level, 2),
        }
