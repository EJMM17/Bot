# Test Coverage Analysis — Darwin Agent v2.3

## Current State: Zero Automated Tests

The codebase has **~4,200 lines of production code across 14 modules** and **no automated tests**. There is no test framework in dependencies, no test directory, no CI pipeline, and no coverage tooling.

Paper trading mode (`--mode test`) exercises one integrated path through the system but provides no regression protection, no boundary testing, and no isolation of faulty components.

---

## Priority 1 — Critical (Financial Risk, Data Integrity)

These modules directly control money and agent survival. Bugs here cause real financial loss or corrupted evolutionary state.

### 1. `risk/manager.py` — RiskManager (105 lines)

The single gate that approves or denies every trade.

| Test Case | Why It Matters |
|-----------|----------------|
| `approve_trade()` rejects at daily trade limit | Prevents overtrading |
| `approve_trade()` rejects at max open positions | Concentration risk |
| `approve_trade()` rejects at daily loss % threshold | Stops bleeding |
| `approve_trade()` rejects bad R:R ratio (BUY and SELL sides) | Risk/reward calculation differs by side (lines 56-65) |
| `approve_trade()` rejects low-confidence signals (< 0.55) | Quality filter |
| `approve_trade()` applies stricter threshold (0.70) when capital < $15 | Capital preservation |
| `calculate_position_size()` respects max position % | Position sizing |
| `calculate_position_size()` handles zero `risk_per_unit` | Division by zero guard |
| `calculate_position_size()` caps at 90% of affordable size | Over-leverage prevention |
| `_today()` prunes old daily stats (keep 7 days) | Memory management |
| `get_risk_report()` with zero capital | Division by zero at line 100 |

### 2. `core/health.py` — HealthSystem (141 lines)

Controls agent life/death. A bug here could kill a profitable agent or keep a failing one alive.

| Test Case | Why It Matters |
|-----------|----------------|
| `record_trade()` HP gain clamped to [2, 10] | Prevents HP explosion |
| `record_trade()` HP loss clamped to [5, 20] | Prevents instant death from small loss |
| Win streak >= 5 triggers +15 HP bonus | Reward consistency |
| Loss streak >= 3 triggers -25 HP penalty | Correct penalty escalation |
| Streaks reset on win/loss transitions | State machine correctness |
| `update_capital()` instant death at capital <= $10 | Death threshold |
| `update_capital()` critical penalty at capital <= $25 | Warning threshold |
| `update_capital()` drawdown penalty at >= 20% | Drawdown protection |
| `_apply_change()` clamps HP to [0, max_hp] | No HP overflow |
| `_apply_change()` no-op when dead | Dead agents don't change |
| `get_status()` at boundaries (HP = 70, 40, 0) | Correct status classification |
| `current_drawdown_pct` with peak_capital = 0 | Edge case |

### 3. `evolution/dna.py` — DNA + EvolutionEngine (187 lines)

Persistence and inheritance. Corruption here means lost evolutionary progress.

| Test Case | Why It Matters |
|-----------|----------------|
| `DNA.to_dict()` / `from_dict()` roundtrip | Serialization integrity |
| `StrategyGene.update_confidence()` with < 5 trades | Returns 0.5 default |
| `StrategyGene.update_confidence()` weighted formula | Confidence accuracy |
| `get_latest_generation()` with empty dir | Bootstrap case |
| `synthesize_inherited_dna()` aggregates strategy genes | Cross-gen learning |
| `synthesize_inherited_dna()` deduplicates rules | No rule bloat |
| `synthesize_inherited_dna()` adds death warnings | Learn from ancestors |
| `save_dna()` / `load_dna()` roundtrip | File persistence |

---

## Priority 2 — High (ML Correctness)

Subtle ML bugs silently degrade trading performance without visible errors.

### 4. `ml/brain.py` — QLearningBrain (192 lines)

| Test Case | Why It Matters |
|-----------|----------------|
| `_decode()` index 0 → ("momentum", "conservative") | Action mapping correctness |
| `_decode()` index 14 → ("breakout", "aggressive") | Boundary case |
| `predict_q()` applies regime bonuses | Regime-aware decisions |
| `choose_action()` higher epsilon when health < 30% | Survival exploration |
| `choose_action()` safe actions only when health < 40% | Risk aversion |
| `choose_action()` dampens aggressive Q-values when health < 50% | Conservative when hurt |
| `_safe_actions()` returns correct indices | Safe action filter |
| `_update()` clips weights to [-10, 10] | Gradient explosion prevention |
| `calculate_reward()` 1.5x penalty for losses | Asymmetric reward |
| `export_brain()` / `import_brain()` with mutation | Inheritance integrity |
| `import_brain()` handles shape mismatch | Cross-gen compatibility |

### 5. `ml/features.py` — FeatureEngineer (250 lines)

| Test Case | Why It Matters |
|-----------|----------------|
| `extract()` returns None for < 50 candles | Minimum data guard |
| `extract()` returns exactly 30 features | Feature vector shape |
| `extract()` handles all-zero volumes | Division by zero |
| `_ema()` against known reference data | Indicator accuracy |
| `_rsi()` stays in [0, 100] range | Valid RSI range |
| `_detect_regime()` classification | Correct regime labels |
| NaN/Inf sanitization (line 170) | No corrupted features |

### 6. `ml/selector.py` — AdaptiveSelector (191 lines)

| Test Case | Why It Matters |
|-----------|----------------|
| `decide()` returns no-trade when features unavailable | Graceful degradation |
| `decide()` confidence blend: `signal*0.6 + action*0.2 + fit*0.2` | Confidence formula |
| `_regime_fit()` hardcoded values without history | Default behavior |
| `_regime_fit()` blends learned + base with >= 5 trades | Adaptive behavior |
| `export_for_dna()` / `import_from_dna()` roundtrip | State persistence |

---

## Priority 3 — Medium (Strategy Signals)

### 7. `strategies/base.py` — All 4 Strategies (294 lines)

| Test Case | Why It Matters |
|-----------|----------------|
| Indicator functions match `features.py` equivalents | Two implementations must agree |
| MomentumStrategy: bullish/bearish crossover detection | Signal correctness |
| MeanReversionStrategy: triggers at RSI extremes + BB | Signal correctness |
| ScalpingStrategy: VWAP + engulfing pattern | Signal correctness |
| BreakoutStrategy: 20-bar range + volume surge | Signal correctness |
| All strategies: SL/TP on correct side of entry | Wrong SL/TP = instant loss |
| All strategies: confidence in valid range | Feeds into risk approval |

---

## Priority 4 — Lower (Config, Data Classes)

### 8. `utils/config.py` (104 lines)

- `validate()` catches all 5 error conditions
- `load_config()` with missing file returns defaults
- `load_config()` partial YAML merges with defaults
- Type coercion edge cases

### 9. `markets/base.py` — Data Classes (139 lines)

- `Position.update_pnl()` for BUY and SELL sides
- `Candle.body_pct` and `is_bullish` properties
- Zero entry price edge case

---

## Structural Recommendations

1. **Add test infrastructure**: `pytest` + `pytest-asyncio` to a `requirements-dev.txt`. The codebase is heavily async.

2. **Duplicate indicator implementations**: `strategies/base.py` and `ml/features.py` both implement EMA, RSI, ATR, and Bollinger independently (list-based vs numpy). Tests should cross-validate. Consider extracting a shared implementation.

3. **Inject time dependencies**: Three modules define `_utcnow()` independently (`health.py:9`, `base.py:10`, `agent_v2.py:33`). `RiskManager._today()` calls `datetime.now()` directly. These must be mockable for testing cooldown logic, daily stat rollover, and death timestamps.

4. **State persistence**: The atomic write pattern (`.tmp` + `os.replace`) in `agent_v2.py:156-189` should be tested, including failure mid-write.

5. **Suggested test directory structure**:
   ```
   tests/
   ├── conftest.py            # Shared fixtures (mock candles, configs, etc.)
   ├── test_health.py         # Priority 1
   ├── test_risk_manager.py   # Priority 1
   ├── test_dna.py            # Priority 1
   ├── test_brain.py          # Priority 2
   ├── test_features.py       # Priority 2
   ├── test_selector.py       # Priority 2
   ├── test_strategies.py     # Priority 3
   ├── test_config.py         # Priority 4
   └── test_data_classes.py   # Priority 4
   ```
