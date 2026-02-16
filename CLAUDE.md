# CLAUDE.md — Darwin Agent v2.3

## Project Overview

Darwin Agent is an autonomous evolutionary crypto trading system. It deploys AI trading agents that evolve across generations using genetic algorithms and reinforcement learning. Each agent trades USDT perpetual futures on Bybit, tracks its own health (HP), and when it "dies" (HP reaches 0), its learned knowledge is encoded into DNA and inherited by the next generation.

**Core philosophy:** "Survive. Adapt. Evolve." — start with $50, compound through consistent gains over months/years on a low-cost VPS.

## Repository Structure

The repository root contains a single zip archive (`darwin agent v2.3`) holding the full project. When extracted, the layout is:

```
darwin_agent/               # Main Python package
├── __init__.py             # Package version (2.3.0)
├── __main__.py             # Entry point: from darwin_agent.main import main
├── main.py                 # CLI, arg parsing, run loop, generation spawning
├── dashboard.py            # aiohttp web dashboard (mobile-friendly, port 8080)
├── core/
│   ├── agent_v2.py         # DarwinAgentV2 — main agent class, lifecycle, trading loop
│   └── health.py           # HealthSystem — HP tracking, death conditions
├── evolution/
│   └── dna.py              # DNA encoding, EvolutionEngine, cross-gen inheritance
├── markets/
│   ├── base.py             # Abstract MarketAdapter, data classes (Candle, Position, etc.)
│   ├── crypto.py           # BybitAdapter (V5 API) + PaperTradingAdapter
│   └── bybit_errors.py     # Bybit error diagnostics + migration readiness checks
├── ml/
│   ├── brain.py            # QLearningBrain — linear function approximation Q-learning
│   ├── features.py         # FeatureEngineer — 30-dimensional feature vector
│   └── selector.py         # AdaptiveSelector — strategy selection per market regime
├── strategies/
│   └── base.py             # Strategy ABC + implementations (Momentum, MeanReversion, Scalping, Breakout)
├── risk/
│   └── manager.py          # RiskManager — position sizing, daily limits, R:R enforcement
└── utils/
    ├── config.py           # Typed dataclass config (AgentConfig, MarketConfig, etc.)
    └── logger.py           # DarwinLogger — structured logging + JSONL trade journal

config_example.yaml         # Configuration template
requirements.txt            # Dependencies (aiohttp, pyyaml, numpy)
Dockerfile                  # Python 3.11-slim container
docker-compose.yml          # Orchestration with resource limits
deploy.sh                   # Server deployment script (Ubuntu/Debian + systemd)
GUIA_DEPLOY.md              # Deployment guide (Spanish)
```

## Tech Stack

- **Language:** Python 3.11+
- **Async framework:** asyncio (native) + aiohttp for HTTP/dashboard
- **ML:** numpy for Q-learning computations and feature engineering
- **Config:** PyYAML
- **Exchange:** Bybit V5 API (USDT perpetual futures)
- **Deployment:** Docker or systemd on Linux VPS

## Key Architecture Concepts

### Agent Lifecycle

1. **Spawn** — `EvolutionEngine.spawn_new_generation()` creates a new `DNA` inheriting from ancestors
2. **Incubation** — Agent paper-trades; must reach `min_graduation_winrate` (default 52%) within `incubation_candles` (default 200) trades to graduate
3. **Live** — Agent trades with real capital (or continues paper trading in test mode)
4. **Death** — HP hits 0 from losses, drawdown, or capital depletion; DNA is saved, brain state persisted
5. **Inheritance** — Next generation inherits ML brain weights (with mutation) and accumulated rules

### Health System (`core/health.py`)

- HP starts at 100, capped at 100
- Wins add 2-10 HP; losses subtract 5-20 HP
- Win streak (5+): +15 HP bonus; loss streak (3+): -25 HP penalty
- Instant death: capital falls below $10
- Critical penalties: capital below $25 or drawdown exceeds 20%
- Loss streaks trigger cooldown periods (streak * 5 minutes)

### ML Pipeline

- **FeatureEngineer** (`ml/features.py`): extracts 30 features from candle data across 6 categories — Trend (6), Momentum (7), Volatility (5), Volume (4), Pattern (4), Context (4)
- **QLearningBrain** (`ml/brain.py`): 15 actions (5 strategies x 3 sizings), linear Q-function with experience replay
- **AdaptiveSelector** (`ml/selector.py`): combines brain output with strategy signals and regime fitness scores
- **Market regimes:** trending_volatile, trending_calm, choppy, ranging_tight, ranging_normal

### Trading Strategies (`strategies/base.py`)

Registered in `STRATEGY_REGISTRY`:
- **momentum** — EMA(9)/EMA(21) crossover + RSI confirmation + volume filter
- **mean_reversion** — Bollinger Band bounces at RSI extremes (<25 or >75)
- **scalping** — VWAP proximity + engulfing candle patterns
- **breakout** — 20-bar range breakout with volume surge confirmation

### Risk Management (`risk/manager.py`)

All trades must pass through `RiskManager.approve_trade()`:
- Max 2% capital per position
- Max 3 concurrent positions
- Max 20 trades per day
- Max 5% daily loss
- Min 1.5:1 risk/reward ratio
- Min confidence: 0.55 (0.70 when capital < $15)

### Market Adapters (`markets/base.py`)

Abstract `MarketAdapter` defines the interface: `connect`, `disconnect`, `get_balance`, `get_candles`, `get_current_price`, `place_order`, `close_position`, `get_open_positions`, `get_min_trade_size`.

Implementations:
- `BybitAdapter` — real Bybit V5 API with HMAC-SHA256 signing
- `PaperTradingAdapter` — wraps a real adapter for simulated execution

## Running the Project

```bash
# Setup
cp config_example.yaml config.yaml
# Edit config.yaml with Bybit API keys
pip install -r requirements.txt

# Paper trading (default)
python -m darwin_agent --mode test

# Diagnostics
python -m darwin_agent --diagnose

# Migration readiness check
python -m darwin_agent --migrate

# Evolution history
python -m darwin_agent --status

# Live trading (real money)
python -m darwin_agent --mode live
```

### Docker

```bash
docker-compose up -d          # Start (paper trading by default)
docker-compose logs -f        # Monitor
```

### Server deployment

```bash
bash deploy.sh                # Sets up venv, systemd, firewall, NTP
sudo systemctl start darwin-agent
```

## Configuration (`config.yaml`)

Key sections: `starting_capital`, `markets` (Bybit API keys, testnet toggle), `health` (HP parameters, death thresholds), `risk` (position limits, R:R ratio), `evolution` (incubation period, graduation winrate, DNA storage path).

Config is loaded by `utils/config.py` into typed dataclasses (`AgentConfig`, `MarketConfig`, `HealthConfig`, `RiskConfig`, `EvolutionConfig`) with `.validate()` for safety checks.

## Data Persistence

- `data/generations/gen_XXXX.json` — DNA records per generation
- `data/generations/brain_gen_XXXX.json` — Q-learning brain weights
- `data/generations/selector_gen_XXXX.json` — Selector regime stats
- `data/agent_state.json` — Live agent state (atomic write with tmp+rename)
- `data/logs/gen_X_*.log` — Per-generation log files
- `data/logs/trades_gen_X.jsonl` — JSONL trade journal

## Code Conventions

- **snake_case** for functions and variables, **PascalCase** for classes
- **Dataclasses** for domain objects (DNA, Position, Candle, Config, etc.)
- **Async/await** for all I/O (market calls, dashboard, network)
- **UTC timestamps** everywhere — `datetime.now(timezone.utc)` via `_utcnow()` helpers
- **Enums** for fixed sets: `OrderSide`, `OrderType`, `TimeFrame`, `AgentPhase`, `HealthStatus`
- **Abstract base classes** for extensible interfaces: `MarketAdapter`, `Strategy`
- **No external test framework** — verification through paper trading mode (`--mode test`), diagnostics (`--diagnose`), and migration checks (`--migrate`)
- Indicator implementations (EMA, RSI, ATR, Bollinger, VWAP) are self-contained — no TA library dependency
- Errors on individual symbols are counted and symbols are skipped after 5 consecutive failures (graceful degradation)
- State saves are atomic (write to `.tmp` then `os.replace`) and throttled to once per minute

## Resilience Patterns

- Periodic state save to disk — survives server restart
- Auto-reconnection to markets on disconnect (checked every 10 cycles)
- Per-symbol error tracking with auto-skip after threshold
- Exponential backoff on repeated failures
- Graceful signal handling (SIGINT/SIGTERM)
- Docker memory limit (512M) and CPU quota (50%)
- NTP sync required for Bybit API signature timestamps

## Safety Guards

- Testnet is the default; mainnet requires explicit `--mode live`
- Pre-flight diagnostics run before trading starts
- Mainnet mode refuses to start with failing diagnostics
- API keys validated before live trading
- No API secrets in logs
- `config.yaml` excluded from Docker builds and git via `.dockerignore`

## Dashboard

Web UI served on port 8080 via aiohttp. Single-page HTML with inline CSS/JS. Auto-refreshes every 5 seconds via `/api/status`, `/api/history`, `/api/trades` endpoints. Mobile-responsive grid layout showing HP bar, capital, trading stats, risk metrics, ML brain stats, regime playbook, evolution history, and recent trades.

## Adding New Exchange Adapters

1. Create a new file in `darwin_agent/markets/`
2. Subclass `MarketAdapter` from `markets/base.py`
3. Implement all abstract methods: `connect`, `disconnect`, `get_balance`, `get_candles`, `get_current_price`, `place_order`, `close_position`, `get_open_positions`, `get_min_trade_size`
4. Register in `core/agent_v2.py` `_init_markets()` method

## Adding New Strategies

1. Subclass `Strategy` from `strategies/base.py`
2. Implement `analyze(candles, symbol, timeframe) -> Optional[MarketSignal]`
3. Add instance to `STRATEGY_REGISTRY` dict in `strategies/base.py`
4. Add strategy name to `QLearningBrain.STRATEGIES` list in `ml/brain.py` (changes action space)
5. Update regime fitness map in `AdaptiveSelector._regime_fit()` in `ml/selector.py`
