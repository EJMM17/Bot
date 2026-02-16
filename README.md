# Darwin Agent v2.3

Autonomous evolutionary crypto trading system. AI agents trade USDT perpetual futures on Bybit, evolve across generations using genetic algorithms and reinforcement learning.

**Core philosophy:** Survive. Adapt. Evolve. Start with $50, compound through consistent gains.

## Features

- **Evolutionary lifecycle** — agents are born, trade, die, and pass learned knowledge to the next generation via DNA encoding
- **Q-Learning brain** — 30-dimensional feature engineering with linear function approximation and experience replay
- **4 trading strategies** — Momentum (EMA crossover), Mean Reversion (Bollinger), Scalping (VWAP), Breakout (range)
- **Adaptive regime detection** — automatically selects strategies based on market conditions (trending, choppy, ranging)
- **Health system** — HP-based survival mechanic that enforces discipline and kills underperforming agents
- **Risk management** — 2% max position size, 5% daily loss limit, 1.5:1 min R:R ratio
- **Web dashboard** — real-time monitoring on port 8080 (mobile-friendly)
- **Safe defaults** — testnet and paper trading by default; live mode requires explicit opt-in

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/EJMM17/Bot.git
cd Bot
pip install -r requirements.txt

# 2. Configure
cp config_example.yaml config.yaml
# Edit config.yaml with your Bybit API keys

# 3. Run (paper trading)
python -m darwin_agent --mode test
```

## Usage

```bash
# Paper trading (default, safe)
python -m darwin_agent --mode test

# Pre-flight diagnostics
python -m darwin_agent --diagnose

# Migration readiness check
python -m darwin_agent --migrate

# Evolution history
python -m darwin_agent --status

# Live trading (real money — use with caution)
python -m darwin_agent --mode live
```

## Docker

```bash
# Start (paper trading by default)
docker-compose up -d

# Monitor logs
docker-compose logs -f

# Stop
docker-compose down
```

## Server Deployment (Ubuntu/Debian)

```bash
bash deploy.sh
sudo systemctl start darwin-agent
sudo systemctl status darwin-agent
```

See [GUIA_DEPLOY.md](GUIA_DEPLOY.md) for a detailed deployment guide.

## Project Structure

```
darwin_agent/               # Main Python package
├── main.py                 # CLI, arg parsing, run loop
├── dashboard.py            # Web dashboard (port 8080)
├── core/
│   ├── agent_v2.py         # Main agent class, lifecycle, trading loop
│   └── health.py           # HP tracking, death conditions
├── evolution/
│   └── dna.py              # DNA encoding, cross-generation inheritance
├── markets/
│   ├── base.py             # Abstract MarketAdapter interface
│   ├── crypto.py           # Bybit V5 API + Paper trading adapter
│   └── bybit_errors.py     # Error diagnostics + migration checks
├── ml/
│   ├── brain.py            # Q-Learning with experience replay
│   ├── features.py         # 30-dimensional feature engineering
│   └── selector.py         # Adaptive strategy selection per regime
├── strategies/
│   └── base.py             # Strategy ABC + 4 implementations
├── risk/
│   └── manager.py          # Position sizing, daily limits, R:R enforcement
└── utils/
    ├── config.py           # Typed dataclass configuration
    └── logger.py           # Structured logging + JSONL trade journal
```

## Configuration

Copy `config_example.yaml` to `config.yaml` and configure:

| Section | Key settings |
|---------|-------------|
| `markets` | Bybit API keys, testnet toggle |
| `health` | Starting HP, death thresholds, drawdown limits |
| `risk` | Position size %, max trades/day, R:R ratio |
| `evolution` | Incubation period, graduation winrate, DNA storage |

## Tech Stack

- **Python 3.11+** with native asyncio
- **aiohttp** — async HTTP client + web dashboard server
- **numpy** — ML computations and feature engineering
- **PyYAML** — configuration management
- **Bybit V5 API** — USDT perpetual futures

## License

All rights reserved.
