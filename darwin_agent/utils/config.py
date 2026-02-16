"""Configuration — Typed, validated, with sensible defaults."""

import yaml
import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MarketConfig:
    enabled: bool = False
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    max_allocation_pct: float = 60.0


@dataclass
class HealthConfig:
    starting_hp: float = 100.0
    instant_death_capital: float = 10.0
    critical_capital: float = 25.0
    critical_hp_penalty: float = 50.0
    max_drawdown_pct: float = 20.0
    drawdown_hp_penalty: float = 30.0


@dataclass
class RiskConfig:
    max_position_pct: float = 2.0
    max_open_positions: int = 3
    max_daily_trades: int = 20
    max_daily_loss_pct: float = 5.0
    default_stop_loss_pct: float = 1.5
    default_take_profit_pct: float = 3.0
    min_risk_reward_ratio: float = 1.5


@dataclass
class EvolutionConfig:
    incubation_candles: int = 200
    min_graduation_winrate: float = 0.52
    dna_path: str = "data/generations"
    max_generations_memory: int = 50


@dataclass
class AgentConfig:
    starting_capital: float = 50.0
    markets: Dict[str, MarketConfig] = field(default_factory=lambda: {
        "crypto": MarketConfig(enabled=True, testnet=True),
    })
    health: HealthConfig = field(default_factory=HealthConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    log_level: str = "INFO"
    heartbeat_interval: int = 60
    dashboard_port: int = 8080

    def validate(self):
        errors = []
        if self.starting_capital < 10:
            errors.append("Starting capital must be >= $10")
        if self.risk.max_position_pct > 10:
            errors.append("Max position % too high (max 10%)")
        if self.risk.min_risk_reward_ratio < 1.0:
            errors.append("Risk/reward ratio must be >= 1.0")
        if self.health.instant_death_capital >= self.starting_capital:
            errors.append("Death threshold must be below starting capital")
        if not any(m.enabled for m in self.markets.values()):
            errors.append("At least one market must be enabled")
        if errors:
            raise ValueError("Config errors: " + "; ".join(errors))


def load_config(path: str = "config.yaml") -> AgentConfig:
    config = AgentConfig()

    if not os.path.exists(path):
        return config

    with open(path, "r") as f:
        try:
            data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file '{path}': {e}") from e

    for key in ("starting_capital", "heartbeat_interval", "log_level", "dashboard_port"):
        if key in data:
            setattr(config, key, type(getattr(config, key))(data[key]))

    if "markets" in data:
        # If user explicitly provides markets, treat it as authoritative
        # to avoid silently keeping the default "crypto" market enabled.
        config.markets = {}
        for name, mdata in data["markets"].items():
            mc = MarketConfig()
            for k, v in (mdata or {}).items():
                if hasattr(mc, k):
                    setattr(mc, k, v)
            config.markets[name] = mc

    for section, obj in [("health", config.health), ("risk", config.risk),
                         ("evolution", config.evolution)]:
        if section in data and data[section]:
            for k, v in data[section].items():
                if hasattr(obj, k):
                    setattr(obj, k, v)

    return config


def config_to_dict(config: AgentConfig) -> Dict:
    """Serialize AgentConfig to plain dict for APIs/YAML dump."""
    return {
        "starting_capital": config.starting_capital,
        "heartbeat_interval": config.heartbeat_interval,
        "log_level": config.log_level,
        "dashboard_port": config.dashboard_port,
        "markets": {
            name: {
                "enabled": market.enabled,
                "api_key": market.api_key,
                "api_secret": market.api_secret,
                "testnet": market.testnet,
                "max_allocation_pct": market.max_allocation_pct,
            }
            for name, market in config.markets.items()
        },
        "health": {
            "starting_hp": config.health.starting_hp,
            "instant_death_capital": config.health.instant_death_capital,
            "critical_capital": config.health.critical_capital,
            "critical_hp_penalty": config.health.critical_hp_penalty,
            "max_drawdown_pct": config.health.max_drawdown_pct,
            "drawdown_hp_penalty": config.health.drawdown_hp_penalty,
        },
        "risk": {
            "max_position_pct": config.risk.max_position_pct,
            "max_open_positions": config.risk.max_open_positions,
            "max_daily_trades": config.risk.max_daily_trades,
            "max_daily_loss_pct": config.risk.max_daily_loss_pct,
            "default_stop_loss_pct": config.risk.default_stop_loss_pct,
            "default_take_profit_pct": config.risk.default_take_profit_pct,
            "min_risk_reward_ratio": config.risk.min_risk_reward_ratio,
        },
        "evolution": {
            "incubation_candles": config.evolution.incubation_candles,
            "min_graduation_winrate": config.evolution.min_graduation_winrate,
            "dna_path": config.evolution.dna_path,
            "max_generations_memory": config.evolution.max_generations_memory,
        },
    }


def save_config(config: AgentConfig, path: str = "config.yaml"):
    """Persist AgentConfig into YAML file."""
    with open(path, "w") as f:
        yaml.safe_dump(config_to_dict(config), f, sort_keys=False)
