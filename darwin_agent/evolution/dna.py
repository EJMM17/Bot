"""Evolution — DNA encoding, death reports, and cross-generation inheritance."""

import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from collections import Counter


@dataclass
class StrategyGene:
    name: str
    times_used: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    confidence_score: float = 0.5

    @property
    def win_rate(self):
        return self.wins / self.times_used if self.times_used > 0 else 0.0

    def update_confidence(self):
        if self.times_used < 5:
            self.confidence_score = 0.5
            return
        wr = self.win_rate
        ps = min(1.0, max(0.0, (self.avg_pnl_per_trade + 5) / 10))
        ss = min(1.0, self.times_used / 50)
        self.confidence_score = wr * 0.4 + ps * 0.4 + ss * 0.2


@dataclass
class DNA:
    generation: int
    born_at: str = ""
    died_at: Optional[str] = None
    cause_of_death: Optional[str] = None
    lifespan_seconds: float = 0.0
    starting_capital: float = 50.0
    peak_capital: float = 50.0
    final_capital: float = 50.0
    total_pnl: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    strategy_genes: Dict[str, StrategyGene] = field(default_factory=dict)
    rules: List[str] = field(default_factory=list)
    blacklisted_pairs: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "generation": self.generation, "born_at": self.born_at,
            "died_at": self.died_at, "cause_of_death": self.cause_of_death,
            "lifespan_seconds": self.lifespan_seconds,
            "starting_capital": self.starting_capital,
            "peak_capital": self.peak_capital, "final_capital": self.final_capital,
            "total_pnl": self.total_pnl, "total_trades": self.total_trades,
            "win_rate": self.win_rate, "max_drawdown_pct": self.max_drawdown_pct,
            "strategy_genes": {k: asdict(v) for k, v in self.strategy_genes.items()},
            "rules": self.rules, "blacklisted_pairs": self.blacklisted_pairs,
        }

    @classmethod
    def from_dict(cls, data):
        dna = cls(generation=data["generation"])
        for k, v in data.items():
            if k == "strategy_genes":
                dna.strategy_genes = {n: StrategyGene(**g) for n, g in v.items()}
            elif hasattr(dna, k):
                setattr(dna, k, v)
        return dna


class EvolutionEngine:
    def __init__(self, dna_path="data/generations"):
        self.dna_path = dna_path
        os.makedirs(dna_path, exist_ok=True)

    def get_latest_generation(self) -> int:
        mx = -1
        for f in os.listdir(self.dna_path):
            if f.startswith("gen_") and f.endswith(".json"):
                try:
                    mx = max(mx, int(f.split("_")[1].split(".")[0]))
                except ValueError:
                    pass
        return mx

    def save_dna(self, dna: DNA):
        path = os.path.join(self.dna_path, f"gen_{dna.generation:04d}.json")
        with open(path, "w") as f:
            json.dump(dna.to_dict(), f, indent=2)

    def load_dna(self, gen: int) -> Optional[DNA]:
        path = os.path.join(self.dna_path, f"gen_{gen:04d}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return DNA.from_dict(json.load(f))

    def load_all_dna(self) -> List[DNA]:
        result = []
        for fn in sorted(os.listdir(self.dna_path)):
            if fn.startswith("gen_") and fn.endswith(".json"):
                path = os.path.join(self.dna_path, fn)
                try:
                    with open(path) as f:
                        result.append(DNA.from_dict(json.load(f)))
                except Exception:
                    pass
        return result

    def create_death_report(self, dna: DNA) -> str:
        lines = [
            "=" * 50,
            f"  DEATH REPORT — Generation {dna.generation}",
            "=" * 50,
            f"  Born:    {dna.born_at}",
            f"  Died:    {dna.died_at}",
            f"  Cause:   {dna.cause_of_death}",
            f"  Life:    {dna.lifespan_seconds / 3600:.1f}h",
            "-" * 50,
            f"  Capital: ${dna.starting_capital:.2f} -> ${dna.final_capital:.2f} (peak ${dna.peak_capital:.2f})",
            f"  P&L:     ${dna.total_pnl:+.2f}",
            f"  Trades:  {dna.total_trades} | WR: {dna.win_rate:.1%} | DD: {dna.max_drawdown_pct:.1f}%",
            "-" * 50,
        ]
        for name, gene in sorted(dna.strategy_genes.items(), key=lambda x: x[1].confidence_score, reverse=True):
            e = "+" if gene.confidence_score > 0.6 else "~" if gene.confidence_score > 0.4 else "-"
            lines.append(f"  [{e}] {name}: {gene.win_rate:.0%} WR | ${gene.total_pnl:+.2f} | conf:{gene.confidence_score:.2f}")
        if dna.rules:
            lines.append("-" * 50)
            for r in dna.rules[-5:]:
                lines.append(f"  * {r}")
        lines.append("=" * 50)
        return "\n".join(lines)

    def synthesize_inherited_dna(self, max_gens=50) -> DNA:
        all_dna = self.load_all_dna()
        if not all_dna:
            return DNA(generation=0, born_at=datetime.now(timezone.utc).isoformat())

        recent = all_dna[-max_gens:]
        new_gen = recent[-1].generation + 1
        inherited = DNA(generation=new_gen, born_at=datetime.now(timezone.utc).isoformat())

        agg: Dict[str, Dict] = {}
        for d in recent:
            for name, gene in d.strategy_genes.items():
                if name not in agg:
                    agg[name] = {"used": 0, "wins": 0, "losses": 0, "pnl": 0.0}
                a = agg[name]
                a["used"] += gene.times_used
                a["wins"] += gene.wins
                a["losses"] += gene.losses
                a["pnl"] += gene.total_pnl

        for name, a in agg.items():
            g = StrategyGene(name=name, times_used=a["used"], wins=a["wins"],
                             losses=a["losses"], total_pnl=a["pnl"],
                             avg_pnl_per_trade=a["pnl"] / a["used"] if a["used"] > 0 else 0)
            g.update_confidence()
            inherited.strategy_genes[name] = g

        all_rules = []
        for d in recent:
            all_rules.extend(d.rules)
        inherited.rules = list(dict.fromkeys(all_rules))

        bp = set()
        for d in recent:
            bp.update(d.blacklisted_pairs)
        inherited.blacklisted_pairs = list(bp)

        deaths = [d.cause_of_death for d in recent if d.cause_of_death]
        for cause, count in Counter(deaths).most_common(3):
            rule = f"WARNING: {count} ancestors died from: {cause}"
            if rule not in inherited.rules:
                inherited.rules.append(rule)

        return inherited

    def spawn_new_generation(self) -> DNA:
        return self.synthesize_inherited_dna()
