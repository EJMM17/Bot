"""Logging system with structured trade journal."""

import logging
import json
import os
from datetime import datetime, timezone
from typing import Optional


class DarwinLogger:
    def __init__(self, generation: int = 0, log_level: str = "INFO"):
        self.generation = generation
        self.log_dir = "data/logs"
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = logging.getLogger(f"Darwin.Gen{generation}")
        self.logger.setLevel(getattr(logging, log_level, logging.INFO))
        self.logger.handlers.clear()

        fmt = logging.Formatter(
            f"[Gen-{generation}] %(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S"
        )

        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level, logging.INFO))
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(f"{self.log_dir}/gen_{generation}_{ts}.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(fh)

        self.trade_journal = f"{self.log_dir}/trades_gen_{generation}.jsonl"

    def born(self, capital: float, inherited_from: Optional[int] = None):
        msg = f"BORN Capital: ${capital:.2f}"
        if inherited_from is not None:
            msg += f" | DNA from Gen-{inherited_from}"
        self.logger.info(msg)

    def trade(self, action: str, market: str, symbol: str, amount: float,
              price: float, reason: str, result: Optional[dict] = None):
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "gen": self.generation,
            "action": action,
            "market": market,
            "symbol": symbol,
            "amount": round(amount, 6),
            "price": round(price, 4),
            "reason": reason,
            "result": result
        }
        try:
            with open(self.trade_journal, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
        side = "BUY" if action == "BUY" else "SELL"
        self.logger.info(f"{side} {symbol} x{amount:.6f} @ ${price:.4f} | {reason}")

    def health_update(self, hp: float, change: float, reason: str):
        self.logger.info(f"HP: {hp:.1f}/100 ({change:+.1f}) | {reason}")

    def death(self, cause: str, final_capital: float, trades_total: int,
              win_rate: float, lifespan_hours: float):
        self.logger.critical(
            f"DEATH Cause: {cause} | Capital: ${final_capital:.2f} | "
            f"Trades: {trades_total} | WR: {win_rate:.1%} | Life: {lifespan_hours:.1f}h"
        )

    def evolution(self, msg: str):
        self.logger.info(f"EVO {msg}")

    def warning(self, msg: str):
        self.logger.warning(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def error(self, msg: str):
        self.logger.error(msg)
