"""Market Adapters — Abstract interface for different exchanges."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from enum import Enum


def _utcnow():
    return datetime.now(timezone.utc)



class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class TimeFrame(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def body_pct(self) -> float:
        return abs(self.close - self.open) / self.open * 100 if self.open > 0 else 0

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open


@dataclass
class Position:
    id: str
    symbol: str
    side: OrderSide
    entry_price: float
    quantity: float
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: datetime = field(default_factory=_utcnow)
    pnl: float = 0.0
    pnl_pct: float = 0.0
    has_server_sltp: bool = False  # True if exchange handles SL/TP server-side

    def update_pnl(self, current_price: float):
        self.current_price = current_price
        if self.side == OrderSide.BUY:
            self.pnl = (current_price - self.entry_price) * self.quantity
            self.pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100 if self.entry_price else 0
        else:
            self.pnl = (self.entry_price - current_price) * self.quantity
            self.pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100 if self.entry_price else 0


@dataclass
class TradeResult:
    success: bool
    order_id: Optional[str] = None
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    price: float = 0.0
    quantity: float = 0.0
    fee: float = 0.0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=_utcnow)


@dataclass
class MarketSignal:
    symbol: str
    side: OrderSide
    strategy: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timeframe: TimeFrame
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarketAdapter(ABC):
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.is_connected = False
        self.is_testnet = config.get("testnet", True)

    @abstractmethod
    async def connect(self) -> bool: ...

    @abstractmethod
    async def disconnect(self): ...

    @abstractmethod
    async def get_balance(self) -> float: ...

    @abstractmethod
    async def get_candles(self, symbol: str, timeframe: TimeFrame, limit: int = 100) -> List[Candle]: ...

    @abstractmethod
    async def get_current_price(self, symbol: str) -> float: ...

    @abstractmethod
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                          quantity: float, price: Optional[float] = None,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> TradeResult: ...

    @abstractmethod
    async def close_position(self, position: Position) -> TradeResult: ...

    @abstractmethod
    async def get_open_positions(self) -> List[Position]: ...

    @abstractmethod
    async def get_min_trade_size(self, symbol: str) -> float: ...
