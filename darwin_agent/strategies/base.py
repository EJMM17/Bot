"""Trading strategies — Momentum, Mean Reversion, Scalping, Breakout."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from darwin_agent.markets.base import Candle, MarketSignal, OrderSide, TimeFrame
import math


class Strategy(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def analyze(self, candles: List[Candle], symbol: str,
                timeframe: TimeFrame) -> Optional[MarketSignal]:
        pass

    @staticmethod
    def ema(values: List[float], period: int) -> List[float]:
        if len(values) < period:
            return []
        result = [sum(values[:period]) / period]
        m = 2 / (period + 1)
        for p in values[period:]:
            result.append((p - result[-1]) * m + result[-1])
        return result

    @staticmethod
    def sma(values: List[float], period: int) -> List[float]:
        if len(values) < period:
            return []
        return [sum(values[i:i + period]) / period for i in range(len(values) - period + 1)]

    @staticmethod
    def rsi(values: List[float], period: int = 14) -> List[float]:
        if len(values) < period + 1:
            return []
        deltas = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        gains = [max(0, d) for d in deltas]
        losses = [max(0, -d) for d in deltas]
        ag = sum(gains[:period]) / period
        al = sum(losses[:period]) / period
        result = []
        for i in range(period, len(deltas)):
            ag = (ag * (period - 1) + gains[i]) / period
            al = (al * (period - 1) + losses[i]) / period
            result.append(100 if al == 0 else 100 - 100 / (1 + ag / al))
        return result

    @staticmethod
    def atr(candles: List[Candle], period: int = 14) -> List[float]:
        if len(candles) < period + 1:
            return []
        trs = []
        for i in range(1, len(candles)):
            trs.append(max(
                candles[i].high - candles[i].low,
                abs(candles[i].high - candles[i - 1].close),
                abs(candles[i].low - candles[i - 1].close)
            ))
        result = [sum(trs[:period]) / period]
        for i in range(period, len(trs)):
            result.append((result[-1] * (period - 1) + trs[i]) / period)
        return result

    @staticmethod
    def bollinger(values: List[float], period: int = 20, std_mult: float = 2.0):
        if len(values) < period:
            return [], [], []
        mid = [sum(values[i:i + period]) / period for i in range(len(values) - period + 1)]
        upper, lower = [], []
        for i in range(len(mid)):
            w = values[i:i + period]
            m = mid[i]
            std = math.sqrt(sum((x - m) ** 2 for x in w) / period)
            upper.append(m + std_mult * std)
            lower.append(m - std_mult * std)
        return upper, mid, lower


class MomentumStrategy(Strategy):
    def __init__(self):
        super().__init__("momentum", "EMA crossover + RSI confirmation + volume filter")

    def analyze(self, candles: List[Candle], symbol: str,
                timeframe: TimeFrame) -> Optional[MarketSignal]:
        if len(candles) < 35:
            return None
        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]
        fast = self.ema(closes, 9)
        slow = self.ema(closes, 21)
        rsi_vals = self.rsi(closes, 14)
        atr_vals = self.atr(candles, 14)

        if not fast or not slow or not rsi_vals or not atr_vals:
            return None

        cf, pf = fast[-1], fast[-2] if len(fast) > 1 else fast[-1]
        cs, ps = slow[-1], slow[-2] if len(slow) > 1 else slow[-1]
        cur_rsi = rsi_vals[-1]
        cur_atr = atr_vals[-1]
        price = closes[-1]
        avg_vol = sum(volumes[-20:]) / min(20, len(volumes))
        vol_ok = volumes[-1] > avg_vol * 0.8

        # Bullish
        bull_cross = pf <= ps and cf > cs
        bull_mom = cf > cs and cur_rsi < 65
        if (bull_cross or bull_mom) and cur_rsi > 35 and vol_ok:
            conf = 0.5
            if bull_cross: conf += 0.15
            if cur_rsi < 45: conf += 0.1
            if volumes[-1] > avg_vol * 1.5: conf += 0.1
            return MarketSignal(
                symbol=symbol, side=OrderSide.BUY, strategy=self.name,
                confidence=min(0.95, conf), entry_price=price,
                stop_loss=price - cur_atr * 2, take_profit=price + cur_atr * 3,
                timeframe=timeframe,
                reason=f"Bull momentum RSI:{cur_rsi:.0f} cross:{bull_cross}",
            )

        # Bearish
        bear_cross = pf >= ps and cf < cs
        bear_mom = cf < cs and cur_rsi > 35
        if (bear_cross or bear_mom) and cur_rsi < 65 and vol_ok:
            conf = 0.5
            if bear_cross: conf += 0.15
            if cur_rsi > 55: conf += 0.1
            if volumes[-1] > avg_vol * 1.5: conf += 0.1
            return MarketSignal(
                symbol=symbol, side=OrderSide.SELL, strategy=self.name,
                confidence=min(0.95, conf), entry_price=price,
                stop_loss=price + cur_atr * 2, take_profit=price - cur_atr * 3,
                timeframe=timeframe,
                reason=f"Bear momentum RSI:{cur_rsi:.0f} cross:{bear_cross}",
            )
        return None


class MeanReversionStrategy(Strategy):
    def __init__(self):
        super().__init__("mean_reversion", "Bollinger Band bounces + RSI extremes")

    def analyze(self, candles: List[Candle], symbol: str,
                timeframe: TimeFrame) -> Optional[MarketSignal]:
        if len(candles) < 30:
            return None
        closes = [c.close for c in candles]
        price = closes[-1]
        upper, mid, lower = self.bollinger(closes, 20, 2.0)
        rsi_vals = self.rsi(closes, 14)
        atr_vals = self.atr(candles)

        if not upper or not rsi_vals or not atr_vals:
            return None

        cur_rsi = rsi_vals[-1]
        cur_atr = atr_vals[-1]

        if price <= lower[-1] and cur_rsi <= 25:
            conf = 0.55
            if cur_rsi < 20: conf += 0.15
            if price < lower[-1] * 0.99: conf += 0.1
            return MarketSignal(
                symbol=symbol, side=OrderSide.BUY, strategy=self.name,
                confidence=min(0.9, conf), entry_price=price,
                stop_loss=price - cur_atr * 1.5, take_profit=mid[-1],
                timeframe=timeframe,
                reason=f"Oversold RSI:{cur_rsi:.0f} below BB",
            )

        if price >= upper[-1] and cur_rsi >= 75:
            conf = 0.55
            if cur_rsi > 80: conf += 0.15
            if price > upper[-1] * 1.01: conf += 0.1
            return MarketSignal(
                symbol=symbol, side=OrderSide.SELL, strategy=self.name,
                confidence=min(0.9, conf), entry_price=price,
                stop_loss=price + cur_atr * 1.5, take_profit=mid[-1],
                timeframe=timeframe,
                reason=f"Overbought RSI:{cur_rsi:.0f} above BB",
            )
        return None


class ScalpingStrategy(Strategy):
    def __init__(self):
        super().__init__("scalping", "Quick micro-profit with VWAP + engulfing")

    def analyze(self, candles: List[Candle], symbol: str,
                timeframe: TimeFrame) -> Optional[MarketSignal]:
        if len(candles) < 30:
            return None
        closes = [c.close for c in candles]
        price = closes[-1]
        atr_vals = self.atr(candles, 10)
        if not atr_vals:
            return None
        cur_atr = atr_vals[-1]

        tp = [(c.high + c.low + c.close) / 3 for c in candles[-20:]]
        vols = [c.volume for c in candles[-20:]]
        total_vol = sum(vols)
        if total_vol == 0:
            return None
        vwap = sum(t * v for t, v in zip(tp, vols)) / total_vol

        prev, curr = candles[-2], candles[-1]
        bull_eng = (not prev.is_bullish and curr.is_bullish and
                    curr.close > prev.open and curr.open < prev.close)
        bear_eng = (prev.is_bullish and not curr.is_bullish and
                    curr.close < prev.open and curr.open > prev.close)

        if price < vwap * 0.999 and bull_eng:
            return MarketSignal(
                symbol=symbol, side=OrderSide.BUY, strategy=self.name,
                confidence=0.6, entry_price=price,
                stop_loss=price - cur_atr, take_profit=price + cur_atr * 1.5,
                timeframe=timeframe, reason="Scalp long: below VWAP + bull engulf",
            )
        if price > vwap * 1.001 and bear_eng:
            return MarketSignal(
                symbol=symbol, side=OrderSide.SELL, strategy=self.name,
                confidence=0.6, entry_price=price,
                stop_loss=price + cur_atr, take_profit=price - cur_atr * 1.5,
                timeframe=timeframe, reason="Scalp short: above VWAP + bear engulf",
            )
        return None


class BreakoutStrategy(Strategy):
    """Breakout strategy — trades range breaks with volume confirmation."""

    def __init__(self):
        super().__init__("breakout", "Range breakout with volume surge")

    def analyze(self, candles: List[Candle], symbol: str,
                timeframe: TimeFrame) -> Optional[MarketSignal]:
        if len(candles) < 40:
            return None
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [c.volume for c in candles]
        price = closes[-1]
        atr_vals = self.atr(candles, 14)
        if not atr_vals:
            return None
        cur_atr = atr_vals[-1]

        # 20-bar range
        range_high = max(highs[-21:-1])
        range_low = min(lows[-21:-1])
        avg_vol = sum(volumes[-20:]) / 20
        vol_surge = volumes[-1] > avg_vol * 1.5

        # Bullish breakout
        if price > range_high and vol_surge:
            conf = 0.6
            if price > range_high * 1.005: conf += 0.1
            if volumes[-1] > avg_vol * 2: conf += 0.1
            return MarketSignal(
                symbol=symbol, side=OrderSide.BUY, strategy=self.name,
                confidence=min(0.9, conf), entry_price=price,
                stop_loss=range_high - cur_atr * 0.5,
                take_profit=price + (price - range_low) * 0.8,
                timeframe=timeframe,
                reason=f"Bullish breakout above {range_high:.2f} vol:{volumes[-1]/avg_vol:.1f}x",
            )

        # Bearish breakout
        if price < range_low and vol_surge:
            conf = 0.6
            if price < range_low * 0.995: conf += 0.1
            if volumes[-1] > avg_vol * 2: conf += 0.1
            return MarketSignal(
                symbol=symbol, side=OrderSide.SELL, strategy=self.name,
                confidence=min(0.9, conf), entry_price=price,
                stop_loss=range_low + cur_atr * 0.5,
                take_profit=price - (range_high - price) * 0.8,
                timeframe=timeframe,
                reason=f"Bearish breakdown below {range_low:.2f} vol:{volumes[-1]/avg_vol:.1f}x",
            )
        return None


STRATEGY_REGISTRY: Dict[str, Strategy] = {
    "momentum": MomentumStrategy(),
    "mean_reversion": MeanReversionStrategy(),
    "scalping": ScalpingStrategy(),
    "breakout": BreakoutStrategy(),
}
