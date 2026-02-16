"""Feature Engineering — Converts candles into a 30-dim ML feature vector.

AI enhancements:
- Adaptive regime detection with online threshold learning
- Better normalization for EMA distances and ROC
- OBV magnitude preserved (not just sign)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from darwin_agent.markets.base import Candle

# Canonical feature order — must always be 30 features
FEATURE_NAMES = [
    # TREND (6)
    "trend_ema9_dist", "trend_ema21_dist", "trend_ema50_dist",
    "trend_alignment", "trend_slope_20", "trend_strength",
    # MOMENTUM (7)
    "momentum_rsi_14", "momentum_rsi_7", "momentum_rsi_divergence",
    "momentum_roc_5", "momentum_roc_10", "momentum_roc_20", "momentum_macd",
    # VOLATILITY (5)
    "vol_atr_14", "vol_bb_width", "vol_bb_position", "vol_ratio", "vol_range_expansion",
    # VOLUME (4)
    "volume_ratio", "volume_trend", "volume_price_corr", "volume_obv_slope",
    # PATTERN (4)
    "pattern_body_ratio", "pattern_upper_wick", "pattern_lower_wick", "pattern_engulfing",
    # CONTEXT (4)
    "context_hour_sin", "context_hour_cos", "context_dow_sin", "context_dow_cos",
]

N_FEATURES = len(FEATURE_NAMES)  # 30


@dataclass
class MarketFeatures:
    timestamp: datetime
    symbol: str
    features: np.ndarray
    feature_names: List[str]
    raw_price: float
    regime: str = "unknown"


class FeatureEngineer:

    def __init__(self):
        # Adaptive regime thresholds: updated online as agent sees more data
        self._regime_trend_threshold = 0.6
        self._regime_vol_high = 2.0
        self._regime_vol_very_high = 3.0
        self._regime_vol_low = 1.5
        # Exponential moving averages for adaptive thresholds
        self._ema_trend = 0.5
        self._ema_vol = 2.0
        self._regime_samples = 0

    def extract(self, candles: List[Candle], symbol: str) -> Optional[MarketFeatures]:
        if len(candles) < 50:
            return None

        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])
        price = closes[-1]

        f = {}

        # ATR for normalization
        atr_14 = self._atr(highs, lows, closes, 14)
        atr_val = atr_14[-1] if len(atr_14) > 0 else max(price * 0.01, 0.01)

        # === TREND ===
        ema9 = self._ema(closes, 9)
        ema21 = self._ema(closes, 21)
        ema50 = self._ema(closes, min(50, len(closes) - 1))

        # FIX: Normalize EMA distance by price (percentage), not ATR
        f["trend_ema9_dist"] = (price - ema9[-1]) / price * 100 if len(ema9) > 0 and price > 0 else 0
        f["trend_ema21_dist"] = (price - ema21[-1]) / price * 100 if len(ema21) > 0 and price > 0 else 0
        f["trend_ema50_dist"] = (price - ema50[-1]) / price * 100 if len(ema50) > 0 and price > 0 else 0

        alignment = 0.0
        if len(ema9) > 0 and len(ema21) > 0 and len(ema50) > 0:
            if ema9[-1] > ema21[-1]: alignment += 0.33
            else: alignment -= 0.33
            if ema21[-1] > ema50[-1]: alignment += 0.33
            else: alignment -= 0.33
            if price > ema9[-1]: alignment += 0.34
            else: alignment -= 0.34
        f["trend_alignment"] = alignment

        x = np.arange(20)
        y = closes[-20:]
        slope = np.polyfit(x, y, 1)[0] if len(y) == 20 else 0
        f["trend_slope_20"] = slope / atr_val if atr_val > 0 else 0

        f["trend_strength"] = self._adx_approx(highs, lows, closes, 14)

        # === MOMENTUM ===
        rsi14 = self._rsi(closes, 14)
        rsi7 = self._rsi(closes, 7)
        f["momentum_rsi_14"] = (rsi14[-1] - 50) / 50 if len(rsi14) > 0 else 0
        f["momentum_rsi_7"] = (rsi7[-1] - 50) / 50 if len(rsi7) > 0 else 0

        if len(rsi14) >= 10:
            ph = closes[-1] > closes[-10]
            rh = rsi14[-1] > rsi14[-10]
            f["momentum_rsi_divergence"] = -1.0 if ph and not rh else (1.0 if not ph and rh else 0.0)
        else:
            f["momentum_rsi_divergence"] = 0.0

        # FIX: ROC with tanh normalization to handle extreme values
        def safe_roc(c, n):
            if len(c) >= n and c[-n] > 0:
                raw = (c[-1] / c[-n] - 1) * 100
                return float(np.tanh(raw / 5))  # Soft clip to [-1, 1]
            return 0.0

        f["momentum_roc_5"] = safe_roc(closes, 5)
        f["momentum_roc_10"] = safe_roc(closes, 10)
        f["momentum_roc_20"] = safe_roc(closes, 20)

        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        f["momentum_macd"] = (ema12[-1] - ema26[-1]) / atr_val if len(ema12) > 0 and len(ema26) > 0 and atr_val > 0 else 0

        # === VOLATILITY ===
        f["vol_atr_14"] = atr_val / price * 100 if price > 0 else 0

        bb_u, bb_m, bb_l = self._bollinger(closes, 20, 2.0)
        if len(bb_u) > 0:
            bb_w = bb_u[-1] - bb_l[-1]
            f["vol_bb_width"] = bb_w / bb_m[-1] * 100 if bb_m[-1] > 0 else 0
            f["vol_bb_position"] = (price - bb_l[-1]) / bb_w * 2 - 1 if bb_w > 0 else 0
        else:
            f["vol_bb_width"] = 0
            f["vol_bb_position"] = 0

        atr7 = self._atr(highs, lows, closes, 7)
        f["vol_ratio"] = atr7[-1] / atr_val if len(atr7) > 0 and atr_val > 0 else 1.0

        rec_r = np.mean((highs[-5:] - lows[-5:]) / np.maximum(closes[-5:], 1e-8) * 100)
        old_r = np.mean((highs[-20:-5] - lows[-20:-5]) / np.maximum(closes[-20:-5], 1e-8) * 100)
        f["vol_range_expansion"] = rec_r / old_r if old_r > 0 else 1.0

        # === VOLUME ===
        avg_vol = np.mean(volumes[-20:])
        f["volume_ratio"] = volumes[-1] / avg_vol if avg_vol > 0 else 1
        f["volume_trend"] = np.mean(volumes[-5:]) / avg_vol if avg_vol > 0 else 1

        pc = np.diff(closes[-10:])
        vc = np.diff(volumes[-10:])
        if len(pc) > 0 and np.std(pc) > 0 and np.std(vc) > 0:
            f["volume_price_corr"] = float(np.corrcoef(pc, vc)[0, 1])
        else:
            f["volume_price_corr"] = 0

        # FIX: OBV slope preserves magnitude (not just sign)
        obv = np.where(np.diff(closes[-20:]) > 0, volumes[-19:], -volumes[-19:])
        if len(obv) > 1:
            cumobv = np.cumsum(obv)
            slope_val = np.polyfit(np.arange(len(cumobv)), cumobv, 1)[0]
            # Normalize by average volume to get comparable values across symbols
            f["volume_obv_slope"] = float(np.tanh(slope_val / max(avg_vol, 1)))
        else:
            f["volume_obv_slope"] = 0

        # === PATTERN ===
        last = candles[-1]
        prev = candles[-2]
        tr = last.high - last.low
        body = abs(last.close - last.open)
        f["pattern_body_ratio"] = body / tr if tr > 0 else 0
        if tr > 0:
            f["pattern_upper_wick"] = (last.high - max(last.open, last.close)) / tr
            f["pattern_lower_wick"] = (min(last.open, last.close) - last.low) / tr
        else:
            f["pattern_upper_wick"] = 0
            f["pattern_lower_wick"] = 0

        bull_eng = (not prev.is_bullish and last.is_bullish and
                    last.close > prev.open and last.open < prev.close)
        bear_eng = (prev.is_bullish and not last.is_bullish and
                    last.close < prev.open and last.open > prev.close)
        f["pattern_engulfing"] = 1.0 if bull_eng else (-1.0 if bear_eng else 0.0)

        # === CONTEXT ===
        now = candles[-1].timestamp
        f["context_hour_sin"] = np.sin(2 * np.pi * now.hour / 24)
        f["context_hour_cos"] = np.cos(2 * np.pi * now.hour / 24)
        f["context_dow_sin"] = np.sin(2 * np.pi * now.weekday() / 7)
        f["context_dow_cos"] = np.cos(2 * np.pi * now.weekday() / 7)

        # Build vector in canonical order
        vec = np.array([f.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float32)
        vec = np.nan_to_num(vec, nan=0.0, posinf=3.0, neginf=-3.0)
        vec = np.clip(vec, -5.0, 5.0)

        regime = self._detect_regime(f)
        return MarketFeatures(
            timestamp=now, symbol=symbol, features=vec,
            feature_names=list(FEATURE_NAMES), raw_price=price, regime=regime,
        )

    def _detect_regime(self, f: Dict[str, float]) -> str:
        """Adaptive regime detection with online threshold learning."""
        trend = abs(f.get("trend_alignment", 0))
        vol = f.get("vol_bb_width", 0)

        # Update adaptive thresholds via exponential moving average
        alpha = 0.01  # Slow adaptation
        self._ema_trend = self._ema_trend * (1 - alpha) + trend * alpha
        self._ema_vol = self._ema_vol * (1 - alpha) + vol * alpha
        self._regime_samples += 1

        # After enough samples, start adapting thresholds
        if self._regime_samples > 100:
            self._regime_trend_threshold = max(0.3, min(0.8, self._ema_trend * 1.2))
            self._regime_vol_high = max(1.0, min(4.0, self._ema_vol * 1.0))
            self._regime_vol_very_high = self._regime_vol_high * 1.5
            self._regime_vol_low = self._regime_vol_high * 0.75

        if trend > self._regime_trend_threshold and vol > self._regime_vol_high:
            return "trending_volatile"
        if trend > self._regime_trend_threshold:
            return "trending_calm"
        if vol > self._regime_vol_very_high:
            return "choppy"
        if vol < self._regime_vol_low:
            return "ranging_tight"
        return "ranging_normal"

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        if len(data) < period:
            return np.array([])
        ema = np.zeros(len(data) - period + 1)
        ema[0] = np.mean(data[:period])
        m = 2 / (period + 1)
        for i in range(1, len(ema)):
            ema[i] = (data[period - 1 + i] - ema[i - 1]) * m + ema[i - 1]
        return ema

    @staticmethod
    def _rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        if len(data) < period + 1:
            return np.array([])
        d = np.diff(data)
        g = np.where(d > 0, d, 0)
        l = np.where(d < 0, -d, 0)
        ag = np.mean(g[:period])
        al = np.mean(l[:period])
        r = []
        for i in range(period, len(d)):
            ag = (ag * (period - 1) + g[i]) / period
            al = (al * (period - 1) + l[i]) / period
            r.append(100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al))
        return np.array(r)

    @staticmethod
    def _atr(highs, lows, closes, period=14):
        if len(highs) < period + 1:
            return np.array([])
        tr = np.maximum(highs[1:] - lows[1:],
                        np.maximum(np.abs(highs[1:] - closes[:-1]),
                                   np.abs(lows[1:] - closes[:-1])))
        atr = np.zeros(len(tr) - period + 1)
        atr[0] = np.mean(tr[:period])
        for i in range(1, len(atr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[period - 1 + i]) / period
        return atr

    @staticmethod
    def _bollinger(data, period=20, std_mult=2.0):
        if len(data) < period:
            return np.array([]), np.array([]), np.array([])
        mid = np.convolve(data, np.ones(period) / period, mode="valid")
        u, l = np.zeros_like(mid), np.zeros_like(mid)
        for i in range(len(mid)):
            s = np.std(data[i:i + period])
            u[i] = mid[i] + std_mult * s
            l[i] = mid[i] - std_mult * s
        return u, mid, l

    @staticmethod
    def _adx_approx(highs, lows, closes, period=14) -> float:
        if len(closes) < period * 2:
            return 0.5
        pr = np.max(highs[-period:]) - np.min(lows[-period:])
        d = abs(closes[-1] - closes[-period])
        return min(1.0, d / pr) if pr > 0 else 0.0
