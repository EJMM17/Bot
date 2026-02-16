"""
Crypto Market Adapter — Bybit V5 API (Production-Grade).

Bybit compliance:
- Server time synchronization (prevents signature rejection from clock drift)
- Quantity rounding to qtyStep (Bybit rejects non-compliant qty)
- Price rounding to tickSize (Bybit rejects non-compliant SL/TP)
- Rate limit handling (retCode 10006, 10018)
- recv_window validation
- Proper HMAC-SHA256 signature per V5 spec
- Session reconnection on disconnect/timeout
- Graceful handling of maintenance windows

Server resilience:
- Auto-reconnect with exponential backoff
- Session health checks before every request
- Timeout per request (15s) + overall session timeout
- Cached instrument info to avoid repeated calls
"""

import asyncio
import hmac
import hashlib
import time
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict
from dataclasses import dataclass

try:
    import aiohttp
except ImportError:
    aiohttp = None

from darwin_agent.markets.base import (
    MarketAdapter, OrderSide, OrderType, TimeFrame,
    Candle, Position, TradeResult
)
from darwin_agent.markets.bybit_errors import (
    get_error_info, ErrorAction, preflight_check
)


@dataclass
class InstrumentInfo:
    """Cached instrument trading rules from Bybit."""
    symbol: str
    min_qty: float
    qty_step: float
    tick_size: float
    min_notional: float = 5.0
    fetched_at: float = 0.0


class BybitAdapter(MarketAdapter):
    """
    Bybit V5 API for USDT linear perpetual futures.
    Docs: https://bybit-exchange.github.io/docs/v5/intro
    """

    TESTNET_URL = "https://api-testnet.bybit.com"
    MAINNET_URL = "https://api.bybit.com"
    RECV_WINDOW = "5000"

    TIMEFRAME_MAP = {
        TimeFrame.M1: "1", TimeFrame.M5: "5", TimeFrame.M15: "15",
        TimeFrame.H1: "60", TimeFrame.H4: "240", TimeFrame.D1: "D",
    }

    def __init__(self, config: dict):
        super().__init__(name="crypto_bybit", config=config)
        self.api_key = config.get("api_key", "")
        self.api_secret = config.get("api_secret", "")
        self.base_url = self.TESTNET_URL if self.is_testnet else self.MAINNET_URL
        self.session: Optional[aiohttp.ClientSession] = None
        self.category = "linear"

        # Time sync: difference between local clock and Bybit server (ms)
        self._time_offset_ms: int = 0
        self._last_time_sync: float = 0.0

        # Instrument cache
        self._instruments: Dict[str, InstrumentInfo] = {}
        self._instruments_cache_ttl = 3600  # 1 hour

        # Reconnection state
        self._reconnect_lock = asyncio.Lock()
        self._consecutive_errors = 0
        self._max_consecutive_errors = 10

    # ═══════════════════════════════════════════
    # Time Synchronization
    # ═══════════════════════════════════════════

    async def _sync_time(self):
        """
        Sync local clock with Bybit server.
        Bybit rejects signatures if timestamp differs by > recv_window (5s).
        Handles: VPS clock drift, NTP misconfiguration, container desync.
        """
        try:
            if not self.session or self.session.closed:
                return
            local_before = int(time.time() * 1000)
            async with self.session.get(
                f"{self.base_url}/v5/market/time",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                data = await resp.json()
            local_after = int(time.time() * 1000)

            if data.get("retCode") == 0:
                server_time = int(data["result"]["timeSecond"]) * 1000
                local_mid = (local_before + local_after) // 2
                self._time_offset_ms = server_time - local_mid
                self._last_time_sync = time.time()
        except Exception:
            pass  # Keep last known offset

    def _get_timestamp_ms(self) -> str:
        """Current timestamp adjusted for server time offset."""
        return str(int(time.time() * 1000) + self._time_offset_ms)

    async def _maybe_sync_time(self):
        """Re-sync time every 5 minutes."""
        if time.time() - self._last_time_sync > 300:
            await self._sync_time()

    # ═══════════════════════════════════════════
    # Authentication
    # ═══════════════════════════════════════════

    def _sign(self, timestamp: str, params: str) -> str:
        """Bybit V5 HMAC-SHA256: timestamp + api_key + recv_window + payload"""
        sign_str = f"{timestamp}{self.api_key}{self.RECV_WINDOW}{params}"
        return hmac.new(
            self.api_secret.encode("utf-8"),
            sign_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _auth_headers(self, timestamp: str, params: str) -> dict:
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": self._sign(timestamp, params),
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": self.RECV_WINDOW,
            "Content-Type": "application/json",
        }

    # ═══════════════════════════════════════════
    # Connection Management
    # ═══════════════════════════════════════════

    async def connect(self) -> bool:
        """
        Connect to Bybit with time synchronization.
        On failure, prints EXACTLY what went wrong.
        """
        if aiohttp is None:
            raise ImportError("aiohttp required: pip install aiohttp")

        timeout = aiohttp.ClientTimeout(total=15, connect=10)
        env = "TESTNET" if self.is_testnet else "MAINNET"

        for attempt in range(3):
            try:
                if self.session and not self.session.closed:
                    await self.session.close()
                    self.session = None

                connector = aiohttp.TCPConnector(
                    limit=10, ttl_dns_cache=300, enable_cleanup_closed=True,
                )

                self.session = aiohttp.ClientSession(
                    timeout=timeout, connector=connector, trust_env=True
                )
                await self._sync_time()

                async with self.session.get(f"{self.base_url}/v5/market/time") as resp:
                    # HTTP 403 = geo-blocked
                    if resp.status == 403:
                        print(f"\n❌ [Bybit {env}] HTTP 403 — Your server IP is GEO-BLOCKED")
                        print(f"   Bybit blocks US and Mainland China IPs.")
                        print(f"   Solution: Deploy on a server in EU, Singapore, or Japan.")
                        print(f"   Recommended: DigitalOcean Singapore ($6/mo) or Hetzner EU (€4/mo)")
                        await self.disconnect()
                        return False

                    data = await resp.json()
                    if data.get("retCode") == 0:
                        self.is_connected = True
                        self._consecutive_errors = 0
                        offset_s = abs(self._time_offset_ms) / 1000
                        print(f"✅ [Bybit {env}] Connected | Clock offset: {offset_s:.2f}s")
                        return True
                    else:
                        ret = data.get("retCode", -1)
                        err = get_error_info(ret)
                        print(f"⚠️ [Bybit {env}] Server error: {ret} {err.name}")
                        if err.user_message:
                            print(f"   {err.user_message}")

            except aiohttp.ClientConnectorError as e:
                print(f"⚠️ [Bybit {env}] Cannot connect (attempt {attempt+1}/3): {e}")
                if "Name or service not known" in str(e):
                    print(f"   DNS resolution failed for {self.base_url}")
                    print(f"   Check: 1) Internet  2) DNS config  3) Firewall")
                elif "Connection refused" in str(e):
                    print(f"   Connection refused. Bybit may be down or IP blocked.")

            except asyncio.TimeoutError:
                print(f"⚠️ [Bybit {env}] Timeout (attempt {attempt+1}/3)")
                if attempt == 2:
                    print(f"   Your server may have high latency to {self.base_url}")
                    print(f"   Bybit servers are in AWS Singapore/Tokyo.")

            except Exception as e:
                print(f"⚠️ [Bybit {env}] Error (attempt {attempt+1}/3): {e}")

            finally:
                if not self.is_connected and self.session and not self.session.closed:
                    await self.session.close()
                    self.session = None

            if attempt < 2:
                await asyncio.sleep(2 ** attempt)

        print(f"\n❌ [Bybit {env}] Failed to connect after 3 attempts.")
        print(f"   Run diagnostics: python -m darwin_agent --diagnose")
        return False

    async def _ensure_connected(self) -> bool:
        """Ensure session alive, reconnect if dead."""
        if self.session and not self.session.closed and self.is_connected:
            return True
        async with self._reconnect_lock:
            if self.session and not self.session.closed and self.is_connected:
                return True
            return await self.connect()

    async def disconnect(self):
        if self.session and not self.session.closed:
            await self.session.close()
        self.is_connected = False

    # ═══════════════════════════════════════════
    # Request Engine
    # ═══════════════════════════════════════════

    async def _request(self, method: str, endpoint: str,
                       params: dict = None, auth: bool = False) -> dict:
        """
        Request with: auto-reconnect, time re-sync, rate limit handling,
        exponential backoff, Bybit error catalog integration.

        Every Bybit retCode is handled with a specific action:
        - RETRY: wait and retry
        - RESYNC_TIME: re-sync clock, then retry
        - FATAL_AUTH: log detailed fix instructions, stop retrying
        - RATE_LIMIT: exponential backoff
        - MAINTENANCE: long wait + retry
        - GEO_BLOCKED: log and stop (need different server)
        """
        if not await self._ensure_connected():
            return {"retCode": -1, "retMsg": "Connection failed"}

        params = params or {}

        for attempt in range(3):
            try:
                if auth:
                    await self._maybe_sync_time()

                ts = self._get_timestamp_ms()

                if method == "GET":
                    sorted_params = sorted(params.items())
                    param_str = "&".join(f"{k}={v}" for k, v in sorted_params)
                    headers = self._auth_headers(ts, param_str) if auth else {}
                    async with self.session.get(
                        f"{self.base_url}{endpoint}", params=params, headers=headers
                    ) as resp:
                        # Check for HTTP-level errors (403 = geo-block)
                        if resp.status == 403:
                            return {"retCode": 10024, "retMsg": "HTTP 403 — IP geo-blocked by Bybit"}
                        data = await resp.json()
                else:
                    body = json.dumps(params, separators=(',', ':'))
                    headers = self._auth_headers(ts, body) if auth else {}
                    async with self.session.post(
                        f"{self.base_url}{endpoint}", data=body, headers=headers
                    ) as resp:
                        if resp.status == 403:
                            return {"retCode": 10024, "retMsg": "HTTP 403 — IP geo-blocked by Bybit"}
                        data = await resp.json()

                ret = data.get("retCode", -1)

                # Success
                if ret == 0:
                    self._consecutive_errors = 0
                    return data

                # Look up error in catalog
                err_info = get_error_info(ret)
                ret_msg = data.get("retMsg", err_info.description)

                # Log the error with Bybit-specific context
                if err_info.is_critical:
                    print(f"[Bybit] ❌ CRITICAL ERROR {ret}: {err_info.name}")
                    print(f"[Bybit] {err_info.user_message}")

                # Decide action based on error catalog
                if err_info.action == ErrorAction.RETRY:
                    delay = err_info.retry_delay_seconds or (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue

                elif err_info.action == ErrorAction.RESYNC_TIME:
                    print(f"[Bybit] ⏰ Clock drift detected (retCode {ret}). Re-syncing...")
                    await self._sync_time()
                    if attempt < 2:
                        continue
                    return data

                elif err_info.action == ErrorAction.RATE_LIMIT:
                    delay = err_info.retry_delay_seconds * (attempt + 1)
                    print(f"[Bybit] ⚡ Rate limited. Waiting {delay:.0f}s...")
                    await asyncio.sleep(delay)
                    continue

                elif err_info.action == ErrorAction.MAINTENANCE:
                    print(f"[Bybit] 🔧 Maintenance detected. Waiting 30s...")
                    await asyncio.sleep(30)
                    if attempt < 2:
                        continue
                    return data

                elif err_info.action in (ErrorAction.FATAL_AUTH, ErrorAction.BANNED,
                                          ErrorAction.GEO_BLOCKED, ErrorAction.UPGRADE_ACCOUNT):
                    # Non-recoverable — return immediately, don't retry
                    return {"retCode": ret, "retMsg": ret_msg,
                            "_error_info": err_info.user_message}

                elif err_info.action in (ErrorAction.SKIP, ErrorAction.FIX_PARAMS,
                                          ErrorAction.INSUFFICIENT, ErrorAction.REDUCE_SIZE):
                    # Trade-specific error — return to caller to handle
                    return data

                else:
                    # Unknown action — retry once then return
                    if attempt < 1:
                        await asyncio.sleep(2)
                        continue
                    return data

            except asyncio.TimeoutError:
                self._consecutive_errors += 1
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {"retCode": -1, "retMsg": "Timeout after 3 attempts"}

            except (aiohttp.ClientError, ConnectionError, OSError) as e:
                self._consecutive_errors += 1
                if self._consecutive_errors >= self._max_consecutive_errors:
                    self.is_connected = False
                if attempt < 2:
                    self.is_connected = False
                    if await self._ensure_connected():
                        continue
                return {"retCode": -1, "retMsg": f"Network error: {e}"}

            except Exception as e:
                self._consecutive_errors += 1
                return {"retCode": -1, "retMsg": f"Unexpected: {e}"}

        return {"retCode": -1, "retMsg": "Max retries exhausted"}

    # ═══════════════════════════════════════════
    # Instrument Info (qty/price rounding)
    # ═══════════════════════════════════════════

    async def _get_instrument(self, symbol: str) -> Optional[InstrumentInfo]:
        """Get instrument trading rules (cached 1h)."""
        cached = self._instruments.get(symbol)
        if cached and (time.time() - cached.fetched_at) < self._instruments_cache_ttl:
            return cached

        data = await self._request("GET", "/v5/market/instruments-info", {
            "category": self.category, "symbol": symbol
        })
        if data.get("retCode") == 0 and data["result"]["list"]:
            info = data["result"]["list"][0]
            lot = info.get("lotSizeFilter", {})
            pf = info.get("priceFilter", {})
            inst = InstrumentInfo(
                symbol=symbol,
                min_qty=float(lot.get("minOrderQty", "0.001")),
                qty_step=float(lot.get("qtyStep", "0.001")),
                tick_size=float(pf.get("tickSize", "0.01")),
                min_notional=float(lot.get("minNotionalValue", "5")),
                fetched_at=time.time(),
            )
            self._instruments[symbol] = inst
            return inst
        return None

    @staticmethod
    def _round_step(value: float, step: float) -> float:
        """Round down to nearest step. Avoids floating point drift."""
        if step <= 0:
            return value
        step_str = f"{step:.10f}".rstrip('0')
        decimals = len(step_str.split('.')[1]) if '.' in step_str else 0
        rounded = int(value / step) * step
        return round(rounded, decimals)

    def _fmt_qty(self, qty: float, inst: InstrumentInfo) -> str:
        qty = self._round_step(qty, inst.qty_step)
        qty = max(qty, inst.min_qty)
        step_str = f"{inst.qty_step:.10f}".rstrip('0')
        dec = len(step_str.split('.')[1]) if '.' in step_str else 0
        return f"{qty:.{dec}f}"

    def _fmt_price(self, price: float, inst: InstrumentInfo) -> str:
        price = self._round_step(price, inst.tick_size)
        tick_str = f"{inst.tick_size:.10f}".rstrip('0')
        dec = len(tick_str.split('.')[1]) if '.' in tick_str else 0
        return f"{price:.{dec}f}"

    # ═══════════════════════════════════════════
    # Market Data
    # ═══════════════════════════════════════════

    async def get_balance(self) -> float:
        data = await self._request("GET", "/v5/account/wallet-balance",
                                   {"accountType": "UNIFIED"}, auth=True)
        if data.get("retCode") == 0:
            try:
                for coin in data["result"]["list"][0]["coin"]:
                    if coin["coin"] == "USDT":
                        return float(coin["walletBalance"])
            except (KeyError, IndexError):
                pass
        return 0.0

    async def get_candles(self, symbol: str, timeframe: TimeFrame,
                          limit: int = 100) -> List[Candle]:
        # Bybit V5 kline limit: max 200 for linear
        limit = min(limit, 200)
        data = await self._request("GET", "/v5/market/kline", {
            "category": self.category, "symbol": symbol,
            "interval": self.TIMEFRAME_MAP[timeframe], "limit": limit,
        })
        candles = []
        if data.get("retCode") == 0:
            try:
                for item in reversed(data["result"]["list"]):
                    candles.append(Candle(
                        timestamp=datetime.fromtimestamp(int(item[0]) / 1000, tz=timezone.utc),
                        open=float(item[1]), high=float(item[2]),
                        low=float(item[3]), close=float(item[4]),
                        volume=float(item[5]),
                    ))
            except (KeyError, IndexError, ValueError):
                pass
        return candles

    async def get_current_price(self, symbol: str) -> float:
        data = await self._request("GET", "/v5/market/tickers", {
            "category": self.category, "symbol": symbol
        })
        if data.get("retCode") == 0:
            try:
                return float(data["result"]["list"][0]["lastPrice"])
            except (KeyError, IndexError, ValueError):
                pass
        return 0.0

    # ═══════════════════════════════════════════
    # Order Execution
    # ═══════════════════════════════════════════

    async def place_order(self, symbol: str, side: OrderSide,
                          order_type: OrderType, quantity: float,
                          price: Optional[float] = None,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> TradeResult:
        """Place order with Bybit qty/price compliance."""
        inst = await self._get_instrument(symbol)
        if inst is None:
            return TradeResult(success=False, symbol=symbol, error=f"No instrument info: {symbol}")

        qty_str = self._fmt_qty(quantity, inst)
        qty_f = float(qty_str)
        if qty_f < inst.min_qty:
            return TradeResult(success=False, symbol=symbol, error=f"Qty {qty_f} < min {inst.min_qty}")

        cur_price = price if price else await self.get_current_price(symbol)
        if cur_price > 0 and qty_f * cur_price < inst.min_notional:
            return TradeResult(success=False, symbol=symbol,
                               error=f"Notional {qty_f * cur_price:.2f} < {inst.min_notional}")

        params = {
            "category": self.category, "symbol": symbol,
            "side": "Buy" if side == OrderSide.BUY else "Sell",
            "orderType": "Market" if order_type == OrderType.MARKET else "Limit",
            "qty": qty_str,
        }

        if order_type == OrderType.MARKET:
            params["timeInForce"] = "IOC"
        else:
            params["timeInForce"] = "GTC"
            if price:
                params["price"] = self._fmt_price(price, inst)

        if stop_loss and stop_loss > 0:
            params["stopLoss"] = self._fmt_price(stop_loss, inst)
        if take_profit and take_profit > 0:
            params["takeProfit"] = self._fmt_price(take_profit, inst)

        data = await self._request("POST", "/v5/order/create", params, auth=True)
        if data.get("retCode") == 0:
            return TradeResult(
                success=True, order_id=data.get("result", {}).get("orderId", ""),
                symbol=symbol, side=side, quantity=qty_f, price=cur_price,
            )
        return TradeResult(success=False, symbol=symbol, error=data.get("retMsg", "Unknown"))

    async def close_position(self, position: Position) -> TradeResult:
        close_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
        return await self.place_order(position.symbol, close_side, OrderType.MARKET, position.quantity)

    async def get_open_positions(self) -> List[Position]:
        data = await self._request("GET", "/v5/position/list",
                                   {"category": self.category, "settleCoin": "USDT"}, auth=True)
        positions = []
        if data.get("retCode") == 0:
            try:
                for item in data["result"]["list"]:
                    size = float(item.get("size", "0"))
                    if size <= 0:
                        continue
                    side_str = item.get("side", "")
                    if side_str not in ("Buy", "Sell"):
                        continue
                    sl_raw = item.get("stopLoss", "0") or "0"
                    tp_raw = item.get("takeProfit", "0") or "0"
                    sl_val = float(sl_raw)
                    tp_val = float(tp_raw)
                    # Use symbol+side+idx as unique ID (positionIdx alone is not unique)
                    pos_id = f"{item['symbol']}_{side_str}_{item.get('positionIdx', '0')}"
                    positions.append(Position(
                        id=pos_id,
                        symbol=item["symbol"],
                        side=OrderSide.BUY if side_str == "Buy" else OrderSide.SELL,
                        entry_price=float(item.get("avgPrice", "0")),
                        quantity=size,
                        current_price=float(item.get("markPrice", "0")),
                        pnl=float(item.get("unrealisedPnl", "0")),
                        stop_loss=sl_val if sl_val > 0 else None,
                        take_profit=tp_val if tp_val > 0 else None,
                        # Track if Bybit is handling SL/TP server-side
                        has_server_sltp=(sl_val > 0 or tp_val > 0),
                    ))
            except (KeyError, ValueError):
                pass
        return positions

    async def get_min_trade_size(self, symbol: str) -> float:
        inst = await self._get_instrument(symbol)
        return inst.min_qty if inst else 0.001

    async def get_tradeable_symbols(self) -> List[str]:
        data = await self._request("GET", "/v5/market/instruments-info", {"category": self.category})
        symbols = []
        if data.get("retCode") == 0:
            try:
                for item in data["result"]["list"]:
                    if item.get("status") == "Trading" and item.get("quoteCoin") == "USDT":
                        symbols.append(item["symbol"])
            except (KeyError, TypeError):
                pass
        return symbols


# ═══════════════════════════════════════════════════════
# Paper Trading
# ═══════════════════════════════════════════════════════

class PaperTradingAdapter(MarketAdapter):
    """Paper trading with realistic fees/slippage."""

    SLIPPAGE_PCT = 0.0005    # 0.05%
    TAKER_FEE_PCT = 0.00055  # 0.055% (Bybit default)

    def __init__(self, real_adapter: MarketAdapter, starting_balance: float = 50.0):
        super().__init__(name=f"paper_{real_adapter.name}", config=real_adapter.config)
        self.real = real_adapter
        self.balance = starting_balance
        self.positions: List[Position] = []
        self.trade_count = 0

    async def connect(self) -> bool:
        r = await self.real.connect()
        self.is_connected = r
        return r

    async def disconnect(self):
        await self.real.disconnect()
        self.is_connected = False

    async def get_balance(self) -> float:
        return self.balance

    async def get_candles(self, symbol: str, timeframe: TimeFrame, limit: int = 100) -> List[Candle]:
        return await self.real.get_candles(symbol, timeframe, limit)

    async def get_current_price(self, symbol: str) -> float:
        return await self.real.get_current_price(symbol)

    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                          quantity: float, price: Optional[float] = None,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> TradeResult:
        cur = await self.get_current_price(symbol)
        if cur <= 0:
            return TradeResult(success=False, symbol=symbol, error="No price")

        ep = price if (price and order_type == OrderType.LIMIT) else cur
        slip = ep * self.SLIPPAGE_PCT
        ep += slip if side == OrderSide.BUY else -slip

        notional = ep * quantity
        fee = notional * self.TAKER_FEE_PCT

        # Check if we can afford this trade (margin = notional value for 1x leverage)
        required = notional + fee
        if required > self.balance:
            return TradeResult(success=False, symbol=symbol,
                               error=f"Insufficient balance: need ${required:.2f}, have ${self.balance:.2f}")

        self.trade_count += 1
        pos = Position(id=f"paper_{self.trade_count}", symbol=symbol, side=side,
                       entry_price=ep, quantity=quantity, current_price=ep,
                       stop_loss=stop_loss, take_profit=take_profit)
        self.positions.append(pos)

        # Deduct margin + fee from balance
        self.balance -= required

        return TradeResult(success=True, order_id=pos.id, symbol=symbol,
                           side=side, price=ep, quantity=quantity, fee=fee)

    async def close_position(self, position: Position) -> TradeResult:
        cur = await self.get_current_price(position.symbol)
        if cur <= 0:
            return TradeResult(success=False, symbol=position.symbol, error="No price")

        cp = cur
        slip = cp * self.SLIPPAGE_PCT
        # Slippage works against us: closing LONG = sell lower, closing SHORT = buy higher
        cp += -slip if position.side == OrderSide.BUY else slip
        position.update_pnl(cp)
        fee = cp * position.quantity * self.TAKER_FEE_PCT

        # Return margin (entry * qty) + realized PnL - closing fee
        margin_returned = position.entry_price * position.quantity
        self.balance += margin_returned + position.pnl - fee

        self.positions = [p for p in self.positions if p.id != position.id]
        return TradeResult(success=True, symbol=position.symbol,
                           side=OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY,
                           price=cp, quantity=position.quantity, fee=fee)

    async def get_open_positions(self) -> List[Position]:
        for pos in self.positions:
            try:
                p = await self.get_current_price(pos.symbol)
                if p > 0:
                    pos.update_pnl(p)
            except Exception:
                pass
        return list(self.positions)

    async def get_min_trade_size(self, symbol: str) -> float:
        return await self.real.get_min_trade_size(symbol)

    async def get_tradeable_symbols(self) -> List[str]:
        return await self.real.get_tradeable_symbols()
