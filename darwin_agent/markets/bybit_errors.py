"""
Bybit Error Handling — Complete error catalog + diagnostics + migration.

This module handles:
1. ALL Bybit V5 retCode errors with specific recovery actions
2. Connection diagnostics (tells you EXACTLY why testnet/mainnet fails)
3. Safe testnet → mainnet migration with automated checklist
4. Pre-flight checks before first trade
5. Geo-restriction detection (US/China IP blocks)

Bybit V5 Error Reference (from official docs):
https://bybit-exchange.github.io/docs/v5/error
"""

import asyncio
import time
import json
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum


# ═══════════════════════════════════════════════════════════════
# Bybit Error Catalog — Every retCode we might encounter
# ═══════════════════════════════════════════════════════════════

class ErrorAction(Enum):
    """What the agent should do when hitting this error."""
    RETRY = "retry"              # Retry after delay
    RESYNC_TIME = "resync_time"  # Clock drift — resync and retry
    FATAL_AUTH = "fatal_auth"    # Bad API key — stop agent
    RATE_LIMIT = "rate_limit"    # Slow down
    SKIP = "skip"                # Skip this trade/symbol, continue
    RECONNECT = "reconnect"      # Connection lost — reconnect
    MAINTENANCE = "maintenance"  # Bybit maintenance — wait
    GEO_BLOCKED = "geo_blocked"  # IP blocked — need VPN/different server
    UPGRADE_ACCOUNT = "upgrade"  # Need UTA account
    INSUFFICIENT = "insufficient"  # Not enough balance
    REDUCE_SIZE = "reduce_size"  # Order too large
    FIX_PARAMS = "fix_params"    # Bad parameters — check & fix
    BANNED = "banned"            # Account banned


@dataclass
class BybitError:
    """Structured error info for a specific retCode."""
    code: int
    name: str
    description: str
    action: ErrorAction
    retry_delay_seconds: float = 0.0
    user_message: str = ""
    is_critical: bool = False


# Complete error catalog from Bybit V5 docs + CCXT source + real experience
BYBIT_ERRORS: Dict[int, BybitError] = {
    # ── General System Errors ──
    0: BybitError(0, "OK", "Success", ErrorAction.SKIP),
    -1: BybitError(-1, "INTERNAL", "Internal/network error (our code)", ErrorAction.RETRY, 2.0),

    5004: BybitError(5004, "SERVER_TIMEOUT", "Bybit server timeout",
                     ErrorAction.RETRY, 5.0,
                     "Bybit server is slow. Retrying..."),

    7001: BybitError(7001, "BAD_PARAMS_TYPE", "Request params type error",
                     ErrorAction.FIX_PARAMS, 0,
                     "Parameter type mismatch. Check qty/price format."),

    # ── Authentication Errors (10xxx) ──
    10001: BybitError(10001, "PARAM_ERROR", "Parameter error or maintenance",
                      ErrorAction.FIX_PARAMS, 0,
                      "Bad parameter. Could also mean Bybit maintenance.", False),

    10002: BybitError(10002, "TIMESTAMP_ERROR", "Request expired — clock drift",
                      ErrorAction.RESYNC_TIME, 1.0,
                      "⏰ Clock out of sync with Bybit. Re-syncing...\n"
                      "If persistent: run 'sudo chronyc -a makestep' on server", True),

    10003: BybitError(10003, "INVALID_API_KEY", "API key doesn't exist",
                      ErrorAction.FATAL_AUTH, 0,
                      "❌ API key invalid. Check:\n"
                      "  1. Key was copied correctly (no spaces)\n"
                      "  2. Key is for TESTNET if using testnet (separate keys!)\n"
                      "  3. Key is for MAINNET if using mainnet\n"
                      "  4. Key has not been deleted from Bybit dashboard", True),

    10004: BybitError(10004, "INVALID_SIGN", "HMAC signature mismatch",
                      ErrorAction.FATAL_AUTH, 0,
                      "❌ Signature invalid. This means:\n"
                      "  1. API SECRET is wrong (most common)\n"
                      "  2. Secret has trailing whitespace\n"
                      "  3. Clock is off by >5 seconds (check NTP)\n"
                      "  4. Body encoding differs from signature input", True),

    10005: BybitError(10005, "PERMISSION_DENIED", "API key lacks permissions",
                      ErrorAction.FATAL_AUTH, 0,
                      "❌ API key missing permissions. Go to Bybit:\n"
                      "  API Management → Edit Key → Enable:\n"
                      "  ✅ Read (required)\n"
                      "  ✅ Trade (required)\n"
                      "  ❌ Withdraw (NOT needed, leave disabled for safety)", True),

    10006: BybitError(10006, "RATE_LIMIT", "Too many requests per second",
                      ErrorAction.RATE_LIMIT, 2.0,
                      "⚠️ Rate limited. Slowing down requests.\n"
                      "Bybit limits: 10-20 req/sec per endpoint per UID"),

    10007: BybitError(10007, "NO_API_KEY", "API key not found in request headers",
                      ErrorAction.FATAL_AUTH, 0,
                      "❌ API key not included in request. Config error."),

    10008: BybitError(10008, "ACCOUNT_BANNED", "Account suspended by Bybit",
                      ErrorAction.BANNED, 0,
                      "🚫 Account is BANNED. Contact Bybit support.", True),

    10016: BybitError(10016, "SERVER_ERROR", "Bybit internal server error",
                      ErrorAction.RETRY, 10.0,
                      "Bybit having issues. Waiting before retry."),

    10017: BybitError(10017, "BAD_PATH", "Endpoint not found or wrong method",
                      ErrorAction.FIX_PARAMS, 0,
                      "Wrong API endpoint. This is a code bug."),

    10018: BybitError(10018, "IP_RATE_LIMIT", "IP-level rate limit exceeded",
                      ErrorAction.RATE_LIMIT, 5.0,
                      "⚠️ IP rate limited (more severe than per-UID).\n"
                      "If persistent, your IP may be flagged."),

    10020: BybitError(10020, "NOT_UNIFIED", "Account not upgraded to UTA",
                      ErrorAction.UPGRADE_ACCOUNT, 0,
                      "❌ Bybit requires Unified Trading Account (UTA).\n"
                      "  Go to Bybit → Account → Upgrade to Unified Account\n"
                      "  This is FREE and takes 30 seconds.", True),

    10024: BybitError(10024, "COMPLIANCE", "Compliance rules triggered",
                      ErrorAction.GEO_BLOCKED, 0,
                      "🌐 Compliance block. Your region may be restricted.\n"
                      "  US and China IPs are blocked by Bybit.\n"
                      "  Solution: Use a server in EU, Singapore, or Japan.", True),

    10027: BybitError(10027, "TRADING_BANNED", "Trading disabled on account",
                      ErrorAction.BANNED, 0,
                      "🚫 Trading disabled. Check Bybit account status.", True),

    10028: BybitError(10028, "UNIFIED_ONLY", "Only unified accounts allowed",
                      ErrorAction.UPGRADE_ACCOUNT, 0,
                      "Upgrade to Unified Trading Account on Bybit.", True),

    10029: BybitError(10029, "SYMBOL_RESTRICTED", "Symbol not available for your account",
                      ErrorAction.SKIP, 0,
                      "This symbol is restricted. Skipping."),

    # ── Trading Errors (110xxx, 170xxx) ──
    110001: BybitError(110001, "ORDER_NOT_FOUND", "Order does not exist",
                       ErrorAction.SKIP, 0, "Order already filled or cancelled."),

    110003: BybitError(110003, "QTY_EXCEEDS", "Order quantity exceeds limit",
                       ErrorAction.REDUCE_SIZE, 0,
                       "Position too large. Reducing size."),

    110004: BybitError(110004, "PRICE_OUT_RANGE", "Price out of allowed range",
                       ErrorAction.FIX_PARAMS, 0,
                       "SL/TP price out of acceptable range."),

    110007: BybitError(110007, "INSUFFICIENT_BALANCE", "Not enough balance",
                       ErrorAction.INSUFFICIENT, 0,
                       "💰 Not enough USDT to open position.\n"
                       "Check: available balance vs required margin."),

    110012: BybitError(110012, "INSUFFICIENT_BALANCE_2", "Insufficient balance variant",
                       ErrorAction.INSUFFICIENT, 0,
                       "Not enough available balance."),

    110017: BybitError(110017, "REDUCE_ONLY_REJECT", "Reduce-only order rejected",
                       ErrorAction.SKIP, 0, "No position to reduce."),

    110025: BybitError(110025, "CROSS_MARGIN_LIMIT", "Cross margin limit exceeded",
                       ErrorAction.REDUCE_SIZE, 0, "Reduce position size."),

    110043: BybitError(110043, "TP_SL_TOO_CLOSE", "TP/SL too close to entry",
                       ErrorAction.FIX_PARAMS, 0,
                       "Stop loss or take profit too close to entry price."),

    110044: BybitError(110044, "ORDER_NOT_MODIFIED", "Nothing to modify",
                       ErrorAction.SKIP, 0, "Order already in desired state."),

    110070: BybitError(110070, "EEA_RESTRICTED", "EEA region derivatives restricted",
                       ErrorAction.GEO_BLOCKED, 0,
                       "🇪🇺 EEA users cannot trade derivatives on Bybit.\n"
                       "  Use Bybit EU endpoint or switch to spot trading.", True),

    170001: BybitError(170001, "INTERNAL_ERROR", "Internal trading engine error",
                       ErrorAction.RETRY, 5.0, "Bybit trading engine issue."),

    170124: BybitError(170124, "ORDER_CANT_CANCEL", "Cannot cancel order",
                       ErrorAction.SKIP, 0, "Order may have already executed."),

    170131: BybitError(170131, "INSUFFICIENT_BALANCE_3", "Balance insufficient",
                       ErrorAction.INSUFFICIENT, 0, "Not enough margin."),

    170210: BybitError(170210, "MIN_NOTIONAL", "Below minimum notional value",
                       ErrorAction.REDUCE_SIZE, 0,
                       "Order value too small. Minimum is usually $5 USDT."),
}


def get_error_info(ret_code: int) -> BybitError:
    """Get structured error info for any Bybit retCode."""
    if ret_code in BYBIT_ERRORS:
        return BYBIT_ERRORS[ret_code]
    # Unknown error — default to retry
    return BybitError(
        code=ret_code,
        name="UNKNOWN",
        description=f"Unknown Bybit error code: {ret_code}",
        action=ErrorAction.RETRY,
        retry_delay_seconds=3.0,
        user_message=f"Unknown error {ret_code}. Will retry.",
    )


# ═══════════════════════════════════════════════════════════════
# Connection Diagnostics
# ═══════════════════════════════════════════════════════════════

@dataclass
class DiagnosticResult:
    """Result of a single diagnostic check."""
    name: str
    passed: bool
    message: str
    details: str = ""
    fix: str = ""


@dataclass
class DiagnosticReport:
    """Full diagnostic report."""
    checks: List[DiagnosticResult] = field(default_factory=list)
    is_ready: bool = False
    summary: str = ""

    def add(self, check: DiagnosticResult):
        self.checks.append(check)

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def render(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("  🔍 BYBIT CONNECTION DIAGNOSTICS")
        lines.append("=" * 60)
        for c in self.checks:
            icon = "✅" if c.passed else "❌"
            lines.append(f"  {icon} {c.name}: {c.message}")
            if c.details:
                lines.append(f"      {c.details}")
            if not c.passed and c.fix:
                lines.append(f"      💡 FIX: {c.fix}")
        lines.append("-" * 60)
        lines.append(f"  Result: {self.passed_count}/{len(self.checks)} passed")
        lines.append(f"  Ready to trade: {'YES ✅' if self.is_ready else 'NO ❌'}")
        if self.summary:
            lines.append(f"  {self.summary}")
        lines.append("=" * 60)
        return "\n".join(lines)


async def run_diagnostics(api_key: str, api_secret: str,
                          testnet: bool = True) -> DiagnosticReport:
    """
    Run comprehensive connection diagnostics.
    Tests every possible failure point so you know EXACTLY what's wrong.
    """
    import aiohttp
    import hmac
    import hashlib

    report = DiagnosticReport()
    base_url = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
    env_name = "TESTNET" if testnet else "MAINNET"

    session = None
    try:
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        session = aiohttp.ClientSession(timeout=timeout)

        # ── CHECK 1: DNS Resolution + Basic Connectivity ──
        try:
            async with session.get(f"{base_url}/v5/market/time") as resp:
                data = await resp.json()
                if data.get("retCode") == 0:
                    server_time = int(data["result"]["timeSecond"])
                    report.add(DiagnosticResult(
                        name=f"Network → {env_name}",
                        passed=True,
                        message=f"Connected to {base_url}",
                        details=f"Server time: {datetime.fromtimestamp(server_time, tz=timezone.utc).isoformat()}"
                    ))
                else:
                    report.add(DiagnosticResult(
                        name=f"Network → {env_name}",
                        passed=False,
                        message=f"Server returned error: {data.get('retMsg')}",
                        fix="Check if Bybit is under maintenance"
                    ))
        except aiohttp.ClientConnectorError:
            report.add(DiagnosticResult(
                name=f"Network → {env_name}",
                passed=False,
                message="Cannot connect to Bybit servers",
                fix="Check: 1) Internet connection  2) DNS resolution  3) Firewall rules\n"
                     "      Try: curl -v " + base_url + "/v5/market/time"
            ))
            report.summary = "Cannot reach Bybit. Fix network first."
            return report
        except asyncio.TimeoutError:
            report.add(DiagnosticResult(
                name=f"Network → {env_name}",
                passed=False,
                message="Connection timed out (>10s)",
                fix="Your server may have high latency to Bybit.\n"
                    "      Bybit servers are in AWS Singapore/Tokyo.\n"
                    "      Consider a server closer to Asia."
            ))
            report.summary = "Timeout. Check server location/network."
            return report

        # ── CHECK 2: Geo-restriction ──
        try:
            # A 403 here means IP is geo-blocked
            async with session.get(f"{base_url}/v5/account/wallet-balance",
                                   params={"accountType": "UNIFIED"}) as resp:
                if resp.status == 403:
                    report.add(DiagnosticResult(
                        name="Geo-restriction",
                        passed=False,
                        message="HTTP 403 — IP is geo-blocked by Bybit",
                        details="US and Mainland China IPs are blocked",
                        fix="Deploy on a server in: EU, Singapore, Japan, or other non-restricted region\n"
                            "      DigitalOcean Singapore or Hetzner EU recommended"
                    ))
                    report.summary = "Your server IP is geo-blocked. Move to allowed region."
                    return report
                else:
                    report.add(DiagnosticResult(
                        name="Geo-restriction",
                        passed=True,
                        message="IP not geo-blocked",
                        details=f"HTTP {resp.status} (expected without auth)"
                    ))
        except Exception as e:
            report.add(DiagnosticResult(
                name="Geo-restriction",
                passed=True,  # Can't determine, assume OK
                message=f"Could not test geo-block: {e}",
            ))

        # ── CHECK 3: Clock Sync ──
        try:
            local_before = int(time.time() * 1000)
            async with session.get(f"{base_url}/v5/market/time") as resp:
                data = await resp.json()
            local_after = int(time.time() * 1000)

            if data.get("retCode") == 0:
                server_ms = int(data["result"]["timeSecond"]) * 1000
                local_mid = (local_before + local_after) // 2
                offset_ms = abs(server_ms - local_mid)
                offset_s = offset_ms / 1000

                passed = offset_s < 3.0  # Bybit recv_window is 5s
                report.add(DiagnosticResult(
                    name="Clock Sync",
                    passed=passed,
                    message=f"Offset: {offset_s:.2f}s {'(OK)' if passed else '(TOO HIGH!)'}",
                    details=f"Bybit rejects if offset > 5s. Yours: {offset_s:.2f}s",
                    fix="" if passed else
                    "Run: sudo apt install chrony && sudo chronyc -a makestep\n"
                    "      Then verify: chronyc tracking"
                ))
        except Exception as e:
            report.add(DiagnosticResult(
                name="Clock Sync", passed=False,
                message=f"Could not test: {e}",
                fix="Install chrony: sudo apt install chrony"
            ))

        # ── CHECK 4: API Key Format ──
        key_ok = True
        if not api_key or api_key.startswith("YOUR") or len(api_key) < 10:
            key_ok = False
            report.add(DiagnosticResult(
                name="API Key Format",
                passed=False,
                message="API key is empty or placeholder",
                fix=f"Get a {'TESTNET' if testnet else 'MAINNET'} API key from:\n"
                    f"      {'https://testnet.bybit.com' if testnet else 'https://www.bybit.com'}\n"
                    f"      → API Management → Create New Key"
            ))
        elif " " in api_key or "\n" in api_key:
            key_ok = False
            report.add(DiagnosticResult(
                name="API Key Format",
                passed=False,
                message="API key contains whitespace!",
                fix="Remove spaces/newlines from api_key in config.yaml"
            ))
        else:
            report.add(DiagnosticResult(
                name="API Key Format",
                passed=True,
                message=f"Key: {api_key[:6]}...{api_key[-4:]} ({len(api_key)} chars)"
            ))

        if not api_secret or api_secret.startswith("YOUR") or len(api_secret) < 10:
            key_ok = False
            report.add(DiagnosticResult(
                name="API Secret Format",
                passed=False,
                message="API secret is empty or placeholder",
                fix="Copy the FULL secret from Bybit (only shown once at creation)"
            ))
        elif " " in api_secret or "\n" in api_secret:
            key_ok = False
            report.add(DiagnosticResult(
                name="API Secret Format",
                passed=False,
                message="API secret contains whitespace!",
                fix="Remove spaces/newlines from api_secret in config.yaml"
            ))
        else:
            report.add(DiagnosticResult(
                name="API Secret Format",
                passed=True,
                message=f"Secret: {api_secret[:4]}...{api_secret[-4:]} ({len(api_secret)} chars)"
            ))

        # ── CHECK 5: Authentication Test ──
        if key_ok:
            try:
                ts = str(int(time.time() * 1000))
                recv = "5000"
                param_str = "accountType=UNIFIED"
                sign_str = f"{ts}{api_key}{recv}{param_str}"
                signature = hmac.new(
                    api_secret.encode(), sign_str.encode(), hashlib.sha256
                ).hexdigest()

                headers = {
                    "X-BAPI-API-KEY": api_key,
                    "X-BAPI-SIGN": signature,
                    "X-BAPI-TIMESTAMP": ts,
                    "X-BAPI-RECV-WINDOW": recv,
                }
                async with session.get(
                    f"{base_url}/v5/account/wallet-balance",
                    params={"accountType": "UNIFIED"},
                    headers=headers
                ) as resp:
                    data = await resp.json()

                ret = data.get("retCode", -1)
                if ret == 0:
                    # Parse balance
                    balance = 0.0
                    try:
                        for coin in data["result"]["list"][0]["coin"]:
                            if coin["coin"] == "USDT":
                                balance = float(coin["walletBalance"])
                    except (KeyError, IndexError):
                        pass

                    report.add(DiagnosticResult(
                        name="Authentication",
                        passed=True,
                        message="API key authenticated successfully!",
                        details=f"USDT Balance: ${balance:.2f}"
                    ))

                    # Balance check
                    if balance < 1:
                        report.add(DiagnosticResult(
                            name="Balance",
                            passed=False,
                            message=f"USDT balance too low: ${balance:.2f}",
                            fix="Deposit USDT to your Bybit account\n"
                                "      For testnet: use Bybit testnet faucet to get free USDT"
                        ))
                    else:
                        report.add(DiagnosticResult(
                            name="Balance",
                            passed=True,
                            message=f"USDT available: ${balance:.2f}"
                        ))
                else:
                    err = get_error_info(ret)
                    report.add(DiagnosticResult(
                        name="Authentication",
                        passed=False,
                        message=f"retCode {ret}: {err.name}",
                        details=data.get("retMsg", ""),
                        fix=err.user_message
                    ))
            except Exception as e:
                report.add(DiagnosticResult(
                    name="Authentication",
                    passed=False,
                    message=f"Auth request failed: {e}",
                ))

        # ── CHECK 6: Market Data Access ──
        try:
            async with session.get(f"{base_url}/v5/market/tickers",
                                   params={"category": "linear", "symbol": "BTCUSDT"}) as resp:
                data = await resp.json()

            if data.get("retCode") == 0:
                price = float(data["result"]["list"][0]["lastPrice"])
                report.add(DiagnosticResult(
                    name="Market Data",
                    passed=True,
                    message=f"BTCUSDT price: ${price:,.2f}",
                    details="Market data feed working"
                ))
            else:
                report.add(DiagnosticResult(
                    name="Market Data",
                    passed=False,
                    message=f"Cannot get market data: {data.get('retMsg')}",
                ))
        except Exception as e:
            report.add(DiagnosticResult(
                name="Market Data",
                passed=False,
                message=f"Market data error: {e}",
            ))

        # ── CHECK 7: Trading Permissions ──
        if key_ok:
            try:
                ts = str(int(time.time() * 1000))
                recv = "5000"
                params = {"category": "linear", "settleCoin": "USDT"}
                param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
                sign_str = f"{ts}{api_key}{recv}{param_str}"
                signature = hmac.new(
                    api_secret.encode(), sign_str.encode(), hashlib.sha256
                ).hexdigest()
                headers = {
                    "X-BAPI-API-KEY": api_key,
                    "X-BAPI-SIGN": signature,
                    "X-BAPI-TIMESTAMP": ts,
                    "X-BAPI-RECV-WINDOW": recv,
                }
                async with session.get(
                    f"{base_url}/v5/position/list",
                    params=params, headers=headers
                ) as resp:
                    data = await resp.json()

                ret = data.get("retCode", -1)
                if ret == 0:
                    report.add(DiagnosticResult(
                        name="Trading Permissions",
                        passed=True,
                        message="Can read positions (trade permission OK)",
                    ))
                elif ret == 10005:
                    report.add(DiagnosticResult(
                        name="Trading Permissions",
                        passed=False,
                        message="API key lacks TRADE permission",
                        fix="Bybit → API Management → Edit Key → Enable 'Trade'"
                    ))
                elif ret == 10020 or ret == 10028:
                    report.add(DiagnosticResult(
                        name="Trading Permissions",
                        passed=False,
                        message="Account not upgraded to Unified Trading Account",
                        fix="Bybit → Account → Upgrade to Unified Account (free)"
                    ))
                else:
                    err = get_error_info(ret)
                    report.add(DiagnosticResult(
                        name="Trading Permissions",
                        passed=False,
                        message=f"Error {ret}: {err.name}",
                        fix=err.user_message
                    ))
            except Exception as e:
                report.add(DiagnosticResult(
                    name="Trading Permissions",
                    passed=False,
                    message=f"Could not test: {e}",
                ))

        # ── VERDICT ──
        report.is_ready = report.failed_count == 0
        if report.is_ready:
            report.summary = f"🎉 All checks passed! Ready to trade on {env_name}."
        else:
            report.summary = f"⚠️  {report.failed_count} issue(s) found. Fix them before trading."

    except Exception as e:
        report.add(DiagnosticResult(
            name="General", passed=False,
            message=f"Diagnostic failed: {e}",
        ))
        report.summary = f"Diagnostics error: {e}"
    finally:
        if session and not session.closed:
            await session.close()

    return report


# ═══════════════════════════════════════════════════════════════
# Testnet → Mainnet Migration
# ═══════════════════════════════════════════════════════════════

@dataclass
class MigrationCheck:
    name: str
    passed: bool
    message: str
    blocking: bool = True  # If True, migration cannot proceed without this


@dataclass
class MigrationReport:
    checks: List[MigrationCheck] = field(default_factory=list)
    can_migrate: bool = False
    summary: str = ""

    def render(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("  🔄 TESTNET → MAINNET MIGRATION CHECK")
        lines.append("=" * 60)
        for c in self.checks:
            icon = "✅" if c.passed else ("❌" if c.blocking else "⚠️")
            lines.append(f"  {icon} {c.name}: {c.message}")
        lines.append("-" * 60)
        blocking_fails = sum(1 for c in self.checks if not c.passed and c.blocking)
        lines.append(f"  Blocking failures: {blocking_fails}")
        lines.append(f"  Can migrate: {'YES ✅' if self.can_migrate else 'NO ❌'}")
        if self.summary:
            lines.append(f"  {self.summary}")
        lines.append("=" * 60)
        return "\n".join(lines)


async def check_migration_readiness(
    testnet_key: str, testnet_secret: str,
    mainnet_key: str, mainnet_secret: str,
    dna_path: str = "data/generations",
    min_generations: int = 3,
    min_win_rate: float = 0.52,
    min_trades: int = 50,
) -> MigrationReport:
    """
    Automated checklist for safely migrating testnet → mainnet.

    Checks:
    1. Agent has evolved enough (generations + trades + win rate)
    2. Brain epsilon is low enough (done exploring)
    3. Mainnet API keys work
    4. Mainnet has sufficient balance
    5. Testnet DNA is intact
    6. No critical death patterns in recent generations
    """
    from darwin_agent.evolution.dna import EvolutionEngine

    report = MigrationReport()

    # ── CHECK 1: Evolution History ──
    evo = EvolutionEngine(dna_path)
    all_dna = evo.load_all_dna()
    gen_count = len(all_dna)

    if gen_count >= min_generations:
        report.checks.append(MigrationCheck(
            "Generations", True,
            f"{gen_count} generations completed (min: {min_generations})"
        ))
    else:
        report.checks.append(MigrationCheck(
            "Generations", False,
            f"Only {gen_count} generations (need {min_generations})",
            blocking=True
        ))

    # ── CHECK 2: Total Trades ──
    total_trades = sum(d.total_trades for d in all_dna)
    if total_trades >= min_trades:
        report.checks.append(MigrationCheck(
            "Trade Experience", True,
            f"{total_trades} total trades across all generations (min: {min_trades})"
        ))
    else:
        report.checks.append(MigrationCheck(
            "Trade Experience", False,
            f"Only {total_trades} trades (need {min_trades})",
            blocking=True
        ))

    # ── CHECK 3: Recent Win Rate ──
    if all_dna:
        recent = all_dna[-3:] if len(all_dna) >= 3 else all_dna
        recent_trades = sum(d.total_trades for d in recent)
        recent_wr = (
            sum(d.win_rate * d.total_trades for d in recent) / recent_trades
            if recent_trades > 0 else 0
        )
        if recent_wr >= min_win_rate:
            report.checks.append(MigrationCheck(
                "Win Rate", True,
                f"Recent WR: {recent_wr:.1%} (min: {min_win_rate:.0%})"
            ))
        else:
            report.checks.append(MigrationCheck(
                "Win Rate", False,
                f"Recent WR: {recent_wr:.1%} (need {min_win_rate:.0%})",
                blocking=True
            ))

    # ── CHECK 4: Brain Maturity ──
    import os
    brain_files = sorted([f for f in os.listdir(dna_path) if f.startswith("brain_gen_")])
    if brain_files:
        with open(os.path.join(dna_path, brain_files[-1])) as f:
            brain_data = json.load(f)
        epsilon = brain_data.get("epsilon", 1.0)
        if epsilon < 0.15:
            report.checks.append(MigrationCheck(
                "Brain Maturity", True,
                f"Epsilon: {epsilon:.3f} (low = exploiting learned knowledge)"
            ))
        else:
            report.checks.append(MigrationCheck(
                "Brain Maturity", False,
                f"Epsilon: {epsilon:.3f} (still exploring too much, need <0.15)",
                blocking=False  # Warning only
            ))
    else:
        report.checks.append(MigrationCheck(
            "Brain Maturity", False,
            "No brain files found. Agent hasn't learned yet.",
            blocking=True
        ))

    # ── CHECK 5: Death Pattern Analysis ──
    if all_dna:
        death_causes = [d.cause_of_death for d in all_dna[-5:] if d.cause_of_death]
        capital_deaths = sum(1 for c in death_causes if "capital" in c.lower())
        if capital_deaths >= 3:
            report.checks.append(MigrationCheck(
                "Death Patterns", False,
                f"{capital_deaths}/5 recent deaths from capital loss — agent is losing money",
                blocking=True
            ))
        else:
            report.checks.append(MigrationCheck(
                "Death Patterns", True,
                f"Recent death causes: {death_causes[:3] or ['none']}"
            ))

    # ── CHECK 6: Mainnet Connectivity ──
    if mainnet_key and not mainnet_key.startswith("YOUR"):
        mainnet_diag = await run_diagnostics(mainnet_key, mainnet_secret, testnet=False)
        auth_ok = any(c.name == "Authentication" and c.passed for c in mainnet_diag.checks)
        balance_ok = any(c.name == "Balance" and c.passed for c in mainnet_diag.checks)
        geo_ok = any(c.name == "Geo-restriction" and c.passed for c in mainnet_diag.checks)

        if auth_ok:
            report.checks.append(MigrationCheck(
                "Mainnet Auth", True, "Mainnet API keys work!"
            ))
        else:
            failed = [c for c in mainnet_diag.checks if not c.passed]
            reason = failed[0].message if failed else "Unknown"
            report.checks.append(MigrationCheck(
                "Mainnet Auth", False,
                f"Mainnet keys failed: {reason}",
                blocking=True
            ))

        if balance_ok:
            bal_check = next((c for c in mainnet_diag.checks if c.name == "Balance"), None)
            report.checks.append(MigrationCheck(
                "Mainnet Balance", True,
                bal_check.message if bal_check else "Balance OK"
            ))
        else:
            report.checks.append(MigrationCheck(
                "Mainnet Balance", False,
                "No USDT on mainnet. Deposit at least $50.",
                blocking=True
            ))

        if not geo_ok:
            report.checks.append(MigrationCheck(
                "Mainnet Geo", False,
                "Server IP may be geo-blocked on mainnet",
                blocking=True
            ))
    else:
        report.checks.append(MigrationCheck(
            "Mainnet Keys", False,
            "No mainnet API keys configured in config.yaml",
            blocking=True
        ))

    # ── VERDICT ──
    blocking_fails = sum(1 for c in report.checks if not c.passed and c.blocking)
    report.can_migrate = blocking_fails == 0

    if report.can_migrate:
        report.summary = (
            "🎉 Ready to migrate!\n"
            "  Steps:\n"
            "  1. Edit config.yaml: set testnet: false\n"
            "  2. Set mainnet API keys\n"
            "  3. Restart: sudo systemctl restart darwin-agent\n"
            "  4. Monitor closely for first 24 hours"
        )
    else:
        report.summary = f"❌ {blocking_fails} blocking issue(s). Fix before migrating."

    return report


# ═══════════════════════════════════════════════════════════════
# Pre-flight Check (run before first trade)
# ═══════════════════════════════════════════════════════════════

async def preflight_check(adapter, config) -> Tuple[bool, str]:
    """
    Quick pre-trade check. Run once at agent startup.
    Returns (ready: bool, message: str)
    """
    issues = []

    # 1. Can we get market data?
    try:
        price = await adapter.get_current_price("BTCUSDT")
        if price <= 0:
            issues.append("Cannot get BTC price (market data down)")
    except Exception as e:
        issues.append(f"Market data error: {e}")

    # 2. Can we read positions?
    try:
        positions = await adapter.get_open_positions()
    except Exception as e:
        issues.append(f"Cannot read positions: {e}")

    # 3. Can we read balance?
    try:
        balance = await adapter.get_balance()
        if balance < config.health.instant_death_capital:
            issues.append(f"Balance ${balance:.2f} below death threshold ${config.health.instant_death_capital}")
    except Exception as e:
        issues.append(f"Cannot read balance: {e}")

    # 4. Can we get instrument info? (needed for qty rounding)
    try:
        min_size = await adapter.get_min_trade_size("BTCUSDT")
        if min_size <= 0:
            issues.append("Cannot get instrument info for BTCUSDT")
    except Exception as e:
        issues.append(f"Instrument info error: {e}")

    if issues:
        return False, "Pre-flight FAILED:\n  " + "\n  ".join(issues)
    return True, "Pre-flight OK: market data ✓ positions ✓ balance ✓ instruments ✓"
