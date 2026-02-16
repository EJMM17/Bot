"""Darwin Agent — Entry Point. Run with: python -m darwin_agent [--mode test|live]"""

import asyncio
import argparse
import sys
import signal
import os

from darwin_agent.core.agent_v2 import DarwinAgentV2, AgentPhase
from darwin_agent.utils.config import load_config, AgentConfig
from darwin_agent.evolution.dna import EvolutionEngine
from darwin_agent.dashboard import set_agent, set_config_context, start_dashboard

BANNER = """
╔═══════════════════════════════════════════════════╗
║  🧬  D A R W I N   A G E N T   v2.3  🧬         ║
║  Autonomous Evolutionary Trading System           ║
║  "Survive. Adapt. Evolve."                        ║
╚═══════════════════════════════════════════════════╝
"""


def spawn(config: AgentConfig, mode: str = "test") -> DarwinAgentV2:
    agent = DarwinAgentV2(config)
    if mode == "live":
        agent.phase = AgentPhase.LIVE
    return agent


async def run_forever(config: AgentConfig, mode: str = "test"):
    evo = EvolutionEngine(config.evolution.dna_path)
    gen = evo.get_latest_generation() + 1

    print(BANNER)
    print(f"  Mode: {mode.upper()} | Gen: {gen} | Capital: ${config.starting_capital}")
    print(f"  Markets: {', '.join(k for k, v in config.markets.items() if v.enabled)}")
    print(f"  Dashboard: http://0.0.0.0:{config.dashboard_port}")
    print("=" * 55)

    # ── Pre-flight: Run diagnostics before starting ──
    for name, mc in config.markets.items():
        if mc.enabled:
            env = "TESTNET" if mc.testnet else "⚠️ MAINNET"
            print(f"\n  🔍 Running pre-flight diagnostics for {name} ({env})...")

            from darwin_agent.markets.bybit_errors import run_diagnostics
            diag = await run_diagnostics(mc.api_key, mc.api_secret, mc.testnet)

            if not diag.is_ready:
                print(diag.render())
                if not mc.testnet:
                    print("\n  ❌ REFUSING to start MAINNET with failing diagnostics.")
                    print("  Fix the issues above or switch to testnet.")
                    return
                else:
                    print("\n  ⚠️  Testnet diagnostics failed. Attempting anyway...")
                    print("  (Paper trading adapter may still work for some operations)")
            else:
                print(f"  ✅ All {diag.passed_count} checks passed!")

    # Start web dashboard
    dashboard_task = None
    try:
        dashboard_task = asyncio.create_task(start_dashboard(config.dashboard_port))
        await asyncio.sleep(0.5)
        print(f"\n  📊 Dashboard running on port {config.dashboard_port}")
    except Exception as e:
        print(f"  ⚠️  Dashboard failed: {e}")

    # Run indefinitely for 24/7 evolution unless capital is fully lost or process is stopped.
    while True:
        agent = spawn(config, mode)
        agent_gen = agent.generation

        print(f"\n{'=' * 55}")
        print(f"  🐣 Spawning Generation {agent_gen}")
        print(f"{'=' * 55}\n")

        set_agent(agent)

        try:
            await agent.run()
        except Exception as e:
            print(f"\n❌ Agent crashed: {e}")

        status = agent.get_status()
        if status["health"]["hp"] <= 0:
            print(f"\n💀 Gen-{agent_gen} dead: {status['health']['cause_of_death']}")
            print(f"   Final: ${status['health']['capital']:.2f}")

            if status["health"]["capital"] <= 0:
                print("\n💸 All capital lost.")
                break

            print(f"\n🧬 Encoding DNA → spawning next gen...\n")
            await asyncio.sleep(3)
        else:
            print("\nAgent stopped. Exiting.")
            break

    if dashboard_task and not dashboard_task.done():
        dashboard_task.cancel()
    print("\n🏁 Darwin Agent session complete.")


def show_status(config):
    evo = EvolutionEngine(config.evolution.dna_path)
    all_dna = evo.load_all_dna()
    if not all_dna:
        print("No generations found.")
        return

    print(f"\n📊 Evolution History — {len(all_dna)} generations\n")
    print(f"{'Gen':>4} | {'Capital':>10} | {'Trades':>6} | {'WR':>6} | Cause of Death")
    print("-" * 65)
    for dna in all_dna:
        print(
            f"  {dna.generation:>2} | ${dna.final_capital:>8.2f} | "
            f"{dna.total_trades:>6} | {dna.win_rate:>5.1%} | "
            f"{dna.cause_of_death or 'alive'}"
        )


async def run_diagnose(config):
    """Run full connection diagnostics."""
    from darwin_agent.markets.bybit_errors import run_diagnostics
    for name, mc in config.markets.items():
        if not mc.enabled:
            continue
        env = "TESTNET" if mc.testnet else "MAINNET"
        print(f"\n🔍 Running diagnostics: {name} ({env})\n")
        report = await run_diagnostics(mc.api_key, mc.api_secret, mc.testnet)
        print(report.render())


async def run_migrate(config):
    """Run migration readiness check."""
    from darwin_agent.markets.bybit_errors import check_migration_readiness

    crypto = config.markets.get("crypto")
    if not crypto:
        print("No crypto market configured.")
        return

    # For migration, we need both testnet and mainnet keys
    # Testnet keys are current, mainnet keys we ask for
    print("\n🔄 TESTNET → MAINNET MIGRATION CHECK\n")

    if crypto.testnet:
        print("Current mode: TESTNET")
        print("To check mainnet readiness, provide mainnet keys in config.yaml")
        print("(Set a second market entry or use environment variables)\n")

        # Check with current keys (testnet)
        mainnet_key = os.environ.get("BYBIT_MAINNET_KEY", "")
        mainnet_secret = os.environ.get("BYBIT_MAINNET_SECRET", "")

        if not mainnet_key:
            print("Set environment variables for mainnet check:")
            print("  export BYBIT_MAINNET_KEY='your-mainnet-key'")
            print("  export BYBIT_MAINNET_SECRET='your-mainnet-secret'")
            print("Then run: python -m darwin_agent --migrate\n")

            # Still run evolution checks
            mainnet_key = "PLACEHOLDER"
            mainnet_secret = "PLACEHOLDER"

        report = await check_migration_readiness(
            testnet_key=crypto.api_key,
            testnet_secret=crypto.api_secret,
            mainnet_key=mainnet_key,
            mainnet_secret=mainnet_secret,
            dna_path=config.evolution.dna_path,
        )
        print(report.render())
    else:
        print("Already on MAINNET. No migration needed.")


def main():
    parser = argparse.ArgumentParser(description="Darwin Agent v2.3")
    parser.add_argument("--mode", choices=["test", "live"], default="test",
                        help="test=paper trading, live=real money")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--status", action="store_true",
                        help="Show evolution history")
    parser.add_argument("--diagnose", action="store_true",
                        help="Run Bybit connection diagnostics")
    parser.add_argument("--migrate", action="store_true",
                        help="Check testnet → mainnet migration readiness")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        print("   Fix config file and try again.")
        sys.exit(1)

    set_config_context(config, args.config)

    if args.status:
        show_status(config)
        return

    if args.diagnose:
        asyncio.run(run_diagnose(config))
        return

    if args.migrate:
        asyncio.run(run_migrate(config))
        return

    # Validate config before running
    try:
        config.validate()
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        print("   Edit config.yaml and try again.")
        sys.exit(1)

    # Warn if no API keys
    for name, mc in config.markets.items():
        if mc.enabled and (not mc.api_key or mc.api_key.startswith("YOUR")):
            print(f"\n⚠️  Warning: Market '{name}' enabled but API key not set.")
            print(f"   Edit config.yaml with valid Bybit API keys.")
            if not mc.testnet:
                print("   REFUSING to start in MAINNET mode without valid keys.")
                sys.exit(1)

    # Extra safety: refuse live mode on mainnet without --mode live explicit
    for name, mc in config.markets.items():
        if mc.enabled and not mc.testnet and args.mode != "live":
            print(f"\n⚠️  Market '{name}' is set to MAINNET but mode is 'test'.")
            print("   This will paper-trade using real market data from mainnet.")
            print("   To trade with real money, use: --mode live")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    _shutdown_triggered = False

    def shutdown(sig, frame):
        nonlocal _shutdown_triggered
        if _shutdown_triggered:
            return
        _shutdown_triggered = True
        print(f"\n⚡ Signal {sig} received. Shutting down gracefully...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        loop.run_until_complete(run_forever(config, args.mode))
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("\n👋 Shutdown complete.")
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


if __name__ == "__main__":
    main()
