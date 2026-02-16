#!/bin/bash
# ═══════════════════════════════════════════
# 🧬 Darwin Agent — Server Deploy Script
# For Ubuntu 22.04+ / Debian 12+
# ═══════════════════════════════════════════
set -e

echo "🧬 Darwin Agent — Deployment Starting..."
echo "============================================"

# 1. System deps + NTP (critical for API signature timestamps)
echo "📦 Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y python3.11 python3.11-venv python3-pip git htop chrony

echo "🕐 Configuring NTP time sync (critical for Bybit API)..."
sudo systemctl enable chrony
sudo systemctl start chrony
sleep 2
echo "   Time offset: $(chronyc tracking 2>/dev/null | grep 'System time' || echo 'syncing...')"

# 2. Project directory
PROJECT_DIR="$HOME/darwin_agent"
echo "📁 Setting up $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# 3. Python venv
echo "🐍 Setting up Python environment..."
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q

# 4. Dependencies
echo "📚 Installing Python packages..."
pip install -r requirements.txt -q

# 5. Data directories
mkdir -p data/generations data/logs

# 6. Config
if [ ! -f "config.yaml" ]; then
    echo "⚠️  No config.yaml found. Copying template..."
    cp config_example.yaml config.yaml
    echo "🔑 EDIT config.yaml with your Bybit API keys:"
    echo "   nano $PROJECT_DIR/config.yaml"
fi

# 7. Firewall
echo "🔒 Configuring firewall..."
sudo ufw allow 22/tcp
sudo ufw allow 8080/tcp
echo "y" | sudo ufw enable 2>/dev/null || true

# 8. Systemd service with auto-restart
echo "🔧 Creating systemd service..."
cat > /tmp/darwin-agent.service << SERVICEEOF
[Unit]
Description=Darwin Agent — Autonomous Trading Bot
After=network-online.target chrony.service
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/venv/bin/python -m darwin_agent.main --mode test
Restart=always
RestartSec=30
StartLimitIntervalSec=300
StartLimitBurst=5
StandardOutput=journal
StandardError=journal
MemoryMax=512M
CPUQuota=50%

# Graceful shutdown (agent saves state on SIGTERM)
TimeoutStopSec=30
KillSignal=SIGTERM

# Auto-restart on OOM or crash
OOMPolicy=continue

[Install]
WantedBy=multi-user.target
SERVICEEOF

sudo mv /tmp/darwin-agent.service /etc/systemd/system/darwin-agent.service
sudo systemctl daemon-reload

echo ""
echo "============================================"
echo "✅ Deployment complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Edit API keys:"
echo "   nano $PROJECT_DIR/config.yaml"
echo ""
echo "2. Test run:"
echo "   cd $PROJECT_DIR && source venv/bin/activate"
echo "   python -m darwin_agent.main --mode test"
echo ""
echo "3. Run as service:"
echo "   sudo systemctl start darwin-agent"
echo "   sudo systemctl enable darwin-agent"
echo ""
echo "4. Monitor:"
echo "   journalctl -u darwin-agent -f"
echo "   http://YOUR_IP:8080  (dashboard)"
echo ""
echo "5. Switch to LIVE when ready:"
echo "   # Edit service: change --mode test to --mode live"
echo "   sudo systemctl edit darwin-agent"
echo "   sudo systemctl restart darwin-agent"
echo ""
echo "⚠️  ALWAYS start with --mode test!"
echo "============================================"
