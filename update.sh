#!/bin/bash
# ═══════════════════════════════════════════
# Darwin Agent — Update Script
# Actualiza una instalacion existente sin perder datos ni config
# ═══════════════════════════════════════════
set -e

PROJECT_DIR="${PROJECT_DIR:-/root/darwin_agent}"
BACKUP_DIR="/root/darwin_backup_$(date +%Y%m%d_%H%M%S)"

echo "Darwin Agent — Update"
echo "============================================"
echo "Directorio: $PROJECT_DIR"
echo "Backup en:  $BACKUP_DIR"
echo ""

# 1. Verificar que existe una instalacion previa
if [ ! -d "$PROJECT_DIR/darwin_agent" ]; then
    echo "ERROR: No se encontro instalacion en $PROJECT_DIR"
    echo "Usa deploy.sh para una instalacion nueva."
    exit 1
fi

# 2. Parar el servicio si esta corriendo
echo "[1/7] Parando el agente..."
if systemctl is-active --quiet darwin-agent 2>/dev/null; then
    sudo systemctl stop darwin-agent
    echo "      Agente detenido."
    AGENT_WAS_RUNNING=true
else
    echo "      El agente no estaba corriendo."
    AGENT_WAS_RUNNING=false
fi

# 3. Backup de config y datos
echo "[2/7] Creando backup..."
mkdir -p "$BACKUP_DIR"
if [ -f "$PROJECT_DIR/config.yaml" ]; then
    cp "$PROJECT_DIR/config.yaml" "$BACKUP_DIR/config.yaml"
    echo "      config.yaml respaldado"
fi
if [ -d "$PROJECT_DIR/data" ]; then
    cp -r "$PROJECT_DIR/data" "$BACKUP_DIR/data"
    echo "      data/ respaldado (generaciones, brain, logs)"
fi
echo "      Backup completo en $BACKUP_DIR"

# 4. Copiar archivos nuevos
echo "[3/7] Actualizando codigo..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$SCRIPT_DIR" != "$PROJECT_DIR" ]; then
    # Copiar desde directorio fuente al destino
    cp -r "$SCRIPT_DIR/darwin_agent/"* "$PROJECT_DIR/darwin_agent/"
    cp "$SCRIPT_DIR/requirements.txt" "$PROJECT_DIR/"
    cp "$SCRIPT_DIR/config_example.yaml" "$PROJECT_DIR/"
    [ -f "$SCRIPT_DIR/Dockerfile" ] && cp "$SCRIPT_DIR/Dockerfile" "$PROJECT_DIR/"
    [ -f "$SCRIPT_DIR/docker-compose.yml" ] && cp "$SCRIPT_DIR/docker-compose.yml" "$PROJECT_DIR/"
    [ -f "$SCRIPT_DIR/deploy.sh" ] && cp "$SCRIPT_DIR/deploy.sh" "$PROJECT_DIR/"
    [ -f "$SCRIPT_DIR/update.sh" ] && cp "$SCRIPT_DIR/update.sh" "$PROJECT_DIR/"
    echo "      Codigo copiado desde $SCRIPT_DIR"
else
    echo "      Ejecutando desde $PROJECT_DIR (git pull o copia manual ya hecha)"
fi

# 5. Restaurar config
echo "[4/7] Restaurando configuracion..."
if [ -f "$BACKUP_DIR/config.yaml" ]; then
    cp "$BACKUP_DIR/config.yaml" "$PROJECT_DIR/config.yaml"
    echo "      config.yaml restaurado"
fi

# 6. Actualizar dependencias
echo "[5/7] Actualizando dependencias Python..."
cd "$PROJECT_DIR"
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "      Creando venv nuevo..."
    python3.11 -m venv venv
    source venv/bin/activate
fi
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "      Dependencias actualizadas"

# 7. Verificar instalacion
echo "[6/7] Verificando instalacion..."
python -m darwin_agent.main --diagnose 2>&1 | head -20 || true

# 8. Reiniciar servicio si estaba corriendo
echo "[7/7] Reiniciando servicio..."
if [ "$AGENT_WAS_RUNNING" = true ]; then
    sudo systemctl daemon-reload
    sudo systemctl start darwin-agent
    sleep 2
    if systemctl is-active --quiet darwin-agent 2>/dev/null; then
        echo "      Agente reiniciado correctamente"
    else
        echo "      ADVERTENCIA: El agente no arranco. Revisa los logs:"
        echo "      journalctl -u darwin-agent -n 50"
    fi
else
    echo "      El agente no estaba corriendo. Para iniciarlo:"
    echo "      sudo systemctl start darwin-agent"
fi

echo ""
echo "============================================"
echo "Actualizacion completada"
echo "============================================"
echo ""
echo "Verificar:"
echo "  sudo systemctl status darwin-agent"
echo "  journalctl -u darwin-agent -f"
echo "  http://TU_IP:8080  (dashboard)"
echo ""
echo "Si algo sale mal, restaurar backup:"
echo "  cp $BACKUP_DIR/config.yaml $PROJECT_DIR/config.yaml"
echo "  cp -r $BACKUP_DIR/data/* $PROJECT_DIR/data/"
echo "  sudo systemctl restart darwin-agent"
echo "============================================"
