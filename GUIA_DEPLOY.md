# 🧬 Darwin Agent v2.2 — Guía Completa de Deploy desde el Celular

## 🎯 Resumen: De $50 a Miles

El agente NO va a convertir $50 en $10,000 en una semana. Eso es estafa.
Lo que SÍ hace es evolucionar generación tras generación:

```
Gen 0-5:   Aprende qué funciona (muchas muertes, paper trading)
Gen 5-15:  Empieza a ser consistente (WR > 52%)
Gen 15-30: Compounding real ($50 → $100 → $200)
Gen 30+:   Si sobrevive, compounding exponencial
```

**La clave es paciencia + capital compuesto.** A un 2% diario consistente:
- Semana 1: $50 → $57
- Mes 1: $50 → $91
- Mes 3: $50 → $301
- Mes 6: $50 → $1,814
- Mes 9: $50 → $10,926

Eso es el poder del compounding. El agente necesita SOBREVIVIR y aprender.

---

## 📱 Deploy desde el Celular (Paso a Paso)

### Paso 1: Crear Droplet en DigitalOcean

1. Descargar la app **DigitalOcean** del App Store / Play Store
2. Crear cuenta (te dan $200 de crédito gratis por 60 días)
3. Create Droplet:
   - **Image:** Ubuntu 22.04
   - **Plan:** Basic $6/mes (1GB RAM, 1 CPU)
   - **Region:** New York o el más cercano a ti
   - **Authentication:** Password (más fácil desde el celular)
4. Anotar la IP del droplet

### Paso 2: Conectar por SSH

Descargar **Termius** (gratis) desde App Store / Play Store.

```
Host: TU_IP_DEL_DROPLET
User: root
Password: el que pusiste
```

### Paso 3: Instalar y Configurar

Copiar y pegar estos comandos uno por uno en Termius:

```bash
# 1. Actualizar sistema
apt update && apt upgrade -y

# 2. Instalar dependencias
apt install -y python3.11 python3.11-venv python3-pip git unzip ufw

# 3. Firewall
ufw allow 22
ufw allow 8080
ufw --force enable

# 4. Crear directorio
mkdir -p /root/darwin_agent
cd /root/darwin_agent
```

### Paso 4: Subir los Archivos

**Opción A — Desde GitHub (recomendado):**
Sube el ZIP a un repo privado de GitHub, luego:
```bash
git clone https://github.com/TU_USUARIO/darwin-agent.git .
```

**Opción B — Subir ZIP directo:**
En Termius, usa la función SFTP para subir `darwin_agent_v2.2.zip`, luego:
```bash
cd /root
unzip darwin_agent_v2.2.zip
cp -r darwin_agent_final/* /root/darwin_agent/
cd /root/darwin_agent
```

### Paso 5: Instalar Python y Dependencias

```bash
cd /root/darwin_agent
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install aiohttp pyyaml numpy
```

### Paso 6: Configurar API Keys

```bash
cp config_example.yaml config.yaml
nano config.yaml
```

Cambiar estas líneas:
```yaml
markets:
  crypto:
    enabled: true
    api_key: "TU_API_KEY_DE_BYBIT"
    api_secret: "TU_API_SECRET_DE_BYBIT"
    testnet: true  # ← DEJARLO EN TRUE POR AHORA
```

Guardar: `Ctrl+X`, luego `Y`, luego `Enter`

### Paso 7: Primer Test

```bash
cd /root/darwin_agent
source venv/bin/activate
python -m darwin_agent.main --mode test
```

Debería verse algo como:
```
🧬 D A R W I N   A G E N T 🧬
Mode: TEST
Starting Generation: 0
Capital: $50.00
🐣 BORN — Capital: $50.00
```

Si funciona, `Ctrl+C` para parar.

### Paso 8: Correr como Servicio (Background)

```bash
cat > /etc/systemd/system/darwin-agent.service << 'EOF'
[Unit]
Description=Darwin Agent Trading Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/darwin_agent
ExecStart=/root/darwin_agent/venv/bin/python -m darwin_agent.main --mode test
Restart=on-failure
RestartSec=30
MemoryMax=512M
CPUQuota=50%

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl start darwin-agent
systemctl enable darwin-agent
```

### Paso 9: Ver el Dashboard desde el Celular

Abre en tu navegador del celular:
```
http://TU_IP_DEL_DROPLET:8080
```

Verás el dashboard en tiempo real con:
- HP del agente
- Capital actual
- Trades y win rate
- Estado del brain ML
- Playbook de regímenes
- Historial de evolución

### Paso 10: Monitorear

```bash
# Ver logs en vivo
journalctl -u darwin-agent -f

# Ver estado de evolución
cd /root/darwin_agent && source venv/bin/activate
python -m darwin_agent.main --status

# Reiniciar agente
systemctl restart darwin-agent

# Parar agente
systemctl stop darwin-agent
```

---

## 🔑 Obtener API Keys de Bybit

### Para Testnet (GRATIS, sin dinero real):
1. Ir a https://testnet.bybit.com
2. Crear cuenta
3. API Management → Create New Key
4. Permisos: ✅ Read, ✅ Trade (NO withdrawal)
5. Copiar Key y Secret

### Para Mainnet (dinero real):
1. Ir a https://www.bybit.com
2. Crear cuenta y verificar identidad
3. Depositar $50 USDT
4. API Management → Create Key
5. Permisos: ✅ Read, ✅ Trade, ❌ NO withdrawal
6. IP whitelist: poner la IP de tu droplet

---

## ⚠️ Checklist Antes de Ir a LIVE

```
□ El agente ha completado 5+ generaciones en paper
□ Win rate > 52% consistente
□ El playbook tiene regímenes aprendidos
□ Brain epsilon < 0.15
□ Has revisado logs manualmente
□ API keys de mainnet listas
□ $50 USDT depositados en Bybit
□ IP del server en whitelist de Bybit
```

Cuando todo esté listo:

```bash
# 1. Editar config
nano /root/darwin_agent/config.yaml
# Cambiar: testnet: false
# Cambiar: api_key y api_secret por los de mainnet

# 2. Editar servicio
nano /etc/systemd/system/darwin-agent.service
# Cambiar: --mode test  →  --mode live

# 3. Reiniciar
systemctl daemon-reload
systemctl restart darwin-agent
```

---

## 📊 Cómo Maximizar Ganancias

### 1. Déjalo Morir y Renacer
Cada muerte = más inteligencia. Las primeras 10 generaciones son aprendizaje.

### 2. No Toques la Config Demasiado
El agente aprende solo. Si cambias parámetros constantemente, rompes su evolución.

### 3. Empieza en Testnet por Semanas
No hay prisa. Paper trading es gratis y el agente aprende igual.

### 4. Cuando Vayas Live, Empieza con $50
Si funciona con $50, funciona con $500. Escala gradualmente.

### 5. Monitorea el Playbook
El playbook te dice qué aprendió:
```
trending_volatile → momentum (WR: 62%)
ranging_tight → mean_reversion (WR: 58%)
choppy → hold (protección)
```
Si el playbook se ve sólido, el agente está listo.

### 6. Compounding
El agente reinvierte automáticamente. No retires capital temprano.
$50 compounding al 1.5% diario = $1,000 en ~3 meses.

---

## 🆘 Troubleshooting

**El agente no conecta:**
```bash
# Verificar que Bybit esté accessible
curl -s https://api-testnet.bybit.com/v5/market/time | head
```

**El agente muere muy rápido:**
- Es normal las primeras generaciones
- Cada muerte enseña algo nuevo
- Después de Gen 5-10 debería sobrevivir más

**Dashboard no carga:**
```bash
# Verificar que el puerto está abierto
ufw status
# Debería mostrar 8080 ALLOW

# Verificar que el agente está corriendo
systemctl status darwin-agent
```

**Error de memoria:**
```bash
# Ver uso de memoria
free -m
# Si se agota, reiniciar
systemctl restart darwin-agent
```

---

## 💰 Costos Mensuales

| Servicio | Costo |
|----------|-------|
| DigitalOcean Droplet | $6/mes |
| Bybit fees (0.06%) | ~$0.03/trade |
| **Total** | **~$7/mes** |

Con $200 de crédito gratis de DigitalOcean, tienes ~33 meses gratis.
