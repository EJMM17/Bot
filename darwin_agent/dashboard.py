"""Darwin Agent — Web Dashboard. Mobile-friendly real-time monitoring."""

import asyncio
import json
import glob
import os

try:
    from aiohttp import web
except ImportError:
    web = None

from darwin_agent.utils.config import load_config, save_config, config_to_dict

_agent = None
_config = None
_config_path = "config.yaml"
_MASKED_SECRET = "********"


def _dashboard_token() -> str:
    """Read optional dashboard token from environment."""
    return os.environ.get("DARWIN_DASHBOARD_TOKEN", "").strip()


def _has_token_auth() -> bool:
    return bool(_dashboard_token())


def _is_authorized(req) -> bool:
    """Validate Bearer/X-Dashboard-Token when dashboard auth is enabled."""
    token = _dashboard_token()
    if not token:
        return True

    auth_header = req.headers.get("Authorization", "")
    bearer = auth_header[7:].strip() if auth_header.startswith("Bearer ") else ""
    x_token = req.headers.get("X-Dashboard-Token", "").strip()
    query_token = req.query.get("token", "").strip()
    supplied = bearer or x_token or query_token
    return bool(supplied) and supplied == token


def _require_auth(req):
    if _is_authorized(req):
        return None
    return web.json_response({"ok": False, "error": "Unauthorized"}, status=401)


def set_agent(agent):
    global _agent
    _agent = agent


def set_config_context(config, config_path="config.yaml"):
    global _config, _config_path
    _config = config
    _config_path = config_path


def _load_generations():
    files = sorted(glob.glob("data/generations/gen_*.json"))
    gens = []
    for fp in files:
        try:
            with open(fp) as f:
                gens.append(json.load(f))
        except Exception:
            pass
    return gens


def _load_trades(n=20):
    files = sorted(glob.glob("data/logs/trades_gen_*.jsonl"))
    if not files:
        return []
    trades = []
    with open(files[-1]) as f:
        for line in f:
            try:
                trades.append(json.loads(line.strip()))
            except Exception:
                pass
    return trades[-n:]


async def handle_status(req):
    s = _agent.get_status() if _agent else {"error": "Agent not running"}
    return web.json_response(s)


async def handle_history(req):
    return web.json_response(_load_generations())


async def handle_trades(req):
    return web.json_response(_load_trades())


async def handle_config_get(req):
    auth_error = _require_auth(req)
    if auth_error:
        return auth_error

    cfg = _config if _config else load_config(_config_path)
    cfg_data = config_to_dict(cfg)

    # Never expose API secret over HTTP; allow frontend to preserve existing value via mask.
    for market in cfg_data.get("markets", {}).values():
        if "api_secret" in market and market["api_secret"]:
            market["api_secret"] = _MASKED_SECRET

    cfg_data["dashboard_auth_enabled"] = _has_token_auth()
    return web.json_response(cfg_data)


async def handle_config_post(req):
    global _config

    auth_error = _require_auth(req)
    if auth_error:
        return auth_error

    try:
        payload = await req.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON payload"}, status=400)

    if not isinstance(payload, dict):
        return web.json_response({"ok": False, "error": "Payload must be a JSON object"}, status=400)

    cfg = _config if _config else load_config(_config_path)

    for key in ("starting_capital", "heartbeat_interval", "log_level", "dashboard_port"):
        if key in payload:
            value = payload[key]
            if value is None:
                return web.json_response({"ok": False, "error": f"Field '{key}' cannot be null"}, status=400)
            try:
                setattr(cfg, key, type(getattr(cfg, key))(value))
            except (TypeError, ValueError):
                return web.json_response(
                    {"ok": False, "error": f"Invalid value for '{key}': {value!r}"},
                    status=400,
                )

    if "markets" in payload and payload["markets"] is not None:
        if not isinstance(payload["markets"], dict):
            return web.json_response({"ok": False, "error": "Field 'markets' must be an object"}, status=400)
        for name, mdata in payload["markets"].items():
            if name not in cfg.markets:
                continue
            if not isinstance(mdata, dict):
                return web.json_response(
                    {"ok": False, "error": f"Market '{name}' config must be an object"}, status=400
                )
            market = cfg.markets[name]
            for field in ("enabled", "api_key", "api_secret", "testnet", "max_allocation_pct"):
                if field in mdata:
                    value = mdata[field]
                    # Preserve existing secret if UI sends placeholder or blank value.
                    if field == "api_secret" and value in (None, "", _MASKED_SECRET):
                        continue
                    setattr(market, field, value)

    try:
        cfg.validate()
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)

    try:
        save_config(cfg, _config_path)
    except Exception as e:
        return web.json_response({"ok": False, "error": f"Could not save config: {e}"}, status=500)

    _config = cfg
    return web.json_response({"ok": True})


async def handle_index(req):
    return web.Response(text=HTML, content_type="text/html")


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>Darwin Agent</title>
<style>
:root{--bg:#0a0e14;--s:#131920;--b:#1e2530;--t:#d4d8de;--d:#6b7280;--g:#22c55e;--r:#ef4444;--y:#eab308;--bl:#3b82f6;--p:#8b5cf6}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--t);font-family:-apple-system,system-ui,sans-serif;font-size:14px;-webkit-font-smoothing:antialiased}
.c{max-width:900px;margin:0 auto;padding:12px}
.hdr{display:flex;align-items:center;justify-content:space-between;padding:12px 0;border-bottom:1px solid var(--b);margin-bottom:12px}
.hdr h1{font-size:16px;color:var(--bl);letter-spacing:1px}
.hdr .tag{font-size:11px;padding:3px 8px;border-radius:4px;background:#1e293b;color:var(--d)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
@media(max-width:600px){.grid{grid-template-columns:1fr}}
.card{background:var(--s);border:1px solid var(--b);border-radius:8px;padding:14px}
.card h2{font-size:11px;color:var(--d);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.m{display:flex;justify-content:space-between;padding:4px 0;font-size:13px}
.m .l{color:var(--d)}.m .v{font-weight:600}
.g{color:var(--g)}.r{color:var(--r)}.y{color:var(--y)}
.hp{height:10px;background:var(--b);border-radius:5px;margin:6px 0;overflow:hidden}
.hp .f{height:100%;border-radius:5px;transition:width .5s}
.full{grid-column:1/-1}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;color:var(--d);padding:5px 6px;border-bottom:1px solid var(--b);font-size:11px}
td{padding:5px 6px;border-bottom:1px solid #1a1f28}
.badge{display:inline-block;padding:2px 7px;border-radius:3px;font-size:10px;font-weight:600}
.bg{background:#14532d;color:var(--g)}.br{background:#450a0a;color:var(--r)}.by{background:#422006;color:var(--y)}.bb{background:#172554;color:var(--bl)}
.ft{text-align:center;padding:8px;color:var(--d);font-size:11px}
.dot{width:6px;height:6px;border-radius:50%;display:inline-block;margin-right:4px;animation:pulse 2s infinite}
input,select{width:100%;background:#0f1520;border:1px solid var(--b);border-radius:6px;color:var(--t);padding:8px;font-size:12px}
label{font-size:11px;color:var(--d);display:block;margin-bottom:4px}
.frm{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.frm .full{grid-column:1/-1}
.btn{background:var(--bl);color:white;border:0;border-radius:6px;padding:9px 12px;font-weight:600;cursor:pointer}
#cfg-msg{font-size:12px;margin-top:8px}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.dot-g{background:var(--g)}.dot-r{background:var(--r)}.dot-y{background:var(--y)}
</style>
</head>
<body>
<div class="c">
<div class="hdr">
<h1>DARWIN AGENT</h1>
<div><span id="gen" class="tag"></span> <span id="live" class="tag"></span></div>
</div>

<div class="card full" id="auth-card" style="display:none">
<h2>Dashboard Login</h2>
<div class="frm">
<div class="full"><label>Access Token</label><input id="dash-token" type="password" placeholder="Bearer token"></div>
<div class="full"><button class="btn" onclick="saveTokenAndLoad()">Unlock</button><div id="auth-msg"></div></div>
</div>
</div>

<div class="grid">
<div class="card">
<h2>Health</h2>
<div class="hp"><div class="f" id="hp-fill"></div></div>
<div style="text-align:center;font-size:12px;color:var(--d)" id="hp-text"></div>
<div id="health-m"></div>
</div>
<div class="card">
<h2>Trading</h2>
<div id="trade-m"></div>
</div>
<div class="card">
<h2>Risk</h2>
<div id="risk-m"></div>
</div>
<div class="card">
<h2>Brain (ML)</h2>
<div id="brain-m"></div>
</div>
<div class="card full">
<h2>Dashboard Config (API keys y settings)</h2>
<div class="frm">
<div><label>Starting Capital</label><input id="cfg-starting-capital" type="number" step="0.1"></div>
<div><label>Heartbeat Interval</label><input id="cfg-heartbeat" type="number"></div>
<div><label>Dashboard Port</label><input id="cfg-port" type="number"></div>
<div><label>Log Level</label><input id="cfg-log-level" type="text"></div>
<div><label>Crypto Enabled</label><select id="cfg-enabled"><option value="true">true</option><option value="false">false</option></select></div>
<div><label>Crypto Testnet</label><select id="cfg-testnet"><option value="true">true</option><option value="false">false</option></select></div>
<div><label>Bybit API Key</label><input id="cfg-api-key" type="text"></div>
<div><label>Bybit API Secret</label><input id="cfg-api-secret" type="password"></div>
<div class="full"><label>Max Allocation %</label><input id="cfg-max-allocation" type="number" step="0.1"></div>
<div class="full"><button class="btn" onclick="saveConfig()">Save Config</button><div id="cfg-msg"></div></div>
</div>
</div>
<div class="card full">
<h2>Playbook</h2>
<table><thead><tr><th>Regime</th><th>Strategy</th><th>WR</th><th>#</th><th>P&L</th></tr></thead>
<tbody id="pb"></tbody></table>
</div>
<div class="card full">
<h2>Evolution</h2>
<table><thead><tr><th>Gen</th><th>Capital</th><th>Trades</th><th>WR</th><th>Death</th></tr></thead>
<tbody id="evo"></tbody></table>
</div>
<div class="card full">
<h2>Recent Trades</h2>
<table><thead><tr><th>Time</th><th>Side</th><th>Symbol</th><th>Price</th><th>Reason</th></tr></thead>
<tbody id="trd"></tbody></table>
</div>
</div>
<div class="ft"><span class="dot dot-g" id="dot"></span> <span id="ts"></span></div>
</div>

<script>
function M(l,v,c=''){return `<div class="m"><span class="l">${l}</span><span class="v ${c}">${v}</span></div>`}
function hc(p){return p>.7?'var(--g)':p>.4?'var(--y)':'var(--r)'}
function esc(v){
  return String(v??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
}
function getStoredToken(){
  return localStorage.getItem('darwin-dashboard-token')||'';
}

function authHeaders(base={}){
  const t=getStoredToken();
  if(t) base['Authorization']='Bearer '+t;
  return base;
}

function saveTokenAndLoad(){
  const t=document.getElementById('dash-token').value.trim();
  if(t){
    localStorage.setItem('darwin-dashboard-token',t);
  }
  loadConfig();
}

async function loadConfig(){
  try{
    const r=await fetch('/api/config',{headers:authHeaders()});
    if(r.status===401){
      document.getElementById('auth-card').style.display='block';
      document.getElementById('auth-msg').innerHTML='<span class="y">Token required. Set DARWIN_DASHBOARD_TOKEN and paste token here.</span>';
      return;
    }
    const c=await r.json();
    if(c.dashboard_auth_enabled){
      document.getElementById('auth-card').style.display='block';
    }
    const m=(c.markets||{}).crypto||{};
    document.getElementById('cfg-starting-capital').value=c.starting_capital??50;
    document.getElementById('cfg-heartbeat').value=c.heartbeat_interval??45;
    document.getElementById('cfg-port').value=c.dashboard_port??8080;
    document.getElementById('cfg-log-level').value=c.log_level??'INFO';
    document.getElementById('cfg-enabled').value=String(m.enabled??true);
    document.getElementById('cfg-testnet').value=String(m.testnet??true);
    document.getElementById('cfg-api-key').value=m.api_key??'';
    document.getElementById('cfg-api-secret').value=m.api_secret??'';
    document.getElementById('cfg-max-allocation').value=m.max_allocation_pct??60;
  }catch(e){
    document.getElementById('cfg-msg').innerHTML='<span class="r">Failed to load config</span>';
  }
}

async function saveConfig(){
  const payload={
    starting_capital:parseFloat(document.getElementById('cfg-starting-capital').value),
    heartbeat_interval:parseInt(document.getElementById('cfg-heartbeat').value),
    dashboard_port:parseInt(document.getElementById('cfg-port').value),
    log_level:document.getElementById('cfg-log-level').value,
    markets:{
      crypto:{
        enabled:document.getElementById('cfg-enabled').value==='true',
        testnet:document.getElementById('cfg-testnet').value==='true',
        api_key:document.getElementById('cfg-api-key').value,
        api_secret:document.getElementById('cfg-api-secret').value,
        max_allocation_pct:parseFloat(document.getElementById('cfg-max-allocation').value)
      }
    }
  };

  const el=document.getElementById('cfg-msg');
  el.textContent='Saving...';
  try{
    const r=await fetch('/api/config',{method:'POST',headers:authHeaders({'Content-Type':'application/json'}),body:JSON.stringify(payload)});
    if(r.status===401){
      el.innerHTML='<span class="r">Unauthorized: invalid dashboard token</span>';
      return;
    }
    if(!r.ok){
      const t=await r.text();
      el.innerHTML='<span class="r">Save failed: '+t+'</span>';
      return;
    }
    el.innerHTML='<span class="g">Config saved. Restart recommended if you changed the port.</span>';
  }catch(e){
    el.innerHTML='<span class="r">Save failed</span>';
  }
}

async function R(){
try{
const[sr,hr,tr]=await Promise.all([
  fetch('/api/status'),
  fetch('/api/history'),
  fetch('/api/trades')
]);
const s=await sr.json(),hi=await hr.json(),trades=await tr.json();
if(s.error)return;
const h=s.health||{},r=s.risk||{},b=s.brain||{},pb=s.playbook||{};

document.getElementById('gen').textContent='Gen-'+s.generation;
document.getElementById('live').textContent=s.phase.toUpperCase()+' #'+s.cycle;

const p=h.hp/h.max_hp;
const fl=document.getElementById('hp-fill');
fl.style.width=(p*100)+'%';fl.style.background=hc(p);
document.getElementById('hp-text').textContent=h.hp+'/'+h.max_hp+' HP';

document.getElementById('health-m').innerHTML=
M('Capital','$'+h.capital?.toFixed(2))+
M('Peak','$'+h.peak_capital?.toFixed(2))+
M('Drawdown',h.drawdown_pct?.toFixed(1)+'%',h.drawdown_pct>15?'r':'')+
M('Status','<span class="badge b'+(h.status==='healthy'?'g':h.status==='critical'?'r':'y')+'">'+h.status+'</span>');

document.getElementById('trade-m').innerHTML=
M('Trades',h.total_trades)+
M('Win Rate',(h.win_rate*100).toFixed(1)+'%',h.win_rate>.52?'g':'r')+
M('Win Streak',h.win_streak,'g')+
M('Loss Streak',h.loss_streak,h.loss_streak>=2?'r':'');

document.getElementById('risk-m').innerHTML=
M('Daily',r.daily_trades+'/'+r.daily_limit)+
M('Daily P&L','$'+r.daily_pnl?.toFixed(2),r.daily_pnl>=0?'g':'r')+
M('Capital','<span class="badge b'+(r.capital_status==='NORMAL'?'g':'r')+'">'+r.capital_status+'</span>');

document.getElementById('brain-m').innerHTML=
M('Epsilon',b.current_epsilon)+
M('Decisions',b.total_decisions)+
M('Explore',(b.exploration_rate*100).toFixed(0)+'%')+
M('Memory',b.memory_size)+
M('Regimes',(b.regimes_learned||[]).join(', ')||'learning...');

let ph='';
for(const[rg,d]of Object.entries(pb)){
ph+=`<tr><td>${esc(rg)}</td><td><span class="badge bb">${esc(d.best_strategy)}</span></td>
<td class="${d.win_rate>.5?'g':'r'}">${(d.win_rate*100).toFixed(0)}%</td>
<td>${d.trades}</td><td class="${d.total_pnl>=0?'g':'r'}">$${d.total_pnl?.toFixed(2)}</td></tr>`}
document.getElementById('pb').innerHTML=ph||'<tr><td colspan="5" style="color:var(--d)">Learning...</td></tr>';

let eh='';
for(const g of hi.slice(-8).reverse()){
const c=g.final_capital>50?'g':g.final_capital>25?'y':'r';
eh+=`<tr><td>${esc(g.generation)}</td><td class="${c}">$${g.final_capital?.toFixed(2)}</td>
<td>${g.total_trades}</td><td>${(g.win_rate*100).toFixed(1)}%</td>
<td style="max-width:150px;overflow:hidden;text-overflow:ellipsis">${esc(g.cause_of_death||'-')}</td></tr>`}
document.getElementById('evo').innerHTML=eh||'<tr><td colspan="5" style="color:var(--d)">No history</td></tr>';

let th='';
for(const t of trades.slice(-8).reverse()){
th+=`<tr><td>${esc((t.ts||'').substring(11,19))}</td>
<td class="${t.action==='BUY'?'g':'r'}">${esc(t.action)}</td>
<td>${esc(t.symbol)}</td><td>$${t.price?.toFixed(2)}</td>
<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis">${esc(t.reason||'')}</td></tr>`}
document.getElementById('trd').innerHTML=th||'<tr><td colspan="5" style="color:var(--d)">No trades</td></tr>';

document.getElementById('ts').textContent='Updated '+new Date().toLocaleTimeString();
document.getElementById('dot').className='dot dot-g';
}catch(e){
document.getElementById('dot').className='dot dot-r';
document.getElementById('ts').textContent='Disconnected';
}}

loadConfig();
R();setInterval(R,5000);
</script>
</body>
</html>"""


def create_app():
    if web is None:
        raise ImportError("pip install aiohttp")
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/status", handle_status)
    app.router.add_get("/api/history", handle_history)
    app.router.add_get("/api/trades", handle_trades)
    app.router.add_get("/api/config", handle_config_get)
    app.router.add_post("/api/config", handle_config_post)
    return app


async def start_dashboard(port=8080):
    """Start dashboard as a long-running task. Runs forever until cancelled."""
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    try:
        # Keep running until cancelled
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        await runner.cleanup()
