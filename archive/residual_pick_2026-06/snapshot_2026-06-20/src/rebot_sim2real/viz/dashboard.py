"""Live reBot SAC training dashboard — sim render frames + training metrics in one page.

A TensorBoard-free live view: open ``http://<host>:<port>`` while training runs and watch
the 3 monitor camera views (quarter / top / wrist) next to the SAC curves (ep_return,
success_rate, losses, alpha) and the staged-reward term breakdown.

Tech: stdlib ``http.server`` (threaded, daemon) + Pillow(JPEG) + numpy. No flask / cv2 /
websocket. The browser polls ``/api/stats`` (JSON) and ``/cam/<view>`` (single JPEG) — frames
are cache-busted with ``?t=<ms>`` so each poll fetches a fresh image (no MJPEG/SSE needed).

Wiring (two processes share this server's state — see docs):
    # in scripts/monitor.py (cameras ON): host the server + push frames + relay metrics
    from rebot_sim2real.viz import dashboard as dash
    dash.start(port=8890)                                  # once
    dash.set_camera_image("quarter", img_hxwx3_uint8)      # each captured frame (live, per-frame)
    dash.push_stats(step=..., ep_return=..., success_rate=..., critic_loss=..., reward_terms={...})

    # in src/rebot_sim2real/rl/train.py (headless trainer, no cameras): just dump JSON;
    # the monitor reads it and calls push_stats for us (single combined URL).

Standalone demo:  python -P dashboard.py --demo --port 8890
"""
from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import numpy as np
from PIL import Image

# the three monitor views (must match scripts/monitor.py `cams` keys)
_VIEWS = ("quarter", "top", "wrist")
_MAX_HIST = 600

_cam_frames: dict[str, bytes] = {}
_cam_lock = threading.Lock()
_stats: dict = {"reward_history": [], "phase": "-"}
_stats_lock = threading.Lock()


def _jpeg(img, q: int = 70) -> bytes:
    """HxWx3 (uint8) numpy -> JPEG bytes. Heavy encode runs OUTSIDE the cam lock."""
    b = io.BytesIO()
    Image.fromarray(np.asarray(img).astype(np.uint8)).save(b, format="JPEG", quality=q)
    return b.getvalue()


def set_camera_image(name: str, np_img) -> None:
    """Cache one live view as JPEG. `np_img` is HxWx3 uint8 (exactly monitor.grab() output)."""
    data = _jpeg(np_img)                       # encode first, lock only for the dict assignment
    with _cam_lock:
        _cam_frames[name] = data


def push_stats(**kw) -> None:
    """Merge a flat metrics dict into the live state and append one history point.

    Expected keys (all optional): step, phase, ep_return, success_rate, ep_len,
    critic_loss, actor_loss, alpha, entropy, q_mean, buffer, fps, best_success_rate,
    mean_step_reward, reward_terms(dict). De-dupes history by `step` so repeated relays
    of an unchanged JSON don't inflate the chart.
    """
    with _stats_lock:
        _stats.update(kw)
        step = kw.get("step", _stats.get("step", 0))
        h = _stats.setdefault("reward_history", [])
        if not h or h[-1].get("step") != step:
            h.append({"step": step,
                      "ret": kw.get("ep_return", 0.0),
                      "sr": kw.get("success_rate", 0.0)})
            if len(h) > _MAX_HIST:
                del h[: len(h) - _MAX_HIST]


def _safe_json(obj) -> str:
    # RL early on emits NaN/Inf (alpha, critic_loss); raw NaN breaks browser JSON.parse.
    return (json.dumps(obj, default=str)
            .replace("NaN", "0").replace("-Infinity", "0").replace("Infinity", "9e9"))


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):                 # silence per-request logging
        pass

    def _jpg(self, data):
        if data:
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(204)            # no frame yet
            self.end_headers()

    def do_GET(self):
        p = self.path
        if p == "/api/stats":
            with _stats_lock:
                body = _safe_json(_stats).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
        elif p.startswith("/cam/"):
            name = p.split("/cam/")[-1].split("?")[0]
            with _cam_lock:
                data = _cam_frames.get(name)
            self._jpg(data)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(_HTML.encode())


class _S(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True                   # rebind immediately after a prior monitor (avoid TIME_WAIT "address in use")


def start(port: int = 8890):
    """Launch the dashboard HTTP server on a daemon thread. Returns the server (or None)."""
    try:
        srv = _S(("0.0.0.0", port), _H)
    except OSError as e:                        # port busy -> don't kill the monitor
        print(f"[dash] could not bind :{port} ({e}); dashboard disabled", flush=True)
        return None
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[dash] live dashboard at http://0.0.0.0:{port}  (views: {', '.join(_VIEWS)})", flush=True)
    return srv


_HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>reBot SAC — live</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,'Segoe UI',monospace;padding:16px}
h1{font-size:20px;margin-bottom:2px}#ph{font-size:13px;margin-bottom:14px}
.sec{font-size:12px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin:6px 2px 8px}
.cams{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px}
.cam{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:6px}
.cam img{width:100%;border-radius:4px;display:block;background:#000;aspect-ratio:16/9;object-fit:contain}
.cam .t{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin:4px 2px}
.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:16px}
.box{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;text-align:center}
.label{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px}
.val{font-size:22px;font-weight:700;margin-top:4px;font-variant-numeric:tabular-nums}
.blue{color:#58a6ff}.green{color:#3fb950}.yellow{color:#d29922}.red{color:#f85149}.purple{color:#a371f7}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;margin-bottom:16px}
.card-title{font-size:12px;color:#8b949e;text-transform:uppercase;margin-bottom:10px;letter-spacing:.5px}
svg{width:100%;height:220px;display:block}
.rp{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.rp-h{font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;font-weight:700}
.term{display:flex;justify-content:space-between;font-size:12px;padding:4px 8px;background:#0d1117;border-radius:4px;margin-bottom:3px}
.rp-sub{margin-top:8px;font-size:13px;font-weight:700;text-align:right;border-top:1px solid #21262d;padding-top:6px}
#updated{font-size:11px;color:#484f58;text-align:right}
</style></head><body>
<h1>reBot SAC — live training</h1><div id="ph" class="purple">phase: -</div>

<div class="grid">
  <div class="box"><div class="label">Env steps</div><div class="val" id="v-step">-</div></div>
  <div class="box"><div class="label">Ep Return</div><div class="val blue" id="v-ret">-</div></div>
  <div class="box"><div class="label">Success Rate</div><div class="val green" id="v-sr">-</div></div>
  <div class="box"><div class="label">Best Eval SR</div><div class="val purple" id="v-best">-</div></div>
  <div class="box"><div class="label">Ep Len</div><div class="val" id="v-len">-</div></div>
  <div class="box"><div class="label">Critic Loss</div><div class="val yellow" id="v-cl">-</div></div>
  <div class="box"><div class="label">Actor Loss</div><div class="val yellow" id="v-al">-</div></div>
  <div class="box"><div class="label">Alpha</div><div class="val" id="v-alpha">-</div></div>
  <div class="box"><div class="label">Buffer / FPS</div><div class="val" id="v-buf">-</div></div>
  <div class="box"><div class="label">Q mean / Entropy</div><div class="val" id="v-q">-</div></div>
</div>

<div class="card"><div class="card-title">Live sim — monitored deterministic rollout (quarter / top / wrist)</div>
  <div class="cams">
    <div class="cam"><div class="t">quarter (3rd-person)</div><img id="cam-quarter" src=""></div>
    <div class="cam"><div class="t">top</div><img id="cam-top" src=""></div>
    <div class="cam"><div class="t">wrist (eye-in-hand)</div><img id="cam-wrist" src=""></div>
  </div>
  <div id="cam-status" class="rp-sub">waiting for monitor frames…</div>
</div>

<div class="card"><div class="card-title">Episode return (blue) &amp; success rate (green) over env-steps</div>
  <svg id="chart" viewBox="0 0 800 220" preserveAspectRatio="none"></svg></div>

<div class="sec">Staged reward — term breakdown (batch mean, last log)</div>
<div class="card"><div class="rp">
  <div><div class="rp-h green">Rewards +</div><div id="rew"></div><div class="rp-sub" id="rew-sum"></div></div>
  <div><div class="rp-h red">Penalties −</div><div id="pen"></div><div class="rp-sub" id="pen-sum"></div></div>
</div></div>

<div id="updated"></div>

<script>
function n(v){return(typeof v==='number'&&isFinite(v))?v:0}
function f(v,d){return n(v).toFixed(d===undefined?3:d)}
function renderTerms(terms){
  terms=terms||{};var rew='',pen='',rs=0,ps=0;
  Object.keys(terms).forEach(function(k){var v=n(terms[k]);
    var row='<div class="term"><span>'+k+'</span><span style="font-weight:600">'+v.toFixed(3)+'</span></div>';
    if(v<0){pen+=row;ps+=v;}else{rew+=row;rs+=v;}});
  document.getElementById('rew').innerHTML=rew||'<span style="color:#484f58;font-size:12px">-</span>';
  document.getElementById('pen').innerHTML=pen||'<span style="color:#484f58;font-size:12px">-</span>';
  document.getElementById('rew-sum').innerHTML='Σ <span class="green">'+rs.toFixed(3)+'</span>';
  document.getElementById('pen-sum').innerHTML='Σ <span class="red">'+ps.toFixed(3)+'</span>';
}
function drawChart(hist){
  var svg=document.getElementById('chart');
  if(!hist||hist.length<2){svg.innerHTML='<text x="400" y="110" text-anchor="middle" fill="#484f58" font-size="14">Waiting for data…</text>';return}
  var W=800,H=220,L=52,R=52,T=20,B=30;
  var ret=hist.map(function(h){return n(h.ret)}),sr=hist.map(function(h){return n(h.sr)}),st=hist.map(function(h){return n(h.step)});
  function rng(a){var mn=Math.min.apply(null,a),mx=Math.max.apply(null,a);if(mx-mn<1e-6)mx=mn+1;return[mn,mx];}
  var rr=rng(ret);                       // success rate is always plotted on a fixed [0,1] right axis
  function xS(i){return L+(W-L-R)*i/(hist.length-1)}
  function yR(v){return(H-B)-(H-T-B)*(v-rr[0])/(rr[1]-rr[0])}
  function yS(v){return(H-B)-(H-T-B)*v}  // sr in [0,1]
  var s='<line x1="'+L+'" y1="'+T+'" x2="'+L+'" y2="'+(H-B)+'" stroke="#484f58"/>'
       +'<line x1="'+(W-R)+'" y1="'+T+'" x2="'+(W-R)+'" y2="'+(H-B)+'" stroke="#484f58"/>'
       +'<line x1="'+L+'" y1="'+(H-B)+'" x2="'+(W-R)+'" y2="'+(H-B)+'" stroke="#484f58"/>';
  for(var k=0;k<=4;k++){var gy=T+(H-T-B)*k/4;
    s+='<line x1="'+L+'" y1="'+gy+'" x2="'+(W-R)+'" y2="'+gy+'" stroke="#21262d"/>';
    s+='<text x="'+(L-5)+'" y="'+(gy+3)+'" fill="#58a6ff" font-size="9" text-anchor="end">'+(rr[1]-(rr[1]-rr[0])*k/4).toFixed(1)+'</text>';
    s+='<text x="'+(W-R+5)+'" y="'+(gy+3)+'" fill="#3fb950" font-size="9">'+(100-25*k)+'%</text>';}
  for(var t=0;t<=4;t++){var xi=Math.round((hist.length-1)*t/4);
    s+='<text x="'+xS(xi)+'" y="'+(H-12)+'" fill="#8b949e" font-size="9" text-anchor="middle">'+n(st[xi]).toLocaleString()+'</text>';}
  function poly(arr,yf,col,w){return'<polyline points="'+arr.map(function(v,i){return xS(i)+','+yf(v)}).join(' ')+'" fill="none" stroke="'+col+'" stroke-width="'+w+'"/>';}
  s+=poly(ret,yR,'#58a6ff',2)+poly(sr,yS,'#3fb950',2.5);
  s+='<text x="'+((L+W-R)/2)+'" y="'+(H-1)+'" fill="#8b949e" font-size="10" text-anchor="middle">env-steps · return=blue (L) · success=green (R)</text>';
  svg.innerHTML=s;
}
function update(){
  fetch('/api/stats').then(function(r){return r.json()}).then(function(s){
    document.getElementById('ph').textContent='phase: '+(s.phase||'-');
    document.getElementById('v-step').textContent=n(s.step).toLocaleString();
    document.getElementById('v-ret').textContent=f(s.ep_return,2);
    document.getElementById('v-sr').textContent=(n(s.success_rate)*100).toFixed(1)+'%';
    document.getElementById('v-best').textContent=(n(s.best_success_rate)*100).toFixed(1)+'%';
    document.getElementById('v-len').textContent=f(s.ep_len,0);
    document.getElementById('v-cl').textContent=f(s.critic_loss,3);
    document.getElementById('v-al').textContent=f(s.actor_loss,3);
    document.getElementById('v-alpha').textContent=f(s.alpha,3);
    document.getElementById('v-buf').textContent=n(s.buffer).toLocaleString()+' / '+f(s.fps,0);
    document.getElementById('v-q').textContent=f(s.q_mean,2)+' / '+f(s.entropy,2);
    drawChart(s.reward_history||[]);
    renderTerms(s.reward_terms);
    document.getElementById('updated').textContent='Updated: '+new Date().toLocaleTimeString();
  }).catch(function(e){document.getElementById('updated').textContent='Error: '+e;});
}
function pollCams(){var t=Date.now();var any=false;
  ['quarter','top','wrist'].forEach(function(nm){var img=document.getElementById('cam-'+nm);if(img){img.src='/cam/'+nm+'?t='+t;any=true;}});
  if(any)document.getElementById('cam-status').textContent='live feed · refreshed '+new Date().toLocaleTimeString();}
setInterval(update,1000);setInterval(pollCams,400);
update();pollCams();
</script></body></html>"""


if __name__ == "__main__":
    import argparse
    import math
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8890)
    ap.add_argument("--demo", action="store_true", help="feed synthetic stats + a test pattern frame")
    a = ap.parse_args()
    start(a.port)
    if a.demo:
        # a simple moving test pattern so the cam tiles aren't blank
        H, Wd = 180, 320
        yy, xx = np.mgrid[0:H, 0:Wd]
        step, ep_ret, sr = 0, 0.0, 0.0
        while True:
            step += 200
            ep_ret = math.sin(step / 4000.0) * 5 + 6 + step / 200000.0
            sr = min(1.0, max(0.0, step / 600000.0 + 0.1 * math.sin(step / 8000.0)))
            for vi, nm in enumerate(_VIEWS):
                pat = (((xx + step // 50 + vi * 40) // 16 + (yy // 16)) % 2 * 200 + 40).astype(np.uint8)
                set_camera_image(nm, np.stack([pat, (pat // 2 + vi * 60).astype(np.uint8), 255 - pat], -1))
            push_stats(step=step, phase="full", ep_return=ep_ret, success_rate=sr,
                       best_success_rate=sr, ep_len=120, critic_loss=abs(math.cos(step / 3000.0)) * 5,
                       actor_loss=-ep_ret * 0.3, alpha=0.2 * math.exp(-step / 400000.0) + 0.02,
                       entropy=-2.5, q_mean=ep_ret * 1.2, buffer=min(step * 64, 1_000_000), fps=900,
                       mean_step_reward=ep_ret / 120,
                       reward_terms={"r_reach": 0.6, "r_align": -0.1, "r_grasp": 0.4 * sr,
                                     "r_lift": 2.0 * sr, "r_success": 20 * sr,
                                     "p_action": -0.02, "p_time": -0.001, "p_fall": -0.3 * (1 - sr)})
            time.sleep(0.2)
