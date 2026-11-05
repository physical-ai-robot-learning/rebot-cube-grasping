"""Custom reBot SAC training dashboard.

Live: 3 camera views (front / iso / wrist) + reward progress CURRENT (per-step) and
CUMULATIVE (running) + success count.
Replay: the LAST SUCCESSFUL episode played back in the same 3 views (looping).

Wiring from the training loop:
    from rebot_teacher_student import dashboard as dash
    dash.start(port=8890, target_cube="cube")                 # once
    dash.update_cameras({"front": cam_f, "iso": cam_i, "wrist": cam_w})  # each render step (live)
    dash.push_stats(step=..., episode=..., current_reward=..., episode_reward=...,
                    cumulative_reward=..., best_reward=..., success_count=..., reward_terms={...})
    # on a successful episode, hand over the captured per-step frames for replay:
    dash.set_replay({"front": [img0, img1, ...], "iso": [...], "wrist": [...]})  # imgs = HxWx3 uint8

Standalone demo:  .venv/bin/python -P rebot_dashboard.py --demo --port 8890
"""
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import numpy as np
from PIL import Image

_VIEWS = ("front", "iso", "wrist")

_cam_frames = {}
_cam_lock = threading.Lock()
_replay = {tV: [] for tV in _VIEWS}
_replay_lock = threading.Lock()
_stats = {"target_cube": "cube", "reward_history": [], "success_count": 0}
_stats_lock = threading.Lock()
_MAX_HIST = 600


def _jpeg(img, q=70):
    tB = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(tB, format="JPEG", quality=q)
    return tB.getvalue()


def updateCameras(camMap):
    """cam_map: {name: isaaclab Camera}. Cache live rgb as JPEG."""
    for tName, tCam in camMap.items():
        try:
            tRgb = tCam.data.output.get("rgb")
            if tRgb is None:
                continue
            tImg = tRgb[0, :, :, :3].cpu().numpy()
            if tImg.dtype != np.uint8:
                tImg = (tImg * 255).clip(0, 255).astype(np.uint8)
            with _cam_lock:
                _cam_frames[tName] = _jpeg(tImg)
        except Exception:
            pass


def setCameraImage(name, npImg):
    with _cam_lock:
        _cam_frames[name] = _jpeg(np.asarray(npImg))


def setReplay(viewFrames):
    """Store the last successful episode's frames. view_frames: {view: [HxWx3 uint8, ...]}."""
    tEnc = {}
    for tV in _VIEWS:
        tFrames = viewFrames.get(tV, [])
        tEnc[tV] = [_jpeg(np.asarray(tF)) for tF in tFrames]
    with _replay_lock:
        for tV in _VIEWS:
            _replay[tV] = tEnc[tV]


def pushStats(**kw):
    with _stats_lock:
        _stats.update(kw)
        tH = _stats.setdefault("reward_history", [])
        tH.append({
            "step": kw.get("step", _stats.get("step", 0)),
            "current": kw.get("current_reward", 0.0),
            "cumulative": kw.get("cumulative_reward", 0.0),
        })
        if len(tH) > _MAX_HIST:
            del tH[: len(tH) - _MAX_HIST]


def _safeJson(obj):
    return json.dumps(obj, default=str).replace("NaN", "0").replace("Infinity", "9e9").replace("-Infinity", "0")


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _jpg(self, data):
        if data:
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(204)
            self.end_headers()

    def do_GET(self):
        tP = self.path
        if tP == "/api/stats":
            with _stats_lock:
                tBody = _safeJson(_stats).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(tBody)
        elif tP == "/api/replay":
            with _replay_lock:
                tN = min(len(_replay[tV]) for tV in _VIEWS) if all(_replay[tV] for tV in _VIEWS) else 0
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"n": tN}).encode())
        elif tP.startswith("/cam/"):
            tName = tP.split("/cam/")[-1].split("?")[0]
            with _cam_lock:
                tData = _cam_frames.get(tName)
            self._jpg(tData)
        elif tP.startswith("/replay/"):
            try:
                _, _, tView, tIdx = tP.split("?")[0].split("/")
                with _replay_lock:
                    tFrames = _replay.get(tView, [])
                    tData = tFrames[int(tIdx)] if tFrames else None
            except Exception:
                tData = None
            self._jpg(tData)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(_HTML.encode())


class _S(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def start(port=8890, targetCube="cube"):
    with _stats_lock:
        _stats["target_cube"] = targetCube
    tSrv = _S(("0.0.0.0", port), _H)
    threading.Thread(target=tSrv.serve_forever, daemon=True).start()
    print(f"[reBot-dash] http://0.0.0.0:{port}  (target={targetCube})", flush=True)
    return tSrv


_HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>reBot SAC</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,'Segoe UI',monospace;padding:16px}
h1{font-size:20px;margin-bottom:4px}#tc{font-size:13px;margin-bottom:14px}
.sec{font-size:12px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin:6px 2px 8px}
.cams{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px}
.cam{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:6px}
.cam img{width:100%;border-radius:4px;display:block;background:#000;aspect-ratio:16/9;object-fit:cover}
.cam .t{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin:4px 2px}
.grid{display:grid;grid-template-columns:repeat(7,1fr);gap:10px;margin-bottom:16px}
.box{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;text-align:center}
.label{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px}
.val{font-size:24px;font-weight:700;margin-top:4px;font-variant-numeric:tabular-nums}
.blue{color:#58a6ff}.green{color:#3fb950}.yellow{color:#d29922}.red{color:#f85149}.purple{color:#a371f7}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;margin-bottom:16px}
.card-title{font-size:12px;color:#8b949e;text-transform:uppercase;margin-bottom:10px;letter-spacing:.5px}
svg{width:100%;height:200px;display:block}
.terms{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px}
.term{display:flex;justify-content:space-between;font-size:12px;padding:4px 8px;background:#0d1117;border-radius:4px;margin-bottom:3px}
.rp{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.rp-h{font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;font-weight:700}
.rp-sub{margin-top:8px;font-size:13px;font-weight:700;text-align:right;border-top:1px solid #21262d;padding-top:6px}
.rep .t{color:#a371f7}#rep-status{font-size:11px;color:#8b949e;margin-top:6px}
#updated{font-size:11px;color:#484f58;text-align:right}
</style></head><body>
<h1>reBot SAC — LiftCube</h1><div id="tc" class="purple">target: -</div>

<div class="grid">
  <div class="box"><div class="label">Current Reward</div><div class="val blue" id="v-cur">-</div></div>
  <div class="box"><div class="label">Episode Reward</div><div class="val yellow" id="v-ep">-</div></div>
  <div class="box"><div class="label">Cumulative</div><div class="val green" id="v-cum">-</div></div>
  <div class="box"><div class="label">Best Episode</div><div class="val purple" id="v-best">-</div></div>
  <div class="box"><div class="label">Success</div><div class="val green" id="v-suc">0</div></div>
  <div class="box"><div class="label">Success Rate</div><div class="val green" id="v-srate">0%</div></div>
  <div class="box"><div class="label">Step / Episode</div><div class="val" id="v-step">- / -</div></div>
</div>

<div class="card"><div class="card-title">Reward over time — current (per step) &amp; cumulative</div>
  <svg id="chart" viewBox="0 0 800 200" preserveAspectRatio="none"></svg></div>

<div class="card rep"><div class="card-title">Live policy view — what the student sees (front + wrist)</div>
  <div class="cams">
    <div class="cam"><div class="t">front</div><img id="live-front" src=""></div>
    <div class="cam"><div class="t">wrist (gripper)</div><img id="live-wrist" src=""></div>
  </div>
  <div id="live-status" class="rp-sub">live camera feed</div>
</div>

<div class="card rep"><div class="card-title">Last successful episode — replay (3 views)</div>
  <div class="cams">
    <div class="cam"><div class="t">front</div><img id="rep-front" src=""></div>
    <div class="cam"><div class="t">iso</div><img id="rep-iso" src=""></div>
    <div class="cam"><div class="t">wrist (gripper)</div><img id="rep-wrist" src=""></div>
  </div>
  <div id="rep-status">no successful episode yet</div>
</div>

<div class="sec">Reward (+) / Penalty (−) breakdown</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px">
  <div class="card"><div class="card-title">current (per step)</div>
    <div class="rp">
      <div><div class="rp-h green">Rewards +</div><div id="cur-rew"></div><div class="rp-sub" id="cur-rew-sum"></div></div>
      <div><div class="rp-h red">Penalties −</div><div id="cur-pen"></div><div class="rp-sub" id="cur-pen-sum"></div></div>
    </div></div>
  <div class="card"><div class="card-title">cumulative</div>
    <div class="rp">
      <div><div class="rp-h green">Rewards +</div><div id="cum-rew"></div><div class="rp-sub" id="cum-rew-sum"></div></div>
      <div><div class="rp-h red">Penalties −</div><div id="cum-pen"></div><div class="rp-sub" id="cum-pen-sum"></div></div>
    </div></div>
</div>
<div class="sec">Dopamine (stage mastery → reward decay)</div>
<div class="card"><div id="dopamine" style="display:flex;flex-wrap:wrap;gap:8px"></div>
  <div class="rp-sub">×1.0 = not yet triggered · ×0.3 = mastered, damped by 70% (h = success-rate EMA)</div></div>

<div id="updated"></div>

<script>
function n(v){return(typeof v==='number'&&isFinite(v))?v:0}
function renderTerms(rewId,penId,rewSum,penSum,terms){
  terms=terms||{};var rew='',pen='',rs=0,ps=0;
  Object.keys(terms).forEach(function(k){var v=typeof terms[k]==='object'?n(terms[k].value):n(terms[k]);
    var row='<div class="term"><span>'+k+'</span><span style="font-weight:600">'+v.toFixed(3)+'</span></div>';
    if(v<0){pen+=row;ps+=v;}else{rew+=row;rs+=v;}});
  document.getElementById(rewId).innerHTML=rew||'<span style="color:#484f58;font-size:12px">-</span>';
  document.getElementById(penId).innerHTML=pen||'<span style="color:#484f58;font-size:12px">-</span>';
  document.getElementById(rewSum).innerHTML='Σ <span class="green">'+rs.toFixed(3)+'</span>';
  document.getElementById(penSum).innerHTML='Σ <span class="red">'+ps.toFixed(3)+'</span>';
}
function renderDopamine(dop){
  dop=dop||{};var h='';
  Object.keys(dop).forEach(function(k){var d=dop[k]||{};var m=n(d.mult),hr=n(d.h);
    var col=m<0.5?'#3fb950':(m<0.9?'#d29922':'#8b949e');
    h+='<div class="term" style="min-width:140px"><span>'+k+'</span><span style="font-weight:600;color:'+col+'">×'+m.toFixed(2)+' <span style="color:#6e7681;font-weight:400">h='+hr.toFixed(2)+'</span></span></div>';});
  document.getElementById('dopamine').innerHTML=h||'<span style="color:#484f58;font-size:12px">- (no hits yet)</span>';
}
function drawChart(hist){
  var svg=document.getElementById('chart');
  if(!hist||hist.length<2){svg.innerHTML='<text x="400" y="100" text-anchor="middle" fill="#484f58" font-size="14">Waiting for data...</text>';return}
  var W=800,H=200,L=48,R=56,T=22,B=28;  // axis margins
  var cur=hist.map(function(h){return n(h.current)}),cum=hist.map(function(h){return n(h.cumulative)}),st=hist.map(function(h){return n(h.step)});
  function rng(a){var mn=Math.min.apply(null,a),mx=Math.max.apply(null,a);if(mx-mn<1e-6)mx=mn+1;return [mn,mx];}
  var cr=rng(cur),cm=rng(cum);
  function xS(i){return L+(W-L-R)*i/(hist.length-1)}
  function yC(v){return (H-B)-(H-T-B)*(v-cr[0])/(cr[1]-cr[0])}
  function yM(v){return (H-B)-(H-T-B)*(v-cm[0])/(cm[1]-cm[0])}
  var s='<line x1="'+L+'" y1="'+T+'" x2="'+L+'" y2="'+(H-B)+'" stroke="#484f58"/>'
       +'<line x1="'+(W-R)+'" y1="'+T+'" x2="'+(W-R)+'" y2="'+(H-B)+'" stroke="#484f58"/>'
       +'<line x1="'+L+'" y1="'+(H-B)+'" x2="'+(W-R)+'" y2="'+(H-B)+'" stroke="#484f58"/>';
  for(var k=0;k<=4;k++){var gy=T+(H-T-B)*k/4;
    s+='<line x1="'+L+'" y1="'+gy+'" x2="'+(W-R)+'" y2="'+gy+'" stroke="#21262d"/>';
    s+='<text x="'+(L-5)+'" y="'+(gy+3)+'" fill="#58a6ff" font-size="9" text-anchor="end">'+(cr[1]-(cr[1]-cr[0])*k/4).toFixed(2)+'</text>';
    s+='<text x="'+(W-R+5)+'" y="'+(gy+3)+'" fill="#3fb950" font-size="9">'+(cm[1]-(cm[1]-cm[0])*k/4).toFixed(0)+'</text>';}
  for(var t=0;t<=4;t++){var xi=Math.round((hist.length-1)*t/4);
    s+='<text x="'+xS(xi)+'" y="'+(H-10)+'" fill="#8b949e" font-size="9" text-anchor="middle">'+n(st[xi]).toLocaleString()+'</text>';}
  function poly(arr,yf,col,w){return '<polyline points="'+arr.map(function(v,i){return xS(i)+','+yf(v)}).join(' ')+'" fill="none" stroke="'+col+'" stroke-width="'+w+'"/>';}
  s+=poly(cur,yC,'#58a6ff',1.5)+poly(cum,yM,'#3fb950',2.5);
  // axis titles
  s+='<text x="'+((L+W-R)/2)+'" y="'+(H-0.5)+'" fill="#c9d1d9" font-size="10" text-anchor="middle">step (x-axis)</text>';
  s+='<text x="'+(L-5)+'" y="13" fill="#58a6ff" font-size="10" text-anchor="end">reward (y) ↑</text>';
  s+='<text x="'+(W-R+5)+'" y="13" fill="#3fb950" font-size="10">cumul. (y)</text>';
  s+='<text x="'+((L+W-R)/2)+'" y="13" fill="#8b949e" font-size="10" text-anchor="middle">current=blue · cumulative=green</text>';
  svg.innerHTML=s;
}
function update(){
  fetch('/api/stats').then(function(r){return r.json()}).then(function(s){
    document.getElementById('tc').textContent='target: '+(s.target_cube||'-');
    document.getElementById('v-cur').textContent=n(s.current_reward).toFixed(3);
    document.getElementById('v-ep').textContent=n(s.episode_reward).toFixed(2);
    document.getElementById('v-cum').textContent=n(s.cumulative_reward).toLocaleString(undefined,{maximumFractionDigits:1});
    document.getElementById('v-best').textContent=n(s.best_reward).toFixed(2);
    document.getElementById('v-suc').textContent=n(s.success_count).toLocaleString();
    document.getElementById('v-srate').textContent=(n(s.success_rate)*100).toFixed(1)+'%';
    document.getElementById('v-step').textContent=n(s.step).toLocaleString()+' / '+n(s.episode).toLocaleString();
    drawChart(s.reward_history||[]);
    renderTerms('cur-rew','cur-pen','cur-rew-sum','cur-pen-sum',s.reward_terms);
    renderTerms('cum-rew','cum-pen','cum-rew-sum','cum-pen-sum',s.cumulative_terms);
    renderDopamine(s.dopamine);
    document.getElementById('updated').textContent='Updated: '+new Date().toLocaleTimeString();
  }).catch(function(e){document.getElementById('updated').textContent='Error: '+e;});
}
// live camera streaming: front + wrist (the policy's actual inputs), cache-busted each poll
function pollLiveCams(){var t=Date.now();
  ['front','wrist'].forEach(function(nm){var img=document.getElementById('live-'+nm);if(img)img.src='/cam/'+nm+'?t='+t;});}
// replay player: cycle frames of the last successful episode across the 3 views
var _repN=0,_repI=0;
function pollReplay(){fetch('/api/replay').then(function(r){return r.json()}).then(function(d){_repN=n(d.n);
  document.getElementById('rep-status').textContent=_repN>0?('replaying last success — '+_repN+' frames'):'no successful episode yet';});}
function stepReplay(){if(_repN<=0)return;_repI=(_repI+1)%_repN;
  ['front','iso','wrist'].forEach(function(nm){var img=document.getElementById('rep-'+nm);if(img)img.src='/replay/'+nm+'/'+_repI;});}
setInterval(update,1000);setInterval(pollReplay,3000);setInterval(stepReplay,120);setInterval(pollLiveCams,400);
update();pollReplay();pollLiveCams();
</script></body></html>"""


if __name__ == "__main__":
    import argparse
    import math

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8890)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    start(a.port, targetCube="cube (red)")
    if a.demo:
        imgs = {}
        for nm, path in [("front", "/tmp/scene_front.png"), ("iso", "/tmp/scene_iso.png"), ("wrist", "/tmp/scene_wrist.png")]:
            try:
                im = np.array(Image.open(path).convert("RGB"))
                setCameraImage(nm, im)
                imgs[nm] = im
            except Exception as e:
                print(f"[demo] no {path}: {e}", flush=True)
        # fake "last success replay" = the rendered views repeated as a short clip
        if imgs:
            setReplay({tV: [imgs.get(tV, np.zeros((90, 160, 3), np.uint8))] * 12 for tV in _VIEWS})
        step, ep, cum, best, ep_r, suc = 0, 0, 0.0, 0.0, 0.0, 0
        cum_terms = {}
        while True:
            step += 1
            cur = max(0.0, math.sin(step / 20.0) * 0.5 + 0.5 + step / 5000.0)
            cum += cur; ep_r += cur
            if step % 100 == 0:
                ep += 1
                if ep_r > 8.0:
                    suc += 1
                best = max(best, ep_r); ep_r = 0.0
            cur_terms = {"reaching": cur * 0.4, "approach_open": cur * 0.1, "grasped": cur * 0.3,
                         "lift_hold": cur * 0.2, "table_penalty": -0.05, "not_lifted": -0.03}
            for k, v in cur_terms.items():
                cum_terms[k] = cum_terms.get(k, 0.0) + v
            pushStats(step=step, episode=ep, current_reward=cur, episode_reward=ep_r,
                       cumulative_reward=cum, best_reward=best, success_count=suc,
                       success_rate=suc / max(ep, 1),
                       reward_terms=cur_terms, cumulative_terms=dict(cum_terms))
            time.sleep(0.1)
