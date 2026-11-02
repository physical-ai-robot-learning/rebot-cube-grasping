"""Shared deploy debug dashboard (stdlib HTTP only — no extra deps).

A tiny threaded web server for REAL-ROBOT phase-by-phase debugging:
  - the deploy loop calls ``dash.update(**snapshot)`` each step with proprio / FK TCP / vision /
    gate internals; the browser polls ``/state`` (JSON) and renders them live.
  - the browser's buttons GET ``/cmd?a=<action>``; the deploy loop pops them via ``dash.poll_cmd()``
    and acts (capture / run / follow / align / align_flip / final / close / open / lift / grip_tight / grip_loose /
    stop / reset / zero / jog / abort).

This lets you SEE where a phase breaks (reach error too large? cube mis-detected? on_cube never
true? tilt too high?) and give the robot feedback without editing code. Used by debug_reach.py
and debug_grasp.py. NEVER let the dashboard move the arm on its own — only on an explicit button.
"""
from __future__ import annotations

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs


def _json_default(o):
    """numpy-safe fallback so /state NEVER 500s on a stray numpy value (np scalar / 0-d or N-d ndarray)
    that leaked into a telemetry field. A dead dashboard (blank '?') is worse than a coerced number."""
    tolist = getattr(o, "tolist", None)           # np.ndarray -> list, np.generic (float64/…) -> python scalar
    if callable(tolist):
        return tolist()
    return str(o)

_HTML = """<!doctype html><meta charset=utf-8><title>reBot deploy debug</title>
<style>
 body{font:13px system-ui;margin:14px;background:#0f1115;color:#e5e7eb}
 h1{font-size:15px} table{border-collapse:collapse;margin:6px 0} td,th{border:1px solid #333;padding:3px 8px;text-align:right}
 .ok{color:#4ade80} .bad{color:#f87171} .k{color:#9aa4b2;text-align:left}
 button{font:13px system-ui;margin:3px;padding:6px 10px;background:#1f2530;color:#e5e7eb;border:1px solid #3a4350;border-radius:6px;cursor:pointer}
 button.danger{background:#7f1d1d;border-color:#b91c1c} .phase{font-size:16px;font-weight:700;color:#60a5fa}
</style>
<h1><!--TITLE--> — <span id=phase class=phase>?</span> <span id=msg></span></h1>
<!--BUTTONS-->
<div style="margin:6px 0">OpenVocab query:
 <input id=ovq placeholder="red cube, red block" onkeydown="if(event.key==='Enter')ovset()"
        style="font:13px system-ui;padding:5px 8px;width:240px;background:#1f2530;color:#e5e7eb;border:1px solid #3a4350;border-radius:6px">
 <button onclick=ovset()>Set query</button>
 <span id=ovnow style="color:#9aa4b2;margin-left:8px"></span></div>
<div style="display:flex;gap:10px;flex-wrap:wrap;margin:8px 0">
 <div><div class=k><!--CAP_FRONT--></div>
  <img id=imgF width=512 style="border:1px solid #333;background:#000"
       onload="this.style.opacity=1" onerror="this.style.opacity=.25"></div>
 <div><div class=k><!--CAP_WRIST--></div>
  <img id=imgW width=512 style="border:1px solid #333;background:#000"
       onload="this.style.opacity=1" onerror="this.style.opacity=.25"></div>
</div>
<table id=t></table>
<details style="margin-top:6px"><summary style="cursor:pointer;color:#9aa4b2;font-size:12px">&#9656; details (raw fields)</summary>
 <table id=d></table></details>
<script>
async function cmd(a){await fetch('/cmd?a='+encodeURIComponent(a))}
async function ovset(){let v=document.getElementById('ovq').value; await fetch('/cmd?a=ovquery&q='+encodeURIComponent(v));}
function row(k,v,cls){return '<tr><td class=k>'+k+'</td><td class="'+(cls||'')+'">'+v+'</td></tr>'}
function refreshImgs(){let t=Date.now();
 document.getElementById('imgF').src='/frame?n=front&t='+t;
 document.getElementById('imgW').src='/frame?n=wrist&t='+t;}
const num=v=>(typeof v==='number')?(Math.abs(v)<100?v.toFixed(3):v.toFixed(1)):v;
async function tick(){
 let s=await (await fetch('/state')).json();
 document.getElementById('phase').textContent=s.phase||'?';
 document.getElementById('msg').textContent=s.msg||'';
 let g=s.gates||{};
 let ovn=document.getElementById('ovnow');
 if(ovn){  // the ONE query drives BOTH cams -> show each cam's own live score so you can confirm the wrist got it
   let f = g.ov_q ? ('front:['+g.ov_q+'] s='+(g.ov_score==null?'—':g.ov_score)) : '';
   let w = g.ov_q_wrist ? ('  wrist:['+g.ov_q_wrist+'] s='+(g.ov_score_wrist==null?'—':g.ov_score_wrist)) : '';
   ovn.textContent = f + w;}
 // ---- ESSENTIALS (curated: what you actually watch during a grasp) ----
 let e='';
 if(s.reached!=null) e+=row('reached', s.reached, s.reached?'ok':'bad');
 if(g.z_gap_io!=null) e+=row('z_gap_io (height)', num(g.z_gap_io), Math.abs(g.z_gap_io-(g.grasp_zgap_io!=null?g.grasp_zgap_io:0))<0.015?'ok':'');
 if(s.xy_err!=null) e+=row('xy err', num(s.xy_err), s.xy_err<0.03?'ok':'bad');
 // ---- TCP position via FK (x,y,z): base_ctrl kinematic TCP + robot_io grasp point (~40mm apart) ----
 if(s.tcp) e+=row('TCP FK (x,y,z)', '('+s.tcp.map(x=>x.toFixed(3)).join(', ')+')');
 if(s.tcp_io) e+=row('TCP grasp io (x,y,z)', '('+s.tcp_io.map(x=>x.toFixed(3)).join(', ')+')');
 // ---- DUAL-CAM cube XY: FRONT (OWLv2 ray-plane) vs WRIST (optical-axis unproject) + their gap ----
 if(s.cube_top) e+=row('cube FRONT (x,y)', s.cube_top.map(x=>x.toFixed(3)));
 e+=row('cube WRIST (x,y)', g.wrist_xy ? g.wrist_xy.map(x=>x.toFixed(3)) : '— (no wrist detect)', g.wrist_xy?'ok':'bad');
 if(g.wf_dxy!=null) e+=row('FRONT↔WRIST Δ', (g.wf_dxy*1000).toFixed(0)+' mm', g.wf_dxy<0.03?'ok':'bad');
 let yv=[]; ['latch_yaw_deg','seg_yaw_deg'].forEach(k=>{if(g[k]!=null)yv.push(k[0]+':'+g[k])});
 if(yv.length) e+=row('yaw', yv.join('&nbsp;&nbsp;'));
 if(g.grasped!=null) e+=row('grasped', g.grasped, g.grasped?'ok':'bad');
 if(g.grip_tau!=null) e+=row('grip tau', num(g.grip_tau));
 document.getElementById('t').innerHTML=e;
 // ---- DETAILS (everything else, collapsed) ----
 const shown=new Set(['z_gap_io','grasp_zgap_io','latch_yaw_deg','seg_yaw_deg','grasped','grip_tau','wrist_xy','wf_dxy']);
 let d=row('step',s.step);
 if(s.wrist_uv) d+=row('wrist (u,v,vis)', s.wrist_uv.map(x=>x.toFixed(2)));
 if(s.grasp_xy) d+=row('grasp xy (jaw-fwd)', s.grasp_xy.map(x=>x.toFixed(3)));
 if(s.z_gap!=null) d+=row('z gap (base_ctrl)', num(s.z_gap));
 if(s.at_cube!=null) d+=row('at cube (base)', s.at_cube, s.at_cube?'ok':'bad');
 if(s.target) d+=row('reach target', s.target.map(x=>x.toFixed(3)));
 if(s.err_xy!=null) d+=row('reach err xy', num(s.err_xy), s.err_xy<0.03?'ok':'bad');
 if(s.err_z!=null) d+=row('reach err z', num(s.err_z));
 for(const k in g){if(shown.has(k))continue; let v=g[k]; d+=row(k, num(v), v===true?'ok':v===false?'bad':'')}
 if(s.joints){d+=row('q (deg)', s.joints.q.map(x=>(x*57.3).toFixed(1)));
   if(s.joints.tau) d+=row('torque', s.joints.tau.map(x=>x.toFixed(2)));}
 if(s.grip_q!=null) d+=row('grip qpos', s.grip_q.toFixed(4));
 document.getElementById('d').innerHTML=d;
}
setInterval(tick,200); tick();
setInterval(refreshImgs,150); refreshImgs();
</script>"""

# Default (RULE deep_reach tools: debug_deepreach / deep_reach_world) button set + captions. A tool can pass
# its OWN via DebugDashboard(buttons=..., title=..., front_caption=..., wrist_caption=...) to drop the buttons
# it doesn't use (e.g. the SAC-policy tool has only RUN/Stop/Reset/Zero/open/Abort). Defaults keep the RULE
# tools' dashboard byte-identical.
DEFAULT_TITLE = "reBot deploy debug"
DEFAULT_CAP_FRONT = "FRONT (fixed) — cube(orange) / gripper(green) / FK-box"
DEFAULT_CAP_WRIST = "WRIST (eye-in-hand) — (u,v) circle · seg MASK (green fill) + box + yaw (magenta) · center cross = target"
DEFAULT_BUTTONS = """<div>GRASP FLOW:
 <button onclick=cmd('capture')>Capture cube</button>
 <button onclick=cmd('run')>RUN &#9654;</button>
 <button onclick=cmd('follow')>Follow (hover-track)</button>
 <button style="font-weight:bold" onclick="if(confirm('AUTO-GRASP: from FOLLOW (cube must be centered), keep auto-centering while descending to grasp depth, then close. Cube framed + clear to grasp? Stop/ABORT halts.'))cmd('autograsp')">AUTO-GRASP &#9654;&#9654;</button>
 <button onclick=cmd('approach')>Approach (track &#8595;)</button>
 <button onclick=cmd('align')>Align (center+J6)</button>
 <button onclick=cmd('align_flip')>Align flip</button>
 <button onclick=cmd('final')>Final &#8595;</button>
 <button onclick=cmd('close')>Grip close</button>
 <button onclick=cmd('open')>Grip open</button>
 <button onclick=cmd('lift')>Lift +z</button>
</div>
<div>grip force:
 <button onclick=cmd('grip_tight')>grip&minus;(tight)</button><button onclick=cmd('grip_loose')>grip+(loose)</button>
 &nbsp;&nbsp; home/halt:
 <button onclick=cmd('stop')>STOP &#9632;</button>
 <button onclick=cmd('reset')>&#8617; Rest</button>
 <button onclick=cmd('zero')>&#9678; Zero</button>
 <button class=danger onclick=cmd('abort')>ABORT (estop)</button>
</div>
<div>jog (recovery):
 <button onclick=cmd('jog?j=0&d=1')>J1+</button><button onclick=cmd('jog?j=0&d=-1')>J1-</button>
 <button onclick=cmd('jog?j=3&d=1')>J4+</button><button onclick=cmd('jog?j=3&d=-1')>J4-</button>
 <button onclick=cmd('jog?j=4&d=1')>J5+</button><button onclick=cmd('jog?j=4&d=-1')>J5-</button>
 <button onclick=cmd('jog?j=5&d=1')>J6+</button><button onclick=cmd('jog?j=5&d=-1')>J6-</button>
</div>
<div>Follow cam-center (tune cube@wrist target):
 <button onclick="cmd('followcam?ax=fwd&d=1')">fwd+</button><button onclick="cmd('followcam?ax=fwd&d=-1')">fwd-</button>
 <button onclick="cmd('followcam?ax=lat&d=1')">lat+</button><button onclick="cmd('followcam?ax=lat&d=-1')">lat-</button>
 &nbsp; <button onclick="cmd('followlookat')">Look-at &#8646; (gimbal fixate)</button>
 &nbsp;<span style="color:#9aa4b2">watch FOLLOW msg cube@wrist(u,v)</span>
</div>
<div>AUTO-GRASP depth (goal z_gap_io; lower=deeper, retargets auto-z live):
 <button onclick="cmd('graspz?d=1')">grasp deeper &#8595;</button><button onclick="cmd('graspz?d=-1')">grasp higher &#8593;</button>
 &nbsp;<span style="color:#9aa4b2">watch gates.grasp_zgap_io / z_gap_io</span>
</div>
<div>FK-Z verify (reach bias): &nbsp; position the arm at a hover over a spot, then &rarr;
 <button onclick="if(confirm('FK-TOUCH: the arm descends STRAIGHT DOWN until the J2/J3 torque rises (table contact), records z_gap_io, then lifts. Clear space below + WATCH it (Stop halts). Repeat at 3 distances.'))cmd('fktouch')">FK-touch (this spot) &#8595;</button>
 &nbsp;<span style="color:#9aa4b2">repeat NEAR/MID/FAR &rarr; the msg prints the --z-reach-slope/ref to relaunch with</span>
</div>
<div>wrist calib (A): &nbsp; Capture cube @ safe hover, then &rarr;
 <button onclick="if(confirm('CALIB sweep: arm moves to a tilted hover + swings J1/J4 for ~9s. Cube Captured + clear space? WATCH it (Stop/Abort halts between poses).'))cmd('calib_wrist')">CALIB sweep (this cube pos)</button>
 &nbsp;<span style="color:#9aa4b2">repeat &ge;6 DISTINCT cube positions (move cube &rarr; Capture &rarr; CALIB)</span>
</div>"""


class DebugDashboard:
    def __init__(self, port: int = 8899, buttons=None, title=None,
                 front_caption=None, wrist_caption=None):
        self.port = port
        # bake the (optionally tool-tailored) HTML once — placeholder replace (NOT .format(): the CSS has
        # literal {} braces). Defaults reproduce the RULE tools' dashboard exactly.
        self._html = (_HTML
                      .replace("<!--TITLE-->", title or DEFAULT_TITLE)
                      .replace("<!--BUTTONS-->", DEFAULT_BUTTONS if buttons is None else buttons)
                      .replace("<!--CAP_FRONT-->", front_caption or DEFAULT_CAP_FRONT)
                      .replace("<!--CAP_WRIST-->", wrist_caption or DEFAULT_CAP_WRIST))
        self._state = {"phase": "init", "step": 0, "msg": ""}
        self._frames = {}        # name -> latest JPEG bytes (front/wrist), polled by the <img> tags
        self._cmds = deque()
        self._lock = threading.Lock()
        dash = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_GET(self):
                u = urlparse(self.path)
                if u.path == "/":
                    self._send(200, "text/html", dash._html.encode())
                elif u.path == "/state":
                    with dash._lock:
                        body = json.dumps(dash._state, default=_json_default).encode()
                    self._send(200, "application/json", body)
                elif u.path == "/frame":
                    n = parse_qs(u.query).get("n", [""])[0]      # ?n=front|wrist (&t=<cache-bust>)
                    with dash._lock:
                        buf = dash._frames.get(n)
                    if buf is None:
                        self._send(404, "text/plain", b"no frame")
                    else:
                        self._send(200, "image/jpeg", buf)
                elif u.path == "/cmd":
                    q = parse_qs(u.query)
                    action = q.get("a", [""])[0]
                    # params come inline as a='<action>?k=v&...' (kept simple); split them back out generically
                    extra = {}
                    if "?" in action:
                        action, _, qs = action.partition("?")
                        extra = {k: v[0] for k, v in parse_qs(qs).items()}   # string values (caller casts)
                        if action == "jog":                                  # jog keeps its int convention
                            extra = {"j": int(extra.get("j", 0)), "d": int(extra.get("d", 1))}
                    if "q" in q:                                  # free-text param (e.g. open-vocab query) -> extra['q']
                        extra["q"] = q.get("q", [""])[0]
                    with dash._lock:
                        dash._cmds.append((action, extra))
                    self._send(200, "text/plain", b"ok")
                else:
                    self._send(404, "text/plain", b"nf")

            def _send(self, code, ctype, body):
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass    # client (poller / browser <img>) hung up mid-response -> ignore, don't spam a traceback

            def log_message(self, *a):
                pass        # silence per-request stderr logging (the dashboard is polled many times/sec)

        self._srv = ThreadingHTTPServer(("0.0.0.0", port), H)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)

    def start(self):
        self._thread.start()
        print(f"[debug-dash] http://localhost:{self.port}  (ssh -L {self.port}:localhost:{self.port} for remote)", flush=True)

    def update(self, **kw):
        with self._lock:
            self._state.update(kw)

    def update_frame(self, name, jpeg_bytes):
        """Push the latest annotated camera JPEG (name='front'|'wrist'); the browser <img> polls /frame?n=."""
        if jpeg_bytes is None:
            return
        with self._lock:
            self._frames[name] = jpeg_bytes

    def poll_cmd(self):
        """Return (action, extra_dict) or None. Non-blocking."""
        with self._lock:
            return self._cmds.popleft() if self._cmds else None

    def stop(self):
        self._srv.shutdown()
