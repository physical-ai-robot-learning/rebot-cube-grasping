#!/usr/bin/env python3
# Auto-bench daemon: watch dacq3's approach ckpt dir; on a new step ckpt, resubmit the
# graphic-node bench with that approach + the latest 8951 search, then post the EVAL result to ntfy.
# Runs on <training-node>, submits bench to node:<<workstation>-node> (graphic/5080). NEVER touches another user (5090).
import os, re, time, glob, shutil, subprocess, urllib.request, json
from ray.job_submission import JobSubmissionClient

RAY = "http://<ray-head>:8265"
APPROACH_DIR = "~/workspace/rebot_isaac_rl/sim2real_so101/delivery/rl_output/det_TP304_refac_smoke9t"
SEARCH_LAST  = "~/workspace/rebot_isaac_rl/sim2real_so101/delivery/rl_output/det_TP304_refac_srch2/search_last.pt"
STAGE_BENCH  = "~/_stage_bench"         # new-net code shipped as working_dir
CKPT_IN      = STAGE_BENCH + "/ckpt_in"               # approach_last.pt / search_last.pt injected into bench
EP_FILE      = "~/_bench_graphic_ep.txt"
BENCH_NODE   = "<<workstation>-node>"                           # graphic/5080 — its own venv/isaac/evalgrid
BENCH_SID    = "bench_graphic_auto"
NTFY         = "https://ntfy.sh/<ntfy-topic>"
POLL_S       = 45
c = JobSubmissionClient(RAY)

def newest_approach_step():
    # file-name step RESETS on fresh restart (g=0), so pick the MOST-RECENTLY-WRITTEN ckpt by mtime,
    # not the largest number. Returns (mtime_int, step_k, path) or None.
    fs = glob.glob(APPROACH_DIR + "/approach_det_TP304_refac_smoke9t_*k.pt")
    best = None
    for f in fs:
        m = re.search(r"_(\d+)k\.pt$", f)
        if not m: continue
        try: mt = int(os.path.getmtime(f))
        except OSError: continue
        k = int(m.group(1))
        if best is None or mt > best[0]: best = (mt, k, f)
    return best

def _hsafe(s):
    return s.replace("\n", " | ").encode("ascii", "replace").decode("ascii")

def _put_with_retry(req, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                return r.status
        except urllib.error.HTTPError as e:
            if e.code == 429:   # ntfy.sh rate limit -> back off and retry
                wait = 30 * (i + 1)
                print(f"[ntfy] 429, retry in {wait}s ({i+1}/{tries})", flush=True); time.sleep(wait); continue
            print("[ntfy] http err", e); return None
        except Exception as e:
            print("[ntfy] err", e); time.sleep(10)
    return None

def _grab_heat_tables(html, anchor):
    """extract the <table> of colored cells following an eval-heat anchor -> rows[][]=color.
    robust: big window (a 94-cell table is ~9.2k chars), tolerate <tr attrs>, and a missing
    </table> (grab up to the next <table> or end)."""
    i = html.find(anchor)
    if i < 0: return None
    seg = html[i:i+30000]
    j = seg.find("<table")
    if j < 0: return None
    body = seg[j:]
    end = body.find("</table>")
    if end >= 0:
        tbl = body[:end]
    else:
        nxt = body.find("<table", 6)   # no close tag -> stop at next table
        tbl = body[:nxt] if nxt > 0 else body
    rows = []
    for tr in re.split(r"<tr[^>]*>", tbl):
        # colored cells carry background:#rgb or rgb(...); the row-label cell uses color: (ignored)
        cells = re.findall(r"background:\s*(#[0-9A-Fa-f]{3,6}|rgb[^;'\"]*)", tr)
        if cells: rows.append([c.strip() for c in cells])
    return rows or None

def post_ntfy(title, msg):
    # try to attach the bench eval heatmap image; fall back to text-only.
    png = None
    try:
        from PIL import Image, ImageDraw, ImageFont
        html = urllib.request.urlopen("http://127.0.0.1:8967", timeout=12).read().decode("utf-8", "replace") if False else None
        # bench serves on the graphic node; fetch via its dashboard IP
        html = urllib.request.urlopen("http://<<workstation>-node>:8967", timeout=12).read().decode("utf-8", "replace")
        panels = []
        for anc, lbl in [("RSR heatmap", "RSR"), ("rsr_old heatmap", "rsr_old"), ("SEARCH heatmap", "SEARCH")]:
            rows = _grab_heat_tables(html, anc)
            if rows: panels.append((lbl, rows))
        if panels:
            def font(sz):
                for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",):
                    try: return ImageFont.truetype(p, sz)
                    except Exception: pass
                return ImageFont.load_default()
            F = font(12)
            cell = 20; pad = 2; gap = 22
            def draw(lbl, rows):
                ny = len(rows); nx = max(len(r) for r in rows)
                im = Image.new("RGB", (nx*(cell+pad)+pad, ny*(cell+pad)+pad+20), "#111"); d = ImageDraw.Draw(im)
                d.text((3, 3), lbl, fill="#ddd", font=F)
                for ry, row in enumerate(rows):
                    for rx, col in enumerate(row):
                        x = pad+rx*(cell+pad); y = 20+pad+ry*(cell+pad)
                        fill = col if (col.startswith("#") or col.startswith("rgb")) else "#222"
                        d.rectangle([x, y, x+cell, y+cell], fill=fill)
                return im
            ims = [draw(lbl, rows) for lbl, rows in panels]
            W = sum(im.width for im in ims)+gap*(len(ims)+1); H = max(im.height for im in ims)+26
            canvas = Image.new("RGB", (W, H), "#111"); dd = ImageDraw.Draw(canvas)
            dd.text((6, 6), title, fill="#8fd", font=F)
            x = gap
            for im in ims: canvas.paste(im, (x, 22)); x += im.width+gap
            import io as _io; buf = _io.BytesIO(); canvas.save(buf, "PNG"); png = buf.getvalue()
    except Exception as e:
        print("[ntfy] heat render err", e, flush=True)
    # user 2026-08-02: NO ntfy (rate-limited). Save the heatmap PNG locally with the step in the name.
    outdir = "~/bench_heatmaps"
    try: os.makedirs(outdir, exist_ok=True)
    except Exception: pass
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", title)   # e.g. "bench e1203k" -> "bench_e1203k"
    if png:
        try:
            open(outdir + "/" + safe + ".png", "wb").write(png)
            print("[bench-daemon] saved heatmap", outdir + "/" + safe + ".png", "|", msg, flush=True)
        except Exception as e:
            print("[bench-daemon] png save err", e, flush=True)
    else:
        print("[bench-daemon] (no heatmap rendered)", title, "|", msg, flush=True)

def stop_bench():
    try:
        st = str(c.get_job_status(BENCH_SID))
        if "RUNNING" in st or "PENDING" in st:
            c.stop_job(BENCH_SID)
            for _ in range(30):
                if str(c.get_job_status(BENCH_SID)) in ("STOPPED","FAILED","SUCCEEDED"): break
                time.sleep(2)
    except Exception:
        pass  # not submitted yet

def submit_bench(step_k, ap):
    # inject latest ckpts into the shipped working_dir (ckpt_in), bench copies them into the node's ckpt_bench_mix
    shutil.copy(ap, CKPT_IN + "/approach_last.pt")
    shutil.copy(SEARCH_LAST, CKPT_IN + "/search_last.pt")
    ep = open(EP_FILE, encoding="utf-8").read().strip()
    inject = ("mkdir -p ~/rebot_eval/ckpt_bench_mix && "
              "cp ckpt_in/approach_last.pt ckpt_in/search_last.pt ~/rebot_eval/ckpt_bench_mix/ && "
              "echo '[BENCH] injected fresh ckpts' && ")
    ep2 = 'bash -lc "' + inject + ep[len('bash -lc "'):]
    # unique submission id per step so logs don't collide; but keep BENCH_SID stopped first
    _sid = BENCH_SID + "_e%dk_%d" % (step_k, int(time.time()))
    c.submit_job(entrypoint=ep2, submission_id=_sid,
                 runtime_env={"working_dir": STAGE_BENCH},
                 entrypoint_num_gpus=1, entrypoint_resources={"node:" + BENCH_NODE: 0.001})
    return _sid

def read_eval_result(sid, wait_s=240):
    # wait for at least one full sweep, then parse the newest [EVAL-SET] line
    t0 = time.time(); last = None
    while time.time() - t0 < wait_s:
        try:
            log = c.get_job_logs(sid)
            evals = re.findall(r"\[EVAL-SET\] RSR=([\d.]+)% rsr_old=([\d.]+)% SEARCH=([\d.]+)% \| covered (\d+)/(\d+)", log)
            if evals:
                last = evals[-1]
                cov, M = int(last[3]), int(last[4])
                if cov >= M:  # one full pass done -> stable numbers
                    return last, True
        except Exception:
            pass
        time.sleep(10)
    return last, False

def _auto_bench_jobs(statuses):
    out=[]
    for j in c.list_jobs():
        sid=j.submission_id
        if sid and sid.startswith(BENCH_SID) and str(j.status) in statuses:
            out.append(sid)
    return out

print(f"[bench-daemon] start; watching {APPROACH_DIR}; bench node {BENCH_NODE}; every {POLL_S}s", flush=True)
seen_mtime = 0
_RUNCTR = [0]
# stop any stale auto-bench variants at startup
try:
    for sid in _auto_bench_jobs(("RUNNING","PENDING")):
        c.stop_job(sid); print("[bench-daemon] stopped stale", sid, flush=True)
except Exception as e: print("[bench-daemon] startup cleanup err", e, flush=True)

while True:
    try:
        nb = newest_approach_step()
        if nb and nb[0] > seen_mtime:
            mt, step_k, ap_path = nb
            print(f"[bench-daemon] new approach ckpt @ {step_k}k (mtime {mt}) -> bench", flush=True)
            # stop any running auto-bench, submit fresh
            for sid in _auto_bench_jobs(("RUNNING","PENDING")):
                try: c.stop_job(sid)
                except Exception: pass
            time.sleep(6)
            sid = submit_bench(step_k, ap_path)
            seen_mtime = mt
            res, full = read_eval_result(sid)
            if res:
                rsr, rold, srch, cov, M = res
                title = f"bench e{step_k}k"
                msg = f"RSR {rsr}% | RSR_DCUBE {rold}% | SEARCH {srch}% | {cov}/{M}" + ("" if full else " (partial)")
                post_ntfy(title, msg)
                print(f"[bench-daemon] {title}: {msg}", flush=True)
                # user 2026-08-02: SEARCH hit 100% on a FULL sweep -> snapshot the EXACT benched search ckpt
                # (CKPT_IN/search_last.pt = what this bench used) to a node folder; the Mac pulls it to the Desktop.
                try:
                    if full and float(srch) >= 100.0:
                        outdir = "~/bench_search100"
                        os.makedirs(outdir, exist_ok=True)
                        dst = outdir + "/search_bench100_e%dk_%d.pt" % (step_k, int(time.time()))
                        shutil.copy(CKPT_IN + "/search_last.pt", dst)
                        print("[bench-daemon] *** SEARCH 100% -> saved search ckpt:", dst, flush=True)
                except Exception as e:
                    print("[bench-daemon] search100 save err", e, flush=True)
            else:
                print(f"[bench-daemon] e{step_k}k: no eval result parsed", flush=True)
    except Exception as e:
        print("[bench-daemon] loop err", e, flush=True)
    time.sleep(POLL_S)
