#!/usr/bin/env python3
# reBot sweep -> ntfy notifier (RSR + SEARCH heatmaps combined into one image).
# Polls the training dashboards, detects a new curriculum sweep (step increases),
# re-renders both heatmaps from the dashboard HTML and PUTs one image to the ntfy topic.
# Non-invasive: reads the dashboard only; does NOT touch the training jobs.
import sys, re, time, io, urllib.request
from PIL import Image, ImageDraw, ImageFont

TOPIC = "<ntfy-topic>"
NTFY  = "https://ntfy.sh/" + TOPIC
PORTS = {"8951": "http://127.0.0.1:8951", "8953": "http://127.0.0.1:8953"}
POLL_S = 60

def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()

FONT_CELL = _font(11)
FONT_HDR  = _font(13)

def _cell_val(color):
    """recover success fraction 0..1 from a heatmap cell color, or None if blank/excluded (#222)."""
    m = re.match(r'rgb\(\s*\d+\s*,\s*(\d+)\s*,', color or "")
    return (int(m.group(1)) / 200.0) if m else None

def _fetch(url):
    with urllib.request.urlopen(url, timeout=15) as r:
        return r.read().decode("utf-8", "replace")

def _grab_table_after(html, anchor):
    """Return (step_k, rate_pct, grid[rows][cols]=color_str) for the heatmap table following `anchor`, or None."""
    i = html.find(anchor)
    if i < 0:
        return None
    seg = html[i:i+8000]
    m = re.search(r'@(\d+)k[^%]*?([\d.]+)%', seg)      # label "RSR @Nk · R%"
    step_k = int(m.group(1)) if m else None
    rate = float(m.group(2)) if m else None
    tm = re.search(r'<table[^>]*>(.*?)</table>', seg, re.S)
    if not tm:
        return None
    rows = []
    for tr in re.findall(r'<tr>(.*?)</tr>', tm.group(1), re.S):
        cells = re.findall(r'background:\s*([^;\'"]+)', tr)
        if cells:
            rows.append([c.strip() for c in cells])
    if not rows:
        return None
    # keep dashboard order: it emits rows iy=NY-1 first -> rows[0]=iy9=FAR(0.65m)=TOP, rows[-1]=iy0=NEAR(0.10m)=bottom
    return (step_k, rate, rows)

def _draw_heat(grid, title, cell=30, pad=2):
    ny = len(grid); nx = max(len(r) for r in grid)
    W = nx*(cell+pad)+pad; H = ny*(cell+pad)+pad
    img = Image.new("RGB", (W, H+22), "#111")
    d = ImageDraw.Draw(img)
    d.text((3, 3), title, fill="#ddd", font=FONT_HDR)
    for ry, row in enumerate(grid):
        for rx, col in enumerate(row):
            x = pad+rx*(cell+pad); y = 22+pad+ry*(cell+pad)
            fill = col if (col and (col.startswith("#") or col.startswith("rgb"))) else "#222"
            d.rectangle([x, y, x+cell, y+cell], fill=fill)
            v = _cell_val(col)
            if v is not None:                          # per-cell success % centered in the cell
                txt = f"{v*100:.0f}"
                tc = "#000" if v > 0.5 else "#fff"     # contrast on green(high)/red(low)
                tb = d.textbbox((0, 0), txt, font=FONT_CELL)
                d.text((x+(cell-(tb[2]-tb[0]))/2, y+(cell-(tb[3]-tb[1]))/2-1), txt, fill=tc, font=FONT_CELL)
    return img

def _grid_minmax(grid):
    """min/max active-cell success over a color grid -> (min_pct,(ix,iy)), (max_pct,(ix,iy)) or None."""
    ny = len(grid); best=None; worst=None
    for ry, row in enumerate(grid):
        for rx, col in enumerate(row):
            v = _cell_val(col)
            if v is None: continue
            iy = ny-1-ry; ix = rx                      # rows[0]=iy9(far,top) .. rows[-1]=iy0(near,bottom)
            if worst is None or v < worst[0]: worst=(v,(ix,iy))
            if best  is None or v > best[0]:  best =(v,(ix,iy))
    if best is None: return None
    return (worst[0]*100, worst[1], best[0]*100, best[1])

def _combine(hm_rsr, hm_srch, header):
    imgs = [im for im in (hm_rsr, hm_srch) if im is not None]
    gap = 18; top = 46
    W = max(sum(im.width for im in imgs) + gap*(len(imgs)+1), 420)
    H = max(im.height for im in imgs) + top + 6
    canvas = Image.new("RGB", (W, H), "#111")
    d = ImageDraw.Draw(canvas)
    d.text((6, 6), header, fill="#8fd", font=FONT_HDR)
    d.text((6, 27), "rows: top=FAR(0.65m) .. bottom=NEAR(0.10m)  |  cols: L=x0.0 .. R=x0.8",
           fill="#8899aa", font=FONT_CELL)
    x = gap
    for im in imgs:
        canvas.paste(im, (x, top)); x += im.width + gap
    return canvas

def _hsafe(s):
    """ntfy header values must be single-line and header-encodable -> ASCII, no newlines."""
    return s.replace("\n", " | ").encode("ascii", "replace").decode("ascii")

def _post(png_bytes, title, msg):
    req = urllib.request.Request(NTFY, data=png_bytes, method="PUT", headers={
        "Filename": "sweep_heatmap.png",
        "Title": _hsafe(title),
        "Message": _hsafe(msg),
        "Tags": "robot",
    })
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.status

def handle(tag, url, last):
    html = _fetch(url)
    rsr  = _grab_table_after(html, "latest RSR heatmap")
    srch = _grab_table_after(html, "latest SEARCH heatmap")
    if not rsr:
        return last  # no sweep yet
    step_k, rsr_pct, rsr_grid = rsr
    if step_k is None or step_k == last:
        return last  # already sent this exact sweep (step changes -> post, incl. restart g-reset that lowers it)
    srch_pct = srch[1] if srch else None
    im_rsr  = _draw_heat(rsr_grid,  f"RSR {rsr_pct:.0f}%")
    im_srch = _draw_heat(srch[2], f"SEARCH {srch_pct:.0f}%") if srch else None
    header  = f"reBot {tag} sweep@{step_k}k"
    canvas  = _combine(im_rsr, im_srch, header)
    buf = io.BytesIO(); canvas.save(buf, "PNG")
    title = f"reBot {tag} sweep@{step_k}k"
    msg   = f"RSR {rsr_pct:.0f}%" + (f" | SEARCH {srch_pct:.0f}%" if srch_pct is not None else "")
    mm = _grid_minmax(rsr_grid)
    if mm:
        wp, wc, bp, bc = mm
        msg += f" | RSR min {wp:.0f}%@{wc} max {bp:.0f}%@{bc}"
    code = _post(buf.getvalue(), title, msg)
    print(f"[{time.strftime('%H:%M:%S')}] {tag} sweep@{step_k}k -> ntfy {code} ({msg})", flush=True)
    return step_k

def selftest():
    # synthetic 10x10 grids (corners excluded = #222) to validate parse->render->ntfy end-to-end
    def cell(v): return "#222" if v is None else f"rgb({int(200*(1-v))},{int(200*v)},40)"
    EXCL = {80,89,90,91,98,99}                  # FAR corners (iy=8,9)
    rsr_rows, srch_rows = [], []
    for iy in range(9, -1, -1):                 # dashboard order: rows[0]=iy9(far)=TOP .. iy0(near)=bottom
        r1, r2 = [], []
        for ix in range(10):
            flat = iy*10+ix
            v = None if flat in EXCL else max(0.0, min(1.0, 0.5+0.05*(ix-iy)))
            r1.append(cell(v)); r2.append(cell(None if flat in EXCL else 0.9))
        rsr_rows.append(r1); srch_rows.append(r2)
    im1 = _draw_heat(rsr_rows, "RSR 82%"); im2 = _draw_heat(srch_rows, "SEARCH 95%")
    canvas = _combine(im1, im2, "reBot [TEST] pipeline check")
    buf = io.BytesIO(); canvas.save(buf, "PNG")
    msg = "per-cell % in each cell"
    mm = _grid_minmax(rsr_rows)
    if mm:
        wp, wc, bp, bc = mm
        msg += f" | RSR min {wp:.0f}%@{wc} max {bp:.0f}%@{bc}"
    print("selftest ntfy ->", _post(buf.getvalue(), "reBot [TEST] heatmap pipeline", msg))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        selftest(); sys.exit(0)
    print(f"[ntfy-poller] start; topic={TOPIC}; ports={list(PORTS)}; every {POLL_S}s", flush=True)
    last = {t: 0 for t in PORTS}
    while True:
        for tag, url in PORTS.items():
            try:
                last[tag] = handle(tag, url, last[tag])
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] {tag} ERR {type(e).__name__}: {e}", flush=True)
        time.sleep(POLL_S)
