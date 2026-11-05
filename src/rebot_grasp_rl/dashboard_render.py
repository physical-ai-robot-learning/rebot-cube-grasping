"""ui_render.py — pure HTTP-UI/image rendering helpers extracted from train_sac_det_xattn.py
REFACTOR 2026-07-27: self-contained (no runtime globals). Must be in the ray --working-dir.
"""
import io, math
import math as _mth
import numpy as np
import torch
import imageio

def _png(arr):
    b = io.BytesIO(); imageio.imwrite(b, arr, format="png"); return b.getvalue()

def _spark(vals, color, w=260, h=64):
    if len(vals) < 2:
        return f"<svg width='{w}' height='{h}' style='background:#0b0b0b;border-radius:6px'></svg>"
    lo, hi = min(vals), max(vals); rng = (hi - lo) or 1.0; n = len(vals)
    pts = " ".join(f"{i/(n-1)*(w-6)+3:.1f},{h-3-(v-lo)/rng*(h-12):.1f}" for i, v in enumerate(vals))
    last = vals[-1]
    return (f"<svg width='{w}' height='{h}' style='background:#0b0b0b;border-radius:6px'>"
            f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='2'/>"
            f"<text x='6' y='14' fill='#777' font-size='10' font-family=monospace>{hi:.3g}</text>"
            f"<text x='6' y='{h-4}' fill='#777' font-size='10' font-family=monospace>{lo:.3g}</text>"
            f"<text x='{w-4}' y='14' fill='{color}' font-size='12' font-family=monospace "
            f"text-anchor='end'>{last:.3g}</text></svg>")

def _trimmean(x, k=4):
    """user 2026-07-10: TRIMMED MEAN — drop k lowest + k highest (8 outliers total), mean of the rest. median fallback if <=2k."""
    n = x.numel()
    if n == 0: return 0.0
    if n <= 2 * k: return float(x.median().item())
    xs, _ = torch.sort(x)
    return float(xs[k:n - k].mean().item())

def _card(label, value, color="#4fc3f7", sub=""):
    return (f"<div style='background:#181818;border-radius:10px;padding:10px 18px;margin:5px;"
            f"display:inline-block;min-width:96px;text-align:center;border:1px solid #262626'>"
            f"<div style='color:#888;font-size:11px;letter-spacing:1px'>{label}</div>"
            f"<div style='color:{color};font-size:28px;font-weight:700'>{value}</div>"
            f"<div style='color:#666;font-size:10px'>{sub}</div></div>")

def _draw_det(im, det, c=(0, 255, 0)):
    """Visualise the detection on env-0 image: red-mask TINT (what the detector sees) + BBOX (yellow) +
    centroid crosshair. Mask uses the SAME HSV threshold the policy trains on (R>0.45)."""
    H, W = im.shape[:2]
    x = im.astype(np.float32) / 255.0
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    mask = (R > 0.45) & (R - G > 0.20) & (R - B > 0.20)               # same as _red_det (training)
    if mask.sum() > 1.5:
        ys, xs = np.nonzero(mask)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        im[mask] = (0.45 * im[mask] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)   # green seg tint
        for ch, cc in enumerate((255, 255, 0)):                      # yellow bbox
            im[y0:y1 + 1, max(0, x0 - 1):x0 + 1, ch] = cc; im[y0:y1 + 1, x1:min(W, x1 + 2), ch] = cc
            im[max(0, y0 - 1):y0 + 1, x0:x1 + 1, ch] = cc; im[y1:min(H, y1 + 2), x0:x1 + 1, ch] = cc
    if det[3] > 0.5:                                                  # centroid crosshair
        cx = int(min(max(det[0], 0.0), 1.0) * (W - 1)); cy = int(min(max(det[1], 0.0), 1.0) * (H - 1))
        for ch in range(3):
            im[max(0, cy - 1):cy + 2, :, ch] = c[ch]; im[:, max(0, cx - 1):cx + 2, ch] = c[ch]
    return im

import math as _mth
def _quat_rpy(q):   # (w,x,y,z) -> (roll,pitch,yaw) deg + grasp-axis(+z) world
    w,x,y,z = [float(v) for v in (q[0],q[1],q[2],q[3])]
    roll=_mth.atan2(2*(w*x+y*z),1-2*(x*x+y*y)); pitch=_mth.asin(max(-1.0,min(1.0,2*(w*y-z*x)))); yaw=_mth.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    return (_mth.degrees(roll),_mth.degrees(pitch),_mth.degrees(yaw)), (2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y))
def _draw_arrow(im, cx, cy, dx, dy, col=(0,255,255)):   # 3D-ish: shaft + triangle head; ~straight-down -> into-page ring
    H,W = im.shape[:2]; L=(dx*dx+dy*dy)**0.5; n=int(L)+1
    for t in range(n+1):                                   # thick shaft
        px=int(cx+dx*t/n); py=int(cy+dy*t/n)
        if 0<=py<H and 0<=px<W:
            for ch in range(3): im[max(0,py-2):py+3, max(0,px-2):px+3, ch]=col[ch]
    x1,y1=cx+dx,cy+dy
    if L>5:                                                # filled triangle arrowhead at tip
        ux,uy=dx/L,dy/L; hl=min(16.0,L*0.45); pxv,pyv=-uy,ux; hw=hl*0.6
        a=(x1,y1); b=(x1-ux*hl+pxv*hw,y1-uy*hl+pyv*hw); c=(x1-ux*hl-pxv*hw,y1-uy*hl-pyv*hw)
        _mnx=int(max(0,min(a[0],b[0],c[0]))); _mxx=int(min(W-1,max(a[0],b[0],c[0])))
        _mny=int(max(0,min(a[1],b[1],c[1]))); _mxy=int(min(H-1,max(a[1],b[1],c[1])))
        def _sg(p,q,r): return (p[0]-r[0])*(q[1]-r[1])-(q[0]-r[0])*(p[1]-r[1])
        for yy in range(_mny,_mxy+1):
            for xx in range(_mnx,_mxx+1):
                pt=(xx,yy); d1=_sg(pt,a,b); d2=_sg(pt,b,c); d3=_sg(pt,c,a)
                if not((d1<0 or d2<0 or d3<0) and (d1>0 or d2>0 or d3>0)):
                    for ch in range(3): im[yy,xx,ch]=col[ch]
    else:                                                  # pointing ~straight DOWN (into image) -> ring marker (3D depth cue)
        for _aa in range(0,360,15):
            px=int(cx+9*_mth.cos(_mth.radians(_aa))); py=int(cy+9*_mth.sin(_mth.radians(_aa)))
            if 0<=py<H and 0<=px<W:
                for ch in range(3): im[py,px,ch]=col[ch]
