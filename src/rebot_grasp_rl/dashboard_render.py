"""ui_render.py — pure HTTP-UI/image rendering helpers extracted from train_sac_det_xattn.py
REFACTOR 2026-07-27: self-contained (no runtime globals). Must be in the ray --working-dir.
"""
import io, math
import math as _mth
import numpy as np
import torch
import imageio

def _png(arr):
    tB = io.BytesIO(); imageio.imwrite(tB, arr, format="png"); return tB.getvalue()

def _spark(vals, color, w=260, h=64):
    if len(vals) < 2:
        return f"<svg width='{w}' height='{h}' style='background:#0b0b0b;border-radius:6px'></svg>"
    tLo, tHi = min(vals), max(vals); tRng = (tHi - tLo) or 1.0; tN = len(vals)
    tPts = " ".join(f"{tI/(tN-1)*(w-6)+3:.1f},{h-3-(tV-tLo)/tRng*(h-12):.1f}" for tI, tV in enumerate(vals))
    tLast = vals[-1]
    return (f"<svg width='{w}' height='{h}' style='background:#0b0b0b;border-radius:6px'>"
            f"<polyline points='{tPts}' fill='none' stroke='{color}' stroke-width='2'/>"
            f"<text x='6' y='14' fill='#777' font-size='10' font-family=monospace>{tHi:.3g}</text>"
            f"<text x='6' y='{h-4}' fill='#777' font-size='10' font-family=monospace>{tLo:.3g}</text>"
            f"<text x='{w-4}' y='14' fill='{color}' font-size='12' font-family=monospace "
            f"text-anchor='end'>{tLast:.3g}</text></svg>")

def _trimmean(x, k=4):
    """user 2026-07-10: TRIMMED MEAN — drop k lowest + k highest (8 outliers total), mean of the rest. median fallback if <=2k."""
    tN = x.numel()
    if tN == 0: return 0.0
    if tN <= 2 * k: return float(x.median().item())
    tXs, _ = torch.sort(x)
    return float(tXs[k:tN - k].mean().item())

def _card(label, value, color="#4fc3f7", sub=""):
    return (f"<div style='background:#181818;border-radius:10px;padding:10px 18px;margin:5px;"
            f"display:inline-block;min-width:96px;text-align:center;border:1px solid #262626'>"
            f"<div style='color:#888;font-size:11px;letter-spacing:1px'>{label}</div>"
            f"<div style='color:{color};font-size:28px;font-weight:700'>{value}</div>"
            f"<div style='color:#666;font-size:10px'>{sub}</div></div>")

def _drawDet(im, det, c=(0, 255, 0)):
    """Visualise the detection on env-0 image: red-mask TINT (what the detector sees) + BBOX (yellow) +
    centroid crosshair. Mask uses the SAME HSV threshold the policy trains on (R>0.45)."""
    tH, tW = im.shape[:2]
    tX = im.astype(np.float32) / 255.0
    tR, tG, tB2 = tX[..., 0], tX[..., 1], tX[..., 2]
    tMask = (tR > 0.45) & (tR - tG > 0.20) & (tR - tB2 > 0.20)               # same as _red_det (training)
    if tMask.sum() > 1.5:
        tYs, tXs = np.nonzero(tMask)
        tX0, tX1, tY0, tY1 = int(tXs.min()), int(tXs.max()), int(tYs.min()), int(tYs.max())
        im[tMask] = (0.45 * im[tMask] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)   # green seg tint
        for tCh, tCc in enumerate((255, 255, 0)):                      # yellow bbox
            im[tY0:tY1 + 1, max(0, tX0 - 1):tX0 + 1, tCh] = tCc; im[tY0:tY1 + 1, tX1:min(tW, tX1 + 2), tCh] = tCc
            im[max(0, tY0 - 1):tY0 + 1, tX0:tX1 + 1, tCh] = tCc; im[tY1:min(tH, tY1 + 2), tX0:tX1 + 1, tCh] = tCc
    if det[3] > 0.5:                                                  # centroid crosshair
        tCx = int(min(max(det[0], 0.0), 1.0) * (tW - 1)); tCy = int(min(max(det[1], 0.0), 1.0) * (tH - 1))
        for tCh in range(3):
            im[max(0, tCy - 1):tCy + 2, :, tCh] = c[tCh]; im[:, max(0, tCx - 1):tCx + 2, tCh] = c[tCh]
    return im

import math as _mth
def _quatRpy(q):   # (w,x,y,z) -> (roll,pitch,yaw) deg + grasp-axis(+z) world
    tW2,tX,tY,tZ = [float(tV) for tV in (q[0],q[1],q[2],q[3])]
    tRoll=_mth.atan2(2*(tW2*tX+tY*tZ),1-2*(tX*tX+tY*tY)); tPitch=_mth.asin(max(-1.0,min(1.0,2*(tW2*tY-tZ*tX)))); tYaw=_mth.atan2(2*(tW2*tZ+tX*tY),1-2*(tY*tY+tZ*tZ))
    return (_mth.degrees(tRoll),_mth.degrees(tPitch),_mth.degrees(tYaw)), (2*(tX*tZ+tW2*tY), 2*(tY*tZ-tW2*tX), 1-2*(tX*tX+tY*tY))
def _drawArrow(im, cx, cy, dx, dy, col=(0,255,255)):   # 3D-ish: shaft + triangle head; ~straight-down -> into-page ring
    tH,tW = im.shape[:2]; tL=(dx*dx+dy*dy)**0.5; tN=int(tL)+1
    for tT in range(tN+1):                                   # thick shaft
        tPx=int(cx+dx*tT/tN); tPy=int(cy+dy*tT/tN)
        if 0<=tPy<tH and 0<=tPx<tW:
            for tCh in range(3): im[max(0,tPy-2):tPy+3, max(0,tPx-2):tPx+3, tCh]=col[tCh]
    tX1,tY1=cx+dx,cy+dy
    if tL>5:                                                # filled triangle arrowhead at tip
        tUx,tUy=dx/tL,dy/tL; tHl=min(16.0,tL*0.45); tPxv,tPyv=-tUy,tUx; tHw=tHl*0.6
        tA=(tX1,tY1); tB=(tX1-tUx*tHl+tPxv*tHw,tY1-tUy*tHl+tPyv*tHw); tC=(tX1-tUx*tHl-tPxv*tHw,tY1-tUy*tHl-tPyv*tHw)
        tMnx=int(max(0,min(tA[0],tB[0],tC[0]))); tMxx=int(min(tW-1,max(tA[0],tB[0],tC[0])))
        tMny=int(max(0,min(tA[1],tB[1],tC[1]))); tMxy=int(min(tH-1,max(tA[1],tB[1],tC[1])))
        def _sg(p,q,r): return (p[0]-r[0])*(q[1]-r[1])-(q[0]-r[0])*(p[1]-r[1])
        for tYy in range(tMny,tMxy+1):
            for tXx in range(tMnx,tMxx+1):
                tPt=(tXx,tYy); tD1=_sg(tPt,tA,tB); tD2=_sg(tPt,tB,tC); tD3=_sg(tPt,tC,tA)
                if not((tD1<0 or tD2<0 or tD3<0) and (tD1>0 or tD2>0 or tD3>0)):
                    for tCh in range(3): im[tYy,tXx,tCh]=col[tCh]
    else:                                                  # pointing ~straight DOWN (into image) -> ring marker (3D depth cue)
        for tAa in range(0,360,15):
            tPx=int(cx+9*_mth.cos(_mth.radians(tAa))); tPy=int(cy+9*_mth.sin(_mth.radians(tAa)))
            if 0<=tPy<tH and 0<=tPx<tW:
                for tCh in range(3): im[tPy,tPx,tCh]=col[tCh]
