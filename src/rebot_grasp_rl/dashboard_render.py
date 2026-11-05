"""HTML and image helpers of the trainer's live web dashboard (was ui_render.py, split out of the trainer on 2026-07-27).

Pure functions, no trainer state: PNG encoding, SVG sparklines, stat cards, a trimmed mean for the dashboard numbers,
and the overlays drawn on the env-0 camera images (detection box / crosshair, gripper-axis arrow). The overlays are
also what the S4 rollout videos show (media/project2/rollouts_s4/).
Inputs: numpy uint8 images (H, W, 3), python lists / torch tensors of numbers. Outputs: bytes, HTML strings, or the
image modified in place.
"""
import io
import math

import imageio
import numpy as np
import torch

SPARK_BACKGROUND = "#0b0b0b"
# red-cube mask of the HSV detector the policy was first trained with (normalised RGB)
RED_MIN = 0.45
RED_MARGIN = 0.20
MIN_MASK_PIXELS = 1.5
TINT_KEEP = 0.45        # image weight under the mask tint
TINT_WEIGHT = 0.55      # tint colour weight
ARROW_HEAD_MAX_PX = 16.0
ARROW_MIN_LENGTH_PX = 5
RING_RADIUS_PX = 9
RING_STEP_DEG = 15


def pngBytes(arr):
    """PNG-encode an (H, W, 3) uint8 image."""
    tBuffer = io.BytesIO()
    imageio.imwrite(tBuffer, arr, format="png")
    return tBuffer.getvalue()


def sparkSvg(vals, color, w=260, h=64):
    """Inline SVG sparkline of `vals` with min / max / last labels (empty frame for fewer than two values)."""
    if len(vals) < 2:
        return f"<svg width='{w}' height='{h}' style='background:{SPARK_BACKGROUND};border-radius:6px'></svg>"
    tLo, tHi = min(vals), max(vals)
    tRng = (tHi - tLo) or 1.0
    tN = len(vals)
    tPts = " ".join(f"{tI/(tN-1)*(w-6)+3:.1f},{h-3-(tV-tLo)/tRng*(h-12):.1f}" for tI, tV in enumerate(vals))
    tLast = vals[-1]
    return (f"<svg width='{w}' height='{h}' style='background:{SPARK_BACKGROUND};border-radius:6px'>"
            f"<polyline points='{tPts}' fill='none' stroke='{color}' stroke-width='2'/>"
            f"<text x='6' y='14' fill='#777' font-size='10' font-family=monospace>{tHi:.3g}</text>"
            f"<text x='6' y='{h-4}' fill='#777' font-size='10' font-family=monospace>{tLo:.3g}</text>"
            f"<text x='{w-4}' y='14' fill='{color}' font-size='12' font-family=monospace "
            f"text-anchor='end'>{tLast:.3g}</text></svg>")


def trimmedMean(x, k=4):
    """Mean of a 1-D tensor without its k lowest and k highest values; the median when it has at most 2k values."""
    tN = x.numel()
    if tN == 0:
        return 0.0
    if tN <= 2 * k:
        return float(x.median().item())
    tXs, _ = torch.sort(x)
    return float(tXs[k:tN - k].mean().item())


def statCard(label, value, color="#4fc3f7", sub=""):
    """HTML card with a label, a large value and a sub-line."""
    return (f"<div style='background:#181818;border-radius:10px;padding:10px 18px;margin:5px;"
            f"display:inline-block;min-width:96px;text-align:center;border:1px solid #262626'>"
            f"<div style='color:#888;font-size:11px;letter-spacing:1px'>{label}</div>"
            f"<div style='color:{color};font-size:28px;font-weight:700'>{value}</div>"
            f"<div style='color:#666;font-size:10px'>{sub}</div></div>")


def drawDetection(im, det, c=(0, 255, 0)):
    """Draw on an env-0 camera image (in place, also returned): green tint over the red-cube mask, a yellow bounding
    box, and a crosshair at the detection centre `det` = (cx, cy, area, visible) when it is visible."""
    tH, tW = im.shape[:2]
    tX = im.astype(np.float32) / 255.0
    tR, tG, tB = tX[..., 0], tX[..., 1], tX[..., 2]
    tMask = (tR > RED_MIN) & (tR - tG > RED_MARGIN) & (tR - tB > RED_MARGIN)
    if tMask.sum() > MIN_MASK_PIXELS:
        tYs, tXs = np.nonzero(tMask)
        tX0, tX1, tY0, tY1 = int(tXs.min()), int(tXs.max()), int(tYs.min()), int(tYs.max())
        im[tMask] = (TINT_KEEP * im[tMask] + TINT_WEIGHT * np.array([0, 255, 0])).astype(np.uint8)
        for tCh, tCc in enumerate((255, 255, 0)):
            im[tY0:tY1 + 1, max(0, tX0 - 1):tX0 + 1, tCh] = tCc
            im[tY0:tY1 + 1, tX1:min(tW, tX1 + 2), tCh] = tCc
            im[max(0, tY0 - 1):tY0 + 1, tX0:tX1 + 1, tCh] = tCc
            im[tY1:min(tH, tY1 + 2), tX0:tX1 + 1, tCh] = tCc
    if det[3] > 0.5:
        tCx = int(min(max(det[0], 0.0), 1.0) * (tW - 1))
        tCy = int(min(max(det[1], 0.0), 1.0) * (tH - 1))
        for tCh in range(3):
            im[max(0, tCy - 1):tCy + 2, :, tCh] = c[tCh]
            im[:, max(0, tCx - 1):tCx + 2, tCh] = c[tCh]
    return im


def quatRpy(q):
    """Quaternion (w, x, y, z) -> ((roll, pitch, yaw) in degrees, gripper +z axis in the world frame)."""
    tW, tX, tY, tZ = [float(tV) for tV in (q[0], q[1], q[2], q[3])]
    tRoll = math.atan2(2 * (tW * tX + tY * tZ), 1 - 2 * (tX * tX + tY * tY))
    tPitch = math.asin(max(-1.0, min(1.0, 2 * (tW * tY - tZ * tX))))
    tYaw = math.atan2(2 * (tW * tZ + tX * tY), 1 - 2 * (tY * tY + tZ * tZ))
    return ((math.degrees(tRoll), math.degrees(tPitch), math.degrees(tYaw)),
            (2 * (tX * tZ + tW * tY), 2 * (tY * tZ - tW * tX), 1 - 2 * (tX * tX + tY * tY)))


def _signedArea(p, q, r):
    return (p[0] - r[0]) * (q[1] - r[1]) - (q[0] - r[0]) * (p[1] - r[1])


def drawArrow(im, cx, cy, dx, dy, col=(0, 255, 255)):
    """Draw (in place) an arrow from (cx, cy) along (dx, dy) pixels: thick shaft + filled triangular head. An arrow
    shorter than ARROW_MIN_LENGTH_PX points into the image (axis nearly straight down) and is drawn as a ring."""
    tH, tW = im.shape[:2]
    tL = (dx * dx + dy * dy) ** 0.5
    tN = int(tL) + 1
    for tT in range(tN + 1):
        tPx = int(cx + dx * tT / tN)
        tPy = int(cy + dy * tT / tN)
        if 0 <= tPy < tH and 0 <= tPx < tW:
            for tCh in range(3):
                im[max(0, tPy - 2):tPy + 3, max(0, tPx - 2):tPx + 3, tCh] = col[tCh]
    tX1, tY1 = cx + dx, cy + dy
    if tL > ARROW_MIN_LENGTH_PX:
        tUx, tUy = dx / tL, dy / tL
        tHl = min(ARROW_HEAD_MAX_PX, tL * 0.45)
        tPxv, tPyv = -tUy, tUx
        tHw = tHl * 0.6
        tA = (tX1, tY1)
        tB = (tX1 - tUx * tHl + tPxv * tHw, tY1 - tUy * tHl + tPyv * tHw)
        tC = (tX1 - tUx * tHl - tPxv * tHw, tY1 - tUy * tHl - tPyv * tHw)
        tMinX = int(max(0, min(tA[0], tB[0], tC[0])))
        tMaxX = int(min(tW - 1, max(tA[0], tB[0], tC[0])))
        tMinY = int(max(0, min(tA[1], tB[1], tC[1])))
        tMaxY = int(min(tH - 1, max(tA[1], tB[1], tC[1])))
        for tYy in range(tMinY, tMaxY + 1):
            for tXx in range(tMinX, tMaxX + 1):
                tPt = (tXx, tYy)
                tD1 = _signedArea(tPt, tA, tB)
                tD2 = _signedArea(tPt, tB, tC)
                tD3 = _signedArea(tPt, tC, tA)
                if not ((tD1 < 0 or tD2 < 0 or tD3 < 0) and (tD1 > 0 or tD2 > 0 or tD3 > 0)):
                    for tCh in range(3):
                        im[tYy, tXx, tCh] = col[tCh]
    else:
        for tAa in range(0, 360, RING_STEP_DEG):
            tPx = int(cx + RING_RADIUS_PX * math.cos(math.radians(tAa)))
            tPy = int(cy + RING_RADIUS_PX * math.sin(math.radians(tAa)))
            if 0 <= tPy < tH and 0 <= tPx < tW:
                for tCh in range(3):
                    im[tPy, tPx, tCh] = col[tCh]
