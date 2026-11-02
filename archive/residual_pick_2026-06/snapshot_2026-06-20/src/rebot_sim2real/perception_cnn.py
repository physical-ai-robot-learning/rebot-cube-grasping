"""Small CNN cube detector: CSP-ish backbone + 2-level FPN + heatmap head -> red-cube pixel (u,v)
via soft-argmax, plus a visibility logit. A clutter/lighting-robust, learned drop-in for the brittle
HSV (u,v) detector. Trained in sim with FREE labels (HSV centroid) + visual DR; the SAME model runs
on sim renders and the real camera. Pure torch (no Isaac), so it imports in both the training and
the deploy venv.

Output (u,v) is in INPUT-IMAGE pixels at the detector's input resolution (default 320x180); the
caller normalizes to the (u,v) token the policy expects, exactly like the HSV path.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cba(ci, co, k=3, s=1):
    return nn.Sequential(nn.Conv2d(ci, co, k, s, k // 2, bias=False), nn.BatchNorm2d(co), nn.SiLU(inplace=True))


class _CSP(nn.Module):
    """Cross-Stage-Partial block: split channels, run a bottleneck on one half, concat."""

    def __init__(self, c):
        super().__init__()
        self.a = _cba(c, c // 2, 1)
        self.b = _cba(c, c // 2, 1)
        self.m = nn.Sequential(_cba(c // 2, c // 2, 3), _cba(c // 2, c // 2, 3))
        self.out = _cba(c, c, 1)

    def forward(self, x):
        return self.out(torch.cat([self.a(x), self.m(self.b(x))], dim=1))


class CubeDetector(nn.Module):
    """Input [B,3,H,W] in 0..1 -> (heatmap [B,1,h,w] at `stride`, visibility logit [B]).
    Use uv_from_heatmap() for the (u,v) in input pixels."""

    def __init__(self, w: int = 32, stride: int = 8):
        super().__init__()
        assert stride == 8
        self.stride = stride
        self.stem = _cba(3, w, 3, 2)                              # /2
        self.c1 = nn.Sequential(_cba(w, w * 2, 3, 2), _CSP(w * 2))    # /4
        self.c2 = nn.Sequential(_cba(w * 2, w * 4, 3, 2), _CSP(w * 4))  # /8
        self.c3 = nn.Sequential(_cba(w * 4, w * 8, 3, 2), _CSP(w * 8))  # /16
        self.lat3 = _cba(w * 8, w * 4, 1)                        # FPN lateral
        self.fpn = _cba(w * 4, w * 4, 3)                         # P2 (/8) = c2 + up(c3)
        self.hm = nn.Conv2d(w * 4, 1, 1)                         # 1-ch heatmap at /8
        self.vis = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(w * 8, 1))

    def forward(self, x):
        s = self.stem(x)
        c2 = self.c2(self.c1(s))
        c3 = self.c3(c2)
        p2 = self.fpn(c2 + F.interpolate(self.lat3(c3), size=c2.shape[-2:], mode="nearest"))
        return self.hm(p2), self.vis(c3).squeeze(-1)

    def uv_from_heatmap(self, hm):
        """Soft-argmax over the heatmap -> (u,v) [B,2] in INPUT-IMAGE pixels (grid coord * stride)."""
        B, _, h, w = hm.shape
        prob = F.softmax(hm.reshape(B, -1), dim=-1).reshape(B, h, w)
        xs = torch.arange(w, device=hm.device, dtype=hm.dtype)
        ys = torch.arange(h, device=hm.device, dtype=hm.dtype)
        gu = (prob.sum(1) * xs).sum(-1)        # expected grid col
        gv = (prob.sum(2) * ys).sum(-1)        # expected grid row
        return torch.stack([gu, gv], dim=-1) * self.stride + self.stride / 2.0


def heatmap_target(uv, h, w, stride, sigma=1.5):
    """Gaussian heatmap target [B,h,w] centered at uv (input px). uv [B,2]; rows where the cube is
    not visible should be zeroed by the caller (multiply by the visible mask)."""
    B = uv.shape[0]
    gx = (uv[:, 0] / stride).clamp(0, w - 1)
    gy = (uv[:, 1] / stride).clamp(0, h - 1)
    xs = torch.arange(w, device=uv.device, dtype=uv.dtype).view(1, 1, w)
    ys = torch.arange(h, device=uv.device, dtype=uv.dtype).view(1, h, 1)
    d2 = (xs - gx.view(B, 1, 1)) ** 2 + (ys - gy.view(B, 1, 1)) ** 2
    return torch.exp(-d2 / (2 * sigma ** 2))


@torch.no_grad()
def detect(model, img_rgb_u8, device=None):
    """Run the detector on a single HxWx3 uint8 RGB image at the model's input size.
    Returns (u, v, visible_prob) in INPUT-IMAGE pixels. Caller resizes the camera frame to the
    model input size first and scales (u,v) back as needed."""
    device = device or next(model.parameters()).device
    import numpy as np
    x = torch.from_numpy(np.ascontiguousarray(img_rgb_u8)).to(device).float().div(255.0)
    x = x.permute(2, 0, 1).unsqueeze(0)
    hm, vlogit = model(x)
    uv = model.uv_from_heatmap(hm)[0]
    return float(uv[0]), float(uv[1]), float(torch.sigmoid(vlogit)[0])
