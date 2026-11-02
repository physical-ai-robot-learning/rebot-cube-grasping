"""STEP 3: deploy-side wrapper for the sim-trained CubeDetector (CSP+FPN+heatmap), the clutter/lighting
-robust drop-in for the brittle HSV (u,v). Loads ckpt/cube_detector_sharp.pt and turns a raw camera
frame (any resolution, 16:9) into the red-cube pixel. Runs in the deploy venv (pure torch, no Isaac).

Two consumers, one detector:
  * uv_normalized(frame)  -> (un, vn, visible) in [-1,1]   == project (b): the policy's (u,v) token
                             (drop-in for run_pick.wrist_feature / a top-cam token)
  * uv_full(frame)        -> (u, v, visible) in FULL-RES px == project (a): feed ray-plane -> cube_pos

The detector input is `input_hw` (320x180); the frame is resized in, and uv_full scales the result
back to the frame resolution so the calibration K (at the camera's native res) stays valid.
"""
from __future__ import annotations
import os
import numpy as np
import torch

from rebot_sim2real.perception_cnn import CubeDetector

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "..", "ckpt", "cube_detector_sharp.pt")


class CubeDetectorIO:
    def __init__(self, ckpt=DEFAULT_CKPT, device="cpu", vis_thresh=0.5):
        sd = torch.load(ckpt, map_location=device)
        self.model = CubeDetector(w=int(sd.get("width", 32)), stride=int(sd.get("stride", 8))).to(device)
        self.model.load_state_dict(sd["state_dict"])
        self.model.eval()
        self.device = device
        self.in_h, self.in_w = sd.get("input_hw", [180, 320])
        self.vis_thresh = vis_thresh
        print(f"[CubeDetectorIO] loaded {os.path.basename(ckpt)} (input {self.in_w}x{self.in_h})")

    @torch.no_grad()
    def _detect_input_px(self, rgb):
        """rgb HxWx3 uint8 (any res) -> (u, v) in INPUT px [in_w,in_h], visible_prob."""
        import cv2
        x = cv2.resize(rgb, (self.in_w, self.in_h))
        t = torch.from_numpy(np.ascontiguousarray(x)).to(self.device).float().div(255.0)
        t = t.permute(2, 0, 1).unsqueeze(0)
        hm, vlogit = self.model(t)
        uv = self.model.uv_from_heatmap(hm)[0]
        return float(uv[0]), float(uv[1]), float(torch.sigmoid(vlogit)[0])

    def uv_normalized(self, rgb):
        """project (b): the policy's (u,v) token, normalized to [-1,1] (same convention as the HSV
        wrist_feature). Not-visible -> (0,0,0)."""
        u, v, p = self._detect_input_px(rgb)
        if p < self.vis_thresh:
            return torch.zeros(1, 3)
        un = (u - self.in_w / 2.0) / (self.in_w / 2.0)
        vn = (v - self.in_h / 2.0) / (self.in_h / 2.0)
        return torch.tensor([[un, vn, 1.0]], dtype=torch.float32)

    def uv_full(self, rgb):
        """project (a): (u, v) in the frame's FULL resolution px (+ visible_prob), for ray-plane ->
        cube_pos with the native-res calibration K. Returns None if not visible."""
        H, W = rgb.shape[:2]
        u, v, p = self._detect_input_px(rgb)
        if p < self.vis_thresh:
            return None
        return u * (W / self.in_w), v * (H / self.in_h), p
