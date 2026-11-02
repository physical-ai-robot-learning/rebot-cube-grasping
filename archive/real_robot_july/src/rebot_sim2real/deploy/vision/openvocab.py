"""Open-vocab (OWLv2) 'red cube' detector used as a DYNAMIC ROI provider for the cam HSV localizer.

Rationale (option B, sim2real). A from-scratch front CNN did NOT transfer to the real rig (cube jumped
~25cm — the sim renders lacked the real intrinsics/distortion/roll warp). Plain HSV DOES transfer but is
fragile to red distractors (tape roll, skin/hand, background red). OWLv2 is pretrained on web images, so it
zero-shots a "red cube" with NO labels and NO sim2real gap — but a detector BOX is not mm-precise. So we
compose the two: OWLv2 proposes a box → the EXISTING HSV red-blob centroid (perception.red_mask /
_largest_blob_centroid) runs ONLY inside that box (via the roi= arg the estimator already supports). Red
outside the box can't out-vote the cube, and the centroid keeps HSV's precision.

Offline-validated on real front frames (deploy/view_front_raw.png etc.): cube scores 0.54–0.68 while the tape
roll / background never boxed (top distractor <0.10); 67–140 ms/call on an RTX 5080. See docs/26.

Structure: ONE heavy OpenVocabModel (weights) is shared across camera streams; each stream owns a cheap
OpenVocabROI (its own period-cache + live-editable query). OPT-IN only (deploy --front-openvocab /
--wrist-openvocab); this module contains NO HSV logic and NEVER touches the baseline paths — the caller runs
the HSV centroid inside the ROIs, so the validated baselines are untouched when this is off.
"""
import time
import threading
import numpy as np

DEFAULT_MODEL   = "google/owlv2-base-patch16-ensemble"
DEFAULT_QUERIES = ("a red cube", "a red block", "a small red toy block")


class OpenVocabModel:
    """Shared OWLv2 weights. `infer(rgb, queries, thresh)` is STATELESS -> one instance serves front + wrist
    (they run sequentially in the single perception thread, so no cross-stream locking is needed on the model)."""

    def __init__(self, model=DEFAULT_MODEL, device="cuda", size=None, do_compile=False):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self._torch  = torch
        self.device  = device
        self.name    = model
        # Auto classes so BOTH owlv2 (base-patch16, 48ms/20FPS) AND owlvit v1 (base-patch32, 18ms/55FPS — 30Hz-capable;
        # box is coarser but it's only the ROI gate for the HSV centroid, so localisation stays precise) load the same way.
        # local_files_only when cached -> stay OFFLINE. AutoProcessor/AutoModel otherwise HEAD the HF hub on every
        # load, which 504s when hub is flaky. Cached model -> offline; first-ever run -> download once, then offline.
        try:
            self._proc = AutoProcessor.from_pretrained(model, local_files_only=True)
            _lfo = {"local_files_only": True}
        except Exception:
            self._proc = AutoProcessor.from_pretrained(model)      # not cached yet -> one-time download
            _lfo = {}
        self._interp = bool(size)      # OWLv2/vit pos-embed is FIXED -> a non-default input square needs pos-embed INTERP
        if size:                       # shrink the input square -> ViT patches ~(size/default)^2 -> faster (small-obj acc ↓)
            try: self._proc.image_processor.size = {"height": int(size), "width": int(size)}
            except Exception: pass
        self._model  = AutoModelForZeroShotObjectDetection.from_pretrained(model, **_lfo).to(device).eval()
        # fp16 on GPU: ~2x faster forward (weights + compute). CPU stays fp32.
        self._half   = (str(device) != "cpu")
        if self._half:
            self._model = self._model.half()
        if do_compile:                 # torch.compile: owlvit-p32 -> 8ms (120 FPS). First infer compiles (~30s), then fast.
            self._model = torch.compile(self._model, mode="default")
        # post-process method was renamed across transformers versions; support both.
        self._pp = getattr(self._proc, "post_process_grounded_object_detection", None) \
            or self._proc.post_process_object_detection

    def infer(self, rgb, queries, thresh):
        """RGB uint8 (COLOR_BGR2RGB) -> list[(box_xyxy_px, score)] best-first, score >= thresh."""
        from PIL import Image
        torch = self._torch
        img = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8))
        W, H = img.size
        qs = [list(queries)]
        inputs = self._proc(text=qs, images=img, return_tensors="pt").to(self.device)
        if self._half:
            inputs["pixel_values"] = inputs["pixel_values"].half()   # match fp16 weights (token ids stay long)
        with torch.no_grad():
            out = self._model(**inputs, interpolate_pos_encoding=self._interp)   # interp pos-embed for non-960 input
        ts = torch.tensor([[H, W]], device=self.device)
        try:
            res = self._pp(outputs=out, threshold=thresh, target_sizes=ts, text_labels=qs)[0]
        except TypeError:
            res = self._pp(outputs=out, threshold=thresh, target_sizes=ts)[0]
        boxes  = res["boxes"].detach().cpu().numpy()
        scores = res["scores"].detach().cpu().numpy()
        order  = np.argsort(-scores)
        return [(boxes[i], float(scores[i])) for i in order]


class OpenVocabROI:
    """Per-camera-stream ROI provider over a shared OpenVocabModel. `.rois(rgb)` → list[(roi_frac, score)]
    best-first, roi_frac = (y0,y1,x0,x1) FRACTIONAL (exactly perception.red_mask/estimate_red_cube_world roi=).

    Re-runs the model at most every `period` s and caches; between re-detects the caller reuses the cached box
    and runs cheap HSV every frame. `set_queries(...)` is thread-safe (the dashboard/control thread edits the
    query live while the perception thread reads it), and invalidates the cache so the next frame re-detects.
    For a FIXED front cam the box barely moves; for --track / eye-in-hand a few-Hz re-detect + per-frame HSV
    inside the padded box tracks fine (a box miss just yields no red that frame → caller fallback / next detect).
    """

    def __init__(self, model, queries=DEFAULT_QUERIES, score_thresh=0.25, pad=0.12, period=0.35, name="cam"):
        self._model   = model
        self.name     = name
        self._queries = [str(q) for q in queries]
        self._score_thresh = float(score_thresh)
        self._pad     = float(pad)
        self._period  = float(period)
        self._lock    = threading.Lock()
        self._last_ts = -1e9
        self._query_version = 0       # query-version, bumped by set_queries -> boxes() discards an infer that finished
        #                               with a now-stale query (else a live query change lags up to one `period`)
        self._cache   = None          # list[(box_xyxy_px, score)] best-first
        self.stamp    = 0             # increments each time OWLv2 ACTUALLY re-ran (not a cache hit) -> the caller's
        #                               detect-then-track loop re-anchors its per-frame tracking box on a new stamp.
        self.last_box_px = None       # (box_xyxy_px, score) most recent top box — dashboard overlay + score readout

    def set_queries(self, q):
        """Live-set the query. q = str ('red cube' or comma-separated 'red cube, red block') or list of str."""
        if isinstance(q, str):
            q = [s.strip() for s in q.split(",")]
        q = [str(s).strip() for s in q if str(s).strip()]
        if not q:
            return False
        with self._lock:
            self._queries = q
            self._last_ts = -1e9      # force a re-detect with the new query on the next frame
            self._cache = None
            self._query_version += 1  # invalidate any infer already in flight (its result is for the OLD query)
        return True

    @property
    def queries(self):
        with self._lock:
            return list(self._queries)

    def boxes(self, rgb, now=None):
        """Cached list[(box_xyxy_px, score)] best-first; re-runs OWLv2 at most every `period` s.
        Thread-safety: set_queries() (control thread) may null self._cache at ANY time. So the cache-hit
        path returns a LOCAL ref captured under the lock — never re-reads self._cache after releasing it —
        else a concurrent set_queries() could null it in the window and this would return None -> rois() crash."""
        now = time.time() if now is None else now
        with self._lock:
            cached = self._cache                              # local ref, immune to a concurrent set_queries() null
            fresh = cached is not None and (now - self._last_ts) < self._period
            qs, thr, query_version = list(self._queries), self._score_thresh, self._query_version
        if fresh:
            return cached
        try:
            res = self._model.infer(rgb, qs, thr)
        except Exception:
            # A FAILED re-detect is NOT a fresh detection: reuse the stale box for THIS frame but do NOT bump
            # stamp/_last_ts, so (a) the caller doesn't re-anchor its tracker on a stale box (stamp unchanged),
            # (b) we retry the real detect next frame instead of suppressing it for a whole period.
            import traceback; traceback.print_exc()
            return cached or []
        with self._lock:
            if query_version != self._query_version:   # the query CHANGED during this infer -> result is for the OLD query;
                return cached or []                    #   discard (don't cache / advance) so the next frame re-infers fresh
            self._cache = res
            self._last_ts = now
            self.stamp += 1               # ONLY a successful OWLv2 re-run bumps stamp -> the caller's re-anchor signal
        return res

    def _box_to_roi(self, box, W, H):
        x0, y0, x1, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        bw, bh = (x1 - x0), (y1 - y0)
        x0 -= self._pad * bw; x1 += self._pad * bw
        y0 -= self._pad * bh; y1 += self._pad * bh
        return (max(0.0, y0 / H), min(1.0, y1 / H), max(0.0, x0 / W), min(1.0, x1 / W))

    def rois(self, rgb, now=None):
        """list[(roi_frac (y0,y1,x0,x1), score)] best-first. Empty if nothing above threshold."""
        H, W = rgb.shape[:2]
        bx = self.boxes(rgb, now)
        self.last_box_px = (bx[0][0], bx[0][1]) if bx else None   # best (box_xyxy_px, score) — overlay + score readout
        return [(self._box_to_roi(b, W, H), s) for (b, s) in bx]
