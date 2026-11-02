"""Read the success-sweep CSV -> precise success-rate SURFACE visualizations (works on partial data too):
  1) filled contour heatmap (success% vs cube xy, with iso-success contour lines)
  2) 3D surface (Z = success%)
  3) radial curve: success% vs distance from the workspace center (answers "how far before it drops")
Pure CPU (no Isaac) -> run anytime while the sweep streams the CSV.

    uv run --no-sync python scripts/plot_success_surface.py --csv viz_out/surface/surface.csv --out viz_out/surface
"""
import os, sys, argparse, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="viz_out/surface/surface.csv")
ap.add_argument("--out", default="viz_out/surface")
ap.add_argument("--config", default="configs/train_residual_realistic.yaml")
ap.add_argument("--sigma", type=float, default=0.9, help="gaussian smoothing (grid cells)")
args = ap.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rebot_sim2real.rl.config import load_train_config
c = load_train_config(args.config).env
rcx, rcy = c.residual_reach_center            # workspace center (table-relative), cm below
cxx, cyy = rcx * 100, rcy * 100

rows = []
with open(args.csv) as f:
    for r in csv.reader(f):
        if r and r[0].isdigit():
            rows.append((float(r[2]), float(r[3]), int(r[4])))
if not rows:
    print("no data yet"); sys.exit(0)
xs = sorted({round(x, 4) for x, _, _ in rows})
ys = sorted({round(y, 4) for _, y, _ in rows})
ix = {x: i for i, x in enumerate(xs)}; iy = {y: i for i, y in enumerate(ys)}
nx, ny = len(xs), len(ys)
S = np.zeros((ny, nx)); N = np.zeros((ny, nx))
for x, y, s in rows:
    j, i = iy[round(y, 4)], ix[round(x, 4)]
    S[j, i] += s; N[j, i] += 1
overall = S.sum() / max(N.sum(), 1)
frac = np.where(N > 0, S / np.maximum(N, 1), np.nan)
# count-weighted gaussian smoothing (robust to partial / uneven coverage)
num = gaussian_filter(np.where(N > 0, S, 0.0), args.sigma)
den = gaussian_filter(N.astype(float), args.sigma)
smooth = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan) * 100.0
Xc = np.array(xs) * 100.0; Yc = np.array(ys) * 100.0
XX, YY = np.meshgrid(Xc, Yc)
ncells = int((N > 0).sum()); minrep = int(N[N > 0].min()) if ncells else 0
sub = f"n={len(rows)} eps | {ncells}/{nx*ny} cells | min {minrep} rep/cell | overall {overall:.0%}"

# 1) contour heatmap
fig, ax = plt.subplots(figsize=(8, 7))
cf = ax.contourf(XX, YY, smooth, levels=np.linspace(0, 100, 21), cmap="RdYlGn", extend="both")
cl = ax.contour(XX, YY, smooth, levels=[40, 50, 60, 70, 80, 90], colors="k", linewidths=0.7)
ax.clabel(cl, fmt="%d%%", fontsize=8)
ax.plot(cxx, cyy, "b*", ms=16, label="workspace center")
ax.set_xlabel("cube x rel. table center (cm)  [+x right]")
ax.set_ylabel("cube y rel. table center (cm)  [+y back]")
ax.set_title(f"Pick success-rate surface\n{sub}")
ax.legend(loc="upper right"); ax.set_aspect("equal")
fig.colorbar(cf, ax=ax, label="success %")
fig.tight_layout(); fig.savefig(os.path.join(args.out, "surface_heatmap.png"), dpi=140); plt.close(fig)

# 2) 3D surface
fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection="3d")
Z = np.nan_to_num(smooth, nan=0.0)
ax.plot_surface(XX, YY, Z, cmap="viridis", linewidth=0, antialiased=True, vmin=0, vmax=100)
ax.set_xlabel("x rel center (cm)"); ax.set_ylabel("y rel center (cm)"); ax.set_zlabel("success %")
ax.set_zlim(0, 100); ax.set_title(f"Success surface (Z=success%)\n{sub}")
ax.view_init(elev=35, azim=-120)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "surface_3d.png"), dpi=140); plt.close(fig)

# 3) radial: success vs distance from workspace center
dx = (Xc[None, :] - cxx); dy = (Yc[:, None] - cyy)          # cm (Xc/Yc and cxx/cyy both in cm)
dist = np.sqrt(dx ** 2 + dy ** 2)
m = N > 0
d_flat = dist[m]; f_flat = frac[m] * 100.0; n_flat = N[m]
order = np.argsort(d_flat)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(d_flat, f_flat, s=14, c="0.6", alpha=0.5, label="per-cell")
# binned weighted mean
bins = np.arange(0, d_flat.max() + 1.2, 1.0)
bc, bm = [], []
for b0, b1 in zip(bins[:-1], bins[1:]):
    sel = (d_flat >= b0) & (d_flat < b1)
    if sel.sum() and n_flat[sel].sum() > 0:
        bc.append((b0 + b1) / 2); bm.append(np.average(f_flat[sel], weights=n_flat[sel]))
ax.plot(bc, bm, "b-o", lw=2, label="1cm-binned mean")
ax.axhline(80, color="g", ls="--", lw=0.8); ax.axhline(50, color="r", ls="--", lw=0.8)
ax.set_xlabel("distance from workspace center (cm)"); ax.set_ylabel("success %")
ax.set_ylim(0, 100); ax.set_title(f"Success vs distance from center\n{sub}")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(args.out, "surface_radial.png"), dpi=140); plt.close(fig)

print(f"[plot] overall {overall:.1%} | {ncells}/{nx*ny} cells | saved 3 PNGs -> {os.path.abspath(args.out)}")
for b, v in zip(bc, bm):
    print(f"  dist {b:.1f}cm -> {v:.0f}%")
