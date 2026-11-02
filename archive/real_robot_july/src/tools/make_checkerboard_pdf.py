#!/usr/bin/env python3
"""Generate an EXACT-SIZE checkerboard calibration PDF (A4) for src/tools/calib_intrinsics_checkerboard.py.

Vector PDF at true physical size: 1 data-mm == 1 page-mm, so if you PRINT AT 100% (Actual size / NO
fit-to-page) each square is exactly --square-mm. A printed 2-square scale bar + a "verify with a ruler"
note lets you confirm the printer didn't rescale (if it did, just pass the MEASURED size to --square-mm).

Default 10x7 squares = 9x6 INNER corners (matches the calib script default), 25mm squares, landscape A4.
  .venv/bin/python src/tools/make_checkerboard_pdf.py            # writes deploy/checkerboard_A4.pdf
  .venv/bin/python src/tools/make_checkerboard_pdf.py --square-mm 20 --sq-x 12 --sq-y 9
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

MM_PER_IN = 25.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sq-x", type=int, default=10, help="squares along the LONG side (inner corners = this-1)")
    ap.add_argument("--sq-y", type=int, default=7, help="squares along the SHORT side (inner corners = this-1)")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--landscape", action="store_true", default=True)
    ap.add_argument("--out", default="deploy/checkerboard_A4.pdf")
    args = ap.parse_args()

    PAGE_W, PAGE_H = (297.0, 210.0) if args.landscape else (210.0, 297.0)   # A4 mm
    sq, nx, ny = float(args.square_mm), int(args.sq_x), int(args.sq_y)
    bw, bh = nx * sq, ny * sq
    if bw > PAGE_W - 10 or bh > PAGE_H - 20:
        raise SystemExit(f"board {bw:.0f}x{bh:.0f}mm doesn't fit A4 {PAGE_W:.0f}x{PAGE_H:.0f} with margin — "
                         f"reduce --square-mm or --sq-x/--sq-y")
    x0 = (PAGE_W - bw) / 2.0
    y0 = (PAGE_H - bh) / 2.0 + 6.0            # nudge up to leave room for the caption + scale bar at the bottom

    fig = plt.figure(figsize=(PAGE_W / MM_PER_IN, PAGE_H / MM_PER_IN))   # exact A4 physical size
    ax = fig.add_axes([0, 0, 1, 1])          # axes fills the whole page
    ax.set_xlim(0, PAGE_W); ax.set_ylim(0, PAGE_H)
    ax.set_aspect("equal")                    # 1mm data == 1mm page (box aspect already matches the page)
    ax.axis("off")

    # checkerboard: black squares on white; the page margin IS the required white quiet-zone for detection
    for i in range(nx):
        for j in range(ny):
            if (i + j) % 2 == 0:
                ax.add_patch(Rectangle((x0 + i * sq, y0 + j * sq), sq, sq, facecolor="black", edgecolor="none"))
    ax.add_patch(Rectangle((x0, y0), bw, bh, fill=False, edgecolor="black", linewidth=0.4))  # thin board outline

    inner = f"{nx - 1}x{ny - 1}"
    # scale bar = exactly 2 squares, labeled -> ruler check
    sbx, sby = x0, y0 - 12.0
    ax.plot([sbx, sbx + 2 * sq], [sby, sby], color="black", linewidth=1.2)
    for xx in (sbx, sbx + sq, sbx + 2 * sq):
        ax.plot([xx, xx], [sby - 1.5, sby + 1.5], color="black", linewidth=1.2)
    ax.text(sbx + 2 * sq + 4, sby, f"= {2 * sq:.0f} mm (2 squares).  Measure with a ruler after printing.",
            va="center", ha="left", fontsize=8)

    ax.text(PAGE_W / 2, y0 + bh + 8,
            f"Checkerboard  {nx}x{ny} squares  =  {inner} INNER corners   |   square = {sq:.0f} mm",
            va="bottom", ha="center", fontsize=11, fontweight="bold")
    ax.text(PAGE_W / 2, 7,
            f"PRINT AT 100% / ACTUAL SIZE  (NOT 'fit to page').   Then:  "
            f"calib_intrinsics_checkerboard.py --cols {nx - 1} --rows {ny - 1} --square-mm {sq:.0f}   "
            f"(use the RULER-MEASURED size if it isn't exactly {sq:.0f}).",
            va="bottom", ha="center", fontsize=8)

    fig.savefig(args.out)   # NOTE: no bbox_inches='tight' — that would crop and rescale, breaking exact size
    print(f"[cb-pdf] wrote {args.out}: A4 {'landscape' if args.landscape else 'portrait'}, "
          f"{nx}x{ny} squares ({inner} inner corners), {sq:.0f}mm. PRINT AT 100%.")


if __name__ == "__main__":
    main()
