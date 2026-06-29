"""Pipeline overview slide -- matplotlib, true 16:9.

Slide-design decisions:
  - Single horizontal pipeline across the upper band; the eye reads left to
    right, no row-wrapping, no diagonal arrows.
  - Cablestack collapsed to ONE node ("4 stages") -- the four-stage detail
    belongs on a zoom slide, not the overview.
  - RVE parked in a separate band below with an italic annotation in place
    of a long crossing arrow. Visual parallelism conveys "runs alongside".
  - Workflow thumbnails (ParaView, conformal, stress-strain) carry the
    visual story; intermediate transcoders sit as quiet text boxes.

Output: data/diagrams/slide_overview.{png, svg} -- 16:9, 1600x900 nominal.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO / "data" / "diagrams"
ICONS_DIR = OUTPUT_DIR / "icons"

TEXT_FILL = "#eef3f8"
TEXT_EDGE = "#6f8ea7"
ACCENT_FILL = "#fff2e2"      # warmer fill for the cablestack node
ACCENT_EDGE = "#c98a3a"
CLUSTER_FILL = "#dbeaf5"
CLUSTER_EDGE = "#6f8ea7"
ARROW_COLOR = "#222222"
DASH_COLOR = "#7a7a7a"

FONT = "Arial"
FS_TITLE = 32
FS_NODE = 18
FS_CAPTION = 17
FS_ANNOT = 17
FS_CLUSTER = 18


def text_box(ax, x, y, label, w=1.7, h=1.05, fc=TEXT_FILL, ec=TEXT_EDGE,
             fs=FS_NODE):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.5, facecolor=fc, edgecolor=ec, zorder=2,
    ))
    ax.text(x, y, label, ha="center", va="center",
            family=FONT, fontsize=fs, color="#1a1a1a", zorder=3)
    return (x, y, w, h)


def image_node(ax, x, y_centre, img_path, caption, w=2.4, h=1.55,
               cap_y=None, cap_gap=0.30, fs=FS_CAPTION):
    """Workflow image with caption. If cap_y is given, the caption is pinned
    to that y -- useful for keeping captions aligned across multiple images
    of differing aspect ratios."""
    img = mpimg.imread(img_path)
    ah, aw = img.shape[:2]
    aspect = aw / ah
    if aspect >= w / h:
        disp_w, disp_h = w, w / aspect
    else:
        disp_w, disp_h = h * aspect, h
    x0, y0 = x - disp_w / 2, y_centre - disp_h / 2
    ax.imshow(img, extent=(x0, x0 + disp_w, y0, y0 + disp_h),
              aspect="auto", interpolation="bilinear", zorder=2)
    ax.add_patch(FancyBboxPatch(
        (x0, y0), disp_w, disp_h,
        boxstyle="round,pad=0,rounding_size=0",
        linewidth=1.0, facecolor="none", edgecolor="#999999", zorder=3,
    ))
    if cap_y is None:
        cap_y = y0 - cap_gap
    ax.text(x, cap_y, caption, ha="center", va="top",
            family=FONT, fontsize=fs, color="#1a1a1a", zorder=3)
    return (x, y_centre, w, h)


def cluster(ax, x0, y0, x1, y1, title):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.02,rounding_size=0.28",
        linewidth=1.0, facecolor=CLUSTER_FILL, edgecolor=CLUSTER_EDGE,
        alpha=0.40, zorder=0,
    ))
    ax.text(x0 + 0.25, y1 - 0.22, title,
            ha="left", va="top", family=FONT, fontsize=FS_CLUSTER,
            color="#1a1a1a", style="italic", zorder=1)


def arrow(ax, p, q, color=ARROW_COLOR, ls="-", lw=2.4,
          connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=22,
        linewidth=lw, color=color, linestyle=ls, zorder=4,
        shrinkA=3, shrinkB=3, connectionstyle=connectionstyle,
    ))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(8, 8.55, "LTS Rutherford pipeline",
            ha="center", va="center", family=FONT, fontsize=FS_TITLE,
            fontweight="bold", color="#1a1a1a")

    # ---- main horizontal flow (one row, 8 elements) --------------------------
    y = 5.70           # centre line for text boxes
    y_img = y + 0.45   # image centres lifted slightly so captions align under
    cap_y = 4.55       # fixed caption y -- keeps labels under all three images
    p_cfg  = text_box(ax, 1.10, y, "Cable\nparams",   w=1.30, h=1.10)
    p_step = text_box(ax, 2.60, y, "FreeCAD\nSTEP",   w=1.30, h=1.10)
    p_mesh = text_box(ax, 4.10, y, "Ansys\nmesh",     w=1.30, h=1.10)
    p_lsd  = text_box(ax, 5.60, y, "LS-DYNA\nsolve",  w=1.30, h=1.10)
    p_pv   = image_node(ax, 7.50, y_img, ICONS_DIR / "paraview.png",
                        "Deformed strands", w=1.85, h=1.40, cap_y=cap_y,
                        fs=FS_CAPTION - 1)
    p_conf = image_node(ax, 9.90, y_img, ICONS_DIR / "conformal.png",
                        "Conformal mesh", w=1.85, h=1.20, cap_y=cap_y,
                        fs=FS_CAPTION - 1)
    p_cs   = text_box(ax, 12.30, y, "Cablestack\nAPDL\n(4 stages)",
                      w=1.85, h=1.65, fc=ACCENT_FILL, ec=ACCENT_EDGE)
    p_pp   = image_node(ax, 14.55, y_img, ICONS_DIR / "subplots.png",
                        "Stress-strain", w=1.55, h=1.25, cap_y=cap_y,
                        fs=FS_CAPTION - 1)

    # ---- arrows on the main flow ---------------------------------------------
    # Connect at the text-row y (y) so the arrow line sits on the conceptual
    # spine of the chain; images live a bit above but the eye accepts that.
    chain = [p_cfg, p_step, p_mesh, p_lsd, p_pv, p_conf, p_cs, p_pp]
    for a, b in zip(chain, chain[1:]):
        x_from = a[0] + a[2] / 2
        x_to = b[0] - b[2] / 2
        arrow(ax, (x_from, y), (x_to, y))

    # ---- RVE band (below main flow) ------------------------------------------
    cluster(ax, 1.20, 0.50, 14.80, 3.00, "RVE  (parallel sub-element)")
    p_rve = image_node(ax, 3.00, 1.65, ICONS_DIR / "rve.png",
                       "RVE strand", w=1.55, h=1.30, cap_gap=0.25,
                       fs=FS_CAPTION)
    ax.text(8.90, 1.65,
            "Independent strand sub-element solve, run in parallel with the\n"
            "LS-DYNA cable simulation.  Produces effective moduli  "
            + r"$E_{xx},\ E_{yy}$" + "\nthat feed the Cablestack APDL stages.",
            ha="center", va="center", family=FONT, fontsize=FS_ANNOT,
            color="#1a1a1a", zorder=3)
    # Short dashed connector down from the Cablestack node into the RVE band.
    arrow(ax, (p_cs[0], p_cs[1] - p_cs[3] / 2 - 0.05),
          (p_cs[0], 3.00 + 0.05),
          ls="--", color=DASH_COLOR, lw=1.8)

    fig.savefig(OUTPUT_DIR / "slide_overview.png", dpi=200,
                bbox_inches="tight", pad_inches=0.18)
    fig.savefig(OUTPUT_DIR / "slide_overview.svg",
                bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR / 'slide_overview.png'}")
    print(f"wrote {OUTPUT_DIR / 'slide_overview.svg'}")


if __name__ == "__main__":
    main()
