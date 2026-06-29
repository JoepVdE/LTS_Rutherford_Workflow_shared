"""Zoom slide: conformal node matching (d3plot -> APDL).

Three-act story across a 16:9 frame:
  1. Input  -- deformed strand outline extracted from the LS-DYNA d3plot
  2. Per-strand    -- B-spline fit, hex-template mesh, project to spline
  3. Per-pair      -- find contact region (kDTree + polygon penetration),
                      align nodes bidirectionally on the shared arc
  4. Output -- conformal mesh + APDL fragments
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
CLUSTER_FILL_A = "#dbeaf5"   # per-strand cluster
CLUSTER_FILL_B = "#f5e6d8"   # per-pair cluster (warmer)
CLUSTER_EDGE_A = "#6f8ea7"
CLUSTER_EDGE_B = "#c98a3a"
ARROW_COLOR = "#222222"

FONT = "Arial"
FS_TITLE = 32
FS_NODE = 19
FS_CAPTION = 18
FS_EDGE = 17
FS_CLUSTER = 19


def text_box(ax, x, y, label, w=1.6, h=1.10, fc=TEXT_FILL, ec=TEXT_EDGE,
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


def cluster(ax, x0, y0, x1, y1, title, fc=CLUSTER_FILL_A, ec=CLUSTER_EDGE_A):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.02,rounding_size=0.28",
        linewidth=1.1, facecolor=fc, edgecolor=ec,
        alpha=0.45, zorder=0,
    ))
    ax.text(x0 + 0.25, y1 - 0.22, title,
            ha="left", va="top", family=FONT, fontsize=FS_CLUSTER,
            color="#1a1a1a", style="italic", zorder=1)


def arrow(ax, p, q, color=ARROW_COLOR, ls="-", lw=2.4, label=None,
          label_xy=None, label_fs=FS_EDGE):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=22,
        linewidth=lw, color=color, linestyle=ls, zorder=4,
        shrinkA=3, shrinkB=3,
    ))
    if label:
        lx, ly = label_xy or ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.30)
        ax.text(lx, ly, label, ha="center", va="bottom",
                family=FONT, fontsize=label_fs, color="#1a1a1a",
                bbox=dict(facecolor="white", edgecolor="#bbbbbb",
                          boxstyle="round,pad=0.25", alpha=0.95), zorder=5)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(8, 8.55, "Conformal node matching",
            ha="center", va="center", family=FONT, fontsize=FS_TITLE,
            fontweight="bold", color="#1a1a1a")
    ax.text(8, 7.85, "from LS-DYNA deformed strands to APDL submodel mesh",
            ha="center", va="center", family=FONT, fontsize=FS_CAPTION,
            color="#555555", style="italic")

    # ---- main row (single horizontal flow) -----------------------------------
    y = 4.95
    y_img = 5.20        # images lifted slightly so captions align under text
    cap_y = 3.85

    p_inp = image_node(ax, 1.30, y_img, ICONS_DIR / "paraview.png",
                       "Deformed\nstrand", w=1.55, h=1.30,
                       cap_y=cap_y, fs=FS_CAPTION - 1)

    # Per-strand cluster -------------------------------------------------------
    cs_a_x0, cs_a_x1 = 2.65, 9.20
    cs_a_y0, cs_a_y1 = 3.10, 6.95
    cluster(ax, cs_a_x0, cs_a_y0, cs_a_x1, cs_a_y1, "Per strand",
            fc=CLUSTER_FILL_A, ec=CLUSTER_EDGE_A)
    p_sp = text_box(ax, 3.80, y, "B-spline\nfit",   w=1.50, h=1.10, fs=17)
    p_hx = text_box(ax, 5.65, y, "Hex\ntemplate",   w=1.45, h=1.10, fs=17)
    p_pr = text_box(ax, 7.75, y, "Project to\nspline", w=1.75, h=1.10, fs=17)

    # Per-pair cluster ---------------------------------------------------------
    cs_b_x0, cs_b_x1 = 9.85, 14.35
    cs_b_y0, cs_b_y1 = 3.10, 6.95
    cluster(ax, cs_b_x0, cs_b_y0, cs_b_x1, cs_b_y1, "Per adjacent pair",
            fc=CLUSTER_FILL_B, ec=CLUSTER_EDGE_B)
    p_cn = image_node(ax, 11.10, y_img, ICONS_DIR / "outer_nodes.png",
                      "Find\ncontact", w=1.75, h=1.30,
                      cap_y=cap_y, fs=FS_CAPTION - 1)
    p_al = image_node(ax, 13.20, y_img, ICONS_DIR / "align_mesh.png",
                      "Align\nnodes", w=1.75, h=1.10,
                      cap_y=cap_y, fs=FS_CAPTION - 1)

    p_out = image_node(ax, 15.05, y_img, ICONS_DIR / "conformal.png",
                       "Conformal\nmesh", w=1.65, h=1.10,
                       cap_y=cap_y, fs=FS_CAPTION - 1)

    # ---- arrows --------------------------------------------------------------
    # input -> per-strand
    arrow(ax, (p_inp[0] + p_inp[2] / 2, y), (p_sp[0] - p_sp[2] / 2, y))
    # within per-strand
    arrow(ax, (p_sp[0] + p_sp[2] / 2, y), (p_hx[0] - p_hx[2] / 2, y))
    arrow(ax, (p_hx[0] + p_hx[2] / 2, y), (p_pr[0] - p_pr[2] / 2, y))
    # per-strand -> per-pair, labelled
    arrow(ax, (p_pr[0] + p_pr[2] / 2, y), (p_cn[0] - p_cn[2] / 2, y),
          label="two strand\nmeshes",
          label_xy=((p_pr[0] + p_pr[2] / 2 + p_cn[0] - p_cn[2] / 2) / 2, y + 0.30))
    # within per-pair
    arrow(ax, (p_cn[0] + p_cn[2] / 2, y), (p_al[0] - p_al[2] / 2, y))
    # per-pair -> output
    arrow(ax, (p_al[0] + p_al[2] / 2, y), (p_out[0] - p_out[2] / 2, y))

    # ---- algorithmic side-notes (below the two captions) ---------------------
    ax.text(11.10, 2.30,
            "kDTree (15 $\\mu$m)\n$\\cup$ polygon",
            ha="center", va="top", family=FONT, fontsize=FS_EDGE - 2,
            color="#555555", style="italic")
    ax.text(13.20, 2.30,
            "bidirectional\nmarch",
            ha="center", va="top", family=FONT, fontsize=FS_EDGE - 2,
            color="#555555", style="italic")

    fig.savefig(OUTPUT_DIR / "slide_nodematching.png", dpi=200,
                bbox_inches="tight", pad_inches=0.18)
    fig.savefig(OUTPUT_DIR / "slide_nodematching.svg",
                bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR / 'slide_nodematching.png'}")
    print(f"wrote {OUTPUT_DIR / 'slide_nodematching.svg'}")


if __name__ == "__main__":
    main()
