"""
Diagnostic: visualise EVERY radial layer of StrandMesh_Hexa to expose the
layer-ordering bug in conformalRutherfordMesh.run.

What you should see:
  - Layers 1..8 are HEX-shaped, monotonically growing.
  - Layers 9, 10, 11 are CIRCULAR transition layers, monotonically SHRINKING.
  - Therefore the OUTER ring (layer 11) is geometrically INSIDE layer 8 — the
    layer that the keypoint writer (line 1781 of conformalRutherfordMesh.py)
    extracts as "middle".

The writer's assumption "middle ring is radially inside outer ring" is wrong
for the current StrandMesh_Hexa template. For D ~ 0.7 mm+ the post-map
deformation is large enough to mask this, but for D < 0.65 mm the inverted
layer ordering survives the mapping and produces a degenerate ASBA topology.

Output: data/runs/_diag_radial_layers.svg
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from strandMeshGenerator import StrandMesh_Hexa
from meshMapping import MeshMapping
from deformedStrandInterpolator import DeformedStrandInterpolator


def get_layers(nodes_arr: np.ndarray, n_inner_hex: int, cd: int, n_layers: int):
    out = []
    for k in range(n_layers):
        out.append(nodes_arr[n_inner_hex + k * cd: n_inner_hex + (k + 1) * cd])
    return out


def draw_panel(ax, nodes_arr, n_inner_hex, cd, n_layers, bspline=None,
               title="", note_writer_layers=True):
    layers = get_layers(nodes_arr, n_inner_hex, cd, n_layers)
    # Plot inner hex region as faint scatter
    inner_pts = nodes_arr[:n_inner_hex]
    ax.scatter(inner_pts[:, 0], inner_pts[:, 1], s=2, color="#cccccc",
               label="inner hex core")
    # Each circular layer with its index
    cmap = plt.get_cmap("turbo")
    for k, L in enumerate(layers):
        loop = np.vstack([L, L[:1]])
        c = cmap(k / max(1, n_layers - 1))
        lw = 1.4 if (note_writer_layers and k in (0, 7, 10)) else 0.7
        alpha = 1.0 if (note_writer_layers and k in (0, 7, 10)) else 0.6
        lbl = None
        if note_writer_layers:
            if k == 0:
                lbl = "layer 1 = writer 'inner'"
            elif k == 7:
                lbl = "layer 8 = writer 'middle'"
            elif k == 10:
                lbl = "layer 11 = writer 'outer'"
        ax.plot(loop[:, 0], loop[:, 1], color=c, lw=lw, alpha=alpha, label=lbl)
    if bspline is not None:
        ax.plot(np.r_[bspline[:, 0], bspline[0, 0]],
                np.r_[bspline[:, 1], bspline[0, 1]],
                "k--", lw=1.6, label="deformed strand B-spline")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)


def main():
    cd = 6
    mesh = StrandMesh_Hexa(diameter=0.5, radial_divisions=3, core_divisions=cd,
                           inner_circumradius_mm=0.1125,
                           outer_circumradius_mm=0.2964)
    circ_div = mesh.circumferential_divisions  # 36
    n_total = len(mesh.nodes)
    n_layers = 11
    n_inner_hex = n_total - n_layers * circ_div  # 91

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # --- panel A: raw, unmapped template ---
    raw_nodes = np.array(mesh.nodes)
    draw_panel(axes[0], raw_nodes, n_inner_hex, circ_div, n_layers,
               title=("RAW template (StrandMesh_Hexa, cd=6, TEST_A radii)\n"
                      "layer 8 (orange) ENCLOSES layer 11 (red): "
                      "writer's 'middle' is geometrically OUTSIDE 'outer'"))

    # --- panel B: same mesh, MAPPED to a real deformed TEST_A_NOM strand ---
    csv = (r"C:\LTS_Rutherford_Workflow\data\runs"
           r"\20260625_022453_TEST_A_thinmany_NOM_apdl_rerun_2"
           r"\stack\Stack_1_Part17.csv")
    s = DeformedStrandInterpolator(csv)
    s.fit_bspline()
    mapper = MeshMapping(mesh, s)
    mapper.translate_mesh_to_barycenter()
    mapper.map_circumferential_layer_to_bspline()
    bs = s.evaluate_bspline(num_points=400)
    draw_panel(axes[1], mapper.mapped_nodes, n_inner_hex, circ_div, n_layers,
               bspline=bs,
               title=("MAPPED to TEST_A_NOM strand17\n"
                      "layer 8 still pokes outside layer 11 -> ASBA fails"))

    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1),
                   fontsize=8, frameon=True)
    fig.suptitle(
        "StrandMesh_Hexa layer ordering: 'middle' (layer 8) vs 'outer' (layer 11). "
        "Each color = one circular layer; bold = writer-extracted layers.",
        fontsize=11)
    out = os.path.join(r"C:\LTS_Rutherford_Workflow\data\runs",
                       "_diag_radial_layers.svg")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
