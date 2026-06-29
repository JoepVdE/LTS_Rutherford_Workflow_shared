"""
Visualise how the mapped strand mesh changes with `core_divisions` (cd).

`cd` is the StrandMesh_Hexa constructor argument; `circumferential_divisions
= cd * 6`. Default cd=6 -> 36 nodes around the outer ring.

Produces a single side-by-side SVG for cd in {2, 3, 4, 5, 6} mapped onto a
real deformed TEST_A_NOM strand (D=0.5 mm) with the SAME wire-derived inner
and outer hex circumradii that production uses. The two failure-relevant
circular layers (the writer's "middle" = layer 8 of 11, and the outer ring)
are highlighted so you can see the radial gap shrink as cd grows.

Output: data/runs/_diag_circ_div_meshes.svg

Run:
  PYTHONIOENCODING=utf-8 \
    "C:/Program Files/Python312/python.exe" -X utf8 \
    scripts/d3plottoapdl_package/diag_circ_div_plot.py
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


# TEST_A wire-derived inner / outer hex circumradii (mm) -- see
# diag_circ_div_sweep.py for the derivation from the JSON wire block.
R_IN_MM = 0.1125
R_OUT_MM = 0.2964
D_MM = 0.5

RUN_FOLDER = (r"C:\LTS_Rutherford_Workflow\data\runs"
              r"\20260625_022453_TEST_A_thinmany_NOM_apdl_rerun_2")
STACK = 1
STRAND = 17  # arbitrary representative deformed strand


def load_strand(run_folder, stack, strand_id):
    csv = os.path.join(run_folder, "stack",
                       f"Stack_{stack}_Part{strand_id}.csv")
    if not os.path.exists(csv):
        return None
    s = DeformedStrandInterpolator(csv)
    s.fit_bspline()
    return s


def panel(ax, cd, strand):
    """Build mesh at the given cd, map it to `strand`, and draw it on ax."""
    mesh = StrandMesh_Hexa(diameter=D_MM, radial_divisions=3, core_divisions=cd,
                           inner_circumradius_mm=R_IN_MM,
                           outer_circumradius_mm=R_OUT_MM)
    mapper = MeshMapping(mesh, strand)
    mapper.translate_mesh_to_barycenter()
    mapper.map_circumferential_layer_to_bspline()

    nodes = mapper.mapped_nodes
    cd_circ = mesh.circumferential_divisions
    n_total = nodes.shape[0]
    n_layers = 11  # constant for radial_divisions=3 across cd in {2..6}
    nodes_square_eff = n_total - n_layers * cd_circ

    # --- draw all mesh elements (faint) ---
    for elem in mesh.elements:
        try:
            poly = [nodes[i] for i in elem] + [nodes[elem[0]]]
            poly = np.array(poly)
            ax.fill(poly[:, 0], poly[:, 1], facecolor="#fff4d6",
                    edgecolor="#999999", linewidth=0.25)
        except (IndexError, TypeError):
            pass

    # --- deformed strand B-spline ---
    bs = strand.evaluate_bspline(num_points=400)
    ax.plot(np.r_[bs[:, 0], bs[0, 0]], np.r_[bs[:, 1], bs[0, 1]],
            "k--", lw=1.0, label="deformed outline (B-spline)")

    # --- highlight the writer-relevant layers ---
    outer = nodes[-cd_circ:]
    middle = nodes[nodes_square_eff + 7 * cd_circ:
                   nodes_square_eff + 8 * cd_circ]
    inner = nodes[nodes_square_eff: nodes_square_eff + cd_circ]
    ax.plot(np.r_[outer[:, 0], outer[0, 0]],
            np.r_[outer[:, 1], outer[0, 1]],
            "r-", lw=1.4, marker="o", ms=3.0,
            label=f"outer ring ({cd_circ} pts)")
    ax.plot(np.r_[middle[:, 0], middle[0, 0]],
            np.r_[middle[:, 1], middle[0, 1]],
            color="#1f77b4", lw=1.0, marker="o", ms=2.5,
            label="writer 'middle' (layer 8)")
    ax.plot(np.r_[inner[:, 0], inner[0, 0]],
            np.r_[inner[:, 1], inner[0, 1]],
            color="#2ca02c", lw=0.7, marker=".", ms=1.5,
            label="writer 'inner' (layer 1)")

    # --- per-angle radial gap, worst-case annotation ---
    gaps = np.hypot(middle[:, 0] - outer[:, 0],
                    middle[:, 1] - outer[:, 1])
    k_worst = int(np.argmin(gaps))
    worst = float(gaps[k_worst])
    ax.plot([middle[k_worst, 0], outer[k_worst, 0]],
            [middle[k_worst, 1], outer[k_worst, 1]],
            "m-", lw=2.0, label=f"min mid-out = {worst*1e3:.2f} um")
    ax.plot(middle[k_worst, 0], middle[k_worst, 1], "m*", ms=10)

    ax.set_aspect("equal")
    ax.set_title(f"cd={cd}  (circ_div={cd_circ})\nworst mid-out = {worst*1e3:.2f} um")
    ax.grid(True, alpha=0.3)


def main():
    strand = load_strand(RUN_FOLDER, STACK, STRAND)
    if strand is None:
        print("Cannot find deformed strand CSV; aborting.")
        return

    cds = [2, 3, 4, 5, 6]
    fig, axes = plt.subplots(1, len(cds), figsize=(4.2 * len(cds), 4.4),
                             sharex=True, sharey=True)
    for ax, cd in zip(axes, cds):
        panel(ax, cd, strand)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1),
                    fontsize=8, frameon=True)
    fig.suptitle(f"Strand mesh on real TEST_A_NOM stack {STACK} strand {STRAND} "
                 f"(D={D_MM} mm), sweeping core_divisions", fontsize=11)
    out_path = os.path.join(r"C:\LTS_Rutherford_Workflow\data\runs",
                            "_diag_circ_div_meshes.svg")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
