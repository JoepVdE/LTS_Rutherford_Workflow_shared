"""
Render the actual FE mesh that MAPDL is solving for a given run folder.

Reads `plots/all_stacks_data.pkl` (saved by conformalRutherfordMesh.run during
the APDL submodel stage) and rebuilds the element connectivity for each
strand using StrandMesh_Hexa (connectivity-only — the deformed node positions
come from the pkl). Overlays the impregnation polygons too.

Usage:
  PYTHONIOENCODING=utf-8 \
    "C:/Program Files/Python312/python.exe" -X utf8 \
    scripts/d3plottoapdl_package/plot_running_mesh.py <run_folder>

Output: <run>/plots/_diag_full_fe_mesh.svg
"""
from __future__ import annotations

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from strandMeshGenerator import StrandMesh_Hexa


# TEST_A wire-derived inner / outer hex circumradii (matches what the workflow
# uses; the clamp inside StrandMesh_Hexa kicks in for D < 2*R_outer/0.9 = 0.66)
R_IN_MM_DEFAULT = 0.1125
R_OUT_MM_DEFAULT = 0.2964


def load_run(run_folder):
    pkl = os.path.join(run_folder, "APDL", "submodel", "apdl_runfolder",
                       "plots", "all_stacks_data.pkl")
    if not os.path.exists(pkl):
        raise FileNotFoundError(pkl)
    with open(pkl, "rb") as f:
        return pickle.load(f)


def cable_D_from_pkl_or_json(run_folder, data):
    # Try cable_parameters.json (next to metadata.json in run root)
    import json
    p = os.path.join(run_folder, "cable_parameters.json")
    if os.path.isfile(p):
        with open(p) as f:
            d = json.load(f)
        return float(d.get("D_Strand_base") or d.get("D_Strand") or 0.5)
    return 0.5


def main(run_folder: str):
    data = load_run(run_folder)
    D = cable_D_from_pkl_or_json(run_folder, data)
    label = data.get("cable_label", "?")
    stack_h_mm = float(data.get("stack_height_mm", 0.0))

    # Build a connectivity template once (same connectivity for all strands)
    # and reuse it. Element indices reference into the per-strand mapped_nodes.
    template = StrandMesh_Hexa(diameter=D, radial_divisions=3, core_divisions=6,
                               inner_circumradius_mm=R_IN_MM_DEFAULT,
                               outer_circumradius_mm=R_OUT_MM_DEFAULT)
    template_elements = template.elements
    n_template_nodes = len(template.nodes)

    n_stacks = max(data["stacks"].keys())
    fig, ax = plt.subplots(figsize=(min(20, 1.6 * 20), 1.6 * n_stacks),
                           dpi=140)

    for stack_nr, stack in data["stacks"].items():
        y_off = (stack_nr - 1) * stack_h_mm
        # --- strands ---
        for strand in stack["mappers"]:
            nodes = np.asarray(strand["nodes"]) * 1.0   # already mm
            if nodes.shape[0] != n_template_nodes:
                continue  # skip if shape mismatch
            polys = []
            for elem in template_elements:
                try:
                    pts = nodes[list(elem)]
                    polys.append(pts)
                except IndexError:
                    pass
            pc = PolyCollection(polys, facecolors="#fff5d6",
                                edgecolors="#b87333", linewidths=0.20)
            pc.set_offsets(np.array([[0.0, y_off]]))
            # PolyCollection.set_offsets doesn't shift each polygon individually;
            # apply offset manually by shifting node coords.
            pc_polys = [p + np.array([0.0, y_off]) for p in polys]
            ax.add_collection(PolyCollection(pc_polys, facecolors="#fff5d6",
                                             edgecolors="#b87333",
                                             linewidths=0.15, alpha=0.9))
        # --- impregnation polygon ---
        ins = stack.get("insulation")
        if ins:
            ins_arr = np.asarray(ins) * 1e3  # m -> mm
            xs = np.r_[ins_arr[:, 0], ins_arr[0, 0]]
            ys = np.r_[ins_arr[:, 1], ins_arr[0, 1]] + y_off
            ax.plot(xs, ys, "b-", lw=1.0, alpha=0.7)

    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"FE mesh as MAPDL sees it — {label}  "
                 f"(D={D} mm, {n_stacks} stacks, 40 strands/stack, "
                 f"copper-orange = element edges)")
    ax.autoscale_view()
    ax.grid(True, alpha=0.3)

    out_svg = os.path.join(run_folder, "APDL", "submodel", "apdl_runfolder",
                           "plots", "_diag_full_fe_mesh.svg")
    out_png = out_svg.replace(".svg", ".png")
    fig.tight_layout()
    fig.savefig(out_svg, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    print(f"Saved {out_svg}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
