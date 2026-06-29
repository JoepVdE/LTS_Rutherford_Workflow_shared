"""
Diagnostic: how does the strand-mesh circumferential resolution affect the
minimum keypoint distance after mapping to a deformed cable strand outline?

Tests the hypothesis that the D=0.5 mm cable's APDL Boolean failure comes
from too many circumferential nodes squeezed onto a small deformed B-spline.
Tries `core_divisions in {2,3,4,5,6,7,8}` (-> circumferential_divisions in
{12,18,24,30,36,42,48}) on the actual deformed strand from the run that
crashed (TEST_A_NOM, D=0.5 mm), measures min radial separation between the
middle and outer circular layers per angular position, and reports.

Run:
  PYTHONIOENCODING=utf-8 \
    "C:/Program Files/Python312/python.exe" -X utf8 \
    scripts/d3plottoapdl_package/diag_circ_div_sweep.py
"""
from __future__ import annotations

import os
import pickle
import sys
import math
import numpy as np

# Make sibling imports work when run directly
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from strandMeshGenerator import StrandMesh_Hexa
from meshMapping import MeshMapping


def load_real_strand(run_folder: str, stack: int, strand: int):
    """Reconstruct the deformed strand B-spline from the source Stack CSV."""
    csv = os.path.join(run_folder, "stack",
                       f"Stack_{stack}_Part{strand}.csv")
    if not os.path.exists(csv):
        return None
    from deformedStrandInterpolator import DeformedStrandInterpolator
    s = DeformedStrandInterpolator(csv)
    s.fit_bspline()
    return s


def synth_bspline_for_D(D_mm: float, eccent: float = 0.65):
    """Synthetic deformed strand: an ellipse, squashed in y, for a given D.

    For the test we just need a small smooth closed curve. eccent < 1 gives a
    lentil-shaped strand similar to a compacted Rutherford strand.
    """

    # MeshMapping calls deformed_strand.evaluate_bspline(num_points=500).
    # Build a tiny stub that returns a closed ellipse sampling at that count.
    class _SynthStrand:
        def __init__(self, rx, ry):
            self.rx, self.ry = rx, ry
        def evaluate_bspline(self, num_points=500):
            t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            return np.column_stack([self.rx * np.cos(t), self.ry * np.sin(t)])

    return _SynthStrand(rx=0.5 * D_mm / eccent, ry=0.5 * D_mm * eccent)


def measure_min_separation(mapped_nodes: np.ndarray, circ_div: int):
    """Mimic the layer extraction in conformalRutherfordMesh.run() (the
    keypoint writer) and measure the separation between the layers ANSYS
    actually receives.

    Production constants (hardcoded in conformalRutherfordMesh.run):
      nodes_square    = 91   (= inner hex node count, NOT scaled with cd)
      n_per_layer     = 36   (NOT scaled with cd)
      radial_layers_nb= 8    (-> writer's 'middle' = circular layer 8 of 11)

    For this diagnostic we adapt these to the active circ_div so the
    comparison is fair across cd:
      nodes_square_eff = num_total_nodes - n_circular_layers * circ_div
      The writer slices: inner=[nodes_square : nodes_square+circ_div],
                        middle=[nodes_square + 7*circ_div : nodes_square + 8*circ_div],
                        outer=[-circ_div:]
    """
    if mapped_nodes.shape[0] < 8 * circ_div:
        return float("nan"), float("nan"), float("nan")
    # Discover where the circular layers start (inner hex comes first)
    n_total = mapped_nodes.shape[0]
    # 11 circular layers in default workflow (radial_layers_nb=8 + radial_layers_cu=3).
    # Detect actual layer count from total node count.
    # The mesh generator always packs the hex core first, then radial circular layers.
    # For StrandMesh_Hexa(radial_divisions=3, core_divisions=cd):
    #   - inner hex node count differs by cd
    #   - circular layers always = 11 (verified empirically with quick probe)
    n_layers = 11
    nodes_square_eff = n_total - n_layers * circ_div
    outer_slice  = mapped_nodes[-circ_div:]
    middle_slice = mapped_nodes[nodes_square_eff + 7 * circ_div :
                                nodes_square_eff + 8 * circ_div]
    # min middle-vs-outer radial distance at same angular index
    min_mo = float("inf")
    for k in range(circ_div):
        d = math.hypot(middle_slice[k, 0] - outer_slice[k, 0],
                       middle_slice[k, 1] - outer_slice[k, 1])
        if d < min_mo:
            min_mo = d
    # min middle-vs-outer pairwise (any angular pair)
    min_pair = float("inf")
    for i in range(circ_div):
        for j in range(circ_div):
            d = math.hypot(middle_slice[i, 0] - outer_slice[j, 0],
                           middle_slice[i, 1] - outer_slice[j, 1])
            if d < min_pair:
                min_pair = d
    # min same-strand outer-vs-outer (catches the contact-zone duplicates)
    min_oo = float("inf")
    for i in range(circ_div):
        for j in range(i + 1, circ_div):
            d = math.hypot(outer_slice[i, 0] - outer_slice[j, 0],
                           outer_slice[i, 1] - outer_slice[j, 1])
            if d < min_oo:
                min_oo = d
    return min_pair, min_mo, min_oo


def run_pair_sweep(strand_a, strand_b, D_mm, label):
    """Build a conformal-pair mesh (the production path) at each cd and
    measure the min outer-vs-middle separation after rotate_outer_layer_nodes
    + align_nodes have done the contact-zone snapping."""
    from conformalRutherfordMesh import ConformalRutherfordMesh
    print(f"\n=== {label} (D={D_mm} mm; full pipeline incl. align) ===")
    header = (f"{'core_div':>9} {'circ_div':>9} "
              f"{'a_min_pair[um]':>15} {'a_min_mid_out[um]':>18} "
              f"{'b_min_pair[um]':>15} {'b_min_mid_out[um]':>18} "
              f"{'status':>10}")
    print(header)
    for cd in (2, 3, 4, 5, 6):
        try:
            mesh_a = StrandMesh_Hexa(diameter=D_mm, radial_divisions=3, core_divisions=cd)
            mesh_b = StrandMesh_Hexa(diameter=D_mm, radial_divisions=3, core_divisions=cd)
            mapper_a = MeshMapping(mesh_a, strand_a)
            mapper_b = MeshMapping(mesh_b, strand_b)
            mapper_a.translate_mesh_to_barycenter()
            mapper_a.map_circumferential_layer_to_bspline()
            mapper_b.translate_mesh_to_barycenter()
            mapper_b.map_circumferential_layer_to_bspline()
            crm = ConformalRutherfordMesh.from_existing(strand_a, mapper_a, strand_b, mapper_b)
            try:
                crm.rotate_outer_layer_nodes()
                status = "ok"
            except Exception as e:
                status = f"rot:{type(e).__name__}"
            min_a_pair, min_a_mo, _ = measure_min_separation(mapper_a.mapped_nodes, mesh_a.circumferential_divisions)
            min_b_pair, min_b_mo, _ = measure_min_separation(mapper_b.mapped_nodes, mesh_b.circumferential_divisions)
            print(f"{cd:>9} {mesh_a.circumferential_divisions:>9} "
                  f"{min_a_pair*1e3:>15.3f} {min_a_mo*1e3:>18.3f} "
                  f"{min_b_pair*1e3:>15.3f} {min_b_mo*1e3:>18.3f} "
                  f"{status:>10}")
        except Exception as e:
            print(f"{cd:>9} -- pair failed: {type(e).__name__}: {e}")


def run_sweep(D_mm: float, label: str, strand_obj=None):
    print(f"\n=== {label} (D={D_mm} mm; pre-align, single strand) ===")
    if strand_obj is None:
        strand_obj = synth_bspline_for_D(D_mm)
    for cd in (2, 3, 4, 5, 6):
        try:
            mesh = StrandMesh_Hexa(diameter=D_mm, radial_divisions=3,
                                   core_divisions=cd)
            mapper = MeshMapping(mesh, strand_obj)
            mapper.translate_mesh_to_barycenter()
            mapper.map_circumferential_layer_to_bspline()
            mp, mo, oo = measure_min_separation(mapper.mapped_nodes,
                                                mesh.circumferential_divisions)
            print(f"  cd={cd} circ={mesh.circumferential_divisions:2d}  "
                  f"mid-out(same-angle)={mo*1e3:6.2f}um  "
                  f"mid-out(any)={mp*1e3:6.2f}um  "
                  f"outer-outer={oo*1e3:6.2f}um")
        except Exception as e:
            print(f"  cd={cd} -- failed: {type(e).__name__}: {e}")


def run_full_stack_sweep(run_folder, stack_nr, D_mm, n_strands):
    """Simulate the production multi-pair loop: load all strands, iterate
    over connections_{stack_nr}.txt, run rotate_outer_layer_nodes per pair,
    then report worst-case middle/outer separation across all strands."""
    from conformalRutherfordMesh import ConformalRutherfordMesh
    conn_path = os.path.join(run_folder, "APDL", "submodel",
                             "apdl_runfolder", f"connections_{stack_nr}.txt")
    if not os.path.exists(conn_path):
        print(f"(no {conn_path})"); return
    connections = []
    with open(conn_path) as f:
        f.readline()  # header
        for line in f:
            a, b = [int(x) for x in line.strip().split(",")]
            connections.append((a, b))
    print(f"\n=== FULL stack {stack_nr} multi-pair sweep (D={D_mm}, "
          f"{len(connections)} connections, {n_strands} strands) ===")
    print(f"{'core_div':>9} {'worst_min_pair[um]':>20} "
          f"{'worst_min_mid_out[um]':>22} {'#strands_below_btol':>22}")
    BTOL_THRESH = 1.0  # um — separations <= this would crash ASBA at btol=1e-6
    for cd in (2, 3, 4, 5, 6):
        try:
            strands = [load_real_strand(run_folder, stack_nr, s)
                       for s in range(1, n_strands + 1)]
            if any(s is None for s in strands):
                print(f"{cd:>9} -- missing strand CSV")
                continue
            # Production uses wire-derived circumradii for TEST_A_thinmany:
            # D_CORE_EQ_UM=50, CU_SLEEVE_THICKNESS_UM=6.75, N_TOTAL=91, N_NB3SN=78
            # -> R_inner_mm = 0.1125, R_outer_mm = 0.2964
            R_in_mm = 0.1125
            R_out_mm = 0.2964
            meshes = [StrandMesh_Hexa(diameter=D_mm, radial_divisions=3, core_divisions=cd,
                                      inner_circumradius_mm=R_in_mm,
                                      outer_circumradius_mm=R_out_mm)
                      for _ in range(n_strands)]
            mappers = [MeshMapping(meshes[i], strands[i]) for i in range(n_strands)]
            for mp in mappers:
                mp.translate_mesh_to_barycenter()
                mp.map_circumferential_layer_to_bspline()
            # Production order: iterate connections, run rotate per pair
            for (a, b) in connections:
                if a > n_strands or b > n_strands:
                    continue
                try:
                    crm = ConformalRutherfordMesh.from_existing(
                        strands[a-1], mappers[a-1],
                        strands[b-1], mappers[b-1])
                    crm._pair_label = f"pair({a},{b})"
                    crm.rotate_outer_layer_nodes()
                except Exception as e:
                    pass  # production also keeps going on per-pair errors
            # Measure post-everything using the production layer slicing
            worst_pair = float("inf"); worst_mo = float("inf"); worst_oo = float("inf")
            below = 0
            for i in range(n_strands):
                mp, mo, oo = measure_min_separation(
                    mappers[i].mapped_nodes, meshes[i].circumferential_divisions)
                if mp < worst_pair: worst_pair = mp
                if mo < worst_mo:   worst_mo = mo
                if oo < worst_oo:   worst_oo = oo
                if mp * 1e3 <= BTOL_THRESH: below += 1
            print(f"{cd:>9}  pair={worst_pair*1e3:7.3f}um  mid-out(same-angle)={worst_mo*1e3:7.3f}um  "
                  f"outer-outer(same-strand)={worst_oo*1e3:7.3f}um  #below={below}")
        except Exception as e:
            print(f"{cd:>9} -- failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    import io as _io
    import contextlib as _cl
    run = (r"C:\LTS_Rutherford_Workflow\data\runs"
           r"\20260625_022453_TEST_A_thinmany_NOM_apdl_rerun_2")

    # Suppress the verbose per-pair print spam from align_nodes so the table is readable
    print("--- (suppressing align_nodes chatter; results below) ---")
    runs_to_test = [
        (run, 1, 0.5, 40, "TEST_A_NOM (D=0.5, failing case)"),
        (r"C:\LTS_Rutherford_Workflow\data\runs\20260625_035933_TEST_C_widethick_LIGHT",
         1, 1.0, 24, "TEST_C_LIGHT (D=1.0, working case)"),
        (r"C:\LTS_Rutherford_Workflow\data\runs\20260624_214844_TEST_B_medstd_NOM",
         1, 0.8, 28, "TEST_B_NOM (D=0.8, working case)"),
    ]
    for run_folder, stack, D, n_str, label in runs_to_test:
        print(f"\n--- {label} ---")
        if not os.path.exists(run_folder):
            print(f"  (run folder missing: {run_folder})")
            continue
        buf = _io.StringIO()
        with _cl.redirect_stdout(buf):
            run_full_stack_sweep(run_folder, stack_nr=stack, D_mm=D, n_strands=n_str)
        out = buf.getvalue()
        for line in out.splitlines():
            if line.startswith("===") or line.lstrip().startswith(("2 ", "3 ", "4 ", "5 ", "6 ")) or \
               line.strip().startswith("core_div") or line.strip().startswith("(") or \
               "missing" in line or "failed" in line:
                print(line)
