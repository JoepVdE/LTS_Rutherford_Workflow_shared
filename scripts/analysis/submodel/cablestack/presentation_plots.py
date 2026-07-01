"""Presentation-ready SVGs of the conformal-mesh pipeline stages.

Loads a single LF strand pair (Stack 1, Part 1 + Part 2) from a completed
run folder and emits seven separate SVG files, one per pipeline stage, into
the temp output directory. No APDL, no Docker, no full pipeline.

Run:
    python scripts/analysis/submodel/cablestack/presentation_plots.py \
        [--run data/runs/<run-folder>] [--out tmp_presentation_plots]

Defaults: latest SMACC_LF run under data/runs, tmp_presentation_plots/ at repo root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

import alphashape
from shapely.geometry import Point

REPO_ROOT = Path(__file__).resolve().parents[3].parent
PKG_DIR = REPO_ROOT / "scripts" / "d3plottoapdl_package"
sys.path.insert(0, str(PKG_DIR))

from deformedStrandInterpolator import DeformedStrandInterpolator
from strandMeshGenerator import StrandMesh_Hexa
from meshMapping import MeshMapping
from conformalRutherfordMesh import (
    ConformalRutherfordMesh,
    compute_hex_circumradii_mm,
)


def _alpha_perimeter(nodes_xy: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Filter (N,2) point cloud to the alpha-shape boundary, sorted CCW.

    Mirrors DeformedStrandInterpolator.fit_bspline's perimeter extraction so
    plots that label points as 'perimeter' truly are. Returns (M, 2) with M<=N.
    """
    coords = np.asarray(nodes_xy)[:, :2]
    shape = alphashape.alphashape(coords, alpha)
    boundary = []
    for c in coords:
        p = Point(c)
        if shape.exterior.contains(p) or shape.exterior.touches(p):
            boundary.append(c)
    boundary = np.array(boundary)
    if boundary.size == 0:
        return boundary
    center = boundary.mean(axis=0)
    angles = np.arctan2(boundary[:, 1] - center[1], boundary[:, 0] - center[0])
    return boundary[np.argsort(angles)]


# ----------------------------------------------------------------------------
# Presentation style: white background, no grid, larger font, thicker lines
# ----------------------------------------------------------------------------
PRES_RC = {
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 1.8,
    "axes.linewidth": 1.2,
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
}

COPPER = "#b87333"
NB3SN  = "#c0c0c0"
RAW1   = "#1f77b4"   # strand 1 colour family (blue)
RAW2   = "#2ca02c"   # strand 2 colour family (green)
ACCENT = "#d62728"   # red for contact arc / highlights
# Light grey strand fill for whole-cable cross-section views (plots 7 + 8),
# where copper/Nb3Sn sub-element detail isn't the point.
STRAND_FILL = "#dadada"
STRAND_EDGE = "#5a5a5a"
INSULATION_FILL = "#fde8c4"


def _setup_axes(ax, title):
    ax.set_aspect("equal")
    ax.set_xlabel("x / mm")
    ax.set_ylabel("y / mm")
    ax.set_title(title, pad=10)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def _legend_below(ax_or_fig, handles=None, labels=None, ncol=None, y=-0.18):
    """Place a horizontal legend below the axis (or figure). Returns the Legend."""
    kwargs = dict(loc="upper center", frameon=True, framealpha=0.95,
                  borderaxespad=0.0)
    if hasattr(ax_or_fig, "transAxes"):
        kwargs["bbox_to_anchor"] = (0.5, y)
        if handles is None:
            handles, labels = ax_or_fig.get_legend_handles_labels()
    else:
        kwargs["bbox_to_anchor"] = (0.5, 0.02)
        if handles is None:
            raise ValueError("Figure-level legend needs explicit handles/labels")
    if ncol is None:
        ncol = max(1, min(4, len(handles)))
    return ax_or_fig.legend(handles, labels, ncol=ncol, **kwargs)


def _save(fig, out_path: Path):
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"[plot] wrote {out_path.relative_to(REPO_ROOT)}")


# ----------------------------------------------------------------------------
# Plot 1: raw deformed strand outline (CSV scatter only)
# ----------------------------------------------------------------------------
def plot_deformed_outline(strand: DeformedStrandInterpolator, out: Path, label: str):
    """Raw CSV points, but filtered to the alpha-shape boundary so 'perimeter'
    means perimeter even when the upstream ParaView extraction leaks interior
    nodes into the CSV."""
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    raw = strand.nodes[:, :2]
    perim = _alpha_perimeter(raw, alpha=1.0)
    ax.scatter(perim[:, 0], perim[:, 1], s=22, color=RAW1, alpha=0.95,
               edgecolor="black", linewidths=0.5,
               label=f"True perimeter nodes (alpha-shape filtered, n={len(perim)})")
    _setup_axes(ax, f"Deformed strand perimeter — {label}")
    _legend_below(ax)
    _save(fig, out)


# ----------------------------------------------------------------------------
# Plot 2: B-spline fit on top of raw perimeter
# ----------------------------------------------------------------------------
def plot_bspline_fit(strand: DeformedStrandInterpolator, out: Path, label: str):
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    raw = strand.nodes[:, :2]
    perim = _alpha_perimeter(raw, alpha=1.0)
    bspline = strand.evaluate_bspline(num_points=400)

    ax.scatter(perim[:, 0], perim[:, 1], s=16, color=RAW1, alpha=0.85,
               edgecolor="black", linewidths=0.4, label="Perimeter nodes")
    ax.plot(bspline[:, 0], bspline[:, 1], color=ACCENT,
            linewidth=2.0, linestyle="-", label="Periodic cubic B-spline")
    _setup_axes(ax, f"B-spline fit through perimeter — {label}")
    _legend_below(ax)
    _save(fig, out)


# ----------------------------------------------------------------------------
# Plot 3: undeformed hex template mesh (single strand)
# ----------------------------------------------------------------------------
def _draw_mesh_elements(ax, nodes_arr, elements, copper_to=72 + 36, silver_to=360):
    """Filled polygons coloured by element index range (copper / Nb3Sn / outer copper)."""
    for ei, elem in enumerate(elements):
        coords = np.array([nodes_arr[k] for k in elem])
        if ei < copper_to:
            fc = COPPER
        elif ei < silver_to:
            fc = NB3SN
        else:
            fc = COPPER
        poly = MplPolygon(coords, closed=True, facecolor=fc, edgecolor="black",
                          linewidth=0.25, alpha=1.0)
        ax.add_patch(poly)


def plot_hex_template(mesh: StrandMesh_Hexa, out: Path):
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    nodes_arr = np.array(mesh.nodes)
    _draw_mesh_elements(ax, nodes_arr, mesh.elements)

    x_vals, y_vals = nodes_arr[:, 0], nodes_arr[:, 1]
    margin = 0.06 * max(np.ptp(x_vals), np.ptp(y_vals))
    ax.set_xlim(x_vals.min() - margin, x_vals.max() + margin)
    ax.set_ylim(y_vals.min() - margin, y_vals.max() + margin)

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="none",
                   markerfacecolor=COPPER, markeredgecolor="black",
                   markersize=11, label="Copper"),
        plt.Line2D([0], [0], marker="s", color="none",
                   markerfacecolor=NB3SN, markeredgecolor="black",
                   markersize=11, label="Nb$_3$Sn sub-element ring"),
    ]
    _setup_axes(ax, "Undeformed hex template mesh (single strand)")
    _legend_below(ax, handles=legend_handles,
                  labels=[h.get_label() for h in legend_handles])
    _save(fig, out)


# ----------------------------------------------------------------------------
# Plot 4: mapped mesh (template -> deformed B-spline)
# ----------------------------------------------------------------------------
def plot_mapped_mesh(mapper: MeshMapping, strand: DeformedStrandInterpolator,
                     out: Path, label: str):
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    nodes_arr = mapper.mapped_nodes
    _draw_mesh_elements(ax, nodes_arr, mapper.mesh.elements)

    bspline = strand.evaluate_bspline(num_points=300)
    ax.plot(bspline[:, 0], bspline[:, 1], color=ACCENT, linewidth=1.6,
            linestyle="--", label="Target B-spline")

    # Mapped barycenter circle of undeformed strand for visual diff (template size)
    barycenter = np.mean(nodes_arr, axis=0)
    template_d = mapper.mesh.diameter
    circle = plt.Circle(barycenter, 0.5 * template_d, fill=False,
                        color="black", linestyle=":", linewidth=1.4,
                        label="Undeformed strand outline (template)")
    ax.add_patch(circle)

    legend_handles = [
        plt.Line2D([0], [0], color=ACCENT, lw=1.6, ls="--",
                   label="Target B-spline (deformed)"),
        plt.Line2D([0], [0], color="black", lw=1.4, ls=":",
                   label="Undeformed strand outline"),
        plt.Line2D([0], [0], marker="s", color="none",
                   markerfacecolor=COPPER, markeredgecolor="black",
                   markersize=11, label="Copper"),
        plt.Line2D([0], [0], marker="s", color="none",
                   markerfacecolor=NB3SN, markeredgecolor="black",
                   markersize=11, label="Nb$_3$Sn"),
    ]

    x_vals, y_vals = nodes_arr[:, 0], nodes_arr[:, 1]
    margin = 0.10 * max(np.ptp(x_vals), np.ptp(y_vals))
    ax.set_xlim(x_vals.min() - margin, x_vals.max() + margin)
    ax.set_ylim(y_vals.min() - margin, y_vals.max() + margin)

    _setup_axes(ax, f"Mapped sub-element mesh — {label}")
    _legend_below(ax, handles=legend_handles,
                  labels=[h.get_label() for h in legend_handles])
    _save(fig, out)


# ----------------------------------------------------------------------------
# Plot 5: two-strand pair with contact arc highlighted
# ----------------------------------------------------------------------------
def plot_contact_region(conformal: ConformalRutherfordMesh, out: Path):
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    bspline1 = conformal.strand1.evaluate_bspline(num_points=400)
    bspline2 = conformal.strand2.evaluate_bspline(num_points=400)

    ax.plot(bspline1[:, 0], bspline1[:, 1], color=RAW1, linewidth=2.0,
            label="Strand 1 B-spline")
    ax.plot(bspline2[:, 0], bspline2[:, 1], color=RAW2, linewidth=2.0,
            label="Strand 2 B-spline")

    # Identify and mark the contact arc (run on bspline1 samples at 200 points,
    # matching what identify_contact_region uses internally)
    from scipy.spatial import cKDTree
    from matplotlib.path import Path as MplPath
    b1 = conformal.strand1.evaluate_bspline(num_points=200)
    b2 = conformal.strand2.evaluate_bspline(num_points=200)
    tree2 = cKDTree(b2)
    dist_mask = tree2.query(b1)[0] <= 0.015
    inside_mask = MplPath(b2).contains_points(b1)
    contact_mask = dist_mask | inside_mask
    contact_pts = b1[contact_mask]

    if contact_pts.size:
        ax.scatter(contact_pts[:, 0], contact_pts[:, 1], s=40,
                   color=ACCENT, edgecolor="black", linewidths=0.4, zorder=3,
                   label=f"Contact arc samples (n={contact_pts.shape[0]}, threshold 15 um)")

        start_point, end_point = conformal.identify_contact_region()
        if start_point is not None:
            ax.scatter([start_point[0], end_point[0]],
                       [start_point[1], end_point[1]],
                       s=140, marker="*", color="gold",
                       edgecolor="black", linewidths=0.8, zorder=4,
                       label="Contact-arc endpoints")

    _setup_axes(ax, "Contact region between adjacent strands")
    _legend_below(ax)
    _save(fig, out)


# ----------------------------------------------------------------------------
# Plots 6 + 7: outer-ring nodes before / after align_nodes
# ----------------------------------------------------------------------------
def _outer_contact_indices(outer_ring, other_bspline, threshold_mm=0.015):
    """Indices on outer_ring within `threshold_mm` of other_bspline."""
    from scipy.spatial import cKDTree
    from matplotlib.path import Path as MplPath
    tree = cKDTree(other_bspline)
    dist = tree.query(outer_ring)[0]
    inside = MplPath(other_bspline).contains_points(outer_ring)
    return np.where((dist <= threshold_mm) | inside)[0]


def _zoom_window(points_list, pad_mm: float = 0.05):
    """Compute (xmin, xmax, ymin, ymax) covering all points + pad."""
    all_pts = np.vstack([p for p in points_list if p is not None and len(p)])
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    dx = xmax - xmin
    dy = ymax - ymin
    side = max(dx, dy) + 2 * pad_mm
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    return cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2


def _draw_outer_rings_panel(ax, outer1, outer2, bspline1, bspline2,
                            contact_idx1, contact_idx2):
    # Dashed B-spline references
    ax.plot(bspline1[:, 0], bspline1[:, 1], color=RAW1, linewidth=1.0,
            linestyle="--", alpha=0.55)
    ax.plot(bspline2[:, 0], bspline2[:, 1], color=RAW2, linewidth=1.0,
            linestyle="--", alpha=0.55)
    # Outlines through outer-ring nodes
    o1c = np.vstack([outer1, outer1[0]])
    o2c = np.vstack([outer2, outer2[0]])
    ax.plot(o1c[:, 0], o1c[:, 1], color=RAW1, linewidth=1.6)
    ax.plot(o2c[:, 0], o2c[:, 1], color=RAW2, linewidth=1.6)
    ax.scatter(outer1[:, 0], outer1[:, 1], s=32, color=RAW1,
               edgecolor="black", linewidths=0.5, zorder=3)
    ax.scatter(outer2[:, 0], outer2[:, 1], s=32, color=RAW2,
               edgecolor="black", linewidths=0.5, zorder=3)
    if len(contact_idx1):
        ax.scatter(outer1[contact_idx1, 0], outer1[contact_idx1, 1],
                   s=110, facecolor="none", edgecolor=ACCENT, linewidths=2.0,
                   zorder=4)
    if len(contact_idx2):
        ax.scatter(outer2[contact_idx2, 0], outer2[contact_idx2, 1],
                   s=110, facecolor="none", edgecolor="orange", linewidths=2.0,
                   zorder=4)


def plot_alignment_before_after(outer1_b, outer2_b, outer1_a, outer2_a,
                                bspline1, bspline2, out: Path,
                                pair_label: str):
    """Side-by-side BEFORE vs AFTER alignment, zoomed on the contact region."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0))

    idx1_b = _outer_contact_indices(outer1_b, bspline2)
    idx2_b = _outer_contact_indices(outer2_b, bspline1)
    idx1_a = _outer_contact_indices(outer1_a, bspline2)
    idx2_a = _outer_contact_indices(outer2_a, bspline1)

    # Single shared zoom window: union of all flagged contact nodes (before+after).
    contact_pts = []
    if len(idx1_b): contact_pts.append(outer1_b[idx1_b])
    if len(idx2_b): contact_pts.append(outer2_b[idx2_b])
    if len(idx1_a): contact_pts.append(outer1_a[idx1_a])
    if len(idx2_a): contact_pts.append(outer2_a[idx2_a])
    xmin, xmax, ymin, ymax = _zoom_window(contact_pts, pad_mm=0.06)

    _draw_outer_rings_panel(axes[0], outer1_b, outer2_b, bspline1, bspline2,
                             idx1_b, idx2_b)
    _draw_outer_rings_panel(axes[1], outer1_a, outer2_a, bspline1, bspline2,
                             idx1_a, idx2_a)

    for ax, sub in zip(axes, ("Before alignment", "After alignment")):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        _setup_axes(ax, sub)

    fig.suptitle(f"Outer-ring nodes around contact arc — {pair_label}",
                 fontsize=15, y=1.02)

    # One shared legend below both panels
    legend_handles = [
        plt.Line2D([0], [0], color=RAW1, lw=1.6, label="Strand 1 outer ring"),
        plt.Line2D([0], [0], color=RAW2, lw=1.6, label="Strand 2 outer ring"),
        plt.Line2D([0], [0], color=RAW1, lw=1.0, ls="--", label="Strand 1 B-spline"),
        plt.Line2D([0], [0], color=RAW2, lw=1.0, ls="--", label="Strand 2 B-spline"),
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="none", markeredgecolor=ACCENT,
                   markersize=12, markeredgewidth=2.0,
                   label="Contact node (strand 1)"),
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="none", markeredgecolor="orange",
                   markersize=12, markeredgewidth=2.0,
                   label="Contact node (strand 2)"),
    ]
    fig.legend(handles=legend_handles,
               labels=[h.get_label() for h in legend_handles],
               loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, frameon=True, framealpha=0.95)

    _save(fig, out)


# ----------------------------------------------------------------------------
# Plots 7 + 8: insulation wrap (single stack and stack-vs-stack comparison)
# ----------------------------------------------------------------------------
def _load_stack_strands(stack_dir: Path, n_parts: int, stack_nr: int):
    """Return list of B-spline outlines (mm) for every strand in a stack."""
    outlines = []
    for i in range(1, n_parts + 1):
        csv = stack_dir / f"Stack_{stack_nr}_Part{i}.csv"
        if not csv.is_file() or csv.stat().st_size == 0:
            continue
        s = DeformedStrandInterpolator(str(csv))
        s.fit_bspline()
        outlines.append(s.evaluate_bspline(num_points=200))
    return outlines


def _run_insulation_pipeline(apdl_runfolder: Path, stack_nr: int,
                             stack_height_m: float):
    """Run the same InsulationLayer pipeline the main flow uses, on the per-stack
    keypoints file. Returns the populated InsulationLayer instance."""
    from insulationlayer import InsulationLayer
    keypoints_file = apdl_runfolder / f"keypoints_nodes_{stack_nr}.txt"
    if not keypoints_file.is_file():
        raise FileNotFoundError(
            f"Need keypoints_nodes_{stack_nr}.txt under {apdl_runfolder} "
            "(re-run the APDL submodel stage if missing)."
        )
    # scale_polygon writes a side-effect PNG; divert it to a tmpdir.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ins = InsulationLayer(str(keypoints_file), stack_nr, plots_dir=Path(tmp))
        ins.read_keypoints()
        ins.generate_alpha_shape(alpha=500)
        ins.select_points_close_to_polygon(tolerance_distance=7.5e-6)
        ins.scale_polygon(
            offset_distance=100e-6, stack_height=stack_height_m,
            stack_nr=stack_nr, stacking=True,
        )
    plt.close("all")  # InsulationLayer leaves stale figures in the global state
    return ins


def _draw_insulation_panel(ax, strand_outlines, ins, center_y_shift_mm: float = 0.0):
    """Draw one cable cross-section (strands + insulation wrap) into ax.

    `center_y_shift_mm` translates the panel content vertically so we can
    plot two stacks in their natural stacking order without overlap.
    Returns the (xmin, xmax, ymin, ymax) covered by the drawing.
    """
    # Insulation polygons (m -> mm); shift y to centre at 0 for clean display.
    inner_xy = np.array(ins.outer_polygon.exterior.coords) * 1e3
    outer_xy = np.array(ins.filtered_outerpoints) * 1e3
    # Re-centre on requested y (cancels the keypoint y-offset of (stack_nr-1)*stack_height)
    cy_existing = 0.5 * (inner_xy[:, 1].min() + inner_xy[:, 1].max())
    dy = center_y_shift_mm - cy_existing
    inner_xy[:, 1] += dy
    outer_xy[:, 1] += dy

    inner_closed = np.vstack([inner_xy, inner_xy[0]])
    outer_closed = np.vstack([outer_xy, outer_xy[0]])

    # Strand outlines are already in their absolute mm positions (Stack_N CSV).
    # Apply the same dy shift so strands sit inside the recentred insulation.
    shifted_strands = [s + np.array([0.0, dy]) for s in strand_outlines]

    # Insulation fill = outer polygon minus inner polygon (paint inner white).
    ax.add_patch(MplPolygon(outer_closed, closed=True, facecolor=INSULATION_FILL,
                             edgecolor="none", alpha=0.95, zorder=0.5))
    ax.add_patch(MplPolygon(inner_closed, closed=True, facecolor="white",
                             edgecolor="none", zorder=0.6))

    # Light-grey strand fills
    for outline in shifted_strands:
        ax.add_patch(MplPolygon(outline, closed=True, facecolor=STRAND_FILL,
                                 edgecolor=STRAND_EDGE, linewidth=0.6,
                                 alpha=0.90, zorder=1.0))

    # Boundary curves on top
    ax.plot(inner_closed[:, 0], inner_closed[:, 1], color=ACCENT,
            linewidth=1.6, linestyle="--", zorder=2.0)
    ax.plot(outer_closed[:, 0], outer_closed[:, 1], color="black",
            linewidth=2.0, zorder=2.0)

    all_pts = np.vstack([outer_closed, *shifted_strands])
    return (all_pts[:, 0].min(), all_pts[:, 0].max(),
            all_pts[:, 1].min(), all_pts[:, 1].max())


def _insulation_legend_handles(thickness_um: float):
    return [
        MplPolygon([(0, 0)], closed=True, facecolor=STRAND_FILL,
                   edgecolor=STRAND_EDGE, alpha=0.90,
                   label="Strand cross-section"),
        MplPolygon([(0, 0)], closed=True, facecolor=INSULATION_FILL,
                   edgecolor="none", alpha=0.95,
                   label=f"Insulation layer ({thickness_um:.0f} um nominal)"),
        plt.Line2D([0], [0], color=ACCENT, lw=1.6, ls="--",
                   label="Inner boundary (alpha hull of strands)"),
        plt.Line2D([0], [0], color="black", lw=2.0,
                   label="Outer boundary (offset + clipped to stack box)"),
    ]


def plot_insulation_wrap(stack_dir: Path, apdl_runfolder: Path,
                         n_parts: int, stack_height_m: float, out: Path,
                         stack_nr: int = 1):
    """Single-stack insulation wrap (Plot 7)."""
    strands = _load_stack_strands(stack_dir, n_parts, stack_nr)
    ins = _run_insulation_pipeline(apdl_runfolder, stack_nr, stack_height_m)

    fig, ax = plt.subplots(figsize=(11.0, 4.5))
    xmin, xmax, ymin, ymax = _draw_insulation_panel(ax, strands, ins,
                                                     center_y_shift_mm=0.0)
    margin = 0.08
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)

    handles = _insulation_legend_handles(ins.insulation_thickness_m * 1e6)
    _setup_axes(ax, f"Insulation layer wrap — Stack {stack_nr}")
    _legend_below(ax, handles=handles,
                  labels=[h.get_label() for h in handles], ncol=2, y=-0.22)
    _save(fig, out)


def plot_stack1_vs_stack2(stack_dir: Path, apdl_runfolder: Path,
                          n_parts: int, stack_height_m: float, out: Path):
    """Side-by-side Stack 1 vs Stack 2 insulation wraps (Plot 8)."""
    strands1 = _load_stack_strands(stack_dir, n_parts, 1)
    strands2 = _load_stack_strands(stack_dir, n_parts, 2)
    ins1 = _run_insulation_pipeline(apdl_runfolder, 1, stack_height_m)
    ins2 = _run_insulation_pipeline(apdl_runfolder, 2, stack_height_m)

    # Strand CSVs are in the LS-DYNA physical frame (no stacking offset). The
    # InsulationLayer keypoints have (stack_nr-1)*stack_height baked into Y for
    # APDL stacking. Apply that same offset to the strand outlines so they sit
    # inside their matching insulation wrap. This is plot-8 specific -- the
    # generic _load_stack_strands stays untouched.
    stack_offset_mm = stack_height_m * 1e3
    strands2 = [s + np.array([0.0, stack_offset_mm]) for s in strands2]

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 6.5),
                              gridspec_kw={"hspace": 0.45})

    # Centre each panel at y=0 so visual comparison is fair (otherwise stack 2
    # would sit one stack-height above stack 1 and the panels would diverge).
    b1 = _draw_insulation_panel(axes[0], strands1, ins1, center_y_shift_mm=0.0)
    b2 = _draw_insulation_panel(axes[1], strands2, ins2, center_y_shift_mm=0.0)

    # Shared zoom: union of both panels, so widths/heights are directly comparable.
    xmin = min(b1[0], b2[0]); xmax = max(b1[1], b2[1])
    ymin = min(b1[2], b2[2]); ymax = max(b1[3], b2[3])
    margin = 0.08
    for ax, sub in zip(axes, ("Stack 1", "Stack 2")):
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)
        _setup_axes(ax, sub)

    fig.suptitle("Stack 1 vs Stack 2 — strand arrangement comparison",
                 fontsize=15, y=0.995)

    handles = _insulation_legend_handles(
        0.5 * (ins1.insulation_thickness_m + ins2.insulation_thickness_m) * 1e6
    )
    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc="upper center", bbox_to_anchor=(0.5, -0.01),
               ncol=2, frameon=True, framealpha=0.95)
    _save(fig, out)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def find_latest_run(runs_dir: Path, cable: str) -> Path:
    """Latest run folder for `cable` that has both extracted CSVs and per-stack
    keypoints files written -- both are required for the full plot set."""
    candidates = sorted(
        [p for p in runs_dir.glob(f"*{cable}*") if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No {cable} run folders under {runs_dir}")
    for c in candidates:
        if ((c / "stack" / "Stack_1_Part1.csv").is_file()
                and (c / "APDL" / "submodel" / "apdl_runfolder"
                     / "keypoints_nodes_1.txt").is_file()):
            return c
    raise FileNotFoundError(
        f"No {cable} run with both CSVs and keypoints_nodes_*.txt found.")


def build_pair(stack_dir: Path, wire: dict | None, diameter_mm: float,
               part_a: int = 1, part_b: int = 2):
    """Load strand pair, fit B-splines, build hex templates, run mapping.

    Returns (strand1, strand2, mapper1, mapper2, template_mesh).
    """
    R_in, R_out = compute_hex_circumradii_mm(wire)

    def _build_single(csv_path: Path, angle: float):
        strand = DeformedStrandInterpolator(str(csv_path))
        strand.fit_bspline()
        mesh = StrandMesh_Hexa(
            diameter=diameter_mm, radial_divisions=3, angle=angle,
            inner_circumradius_mm=R_in, outer_circumradius_mm=R_out,
        )
        mapper = MeshMapping(mesh, strand)
        mapper.translate_mesh_to_barycenter()
        mapper.map_circumferential_layer_to_bspline()
        return strand, mapper

    strand1, mapper1 = _build_single(stack_dir / f"Stack_1_Part{part_a}.csv", angle=0.0)
    strand2, mapper2 = _build_single(stack_dir / f"Stack_1_Part{part_b}.csv", angle=15.0)

    # Independent template mesh used solely for the "undeformed hex template" plot
    template_mesh = StrandMesh_Hexa(
        diameter=diameter_mm, radial_divisions=3, angle=0.0,
        inner_circumradius_mm=R_in, outer_circumradius_mm=R_out,
    )
    return strand1, strand2, mapper1, mapper2, template_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cable", default="SMACC_LF",
                        help="Cable name (SMACC_LF, SMACC_HF, CD1, ...). "
                             "Used to pick the latest run folder and to label "
                             "the output subdirectory.")
    parser.add_argument("--run", type=Path, default=None,
                        help="Explicit run folder (overrides --cable lookup).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: tmp_presentation_plots/<cable>).")
    parser.add_argument("--part-a", type=int, default=1)
    parser.add_argument("--part-b", type=int, default=2)
    args = parser.parse_args()

    run_dir = args.run or find_latest_run(REPO_ROOT / "data" / "runs", args.cable)
    out_dir = args.out or (REPO_ROOT / "tmp_presentation_plots" / args.cable)
    print(f"[run] using {run_dir.relative_to(REPO_ROOT)}")

    cable_params_path = run_dir / "cable_parameters.json"
    cable_params = json.loads(cable_params_path.read_text())
    diameter_mm = float(cable_params["D_Strand"])
    print(f"[run] D_Strand = {diameter_mm*1e3:.1f} um (geometry_correction applied)")

    # The per-run cable_parameters.json does NOT include the `wire` sub-element
    # block — we need to fetch it from the user-config file.
    user_cfg_path = REPO_ROOT / "scripts" / "main" / "cable_parameters_user.json"
    user_cfg = json.loads(user_cfg_path.read_text())
    cable_name = cable_params.get("cable_name", "SMACC_LF")
    wire = user_cfg["cables"][cable_name].get("wire")
    if wire is None:
        print("[run] WARNING: no wire block — falling back to legacy hex ratios")

    stack_dir = run_dir / "stack"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] output: {out_dir}")

    plt.rcParams.update(PRES_RC)

    strand1, strand2, mapper1, mapper2, template_mesh = build_pair(
        stack_dir, wire, diameter_mm,
        part_a=args.part_a, part_b=args.part_b,
    )
    pair_label = f"Stack 1 / Parts {args.part_a},{args.part_b}"

    # -- Plot 1: raw deformed outline (strand 1 used as representative)
    plot_deformed_outline(strand1, out_dir / "01_deformed_outline.svg",
                          label=f"Stack 1, Part {args.part_a}")

    # -- Plot 2: B-spline fit on the same strand
    plot_bspline_fit(strand1, out_dir / "02_bspline_fit.svg",
                     label=f"Stack 1, Part {args.part_a}")

    # -- Plot 3: undeformed hex template (independent mesh)
    plot_hex_template(template_mesh, out_dir / "03_hex_template.svg")

    # -- Plot 4: mapped mesh deformed onto the B-spline
    plot_mapped_mesh(mapper1, strand1, out_dir / "04_mapped_mesh.svg",
                     label=f"Stack 1, Part {args.part_a}")

    # ------------------------------------------------------------------
    # Capture outer-ring snapshot BEFORE align_nodes, then run alignment
    # ------------------------------------------------------------------
    CD = mapper1.mesh.circumferential_divisions
    outer1_before = mapper1.mapped_nodes[-CD:].copy()
    outer2_before = mapper2.mapped_nodes[-CD:].copy()

    conformal = ConformalRutherfordMesh.from_existing(
        strand1, mapper1, strand2, mapper2,
    )
    conformal._pair_label = f"presentation:({args.part_a},{args.part_b})"

    # -- Plot 5: contact region (uses pre-alignment B-splines, identical post-)
    plot_contact_region(conformal, out_dir / "05_contact_region.svg")

    # Now run the alignment in-place; capture after-state
    status = conformal.rotate_outer_layer_nodes()
    if status is not True:
        print(f"[warn] rotate_outer_layer_nodes returned {status!r} — no contact?")
    outer1_after = mapper1.mapped_nodes[-CD:].copy()
    outer2_after = mapper2.mapped_nodes[-CD:].copy()

    bspline1 = strand1.evaluate_bspline(num_points=300)
    bspline2 = strand2.evaluate_bspline(num_points=300)

    # -- Plot 6: side-by-side BEFORE / AFTER alignment, zoomed on contact region
    plot_alignment_before_after(
        outer1_before, outer2_before, outer1_after, outer2_after,
        bspline1, bspline2,
        out_dir / "06_alignment_before_after.svg",
        pair_label=pair_label,
    )

    # -- Plot 7: insulation layer wrapped around stack 1
    apdl_runfolder = run_dir / "APDL" / "submodel" / "apdl_runfolder"
    cable_name = cable_params.get("cable_name", "SMACC_LF")
    cable_cfg = user_cfg["cables"][cable_name]
    stack_height_m = cable_cfg.get("stack_height_mm",
                                   cable_cfg.get("cable_height", 1.31) + 0.3) * 1e-3
    n_parts = int(cable_params.get("N_Strands", 34))
    plot_insulation_wrap(
        stack_dir=stack_dir,
        apdl_runfolder=apdl_runfolder,
        n_parts=n_parts,
        stack_height_m=stack_height_m,
        out=out_dir / "07_insulation_wrap.svg",
        stack_nr=1,
    )

    # -- Plot 8: Stack 1 vs Stack 2 comparison
    plot_stack1_vs_stack2(
        stack_dir=stack_dir,
        apdl_runfolder=apdl_runfolder,
        n_parts=n_parts,
        stack_height_m=stack_height_m,
        out=out_dir / "08_stack1_vs_stack2.svg",
    )

    print(f"\nAll 8 SVGs written to: {out_dir}")


if __name__ == "__main__":
    main()
