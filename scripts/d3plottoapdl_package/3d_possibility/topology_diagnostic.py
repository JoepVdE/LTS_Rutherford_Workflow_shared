"""Topology-invariance diagnostic for 3D cablestack feasibility.

For each z-station (Stack_N) in a run's stack/ folder, build the per-strand
B-spline outline (DeformedStrandInterpolator) and run the same pairwise contact
rule used by conformalRutherfordMesh.identify_contact_region: distance <=
0.015 mm OR geometric penetration. The result is the contact graph G(z).

Comparing G across adjacent z-stations tells us whether the strand-strand
contact topology is stable enough along the cable axis to support a
prismatic-sweep 3D mesh (constant topology = clean sweep zone; transitions =
need a tet/imprint layer or a different approach).

Usage:
    & "C:/Program Files/Python312/python.exe" \
        scripts/d3plottoapdl_package/3d_possibility/topology_diagnostic.py \
        data/runs/<run_folder> [--threshold-mm 0.015] [--samples 200] \
        [--jaccard-warn 0.95]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from deformedStrandInterpolator import DeformedStrandInterpolator  # noqa: E402


def _discover_stack_layout(stack_dir: Path):
    pat = re.compile(r"^Stack_(\d+)_Part(\d+)\.csv$")
    by_stack: dict[int, set[int]] = defaultdict(set)
    for p in stack_dir.iterdir():
        m = pat.match(p.name)
        if m:
            by_stack[int(m.group(1))].add(int(m.group(2)))
    if not by_stack:
        raise SystemExit(f"No Stack_*_Part*.csv files found under {stack_dir}")
    stacks = sorted(by_stack)
    parts_union = sorted(set().union(*by_stack.values()))
    return stacks, parts_union, by_stack


def _bspline_for(csv_path: Path, samples: int) -> np.ndarray | None:
    try:
        interp = DeformedStrandInterpolator(str(csv_path))
        interp.fit_bspline()
        return interp.evaluate_bspline(num_points=samples)
    except Exception as exc:
        print(f"  [warn] {csv_path.name}: B-spline fit failed ({exc!s})")
        return None


def _pair_in_contact(b1: np.ndarray, b2: np.ndarray, threshold_mm: float) -> bool:
    # AABB pre-filter — most pairs are far apart; skip the KDTree build for those.
    if (b1[:, 0].max() + threshold_mm < b2[:, 0].min()
            or b2[:, 0].max() + threshold_mm < b1[:, 0].min()
            or b1[:, 1].max() + threshold_mm < b2[:, 1].min()
            or b2[:, 1].max() + threshold_mm < b1[:, 1].min()):
        return False
    tree2 = cKDTree(b2)
    if tree2.query(b1)[0].min() <= threshold_mm:
        return True
    return bool(MplPath(b2).contains_points(b1).any())


def _contact_graph(stack_dir: Path, stack_nr: int, parts: list[int],
                   samples: int, threshold_mm: float) -> tuple[set[frozenset], int]:
    bsplines: dict[int, np.ndarray] = {}
    for pid in parts:
        csv_path = stack_dir / f"Stack_{stack_nr}_Part{pid}.csv"
        if not csv_path.is_file():
            continue
        b = _bspline_for(csv_path, samples)
        if b is not None:
            bsplines[pid] = b
    edges: set[frozenset] = set()
    for i, j in combinations(sorted(bsplines), 2):
        if _pair_in_contact(bsplines[i], bsplines[j], threshold_mm):
            edges.add(frozenset({i, j}))
    return edges, len(bsplines)


def _fmt_edge(e: frozenset) -> str:
    a, b = sorted(e)
    return f"({a:>3d},{b:>3d})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_folder", type=Path,
                    help="Path to a data/runs/<run> folder containing a stack/ subdir")
    ap.add_argument("--threshold-mm", type=float, default=0.015,
                    help="Contact distance threshold in mm (default 0.015 = 15 um)")
    ap.add_argument("--samples", type=int, default=200,
                    help="B-spline samples per strand (default 200)")
    ap.add_argument("--jaccard-warn", type=float, default=0.95,
                    help="Flag transitions with Jaccard similarity below this")
    args = ap.parse_args(argv)

    run = args.run_folder.resolve()
    stack_dir = run / "stack"
    if not stack_dir.is_dir():
        raise SystemExit(f"No stack/ subdir found under {run}")

    out_dir = _HERE / run.name
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "report.txt"
    json_path = out_dir / "contact_graphs.json"

    stacks, parts, by_stack = _discover_stack_layout(stack_dir)
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit(f"[topology-diagnostic] run:     {run}")
    emit(f"[topology-diagnostic] stacks:  {len(stacks)} (numbers {stacks[0]}..{stacks[-1]})")
    emit(f"[topology-diagnostic] strands: {len(parts)} (parts {parts[0]}..{parts[-1]})")
    emit(f"[topology-diagnostic] rule:    dist <= {args.threshold_mm} mm OR penetration, "
         f"{args.samples} bspline samples")
    emit("")

    graphs: dict[int, set[frozenset]] = {}
    n_fit: dict[int, int] = {}
    for s in stacks:
        emit(f"[stack {s:>3d}] building contact graph...")
        graphs[s], n_fit[s] = _contact_graph(
            stack_dir, s, parts, args.samples, args.threshold_mm)

    # Stacks whose CSVs are empty/corrupt produce zero usable B-splines.
    # They must not poison persistence (intersect-with-empty = empty).
    valid_stacks = [s for s in stacks if n_fit[s] >= max(2, 0.5 * len(parts))]
    invalid_stacks = [s for s in stacks if s not in valid_stacks]

    emit("")
    emit("Per-stack contact counts:")
    for s in stacks:
        tag = "" if s in valid_stacks else "  [excluded: too few B-splines]"
        emit(f"  stack {s:>3d}: {len(graphs[s]):>4d} pairs"
             f"  ({n_fit[s]}/{len(parts)} strands fit){tag}")
    if invalid_stacks:
        emit(f"  -> persistence calc uses valid stacks only: {valid_stacks}")
    emit("")

    emit("Adjacent-stack transitions (valid stacks only):")
    n_unstable = 0
    for a, b in zip(valid_stacks[:-1], valid_stacks[1:]):
        ga, gb = graphs[a], graphs[b]
        same = ga & gb
        added = gb - ga
        removed = ga - gb
        union = ga | gb
        jacc = len(same) / len(union) if union else 1.0
        flag = ""
        if jacc < args.jaccard_warn:
            n_unstable += 1
            flag = "  *** topology change ***"
        emit(f"  {a:>3d} -> {b:<3d}: {len(same):>3d} same, "
             f"+{len(added)}/-{len(removed)}, jaccard={jacc:.3f}{flag}")
        for e in sorted(added, key=lambda x: tuple(sorted(x))):
            emit(f"      added:   {_fmt_edge(e)}")
        for e in sorted(removed, key=lambda x: tuple(sorted(x))):
            emit(f"      removed: {_fmt_edge(e)}")
    emit("")

    valid_graphs = [graphs[s] for s in valid_stacks]
    all_edges: set[frozenset] = set().union(*valid_graphs) if valid_graphs else set()
    persistent = {e for e in all_edges if all(e in g for g in valid_graphs)}
    intermittent = all_edges - persistent

    # Group intermittent edges into stable runs (constant-topology zones).
    # An edge present in stacks [a..b] contributes to that run.
    def _runs_of(present: list[int]) -> list[tuple[int, int]]:
        if not present:
            return []
        runs, lo = [], present[0]
        for x, y in zip(present, present[1:]):
            if y != x + 1:
                runs.append((lo, x)); lo = y
        runs.append((lo, present[-1]))
        return runs

    emit(f"Edge persistence summary (over {len(valid_stacks)} valid stacks):")
    emit(f"  unique edges observed: {len(all_edges)}")
    emit(f"  persistent (in every valid stack): {len(persistent)}")
    emit(f"  intermittent: {len(intermittent)}")
    for e in sorted(intermittent, key=lambda x: tuple(sorted(x))):
        present = [s for s in valid_stacks if e in graphs[s]]
        runs = _runs_of(present)
        runs_str = ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in runs)
        emit(f"    {_fmt_edge(e)} stacks {runs_str}")
    emit("")

    if n_unstable == 0:
        emit(f"VERDICT: all {len(valid_stacks) - 1} valid transitions stable "
             f"(jaccard >= {args.jaccard_warn}).")
        emit("        => clean prismatic sweep across all z is feasible.")
    else:
        emit(f"VERDICT: {n_unstable}/{len(valid_stacks) - 1} transitions violate jaccard >= "
             f"{args.jaccard_warn}.")
        emit(f"        => sweep within stable runs; insert tet/imprint layer at each transition.")
        if all_edges:
            lost = len(intermittent) / len(all_edges)
            emit(f"        => persistent-only graph would drop {len(intermittent)}/{len(all_edges)} "
                 f"edges ({lost:.0%}).")

    emit("")
    emit(f"Wrote: {report_path}")
    emit(f"Wrote: {json_path}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    json_payload = {
        "run_folder": str(run),
        "threshold_mm": args.threshold_mm,
        "samples": args.samples,
        "stacks": stacks,
        "valid_stacks": valid_stacks,
        "invalid_stacks": invalid_stacks,
        "parts": parts,
        "strands_fit_per_stack": {str(s): n_fit[s] for s in stacks},
        "graphs": {
            str(s): sorted(sorted(e) for e in graphs[s]) for s in stacks
        },
        "persistent_edges": sorted(sorted(e) for e in persistent),
        "intermittent_edges": sorted(sorted(e) for e in intermittent),
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
