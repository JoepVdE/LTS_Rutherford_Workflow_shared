"""Print per-loading-branch slopes for every HF case, GPS + PS.

Output is a table the user can scan: for each (formulation, case), one row
per loading branch (virgin or continued), with the strain range it spans,
the sigma range, and the least-squares slope in GPa. If the loading
envelope genuinely stiffens above 150 MPa, the slope numbers will jump
upward at the corresponding branch.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3].parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis" / "submodel" / "cablestack"))

from diagnostic_envelope_slope import (  # noqa: E402
    _read, _segment_branches, _branch_slope,
    SUFFIX, CASE_TITLE, CABLE, GPS_PP, PS_PP,
)


def report(label: str, pp_dir):
    print(f"\n=== {label} ===")
    for case, suf in SUFFIX.items():
        df = _read(pp_dir / f"{CABLE}{suf}_stress_strain.txt")
        if df is None:
            print(f"  [{case}]  (no data)")
            continue
        branches = _segment_branches(df)
        loading = [(k, e, s) for (k, e, s) in branches
                   if k in ("load_virgin", "load_continue")]
        print(f"  [{case}]  {CASE_TITLE[case]}")
        print(f"  {'branch':<16}{'strain %':<18}{'sigma MPa':<22}{'slope (GPa)':>12}")
        for kind, e, s in loading:
            srange = f"{e[0]:.3f} -> {e[-1]:.3f}"
            grange = f"{s[0]:.1f} -> {s[-1]:.1f}"
            slope = _branch_slope(e, s)
            print(f"  {kind:<16}{srange:<18}{grange:<22}{slope:>12.2f}")


def main():
    report("GPS (plane strain) - HF", GPS_PP)
    report("Plane stress - HF", PS_PP)


if __name__ == "__main__":
    main()
