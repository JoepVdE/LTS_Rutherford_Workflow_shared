"""Overlay stress-strain curves from multiple cable runs, one figure per loading case.

Reads <cable>_<suffix>_stress_strain.txt files (written by analyse_pressure.analyse)
from each provided apdl_runfolder. Files are 6-column whitespace-separated
(Set, Time, strain_load, sigma_load_MPa, strain_react, sigma_react_MPa) with a
4-line `#` header.

Output: comparison/<case>_comparison.svg under the first run's analysis_comparison
directory (configurable). Each plot overlays sigma_load_MPa vs strain_load for
every cable that has data for that case; cables with missing data are listed in
the figure title.

Run standalone:
    python compare_cables.py <out_dir> <apdl_runfolder> [<apdl_runfolder> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analysis_utils import cable_label_from_runfolder, read_stress_strain_curve


CASES: List[Tuple[str, str, str]] = [
    ("displacement_transverse", "",            "Y wall, displacement-controlled"),
    ("displacement_radial",     "_disp_radial", "X wall, displacement-controlled"),
    ("pressure_transverse",     "_pressure",   "Y wall, pressure-controlled"),
    ("pressure_radial",         "_radial",     "X wall, pressure-controlled"),
]


def _cable_label(apdl_runfolder: Path) -> str:
    """Pull '<CABLE>' out of '<run>/APDL/submodel/apdl_runfolder' path."""
    return cable_label_from_runfolder(apdl_runfolder)


def _read_curve(path: Path) -> Optional[np.ndarray]:
    """(n_rows, 2) array of [strain_load, sigma_load_MPa]; see analysis_utils."""
    return read_stress_strain_curve(path)


def compare(out_dir: Path, runfolders: List[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cable_data: Dict[str, Dict[str, np.ndarray]] = {}
    for rf in runfolders:
        cable = _cable_label(rf)
        cable_data[cable] = {}
        for case, suffix, _ in CASES:
            txt = rf / "pp" / f"{cable}{suffix}_stress_strain.txt"
            arr = _read_curve(txt)
            if arr is not None:
                cable_data[cable][case] = arr

    summary_rows = []
    for case, suffix, descr in CASES:
        fig, ax = plt.subplots(figsize=(8.5, 6.0))
        plotted = []
        missing = []
        for cable, curves in cable_data.items():
            if case in curves:
                arr = curves[case]
                strain = arr[:, 0] * 100.0  # percent
                stress = arr[:, 1]
                ax.plot(strain, stress, marker="o", markersize=3, linewidth=1.5,
                        label=f"{cable}  (n={len(arr)})")
                plotted.append(cable)
                summary_rows.append((case, cable, len(arr), float(np.abs(stress).max())))
            else:
                missing.append(cable)
        title = f"{case} - sigma_load vs strain_load\n{descr}"
        if missing:
            title += f"  [missing: {', '.join(missing)}]"
        ax.set_title(title)
        ax.set_xlabel("strain_load (%)")
        ax.set_ylabel("sigma_load (MPa)")
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "no data for any cable", ha="center", va="center",
                    transform=ax.transAxes)
        fig.tight_layout()
        out_path = out_dir / f"{case}_comparison.svg"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"-> {out_path}  (cables: {plotted or 'none'})")

    print("\nSummary (case, cable, n_points, peak_sigma_MPa):")
    for row in summary_rows:
        print(f"  {row[0]:<26} {row[1]:<10} {row[2]:>4}  {row[3]:>8.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    out = Path(sys.argv[1])
    runs = [Path(p) for p in sys.argv[2:]]
    compare(out, runs)
