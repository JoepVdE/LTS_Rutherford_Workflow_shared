"""Probe: does the loading-stiffness jump at the LoadStep 5 -> 6 transition?

For each pressure stage (cable HF, both formulations), prints:
  - last 4 substeps of LS5 with local secant slope
  - first 4 substeps of LS6 with local secant slope
  - secant across the step boundary

If the model is physically smooth, the slope just inside LS5 should match
the slope just inside LS6. If the step-boundary is introducing an artefact,
the slope will jump at TIME=5.

Reads pp/*_stress_strain.txt directly (already has strain_load + sigma_load_MPa).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3].parent
CABLE = "R2D2_HF"

RUNS = {
    "GPS pressure_transverse": REPO_ROOT / "data" / "runs" / "20260504_232855_R2D2_HF_apdl_rerun_51" / "APDL" / "submodel" / "apdl_runfolder" / "pp" / f"{CABLE}_pressure_stress_strain.txt",
    "GPS pressure_radial":     REPO_ROOT / "data" / "runs" / "20260504_232855_R2D2_HF_apdl_rerun_51" / "APDL" / "submodel" / "apdl_runfolder" / "pp" / f"{CABLE}_radial_stress_strain.txt",
    "PS  pressure_transverse": REPO_ROOT / "data" / "runs" / "20260504_232855_R2D2_HF_apdl_rerun_52_ps" / "APDL" / "submodel" / "apdl_runfolder" / "pp" / f"{CABLE}_pressure_stress_strain.txt",
    "PS  pressure_radial":     REPO_ROOT / "data" / "runs" / "20260504_232855_R2D2_HF_apdl_rerun_52_ps" / "APDL" / "submodel" / "apdl_runfolder" / "pp" / f"{CABLE}_radial_stress_strain.txt",
}


def _read(p: Path) -> pd.DataFrame:
    rows = []
    cols = None
    with p.open() as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if cols is None:
                cols = s.split()
                continue
            try:
                rows.append([float(x) for x in s.split()])
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=cols)


def secant_GPa(df, i, j):
    """secant slope dsigma_MPa / dstrain (dimensionless) -> GPa."""
    if i == j or i >= len(df) or j >= len(df) or i < 0 or j < 0:
        return float("nan")
    ds = df["sigma_load_MPa"].iloc[j] - df["sigma_load_MPa"].iloc[i]
    de = df["strain_load"].iloc[j] - df["strain_load"].iloc[i]
    if de == 0:
        return float("nan")
    return (ds / de) * 1e-3  # MPa per (1.0 strain) -> GPa


def main():
    for label, path in RUNS.items():
        print(f"\n=== {label}  [{path.parent.parent.parent.parent.parent.name}] ===")
        if not path.is_file():
            print("  (file missing)")
            continue
        df = _read(path)
        df["strain_load"] = df["strain_load"].abs()
        df["sigma_load_MPa"] = df["sigma_load_MPa"].abs()

        # Find indices in LoadStep 5 (last 4) and LoadStep 6 (first 4)
        is_ls5 = df["LoadStep"] == 5
        is_ls6 = df["LoadStep"] == 6
        ls5_idx = df.index[is_ls5].tolist()
        ls6_idx = df.index[is_ls6].tolist()
        if not ls5_idx or not ls6_idx:
            print("  (no LS5 or LS6 data)")
            continue

        print(f"  --- Last 4 substeps of LoadStep 5 ---")
        print(f"  {'SubStep':>8} {'Time':>10} {'strain':>14} {'sigma_MPa':>12} {'local_slope_GPa':>16}")
        for i in ls5_idx[-4:]:
            ss = int(df['SubStep'].iloc[i])
            t = df['Time'].iloc[i]
            e = df['strain_load'].iloc[i]
            g = df['sigma_load_MPa'].iloc[i]
            slope = secant_GPa(df, i-1, i) if i > 0 else float("nan")
            print(f"  {ss:>8} {t:>10.5f} {e:>14.6f} {g:>12.3f} {slope:>16.2f}")

        print(f"  >>> ACROSS BOUNDARY (LS5 end -> LS6 first sub):  "
              f"slope = {secant_GPa(df, ls5_idx[-1], ls6_idx[0]):.2f} GPa")

        print(f"  --- First 4 substeps of LoadStep 6 ---")
        print(f"  {'SubStep':>8} {'Time':>10} {'strain':>14} {'sigma_MPa':>12} {'local_slope_GPa':>16}")
        for i in ls6_idx[:4]:
            ss = int(df['SubStep'].iloc[i])
            t = df['Time'].iloc[i]
            e = df['strain_load'].iloc[i]
            g = df['sigma_load_MPa'].iloc[i]
            slope = secant_GPa(df, i-1, i)
            print(f"  {ss:>8} {t:>10.5f} {e:>14.6f} {g:>12.3f} {slope:>16.2f}")

        # Overall secants
        print(f"  Secant 125 MPa -> 150 MPa (LS5 tail): {secant_GPa(df, ls5_idx[-4], ls5_idx[-1]):.2f} GPa")
        print(f"  Secant 150 MPa -> 218 MPa (LS6 full): {secant_GPa(df, ls5_idx[-1], ls6_idx[-1]):.2f} GPa")


if __name__ == "__main__":
    main()
