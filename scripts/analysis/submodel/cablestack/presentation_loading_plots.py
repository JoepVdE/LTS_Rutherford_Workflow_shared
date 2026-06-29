"""Presentation-ready stress-strain comparison plots, per cable.

For each cable in CABLES, emits up to 12 SVGs into tmp_presentation_plots/
covering three comparison axes:

  Set A  (4 SVGs) - plane-strain (GPS) vs plane-stress, one per loading config
  Set B  (4 SVGs) - pressure vs displacement, fixed (formulation, direction)
  Set C  (4 SVGs) - radial vs transverse,    fixed (formulation, control)

Data:
  R2D2_HF -- full fresh GPS + PS data, 12 SVGs.
  CD1     -- partial fresh GPS (DT + PT only); PS fallback for disp_transverse
             from OneDrive 'fd_good_CD1.txt' (single-ramp legacy PS).
             Panels with no data either side are skipped.

Run:
    python scripts/analysis/submodel/cablestack/presentation_loading_plots.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3].parent

OUT_DIR = REPO_ROOT / "tmp_presentation_plots"

OLD_ONEDRIVE = Path(r"C:/Users/vanden_j/OneDrive - ETH Zurich/Documents/ANSYS/0 - 10-stack/data analysis")

# Per-cable data sources. Each cable has:
#   gps_pp:        Path to new fresh-data pp/ for GPS (formulation=1)
#   ps_pp:         Path to new fresh-data pp/ for PS  (formulation=0), or None if no fresh PS
#   old_ps_fd_dt:  OneDrive raw fd_good_<cable>.txt (legacy PS, single-ramp). Used as fallback
#                  for disp_transverse PS when ps_pp is None or its pp/ is empty.
#   geometry_mm:   (total_width_mm, total_height_mm) for old-data unit conversions
#   has_ps_pressure_kink: True if the fresh PS pressure runs exhibit the LS5/LS6 numerical
#                  artifact (annotated on the corresponding plots).
CABLES = {
    "R2D2_HF": {
        "gps_pp": REPO_ROOT / "data" / "runs" / "20260504_232855_R2D2_HF_apdl_rerun_51" / "APDL" / "submodel" / "apdl_runfolder" / "pp",
        "ps_pp":  REPO_ROOT / "data" / "runs" / "20260504_232855_R2D2_HF_apdl_rerun_52_ps" / "APDL" / "submodel" / "apdl_runfolder" / "pp",
        "old_ps_fd_dt": None,
        "geometry_mm": (13.74, 14.16),
        "has_ps_pressure_kink": True,
    },
    "CD1": {
        "gps_pp": REPO_ROOT / "data" / "runs" / "20260504_232855_CD1_apdl_rerun_46" / "APDL" / "submodel" / "apdl_runfolder" / "pp",
        "ps_pp":  None,
        "old_ps_fd_dt": OLD_ONEDRIVE / "Data" / "fd_good_CD1.txt",
        "geometry_mm": (10.55, 19.20),  # cable_width + 2*margin (9.55+1.0), n_stacks*stack_height (10*1.92)
        "has_ps_pressure_kink": False,  # no fresh PS pressure data; OneDrive PS DT is single-ramp legacy
    },
}

# pp filename suffix per loading case
SUFFIX = {
    "disp_transverse":     "",
    "disp_radial":         "_disp_radial",
    "pressure_transverse": "_pressure",
    "pressure_radial":     "_radial",
}

CASE_TITLE = {
    "disp_transverse":     "Displacement-controlled, transverse (Y)",
    "disp_radial":         "Displacement-controlled, radial (X)",
    "pressure_transverse": "Pressure-controlled, transverse (Y)",
    "pressure_radial":     "Pressure-controlled, radial (X)",
}

# Presentation style
PRES_RC = {
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2.2,
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
}

# Colours: formulation-coded for Set A, loading-coded for Sets B/C
COL_GPS = "#1f77b4"    # blue
COL_PS  = "#d62728"    # red
COL_DISP = "#1f77b4"   # blue
COL_PRES = "#2ca02c"   # green
COL_TRANS = "#1f77b4"  # blue
COL_RAD   = "#ff7f0e"  # orange


def _read_stress_strain(path: Path) -> Optional[pd.DataFrame]:
    """Read a pp/<cable><suffix>_stress_strain.txt; return df[strain%, sigma_MPa]."""
    if not path.is_file():
        print(f"  [miss] {path.name}")
        return None
    rows = []
    strain_col = sigma_col = None
    with path.open() as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if strain_col is None:
                names = s.split()
                try:
                    strain_col = names.index("strain_load")
                    sigma_col = names.index("sigma_load_MPa")
                except ValueError:
                    return None
                continue
            try:
                parts = [float(x) for x in s.split()]
            except ValueError:
                continue
            if max(strain_col, sigma_col) < len(parts):
                rows.append([parts[strain_col], parts[sigma_col]])
    if len(rows) < 2:
        return None
    df = pd.DataFrame(rows, columns=["strain_load", "sigma_load_MPa"])
    # disp_radial conventionally produces negative values; standardise to magnitude
    df["strain_pct"] = np.abs(df["strain_load"]) * 100.0
    df["sigma_MPa"] = np.abs(df["sigma_load_MPa"])
    return df


def _load_case(pp_dir: Optional[Path], cable: str, case: str) -> Optional[pd.DataFrame]:
    """Load a fresh-pipeline pp/<cable><suffix>_stress_strain.txt for the case."""
    if pp_dir is None:
        return None
    suf = SUFFIX[case]
    return _read_stress_strain(pp_dir / f"{cable}{suf}_stress_strain.txt")


def _load_old_ps_dt(path: Optional[Path], geometry_mm: tuple[float, float]) -> Optional[pd.DataFrame]:
    """Load OneDrive raw fd_good_<cable>.txt as a PS disp_transverse curve.

    Format: 6 cols (Set, Time, UY[m], FY_total[N/m], UX[m], FX_total[N/m]).
    Despite the 'mm' label, UY/UX values are in metres (consistent with
    OLD analyse_pressure.py reader). Converts to (strain_pct, sigma_MPa)
    using geometry_mm = (total_width_mm, total_height_mm) and prepends an
    origin row so the curve passes through (0, 0).
    """
    if path is None or not path.is_file():
        return None
    rows = []
    with path.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 6:
                continue
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            rows.append(vals)
    if len(rows) < 2:
        return None
    df = pd.DataFrame(rows, columns=["Set", "Time", "UY", "FY_total", "UX", "FX_total"])
    total_width_m  = geometry_mm[0] * 1e-3
    total_height_m = geometry_mm[1] * 1e-3
    df["strain_load"] = -df["UY"] / total_height_m
    df["sigma_load_MPa"] = df["FY_total"] / (total_width_m * 1e6)
    origin = pd.DataFrame([{c: 0.0 for c in df.columns}])
    df = pd.concat([origin, df], ignore_index=True)
    df["strain_pct"] = np.abs(df["strain_load"]) * 100.0
    df["sigma_MPa"] = np.abs(df["sigma_load_MPa"])
    return df


def _ps_curve(cable_cfg: dict, case: str) -> tuple[Optional[pd.DataFrame], str]:
    """Return (df, source_label) for the PS data of this (cable, case).

    Prefers fresh pp/ data; falls back to OneDrive OLD fd_good_<cable>.txt
    for disp_transverse only (the only OLD file we have a parser for).
    Returns (None, '') if no PS source available.
    """
    fresh = _load_case(cable_cfg.get("ps_pp"), _CUR_CABLE, case)
    if fresh is not None:
        return fresh, "fresh"
    if case == "disp_transverse":
        old = _load_old_ps_dt(cable_cfg.get("old_ps_fd_dt"), cable_cfg["geometry_mm"])
        if old is not None:
            return old, "OneDrive legacy"
    return None, ""


def _annotate_ps_kink(ax, ps_df: pd.DataFrame):
    """Mark the LS5/LS6 numerical kink at ~150 MPa on a PS pressure curve.

    Draws a small arrow + text on the curve near sigma=150 MPa. Only call
    this for plots containing a PS pressure_transverse or pressure_radial
    curve (the kink is absent from PS displacement curves and from GPS).
    """
    diffs = (ps_df["sigma_MPa"] - 150.0).abs()
    idx = diffs.idxmin()
    x_at = float(ps_df["strain_pct"].iloc[idx])
    y_at = 150.0
    ax.annotate(
        "LS5/LS6 numerical artifact\n(PS-only; see backup slide)",
        xy=(x_at, y_at),
        xytext=(x_at + 0.4, y_at + 80),
        fontsize=9, color="black",
        ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                  ec="black", lw=0.7),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0,
                        connectionstyle="arc3,rad=-0.2"),
    )


# Module-level state set inside main(); used by _ps_curve to know which cable's
# fresh pp/ to search. (Cheaper than threading the cable name through every helper.)
_CUR_CABLE: str = ""


def _finalize(ax, title: str, out_path: Path, fig):
    ax.set_xlabel("Compressive strain along load axis (%)")
    ax.set_ylabel(r"Stress along load axis $\sigma_{\mathrm{load}}$ (MPa)")
    ax.set_title(title, pad=10)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"  -> {out_path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Set A: plane strain (GPS) vs plane stress, per loading case
# ---------------------------------------------------------------------------
def plot_set_A(cable: str, cfg: dict):
    print(f"\n[Set A | {cable}] plane-strain (GPS) vs plane-stress")
    for case in SUFFIX:
        gps = _load_case(cfg.get("gps_pp"), cable, case)
        ps, ps_source = _ps_curve(cfg, case)
        if gps is None and ps is None:
            print(f"  [skip] {case} - no data either side")
            continue

        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        if gps is not None:
            peak = gps["sigma_MPa"].max()
            ax.plot(gps["strain_pct"], gps["sigma_MPa"],
                    color=COL_GPS, lw=2.4, marker="o", ms=4.5,
                    label=f"Plane strain (GPS)  -  peak {peak:.0f} MPa")
        if ps is not None:
            peak = ps["sigma_MPa"].max()
            ps_lbl = "Plane stress"
            if ps_source == "OneDrive legacy":
                ps_lbl += " (legacy)"
            ax.plot(ps["strain_pct"], ps["sigma_MPa"],
                    color=COL_PS, lw=2.4, marker="s", ms=4.5,
                    linestyle="--",
                    label=f"{ps_lbl}  -  peak {peak:.0f} MPa")
            if (case in ("pressure_transverse", "pressure_radial")
                and cfg.get("has_ps_pressure_kink") and ps_source == "fresh"):
                _annotate_ps_kink(ax, ps)
        title = f"{cable}  -  {CASE_TITLE[case]}"
        out = OUT_DIR / f"loading_A_{cable}_{case}_PSvsGPS.svg"
        _finalize(ax, title, out, fig)


# ---------------------------------------------------------------------------
# Set B: pressure vs displacement, fixed formulation + direction
# ---------------------------------------------------------------------------
def plot_set_B(cable: str, cfg: dict):
    print(f"\n[Set B | {cable}] pressure vs displacement (per formulation, per direction)")
    combos = [
        ("GPS", "transverse", "disp_transverse",     "pressure_transverse"),
        ("GPS", "radial",     "disp_radial",         "pressure_radial"),
        ("PS",  "transverse", "disp_transverse",     "pressure_transverse"),
        ("PS",  "radial",     "disp_radial",         "pressure_radial"),
    ]
    for formul, direction, disp_case, pres_case in combos:
        if formul == "GPS":
            disp = _load_case(cfg.get("gps_pp"), cable, disp_case)
            pres = _load_case(cfg.get("gps_pp"), cable, pres_case)
            disp_source = pres_source = "fresh"
        else:
            disp, disp_source = _ps_curve(cfg, disp_case)
            pres, pres_source = _ps_curve(cfg, pres_case)

        if disp is None and pres is None:
            print(f"  [skip] {formul}-{direction} - no data either control")
            continue

        formul_title = "Plane strain (GPS)" if formul == "GPS" else "Plane stress"
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        if disp is not None:
            peak = disp["sigma_MPa"].max()
            d_lbl = "Displacement-controlled"
            if disp_source == "OneDrive legacy":
                d_lbl += " (legacy)"
            ax.plot(disp["strain_pct"], disp["sigma_MPa"],
                    color=COL_DISP, lw=2.4, marker="o", ms=4.5,
                    label=f"{d_lbl}  -  peak {peak:.0f} MPa")
        if pres is not None:
            peak = pres["sigma_MPa"].max()
            p_lbl = "Pressure-controlled"
            if pres_source == "OneDrive legacy":
                p_lbl += " (legacy)"
            ax.plot(pres["strain_pct"], pres["sigma_MPa"],
                    color=COL_PRES, lw=2.4, marker="s", ms=4.5,
                    linestyle="--",
                    label=f"{p_lbl}  -  peak {peak:.0f} MPa")
            if formul == "PS" and cfg.get("has_ps_pressure_kink") and pres_source == "fresh":
                _annotate_ps_kink(ax, pres)
        title = f"{cable}  -  {formul_title}, {direction} loading"
        out = OUT_DIR / f"loading_B_{cable}_{formul}_{direction}_PressVsDisp.svg"
        _finalize(ax, title, out, fig)


# ---------------------------------------------------------------------------
# Set C: radial vs transverse, fixed formulation + control method
# ---------------------------------------------------------------------------
def plot_set_C(cable: str, cfg: dict):
    print(f"\n[Set C | {cable}] radial vs transverse (per formulation, per control)")
    combos = [
        ("GPS", "displacement", "disp_transverse",     "disp_radial"),
        ("GPS", "pressure",     "pressure_transverse", "pressure_radial"),
        ("PS",  "displacement", "disp_transverse",     "disp_radial"),
        ("PS",  "pressure",     "pressure_transverse", "pressure_radial"),
    ]
    for formul, control, trans_case, rad_case in combos:
        if formul == "GPS":
            trans = _load_case(cfg.get("gps_pp"), cable, trans_case)
            rad   = _load_case(cfg.get("gps_pp"), cable, rad_case)
            trans_source = rad_source = "fresh"
        else:
            trans, trans_source = _ps_curve(cfg, trans_case)
            rad,   rad_source   = _ps_curve(cfg, rad_case)

        if trans is None and rad is None:
            print(f"  [skip] {formul}-{control} - no data either direction")
            continue

        formul_title = "Plane strain (GPS)" if formul == "GPS" else "Plane stress"
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        if trans is not None:
            peak = trans["sigma_MPa"].max()
            t_lbl = "Transverse (Y)"
            if trans_source == "OneDrive legacy":
                t_lbl += " (legacy)"
            ax.plot(trans["strain_pct"], trans["sigma_MPa"],
                    color=COL_TRANS, lw=2.4, marker="o", ms=4.5,
                    label=f"{t_lbl}  -  peak {peak:.0f} MPa")
        if rad is not None:
            peak = rad["sigma_MPa"].max()
            r_lbl = "Radial (X)"
            if rad_source == "OneDrive legacy":
                r_lbl += " (legacy)"
            ax.plot(rad["strain_pct"], rad["sigma_MPa"],
                    color=COL_RAD, lw=2.4, marker="s", ms=4.5,
                    linestyle="--",
                    label=f"{r_lbl}  -  peak {peak:.0f} MPa")
        if formul == "PS" and control == "pressure" and cfg.get("has_ps_pressure_kink"):
            kink_curve = rad if (rad is not None and rad_source == "fresh") else \
                         (trans if (trans is not None and trans_source == "fresh") else None)
            if kink_curve is not None:
                _annotate_ps_kink(ax, kink_curve)
        title = f"{cable}  -  {formul_title}, {control}-controlled"
        out = OUT_DIR / f"loading_C_{cable}_{formul}_{control}_RadVsTrans.svg"
        _finalize(ax, title, out, fig)


def main():
    global _CUR_CABLE
    plt.rcParams.update(PRES_RC)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[paths] out: {OUT_DIR.relative_to(REPO_ROOT)}")
    for cable, cfg in CABLES.items():
        _CUR_CABLE = cable
        print(f"\n========================================")
        print(f"  CABLE: {cable}")
        print(f"========================================")
        gps_pp = cfg.get("gps_pp")
        ps_pp = cfg.get("ps_pp")
        old_ps = cfg.get("old_ps_fd_dt")
        print(f"  GPS pp/: {gps_pp.relative_to(REPO_ROOT) if gps_pp else '(none)'}")
        print(f"  PS  pp/: {ps_pp.relative_to(REPO_ROOT) if ps_pp else '(none)'}")
        print(f"  OLD PS DT fallback: {old_ps if old_ps else '(none)'}")
        plot_set_A(cable, cfg)
        plot_set_B(cable, cfg)
        plot_set_C(cable, cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
