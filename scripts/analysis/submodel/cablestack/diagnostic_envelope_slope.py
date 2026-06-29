"""Diagnostic: does the loading-branch slope change above 150 MPa?

Splits each pp/*_stress_strain.txt into:
  - LOADING branches (strain strictly increasing AND stress strictly increasing)
  - UNLOAD branches (both decreasing)
  - RELOAD branches (both increasing, but starting from an unload trough,
    until they reach the prior peak)

A reference line is fit through the LOADING branches only (least-squares,
forced through origin) and shown alongside the data. Numerical slopes per
individual loading branch are annotated. If the loading envelope above 150 MPa
genuinely stiffens, the per-branch slopes will jump upward at that threshold.

Emits one SVG per case into tmp_presentation_plots/diagnostics/.

Run:
    python scripts/analysis/submodel/cablestack/diagnostic_envelope_slope.py
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3].parent

CABLE = "R2D2_HF"
GPS_PP = REPO_ROOT / "data" / "runs" / "20260504_232855_R2D2_HF_apdl_rerun_51" / "APDL" / "submodel" / "apdl_runfolder" / "pp"
PS_PP  = REPO_ROOT / "data" / "runs" / "20260504_232855_R2D2_HF_apdl_rerun_52_ps" / "APDL" / "submodel" / "apdl_runfolder" / "pp"

OUT = REPO_ROOT / "tmp_presentation_plots" / "diagnostics"

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


def _read(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    rows, sc, gc = [], None, None
    with path.open() as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if sc is None:
                names = s.split()
                try:
                    sc = names.index("strain_load")
                    gc = names.index("sigma_load_MPa")
                except ValueError:
                    return None
                continue
            try:
                p = [float(x) for x in s.split()]
            except ValueError:
                continue
            if max(sc, gc) < len(p):
                rows.append([p[sc], p[gc]])
    if len(rows) < 2:
        return None
    df = pd.DataFrame(rows, columns=["strain", "sigma"])
    df["strain"] = np.abs(df["strain"]) * 100.0   # percent
    df["sigma"] = np.abs(df["sigma"])
    return df


def _segment_branches(df: pd.DataFrame) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Walk the curve and tag each segment as load/unload/reload.

    A segment ends when the sign of d_strain flips. The first ascending
    segment is 'load_virgin'; subsequent ascending segments are 'reload'
    until they exceed the running max strain, then 'load_continue'.
    Descending segments are 'unload'.
    """
    eps = df["strain"].values
    sig = df["sigma"].values
    branches = []
    i = 0
    max_eps_seen = 0.0
    while i < len(eps) - 1:
        sign = np.sign(eps[i+1] - eps[i])
        if sign == 0:
            i += 1
            continue
        j = i + 1
        while j < len(eps) - 1 and np.sign(eps[j+1] - eps[j]) == sign:
            j += 1
        seg_eps = eps[i:j+1]
        seg_sig = sig[i:j+1]
        if sign > 0:
            # ascending: split into reload (up to previous max) + load_continue (beyond it)
            if seg_eps[-1] <= max_eps_seen + 1e-12:
                kind = "reload"
                branches.append((kind, seg_eps, seg_sig))
            else:
                # Split point: where strain first exceeds previous max
                split_idx = np.searchsorted(seg_eps, max_eps_seen, side="right")
                if split_idx > 1:
                    branches.append(("reload", seg_eps[:split_idx], seg_sig[:split_idx]))
                # load branch
                load_eps = seg_eps[max(0, split_idx - 1):]
                load_sig = seg_sig[max(0, split_idx - 1):]
                kind = "load_virgin" if max_eps_seen == 0 else "load_continue"
                branches.append((kind, load_eps, load_sig))
                max_eps_seen = seg_eps[-1]
        else:
            branches.append(("unload", seg_eps, seg_sig))
        i = j
    return branches


def _branch_slope(eps_pct: np.ndarray, sig_mpa: np.ndarray) -> float:
    """Least-squares slope dsigma/d(strain) on the branch, returned in GPa.

    eps_pct is strain in percent, so dε/d(eps_pct) = 1e-2 → slope_GPa = (dσ_MPa / dε) * 1e-3
    """
    if len(eps_pct) < 2:
        return float("nan")
    # Fit sigma = m * eps_pct + c, with eps_pct in %; convert m (MPa per %) to GPa: m / 10
    A = np.vstack([eps_pct, np.ones_like(eps_pct)]).T
    m, c = np.linalg.lstsq(A, sig_mpa, rcond=None)[0]
    return m / 10.0   # MPa per (% strain) -> GPa (since 1% = 0.01)


def _plot_case(df: pd.DataFrame, label: str, title: str, out_path: Path):
    branches = _segment_branches(df)

    fig, ax = plt.subplots(figsize=(10.0, 7.0))

    # Faint full curve
    ax.plot(df["strain"], df["sigma"], color="lightgrey", lw=1.0, zorder=1)

    colors = {
        "load_virgin":   ("#1f77b4", "Virgin loading"),
        "load_continue": ("#9467bd", "Continued loading (post-cycle)"),
        "reload":        ("#2ca02c", "Elastic reload"),
        "unload":        ("#d62728", "Unload"),
    }
    seen = set()

    # Annotate one slope label per LOADING branch
    load_slopes = []
    for kind, eps_b, sig_b in branches:
        col, lbl = colors[kind]
        first = kind not in seen
        seen.add(kind)
        ax.plot(eps_b, sig_b, color=col, lw=2.2, marker="o", ms=4,
                label=lbl if first else None, zorder=3)
        if kind in ("load_virgin", "load_continue") and len(eps_b) >= 3:
            slope = _branch_slope(eps_b, sig_b)
            load_slopes.append((eps_b[-1], sig_b[-1], slope))
            mid_idx = len(eps_b) // 2
            ax.annotate(f"{slope:.1f} GPa",
                        xy=(eps_b[mid_idx], sig_b[mid_idx]),
                        xytext=(8, -16), textcoords="offset points",
                        fontsize=10, color=col,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, lw=0.6))

    # Reference line: virgin slope extended to peak
    virgin = [(e, s) for (k, e, s) in
              [("load_virgin", b[1], b[2]) for b in branches if b[0] == "load_virgin"]]
    if virgin:
        eps_v, sig_v = virgin[0]
        virgin_slope_GPa = _branch_slope(eps_v, sig_v)
        eps_max = df["strain"].max() * 1.02
        ax.plot([0, eps_max], [0, virgin_slope_GPa * 10 * eps_max],
                color="black", lw=1.2, ls="--", alpha=0.7,
                label=f"Virgin slope extrapolated ({virgin_slope_GPa:.1f} GPa)")

    # 150 MPa reference (the user's threshold)
    ax.axhline(150.0, color="grey", lw=1.0, ls=":", alpha=0.7)
    ax.text(df["strain"].max() * 0.98, 150 + 5,
            "150 MPa", fontsize=10, color="grey", ha="right")

    ax.set_xlabel("Compressive strain along load axis (%)")
    ax.set_ylabel(r"$\sigma_{\mathrm{load}}$ (MPa)")
    ax.set_title(f"{label} - {title}", pad=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)

    # Slope summary box (per-branch loading slopes)
    if load_slopes:
        txt = "Loading-branch slopes:\n"
        for e_end, s_end, slope in load_slopes:
            txt += f"  end ({e_end:.2f}%, {s_end:.0f} MPa) -> {slope:.1f} GPa\n"
        ax.text(0.02, 0.98, txt.strip(),
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="grey", lw=0.6))

    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"  -> {out_path.relative_to(REPO_ROOT)}")


def main():
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
        "legend.fontsize": 10, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })
    OUT.mkdir(parents=True, exist_ok=True)

    for formul_label, pp_dir in [("GPS (plane strain)", GPS_PP), ("Plane stress", PS_PP)]:
        print(f"\n[{formul_label}]")
        for case, suf in SUFFIX.items():
            df = _read(pp_dir / f"{CABLE}{suf}_stress_strain.txt")
            if df is None:
                print(f"  [miss] {case}")
                continue
            tag = "GPS" if "GPS" in formul_label else "PS"
            out = OUT / f"diag_{CABLE}_{tag}_{case}.svg"
            _plot_case(df, f"{CABLE} [{formul_label}]", CASE_TITLE[case], out)

    print("\nDone.")


if __name__ == "__main__":
    main()
