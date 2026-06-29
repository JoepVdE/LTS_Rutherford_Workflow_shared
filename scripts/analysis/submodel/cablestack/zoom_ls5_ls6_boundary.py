"""Zoom plot at LS5/LS6 boundary for PS and GPS pressure_radial, side-by-side.

Marks substep dots, colours LS5 (blue) and LS6 (red), draws local-secant
slope arrows between consecutive substeps, prints slope in GPa next to each
arrow. Title shows the boundary value (150 MPa).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3].parent
OUT = REPO_ROOT / "tmp_presentation_plots" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

GPS_FILE = REPO_ROOT / "data" / "runs" / "20260504_232855_R2D2_HF_apdl_rerun_51"    / "APDL" / "submodel" / "apdl_runfolder" / "pp" / "R2D2_HF_radial_stress_strain.txt"
PS_FILE  = REPO_ROOT / "data" / "runs" / "20260504_232855_R2D2_HF_apdl_rerun_52_ps" / "APDL" / "submodel" / "apdl_runfolder" / "pp" / "R2D2_HF_radial_stress_strain.txt"


def _read(p):
    rows, cols = [], None
    for line in p.read_text().splitlines():
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


def panel(ax, df, title):
    is5 = df["LoadStep"] == 5
    is6 = df["LoadStep"] == 6
    ls5 = df[is5].tail(4).copy()
    ls6 = df[is6].head(4).copy()
    # Convert to %
    ls5["e_pct"] = ls5["strain_load"].abs() * 100
    ls6["e_pct"] = ls6["strain_load"].abs() * 100
    ls5["s_MPa"] = ls5["sigma_load_MPa"].abs()
    ls6["s_MPa"] = ls6["sigma_load_MPa"].abs()

    ax.plot(ls5["e_pct"], ls5["s_MPa"], "o-", color="#1f77b4", ms=10, lw=2.0,
            label="LoadStep 5 (last 4 substeps)")
    ax.plot(ls6["e_pct"], ls6["s_MPa"], "s-", color="#d62728", ms=10, lw=2.0,
            label="LoadStep 6 (first 4 substeps)")
    # Boundary line
    ax.axhline(150.0, color="black", lw=1.0, ls=":", alpha=0.7)
    ax.text(ax.get_xlim()[0], 152, "boundary (150 MPa, t=5.0 s)",
            fontsize=9, color="black", va="bottom")

    # Local slope labels (between consecutive substeps within each LS, plus across boundary)
    all_pts = list(zip(ls5["e_pct"].values, ls5["s_MPa"].values)) + \
              list(zip(ls6["e_pct"].values, ls6["s_MPa"].values))
    for i in range(len(all_pts) - 1):
        e0, s0 = all_pts[i]
        e1, s1 = all_pts[i+1]
        slope_GPa = ((s1 - s0) / (e1 - e0)) * 0.1  # MPa per %strain -> GPa
        midx = 0.5 * (e0 + e1)
        midy = 0.5 * (s0 + s1)
        # Highlight the BOUNDARY interval in bold
        boundary = (i == len(ls5) - 1)
        color = "darkgreen" if boundary else "grey"
        weight = "bold" if boundary else "normal"
        ax.annotate(f"{slope_GPa:.1f} GPa",
                    xy=(midx, midy),
                    xytext=(12, -4), textcoords="offset points",
                    fontsize=10, color=color, fontweight=weight,
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc="lightyellow" if boundary else "white",
                              ec=color, lw=0.6))

    ax.set_title(title, pad=10)
    ax.set_xlabel("Compressive strain (%)")
    ax.set_ylabel("sigma_load (MPa)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)


def main():
    plt.rcParams.update({
        "font.size": 11, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    panel(axes[0], _read(GPS_FILE),
          "GPS pressure_radial - LS5/LS6 boundary at 150 MPa\n(continuous: slope ~36 GPa)")
    panel(axes[1], _read(PS_FILE),
          "PS pressure_radial - LS5/LS6 boundary at 150 MPa\n(KINK: slope jumps 14 -> 25 GPa)")
    out = OUT / "zoom_LS5_LS6_boundary_HF_radial.svg"
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    print(f"-> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
