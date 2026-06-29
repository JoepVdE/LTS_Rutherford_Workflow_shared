"""
Plot stress-strain / pressure-displacement curves from cablestack APDL postprocessing.

Handles two output types:
  7-PP.inp  -> fd_good_<usecase>.txt        (displacement-controlled)
  8-PP-pressure.inp -> fd_pressure_<usecase>.txt + uy_top_<usecase>.txt  (pressure loading)

Usage:
    python plot_fd_good.py                          # auto-detects latest run with fd_good file
    python plot_fd_good.py <run_folder>             # path to a specific run folder
    python plot_fd_good.py <apdl_runfolder>         # path directly to an apdl_runfolder

Output SVGs are saved to <apdl_runfolder>/plots/.
"""

import os
import re
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from analysis_utils import find_apdl_runfolder as _find_apdl_runfolder
from analysis_utils import iter_apdl_runfolders_latest_first as _iter_apdl_runfolders


def _parse_0start(apdl_runfolder: str) -> dict:
    """Extract x_cab, y_cab, n_stacks, usecase from 0-start.inp."""
    inp = os.path.join(apdl_runfolder, "0-start.inp")
    if not os.path.isfile(inp):
        raise FileNotFoundError(f"0-start.inp not found in {apdl_runfolder}")

    params = {}
    with open(inp, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("!"):
                continue
            m = re.match(r"^\s*x_cab\s*=\s*([0-9eE+\-.]+)", stripped)
            if m:
                params["x_cab"] = float(m.group(1))
            m = re.match(r"^\s*y_cab\s*=\s*([0-9eE+\-.]+)", stripped)
            if m:
                params["y_cab"] = float(m.group(1))
            m = re.match(r"^\s*n_stacks\s*=\s*(\d+)", stripped)
            if m:
                params["n_stacks"] = int(m.group(1))
            m = re.match(r"""\s*usecase\s*=\s*['"]([^'"]+)['"]""", stripped)
            if m:
                params["usecase"] = m.group(1)

    required = ["x_cab", "y_cab", "n_stacks", "usecase"]
    missing = [k for k in required if k not in params]
    if missing:
        raise ValueError(f"Could not parse {missing} from 0-start.inp")
    return params


def _read_fd(filepath: str) -> pd.DataFrame:
    """
    Parse fd_good_<usecase>.txt (or fd_temp_<usecase>.txt).

    Header line:  Set  Time (s)  UY (mm)  FY_total top (N)  UX (mm)  Fx_total left (N)
    Data columns: Set, Time, UY[m], FY[N], UX[m], FX[N]
    Note: ANSYS writes displacements in SI metres despite the header saying mm.
    """
    rows = []
    with open(filepath, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("Set"):
                continue
            parts = stripped.split()
            if len(parts) == 6:
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    continue

    df = pd.DataFrame(rows, columns=["Set", "Time_s", "UY_m", "FY_N", "UX_m", "FX_N"])
    # Prepend (0,0,0,0,0,0) origin
    origin = pd.DataFrame([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], columns=df.columns)
    df = pd.concat([origin, df], ignore_index=True)
    return df


def _compute_stress_strain(df: pd.DataFrame, x_cab: float, y_cab: float, n_stacks: int) -> pd.DataFrame:
    """
    Add stress and strain columns.

    Geometry (all SI):
      total_height = n_stacks * 2 * y_cab   [m]
      total_width  = 2 * x_cab              [m]

    Strain Y (compressive positive):
      strain_y = -UY_m / total_height

    Stress Y (plane-strain, unit thickness):
      stress_y_MPa = FY_N / (total_width * 1e6)

    Strain X:
      strain_x = UX_m / total_width

    Stress X:
      stress_x_MPa = FX_N / (total_height * 1e6)
    """
    total_height = n_stacks * 2.0 * y_cab   # m
    total_width  = 2.0 * x_cab              # m

    df = df.copy()
    df["UY_mm"] = df["UY_m"] * 1e3
    df["UX_mm"] = df["UX_m"] * 1e3

    df["strain_y"]    = -df["UY_m"]  / total_height
    df["stress_y_MPa"] = df["FY_N"]  / (total_width  * 1e6)

    df["strain_x"]    =  df["UX_m"]  / total_width
    df["stress_x_MPa"] = df["FX_N"]  / (total_height * 1e6)

    return df, total_height * 1e3, total_width * 1e3  # return dims in mm too


def _plot(df: pd.DataFrame, usecase: str, height_mm: float, width_mm: float,
          plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)
    is_pressure = "_pressure" in usecase

    # ---- 4-panel summary ----
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    if is_pressure:
        axs[0, 0].plot(df["UY_mm"], df["stress_y_MPa"], marker="o")
        axs[0, 0].set_xlabel("Displacement UY (mm)")
        axs[0, 0].set_ylabel("Applied Pressure (MPa)")
        axs[0, 0].set_title("Pressure vs Displacement (Y)")
    else:
        axs[0, 0].plot(-df["UY_mm"], df["FY_N"], marker="o")
        axs[0, 0].set_xlabel("Displacement UY (mm)")
        axs[0, 0].set_ylabel("Total Force FY (N)")
        axs[0, 0].set_title("Force vs Displacement (Y)")
    axs[0, 0].grid(True)

    if is_pressure:
        axs[0, 1].plot(df["strain_y"], df["stress_y_MPa"], marker="o")
        axs[0, 1].set_xlabel("Vertical Strain (-)")
        axs[0, 1].set_ylabel("Applied Pressure (MPa)")
        axs[0, 1].set_title("Pressure vs Strain (Y direction)")
    else:
        axs[0, 1].plot(df["strain_y"], df["stress_y_MPa"], marker="o")
        axs[0, 1].set_xlabel("Vertical Strain (-)")
        axs[0, 1].set_ylabel("Stress Y (MPa)")
        axs[0, 1].set_title("Stress-Strain (Y direction)")
    axs[0, 1].grid(True)

    axs[1, 0].plot(df["UX_mm"], -df["FX_N"], marker="o")
    axs[1, 0].set_xlabel("Displacement UX (mm)")
    axs[1, 0].set_ylabel("Total Force |FX| (N)")
    axs[1, 0].set_title("Force vs Displacement (X)")
    axs[1, 0].grid(True)

    axs[1, 1].plot(df["strain_x"], -df["stress_x_MPa"], marker="o")
    axs[1, 1].set_xlabel("Radial Strain (-)")
    axs[1, 1].set_ylabel("|Stress X| (MPa)")
    axs[1, 1].set_title("Stress-Strain (X direction)")
    axs[1, 1].grid(True)

    fig.suptitle(f"Cablestack — {usecase}  |  h={height_mm:.2f} mm, w={width_mm:.2f} mm",
                 fontsize=13)
    plt.tight_layout()
    out = os.path.join(plots_dir, f"fd_good_{usecase}_summary.svg")
    plt.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")

    # ---- primary curve (publication style) ----
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    if is_pressure:
        ax2.plot(df["UY_mm"], df["stress_y_MPa"], marker="o", color="steelblue")
        ax2.set_xlabel("Displacement UY (mm)", fontsize=14)
        ax2.set_ylabel("Applied Pressure (MPa)", fontsize=14)
        ax2.set_title(f"Pressure-Displacement — {usecase}", fontsize=14)
    else:
        ax2.plot(df["strain_y"], df["stress_y_MPa"], marker="o", color="steelblue")
        ax2.set_xlabel("Vertical Strain (-)", fontsize=14)
        ax2.set_ylabel("Applied Pressure (MPa)", fontsize=14)
        ax2.set_title(f"Stress-Strain — {usecase}", fontsize=14)
    ax2.tick_params(axis="both", labelsize=12)
    ax2.grid(True)
    plt.tight_layout()
    out2 = os.path.join(plots_dir, f"fd_good_{usecase}_stress_strain_y.svg")
    plt.savefig(out2)
    plt.close(fig2)
    print(f"  Saved: {out2}")


# ---------------------------------------------------------------------------
# Pressure postprocessing (8-PP-pressure.inp outputs)
# ---------------------------------------------------------------------------

def _read_fd_pressure(filepath: str) -> pd.DataFrame:
    """
    Parse fd_pressure_<usecase>.txt.

    Columns: LoadStep, SubStep, Time(s), FY_analytical(N/m), FX_reaction(N/m)
    """
    rows = []
    with open(filepath, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("LoadStep"):
                continue
            parts = stripped.split()
            if len(parts) == 5:
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    continue
    if not rows:
        return pd.DataFrame(columns=["LoadStep", "SubStep", "Time_s", "FY_N", "FX_N"])
    return pd.DataFrame(rows, columns=["LoadStep", "SubStep", "Time_s", "FY_N", "FX_N"])


def _read_uy_top(filepath: str) -> pd.DataFrame:
    """
    Parse uy_top_<usecase>.txt.  Returns mean UY per (LoadStep, SubStep).

    Input columns: LoadStep, SubStep, NodeID, UY(m)
    """
    rows = []
    with open(filepath, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("LoadStep"):
                continue
            parts = stripped.split()
            if len(parts) == 4:
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    continue
    if not rows:
        return pd.DataFrame(columns=["LoadStep", "SubStep", "mean_UY_m"])
    df = pd.DataFrame(rows, columns=["LoadStep", "SubStep", "NodeID", "UY_m"])
    mean_uy = df.groupby(["LoadStep", "SubStep"], as_index=False)["UY_m"].mean()
    mean_uy.rename(columns={"UY_m": "mean_UY_m"}, inplace=True)
    return mean_uy


def _compute_pressure_curves(fd_df: pd.DataFrame, uy_df: pd.DataFrame,
                              x_cab: float, y_cab: float, n_stacks: int):
    """
    Merge fd_pressure and uy_top data; add stress/strain columns.

    Geometry (SI):
      total_height = n_stacks * 2 * y_cab  [m]
      total_width  = 2 * x_cab             [m]

    Applied pressure (MPa) = FY_analytical / (total_width * 1e6)
    Vertical strain (compressive positive) = -mean_UY_m / total_height
    Radial reaction stress (MPa) = FX_reaction / (total_height * 1e6)
    """
    df = pd.merge(fd_df, uy_df, on=["LoadStep", "SubStep"], how="inner")

    total_height = n_stacks * 2.0 * y_cab
    total_width  = 2.0 * x_cab

    df["pressure_MPa"]   = df["FY_N"] / (total_width  * 1e6)
    df["mean_UY_mm"]     = df["mean_UY_m"] * 1e3
    df["strain_y"]       = -df["mean_UY_m"] / total_height
    df["stress_x_MPa"]   = df["FX_N"] / (total_height * 1e6)

    origin = pd.DataFrame([{
        "LoadStep": 0.0, "SubStep": 0.0, "Time_s": 0.0,
        "FY_N": 0.0, "FX_N": 0.0, "mean_UY_m": 0.0,
        "pressure_MPa": 0.0, "mean_UY_mm": 0.0,
        "strain_y": 0.0, "stress_x_MPa": 0.0,
    }])
    df = pd.concat([origin, df], ignore_index=True)
    return df, total_height * 1e3, total_width * 1e3


def _plot_pressure(df: pd.DataFrame, usecase: str, height_mm: float,
                   width_mm: float, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(-df["mean_UY_mm"], df["pressure_MPa"], marker="o", color="steelblue")
    axs[0].set_xlabel("Mean compaction |UY| (mm)")
    axs[0].set_ylabel("Applied pressure (MPa)")
    axs[0].set_title("Pressure vs Compaction")
    axs[0].grid(True)

    axs[1].plot(df["strain_y"], df["pressure_MPa"], marker="o", color="darkorange")
    axs[1].set_xlabel("Vertical strain (-)")
    axs[1].set_ylabel("Applied pressure (MPa)")
    axs[1].set_title("Pressure vs Vertical Strain")
    axs[1].grid(True)

    fig.suptitle(
        f"Pressure loading — {usecase}  |  h={height_mm:.2f} mm, w={width_mm:.2f} mm",
        fontsize=13)
    plt.tight_layout()
    out = os.path.join(plots_dir, f"fd_pressure_{usecase}_summary.svg")
    plt.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(df["strain_y"], df["pressure_MPa"], marker="o", color="darkorange")
    ax2.set_xlabel("Vertical strain (-)", fontsize=14)
    ax2.set_ylabel("Applied pressure (MPa)", fontsize=14)
    ax2.set_title(f"Pressure–Strain — {usecase}", fontsize=14)
    ax2.tick_params(axis="both", labelsize=12)
    ax2.grid(True)
    plt.tight_layout()
    out2 = os.path.join(plots_dir, f"fd_pressure_{usecase}_strain_y.svg")
    plt.savefig(out2)
    plt.close(fig2)
    print(f"  Saved: {out2}")


def _read_fd_radial(filepath: str) -> pd.DataFrame:
    """
    Parse fd_radial_<usecase>.txt.

    Columns: LoadStep, SubStep, Time(s), FX_analytical(N/m), FX_reaction(N/m)
    """
    rows = []
    with open(filepath, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("LoadStep"):
                continue
            parts = stripped.split()
            if len(parts) == 5:
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    continue
    if not rows:
        return pd.DataFrame(columns=["LoadStep", "SubStep", "Time_s", "FX_N", "FX_reaction_N"])
    return pd.DataFrame(rows, columns=["LoadStep", "SubStep", "Time_s", "FX_N", "FX_reaction_N"])


def _read_ux_left(filepath: str) -> pd.DataFrame:
    """
    Parse ux_left_<usecase>.txt.  Returns mean UX per (LoadStep, SubStep).

    Input columns: LoadStep, SubStep, NodeID, UX(m)
    """
    rows = []
    with open(filepath, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("LoadStep"):
                continue
            parts = stripped.split()
            if len(parts) == 4:
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    continue
    if not rows:
        return pd.DataFrame(columns=["LoadStep", "SubStep", "mean_UX_m"])
    df = pd.DataFrame(rows, columns=["LoadStep", "SubStep", "NodeID", "UX_m"])
    mean_ux = df.groupby(["LoadStep", "SubStep"], as_index=False)["UX_m"].mean()
    mean_ux.rename(columns={"UX_m": "mean_UX_m"}, inplace=True)
    return mean_ux


def _compute_radial_curves(fd_df: pd.DataFrame, ux_df: pd.DataFrame,
                             x_cab: float, y_cab: float, n_stacks: int):
    """
    Merge fd_radial and ux_left data; add stress/strain columns.

    Geometry (SI):
      total_height = n_stacks * 2 * y_cab  [m]
      total_width  = 2 * x_cab             [m]

    Applied radial pressure (MPa) = FX_analytical / (total_height * 1e6)
    Radial strain (compressive positive) = mean_UX_m / total_width
      (left wall moves in +X as cable is compressed from the left)
    """
    df = pd.merge(fd_df, ux_df, on=["LoadStep", "SubStep"], how="inner")

    total_height = n_stacks * 2.0 * y_cab
    total_width  = 2.0 * x_cab

    df["radial_pressure_MPa"] = df["FX_N"] / (total_height * 1e6)
    df["mean_UX_mm"]           = df["mean_UX_m"] * 1e3
    df["radial_strain"]       = df["mean_UX_m"] / total_width

    origin = pd.DataFrame([{
        "LoadStep": 0.0, "SubStep": 0.0, "Time_s": 0.0,
        "FX_N": 0.0, "FX_reaction_N": 0.0, "mean_UX_m": 0.0,
        "radial_pressure_MPa": 0.0, "mean_UX_mm": 0.0,
        "radial_strain": 0.0,
    }])
    df = pd.concat([origin, df], ignore_index=True)
    return df, total_height * 1e3, total_width * 1e3


def _plot_radial(df: pd.DataFrame, usecase: str, height_mm: float,
                  width_mm: float, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(df["mean_UX_mm"], df["radial_pressure_MPa"], marker="o", color="steelblue")
    axs[0].set_xlabel("Mean radial compaction UX (mm)")
    axs[0].set_ylabel("Applied radial pressure (MPa)")
    axs[0].set_title("Radial Pressure vs Compaction")
    axs[0].grid(True)

    axs[1].plot(df["radial_strain"], df["radial_pressure_MPa"], marker="o", color="darkorange")
    axs[1].set_xlabel("Radial strain (-)")
    axs[1].set_ylabel("Applied radial pressure (MPa)")
    axs[1].set_title("Radial Pressure vs Strain")
    axs[1].grid(True)

    fig.suptitle(
        f"Radial pressure loading — {usecase}  |  h={height_mm:.2f} mm, w={width_mm:.2f} mm",
        fontsize=13)
    plt.tight_layout()
    out = os.path.join(plots_dir, f"fd_radial_{usecase}_summary.svg")
    plt.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(df["radial_strain"], df["radial_pressure_MPa"], marker="o", color="darkorange")
    ax2.set_xlabel("Radial strain (-)", fontsize=14)
    ax2.set_ylabel("Applied radial pressure (MPa)", fontsize=14)
    ax2.set_title(f"Radial Pressure–Strain — {usecase}", fontsize=14)
    ax2.tick_params(axis="both", labelsize=12)
    ax2.grid(True)
    plt.tight_layout()
    out2 = os.path.join(plots_dir, f"fd_radial_{usecase}_strain_x.svg")
    plt.savefig(out2)
    plt.close(fig2)
    print(f"  Saved: {out2}")


def process_radial(apdl_runfolder: str, params: dict):
    """Process all fd_radial_*.txt + ux_left_*.txt files in apdl_runfolder."""
    x_cab    = params["x_cab"]
    y_cab    = params["y_cab"]
    n_stacks = params["n_stacks"]
    usecase  = params["usecase"]
    plots_dir = os.path.join(apdl_runfolder, "plots")

    fd_paths = sorted(glob.glob(os.path.join(apdl_runfolder, "fd_radial_*.txt")))
    if not fd_paths:
        return

    for fd_path in fd_paths:
        fname = os.path.basename(fd_path)
        file_usecase = re.sub(r'^fd_radial_', '', fname)
        file_usecase = re.sub(r'\.txt$', '', file_usecase)
        uc = file_usecase if file_usecase else usecase

        ux_path = os.path.join(apdl_runfolder, f"ux_left_{uc}.txt")
        if not os.path.isfile(ux_path):
            print(f"  Skipping radial plot for '{uc}': ux_left_{uc}.txt not found")
            continue

        print(f"Reading radial data: {fd_path}")
        fd_df = _read_fd_radial(fd_path)
        ux_df = _read_ux_left(ux_path)

        if fd_df.empty or ux_df.empty:
            print(f"  Skipping radial plot for '{uc}': empty data")
            continue

        if fd_df["FX_N"].abs().max() < 1.0:
            print(f"  Skipping radial plot for '{uc}': FX_analytical ~ 0")
            continue

        df, height_mm, width_mm = _compute_radial_curves(
            fd_df, ux_df, x_cab, y_cab, n_stacks)
        print(f"  Radial data points: {len(df) - 1}")
        print(f"  Max radial pressure: {df['radial_pressure_MPa'].max():.2f} MPa  "
              f"at strain {df.loc[df['radial_pressure_MPa'].idxmax(), 'radial_strain']:.5f}")
        _plot_radial(df, uc, height_mm, width_mm, plots_dir)


def process_pressure(apdl_runfolder: str, params: dict):
    """Process all fd_pressure_*.txt + uy_top_*.txt files in apdl_runfolder."""
    x_cab    = params["x_cab"]
    y_cab    = params["y_cab"]
    n_stacks = params["n_stacks"]
    usecase  = params["usecase"]
    plots_dir = os.path.join(apdl_runfolder, "plots")

    fd_paths = sorted(glob.glob(os.path.join(apdl_runfolder, "fd_pressure_*.txt")))
    if not fd_paths:
        return

    for fd_path in fd_paths:
        fname = os.path.basename(fd_path)
        file_usecase = re.sub(r'^fd_pressure_', '', fname)
        file_usecase = re.sub(r'\.txt$', '', file_usecase)
        uc = file_usecase if file_usecase else usecase

        uy_path = os.path.join(apdl_runfolder, f"uy_top_{uc}.txt")
        if not os.path.isfile(uy_path):
            print(f"  Skipping pressure plot for '{uc}': uy_top_{uc}.txt not found")
            continue

        print(f"Reading pressure data: {fd_path}")
        fd_df = _read_fd_pressure(fd_path)
        uy_df = _read_uy_top(uy_path)

        if fd_df.empty or uy_df.empty:
            print(f"  Skipping pressure plot for '{uc}': empty data")
            continue

        if fd_df["FY_N"].abs().max() < 1.0:
            print(f"  Skipping pressure plot for '{uc}': FY_analytical ~ 0 "
                  "(loading_pres not set in BC file?)")
            continue

        df, height_mm, width_mm = _compute_pressure_curves(
            fd_df, uy_df, x_cab, y_cab, n_stacks)
        print(f"  Pressure data points: {len(df) - 1}")
        print(f"  Max pressure: {df['pressure_MPa'].max():.2f} MPa  "
              f"at strain {df.loc[df['pressure_MPa'].idxmax(), 'strain_y']:.5f}")
        _plot_pressure(df, uc, height_mm, width_mm, plots_dir)


# ---------------------------------------------------------------------------
# Auto-detect latest run with an fd_good file
# ---------------------------------------------------------------------------

def _find_latest_fd_good() -> tuple[str, list[str]]:
    """Return (apdl_runfolder, [fd_good_filepath, ...]) for the most-recent run.

    Uses analysis_utils for the data/runs walk (the previous local copy
    resolved the repo root one level short and never found anything).
    """
    for apdl_rf in _iter_apdl_runfolders():
        if not os.path.isdir(apdl_rf):
            continue
        matches = glob.glob(os.path.join(apdl_rf, "pp", "fd_good_*.txt"))
        if not matches:
            matches = glob.glob(os.path.join(apdl_rf, "fd_good_*.txt"))
        if matches:
            return apdl_rf, sorted(matches)
    raise FileNotFoundError("No fd_good_*.txt found in any run folder under data/runs/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process(apdl_runfolder: str, fd_paths: list[str] | None = None):
    params = _parse_0start(apdl_runfolder)
    x_cab    = params["x_cab"]
    y_cab    = params["y_cab"]
    n_stacks = params["n_stacks"]
    usecase  = params["usecase"]

    if fd_paths is None:
        fd_paths = sorted(glob.glob(os.path.join(apdl_runfolder, "pp", "fd_good_*.txt")))
        if not fd_paths:
            fd_paths = sorted(glob.glob(os.path.join(apdl_runfolder, "fd_good_*.txt")))
        if not fd_paths:
            raise FileNotFoundError(
                f"No fd_good file found in {apdl_runfolder}.\n"
                "Run the APDL cablestack simulation (7-PP.inp) first."
            )

    plots_dir = os.path.join(apdl_runfolder, "plots")
    for fd_path in fd_paths:
        # derive usecase from filename: fd_good_<usecase>.txt
        fname = os.path.basename(fd_path)
        file_usecase = re.sub(r'^fd_good_', '', fname)
        file_usecase = re.sub(r'\.txt$', '', file_usecase)
        uc = file_usecase if file_usecase else usecase

        print(f"Reading: {fd_path}")
        print(f"  x_cab={x_cab*1e3:.3f} mm  y_cab={y_cab*1e3:.3f} mm  n_stacks={n_stacks}  usecase={uc}")

        df = _read_fd(fd_path)
        df, height_mm, width_mm = _compute_stress_strain(df, x_cab, y_cab, n_stacks)

        print(f"  Total height: {height_mm:.3f} mm,  Total width: {width_mm:.3f} mm")
        print(f"  Data points: {len(df)}")
        if len(df) > 1:
            print(f"  Max stress Y: {df['stress_y_MPa'].max():.2f} MPa  "
                  f"at strain {df.loc[df['stress_y_MPa'].idxmax(), 'strain_y']:.5f}")

        _plot(df, uc, height_mm, width_mm, plots_dir)

    # Also process pressure and radial outputs if present
    process_pressure(apdl_runfolder, params)
    process_radial(apdl_runfolder, params)


def main():
    if len(sys.argv) >= 2:
        path = sys.argv[1]
        try:
            apdl_rf = _find_apdl_runfolder(path)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        process(apdl_rf)
    else:
        try:
            apdl_rf, fd_paths = _find_latest_fd_good()
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        process(apdl_rf, fd_paths)


if __name__ == "__main__":
    main()
