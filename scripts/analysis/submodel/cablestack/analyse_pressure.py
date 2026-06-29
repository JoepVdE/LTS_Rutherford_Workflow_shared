"""
Cablestack postprocessing — one function per APDL stage, all reading from the
``apdl_runfolder`` (``<run>/APDL/submodel/apdl_runfolder/``).

Stage map (imported from scripts/main/cablestack_stages.py — single source):

  displacement_transverse : ``fd_good_<cable>.txt``                 [7-PP.inp]
  displacement_radial     : ``fd_good_<cable>_disp_radial.txt``     [7-PP.inp under disp-radial restart]
  pressure_transverse     : ``fd_pressure_<cable>_pressure.txt``    [8-PP-pressure.inp]
                            ``uy_top_<cable>_pressure.txt``
  pressure_radial         : ``fd_radial_<cable>_radial.txt``        [8-PP-radial.inp]
                            ``ux_left_<cable>_radial.txt``

Public entry points (called by WorkflowRunner.run_cablestack_postprocess):

  postprocess_displacement_transverse(apdl_runfolder)
  postprocess_displacement_radial    (apdl_runfolder)
  postprocess_pressure_transverse    (apdl_runfolder)
  postprocess_pressure_radial        (apdl_runfolder)
  analyse(apdl_runfolder)            — run every stage that has output present

Each returns True iff it produced output.  Missing files log a notice and
return False; nothing raises on the missing-input path.

Derived quantities (all SI, compressive strain positive):

  total_width   = 2 * x_cab_m                  [m]  from loading_cycle.json
  total_height  = n_stacks * 2 * y_cab         [m]  from 0-start.inp
                  (= n_stacks * stack_height_mm * 1e-3)

  Pressure stages — sigma is the nominal pressure interpolated from
  loading_cycle.json steps (the SFL-driven boundary's FY/FX FSUM is ~0 by
  construction); strain is from the mean UY/UX on the driven boundary nodes.

  Displacement stages — sigma is the FSUM reaction on the driven boundary
  divided by the cable's opposite half-perimeter (plane-strain, unit depth);
  strain is the imposed displacement / cable dimension.

SVGs land in ``<apdl_runfolder>/plots/`` and a ``<usecase>_stress_strain.txt``
export is written next to the APDL dumps.

CLI:
    python analyse_pressure.py                 # auto-detect latest run under data/runs/
    python analyse_pressure.py <run_folder>    # top-level run folder
    python analyse_pressure.py <apdl_runfolder>
"""

import os
import re
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Per-stage usecase suffix appended to the cable label.  Single-sourced from
# scripts/main/cablestack_stages.py (the same registry main.py runs the stages
# from), so the filename convention cannot drift between solve and postprocess.
_SCRIPTS_MAIN = Path(__file__).resolve().parents[3] / "main"
if str(_SCRIPTS_MAIN) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_MAIN))
from cablestack_stages import STAGE_USECASE_SUFFIX as _STAGE_USECASE_SUFFIX


# ── Run metadata detection (formulation + BC variant per stage) ──────────────
# Reads the deck files left in the apdl_runfolder so plot titles / console
# headers / text-export headers all surface "PS vs GPS" and "which 5-BC was
# applied". Without this you cannot tell two sibling runs apart from the SVGs.

_STAGE_DECK_CANDIDATES = {
    # New (00-restart-*) and old (0-start-* / 0-start.inp) deck naming.
    "displacement_transverse": ["00-restart-transverse.inp", "0-start.inp"],
    "displacement_radial":     ["00-restart-radial.inp", "0-start-radial.inp"],
    "pressure_transverse":     ["00-restart-pressure.inp", "0-start-pressure.inp"],
    "pressure_radial":         ["00-restart-pressure-radial.inp", "0-start-pressure-radial.inp"],
    "thermal_cooldown":        ["00-restart-thermal.inp", "0-start-thermal.inp"],
}

_FORM_SHORT = {
    "plane stress": "PS",
    "generalized plane strain": "GPS",
    "unknown": "?",
}


def _detect_formulation(apdl_runfolder):
    """Read `formulation = 0|1` from 0-start.inp. Pre-toggle decks default to GPS."""
    inp = os.path.join(apdl_runfolder, "0-start.inp")
    if not os.path.isfile(inp):
        return "unknown"
    try:
        with open(inp) as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("!"):
                    continue
                m = re.match(r"^\s*formulation\s*=\s*([01])\b", s)
                if m:
                    return "plane stress" if m.group(1) == "0" else "generalized plane strain"
    except OSError:
        pass
    return "generalized plane strain"


def _detect_stage_bc(apdl_runfolder, stage):
    """Return the 5-BC-<variant> name actually invoked by this stage's deck."""
    for fname in _STAGE_DECK_CANDIDATES.get(stage, ()):
        p = os.path.join(apdl_runfolder, fname)
        if not os.path.isfile(p):
            continue
        try:
            with open(p) as fh:
                for line in fh:
                    s = line.strip()
                    if s.startswith("!"):
                        continue
                    m = re.match(r"^/inp\s*,\s*5-BC-([^,\s]+)\s*,\s*inp\b", s, re.IGNORECASE)
                    if m:
                        return m.group(1)
        except OSError:
            continue
    return "unknown"


def _stage_tag(apdl_runfolder, stage):
    """Compact label like 'PS | BC=cyclic' for plot titles + console headers."""
    return f"{_FORM_SHORT[_detect_formulation(apdl_runfolder)]} | BC={_detect_stage_bc(apdl_runfolder, stage)}"


# ── Output-file resolution ────────────────────────────────────────────────────
# ANSYS writes postprocessing files into <apdl_runfolder>/pp/.  Older runs wrote
# them to <apdl_runfolder>/ directly — _pp_in() prefers pp/ but falls back to the
# runfolder root for back-compat.  _pp_out() always targets pp/.

def _pp_in(apdl_runfolder, filename):
    pp = os.path.join(apdl_runfolder, "pp", filename)
    if os.path.exists(pp):
        return pp
    return os.path.join(apdl_runfolder, filename)


def _pp_out(apdl_runfolder, filename):
    pp_dir = os.path.join(apdl_runfolder, "pp")
    os.makedirs(pp_dir, exist_ok=True)
    return os.path.join(pp_dir, filename)


# ── Run-folder / geometry resolution ──────────────────────────────────────────
# Canonical implementations live in analysis_utils; local names kept for the
# existing call sites.

from analysis_utils import find_apdl_runfolder as _find_apdl_runfolder
from analysis_utils import find_latest_apdl_runfolder as _find_latest_runfolder


def _parse_0start_ycab(apdl_runfolder):
    """y_cab [m] (half stack height incl. insulation) from 0-start.inp."""
    inp = os.path.join(apdl_runfolder, "0-start.inp")
    if not os.path.isfile(inp):
        return None
    with open(inp) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("!"):
                continue
            m = re.match(r"^\s*y_cab\s*=\s*([0-9eE+\-.]+)", s)
            if m:
                return float(m.group(1))
    return None


def _total_height_m(apdl_runfolder, n_stacks):
    """Total model height [m] = n_stacks * 2 * y_cab from 0-start.inp."""
    y_cab = _parse_0start_ycab(apdl_runfolder)
    if y_cab is None:
        raise ValueError(f"Could not read y_cab from 0-start.inp in {apdl_runfolder}")
    return n_stacks * 2.0 * y_cab


def _load_jd(apdl_runfolder):
    """Return parsed loading_cycle.json or None if absent."""
    json_path = os.path.join(apdl_runfolder, "loading_cycle.json")
    if not os.path.isfile(json_path):
        print(f"[postprocess] loading_cycle.json not found in {apdl_runfolder}; nothing to do.")
        return None
    with open(json_path) as f:
        return json.load(f)


def _stage_usecase(jd, stage):
    """<cable_label><suffix> for the requested stage."""
    cable = jd["cable"]
    return cable + _STAGE_USECASE_SUFFIX[stage]


# ── Load-schedule helpers ─────────────────────────────────────────────────────

def nominal_pressure_mpa(t, steps):
    """
    Nominal applied pressure [MPa] at APDL time t.

    APDL load step i runs TIME from (i-1) to i.  Within that step pressure
    ramps linearly from steps[i-2].pressure_Pa (0 for i=1) to
    steps[i-1].pressure_Pa.
    """
    if t <= 0.0:
        return 0.0
    ls = int(np.ceil(t))                      # 1-based load step number
    ls = max(1, min(ls, len(steps)))
    p_end   = steps[ls - 1]['pressure_Pa']
    p_start = steps[ls - 2]['pressure_Pa'] if ls > 1 else 0.0
    t_start = float(ls - 1)
    return (p_start + (p_end - p_start) * (t - t_start)) / 1e6


# ── File readers ──────────────────────────────────────────────────────────────
# Generic table parsing lives in analysis_utils (shared with plot_fd_good and
# the ad-hoc comparison scripts); the local names are kept as thin wrappers.

from analysis_utils import read_float_table as _read_float_table
from analysis_utils import read_node_mean as _read_node_mean


def _read_5col(filepath, columns):
    """Generic reader for the 5-column fd_*.txt dumps (header rows are skipped)."""
    return _read_float_table(filepath, 5, columns)


def _read_fd_good(filepath, total_width, total_height):
    """
    fd_good_<usecase>.txt (6-col, 7-PP.inp) → DataFrame with epsilon_y, sigma_y_MPa,
    epsilon_x, sigma_x_MPa added.  UY/UX are metres; FY_total/FX_total are N/m depth.
    """
    df = _read_float_table(filepath, 6, ['Set', 'Time', 'UY', 'FY_total', 'UX', 'FX_total'])
    df['epsilon_y'] = -df['UY'] / total_height
    df['sigma_y_MPa'] = df['FY_total'] / (total_width * 1e6)
    df['epsilon_x'] =  df['UX'] / total_width
    df['sigma_x_MPa'] = df['FX_total'] / (total_height * 1e6)
    # Prepend (0,0) origin
    origin = {c: 0.0 for c in df.columns}
    df = pd.concat([pd.DataFrame([origin]), df], ignore_index=True)
    return df


# ── Step annotation helpers ───────────────────────────────────────────────────

_KIND_COLOR = {'ramp': '#1f77b4', 'unload': '#ff7f0e', 'peak': '#2ca02c', 'other': 'gray'}


def _vlines(ax, steps_ran):
    for s in steps_ran:
        ax.axvline(s['time'], color=_KIND_COLOR.get(s['kind'], 'gray'),
                   linestyle='--', alpha=0.45, linewidth=1)


def _hlines_sigma(ax, steps_ran):
    for s in steps_ran:
        ax.axhline(s['pressure_MPa'], color=_KIND_COLOR.get(s['kind'], 'gray'),
                   linestyle=':', alpha=0.45, linewidth=1)


# ── Pressure (cyclic-SFL) stages ──────────────────────────────────────────────

def _analyse_pressure_axis(apdl_runfolder, jd, plots_dir, *,
                            stage, usecase, fd_prefix, u_prefix, u_var,
                            load_label, strain_label, reaction_label):
    """
    Generic pressure-stage analysis used by both pressure_transverse (Y axis, SFL on
    top → fd_pressure_<usecase>.txt + uy_top_<usecase>.txt) and pressure_radial
    (X axis, SFL on left → fd_radial_<usecase>.txt + ux_left_<usecase>.txt).

    The driven boundary's force FSUM is ~0 (SFL reaction balances the pressure
    distribution); the applied stress is the nominal pressure from loading_cycle.json
    and the strain comes from mean displacement on the driven boundary.
    """
    steps        = jd['steps']
    total_width  = 2.0 * jd['x_cab_m']
    total_height = _total_height_m(apdl_runfolder, jd['n_stacks'])

    fd_path = _pp_in(apdl_runfolder, f'{fd_prefix}_{usecase}.txt')
    u_path  = _pp_in(apdl_runfolder, f'{u_prefix}_{usecase}.txt')
    for p in (fd_path, u_path):
        if not os.path.exists(p):
            print(f'[{stage}] File not found, skipping: {os.path.basename(p)}')
            return None

    fd = _read_5col(fd_path, ['LoadStep', 'SubStep', 'Time', 'F_analytical', 'F_reaction'])
    mean_u = _read_node_mean(u_path, u_var)

    df = fd.merge(mean_u, on=['LoadStep', 'SubStep'], how='inner')
    n_fd_only = len(fd) - len(df)
    if n_fd_only:
        print(f'[{stage}] {n_fd_only} fd substep(s) have no {u_prefix} match and were dropped.')
    if df.empty:
        print(f'[{stage}] No overlapping substeps between {fd_prefix} and {u_prefix}; skipping.')
        return None

    # Driven-axis stress = nominal pressure (interpolated); reaction stress = FSUM
    # on the constrained opposite wall divided by that wall's exposed length.
    if stage == "pressure_transverse":
        # SFL on top (y = ymax): driven axis is Y, opposite wall is left (length=total_height).
        df['sigma_load_MPa']     = df['Time'].apply(lambda t: nominal_pressure_mpa(t, steps))
        df['strain_load']        = -df[u_var] / total_height
        df['sigma_reaction_MPa'] = df['F_reaction'] / (total_height * 1e6)
    else:  # pressure_radial
        # SFL on left (x = -xmax): driven axis is X, opposite wall is right (length=total_height).
        df['sigma_load_MPa']     = df['Time'].apply(lambda t: nominal_pressure_mpa(t, steps))
        df['strain_load']        = df[u_var] / total_width
        df['sigma_reaction_MPa'] = df['F_reaction'] / (total_height * 1e6)

    origin = {c: 0.0 for c in df.columns}
    df = pd.concat([pd.DataFrame([origin]), df], ignore_index=True)

    max_ls    = int(df['LoadStep'].max())
    steps_ran = [s for s in steps if s['index'] <= max_ls]
    n_missing = len(steps) - len(steps_ran)

    tag = _stage_tag(apdl_runfolder, stage)
    print(f'\n[{stage}] {usecase}  ({tag})')
    print(f'  Load steps completed: {len(steps_ran)}/{len(steps)}'
          + (f"  WARNING: {n_missing} step(s) did not run" if n_missing else ""))
    print(f'  Geometry: total_width={total_width*1e3:.2f} mm, total_height={total_height*1e3:.2f} mm')
    print(df[['LoadStep', 'SubStep', 'Time', 'sigma_load_MPa', 'strain_load', 'sigma_reaction_MPa']]
          .to_string(index=False, float_format='{:.4g}'.format))

    # ── Figure 1: 4-subplot overview ──────────────────────────────────────────
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{stage} - {usecase}\n[{tag}]', fontsize=12)

    axs[0, 0].plot(df['strain_load'], df['sigma_load_MPa'], '-o', markersize=5, color='C0', label=stage)
    _hlines_sigma(axs[0, 0], steps_ran)
    axs[0, 0].set_xlabel(strain_label)
    axs[0, 0].set_ylabel(load_label)
    axs[0, 0].set_title('Applied stress vs strain (load axis)')
    axs[0, 0].legend(fontsize=9)
    axs[0, 0].grid(True)

    axs[0, 1].plot(df['strain_load'], df['sigma_reaction_MPa'], '-o', markersize=5, color='C2')
    axs[0, 1].set_xlabel(strain_label)
    axs[0, 1].set_ylabel(reaction_label)
    axs[0, 1].set_title('Opposite-wall reaction vs load-axis strain')
    axs[0, 1].grid(True)

    axs[1, 0].plot(df['Time'], df['sigma_load_MPa'], '-o', markersize=5, color='C0')
    _vlines(axs[1, 0], steps_ran)
    axs[1, 0].set_xlabel('APDL time (s)')
    axs[1, 0].set_ylabel(load_label)
    axs[1, 0].set_title('Load schedule (stress vs time)')
    axs[1, 0].grid(True)

    axs[1, 1].plot(df['Time'], df['strain_load'], '-o', markersize=5, color='C0')
    _vlines(axs[1, 1], steps_ran)
    axs[1, 1].set_xlabel('APDL time (s)')
    axs[1, 1].set_ylabel(strain_label)
    axs[1, 1].set_title('Compaction history (strain vs time)')
    axs[1, 1].grid(True)

    plt.tight_layout()
    out1 = os.path.join(plots_dir, f'{usecase}_subplots.svg')
    fig.savefig(out1)
    plt.close(fig)

    # ── Figure 2: annotated stress-strain ─────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(df['strain_load'], df['sigma_load_MPa'], '-o', linewidth=2, markersize=6,
             color='C0', label=f'{stage} (nominal)')
    eps_max = max(df['strain_load'].max(), 1e-4)
    for s in steps_ran:
        c = _KIND_COLOR.get(s['kind'], 'gray')
        ax2.axhline(s['pressure_MPa'], color=c, linestyle=':', alpha=0.5, linewidth=1)
        ax2.annotate(f" {s['pressure_MPa']:.0f} MPa ({s['kind']})",
                     xy=(eps_max * 1.01, s['pressure_MPa']), fontsize=8, va='center',
                     color=c, annotation_clip=False)
    title = f'{stage} - {usecase}\n[{tag}]'
    if n_missing:
        title += f'  -  Partial run: {len(steps_ran)}/{len(steps)} steps completed'
    ax2.set_title(title, fontsize=12)
    ax2.set_xlabel(strain_label, fontsize=13)
    ax2.set_ylabel(load_label, fontsize=13)
    ax2.tick_params(labelsize=11)
    ax2.legend(fontsize=11)
    ax2.grid(True)
    plt.tight_layout()
    out2 = os.path.join(plots_dir, f'{usecase}_stress_strain.svg')
    fig2.savefig(out2)
    plt.close(fig2)

    # ── Text export ───────────────────────────────────────────────────────────
    out_txt = _pp_out(apdl_runfolder, f'{usecase}_stress_strain.txt')
    export = df[['LoadStep', 'SubStep', 'Time', 'strain_load',
                 'sigma_load_MPa', 'sigma_reaction_MPa']].copy()
    with open(out_txt, 'w') as f:
        f.write(f'# {stage} stress-strain data - {usecase}\n')
        f.write(f'# Formulation: {_detect_formulation(apdl_runfolder)} | BC: {_detect_stage_bc(apdl_runfolder, stage)}\n')
        f.write(f'# sigma_load: nominal applied pressure from loading_cycle.json [MPa]\n')
        f.write(f'# sigma_reaction: opposite-wall reaction stress [MPa]\n')
        f.write(f'# strain_load: compressive strain along load axis (positive = compression)\n')
        f.write(f'# Geometry: total_width={total_width*1e3:.4f} mm, total_height={total_height*1e3:.4f} mm\n')
        f.write(f'{"LoadStep":>10} {"SubStep":>10} {"Time":>12} {"strain_load":>14}'
                f' {"sigma_load_MPa":>16} {"sigma_reaction_MPa":>20}\n')
        for _, row in export.iterrows():
            f.write(f'{row["LoadStep"]:10.0f} {row["SubStep"]:10.0f} {row["Time"]:12.6g}'
                    f' {row["strain_load"]:14.6g} {row["sigma_load_MPa"]:16.6g}'
                    f' {row["sigma_reaction_MPa"]:20.6g}\n')

    print(f'  -> {out1}')
    print(f'  -> {out2}')
    print(f'  -> {out_txt}')
    return df


def postprocess_pressure_transverse(apdl_runfolder):
    jd = _load_jd(apdl_runfolder)
    if jd is None:
        return False
    plots_dir = os.path.join(apdl_runfolder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    df = _analyse_pressure_axis(
        apdl_runfolder, jd, plots_dir,
        stage="pressure_transverse",
        usecase=_stage_usecase(jd, "pressure_transverse"),
        fd_prefix="fd_pressure", u_prefix="uy_top", u_var="UY",
        load_label="Applied pressure sigma_y (MPa)",
        strain_label="Vertical strain epsilon_y (-)",
        reaction_label="Lateral wall reaction sigma_x (MPa)",
    )
    return df is not None


def postprocess_pressure_radial(apdl_runfolder):
    jd = _load_jd(apdl_runfolder)
    if jd is None:
        return False
    plots_dir = os.path.join(apdl_runfolder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    df = _analyse_pressure_axis(
        apdl_runfolder, jd, plots_dir,
        stage="pressure_radial",
        usecase=_stage_usecase(jd, "pressure_radial"),
        fd_prefix="fd_radial", u_prefix="ux_left", u_var="UX",
        load_label="Applied lateral pressure sigma_x (MPa)",
        strain_label="Radial strain epsilon_x (-)",
        reaction_label="Right-wall reaction sigma_x_r (MPa)",
    )
    return df is not None


def postprocess_pressure_combined(apdl_runfolder):
    """Overlay transverse and radial pressure stages on one stress-strain figure."""
    jd = _load_jd(apdl_runfolder)
    if jd is None:
        return False
    steps = jd['steps']
    total_width  = 2.0 * jd['x_cab_m']
    total_height = _total_height_m(apdl_runfolder, jd['n_stacks'])
    cable        = jd['cable']

    def _load(fd_prefix, u_prefix, u_var, stage):
        usecase = _stage_usecase(jd, stage)
        fd_path = _pp_in(apdl_runfolder, f'{fd_prefix}_{usecase}.txt')
        u_path  = _pp_in(apdl_runfolder, f'{u_prefix}_{usecase}.txt')
        if not (os.path.exists(fd_path) and os.path.exists(u_path)):
            return None, usecase
        fd      = _read_5col(fd_path, ['LoadStep', 'SubStep', 'Time', 'F_analytical', 'F_reaction'])
        mean_u  = _read_node_mean(u_path, u_var)
        df      = fd.merge(mean_u, on=['LoadStep', 'SubStep'], how='inner')
        if df.empty:
            return None, usecase
        df['sigma_load_MPa'] = df['Time'].apply(lambda t: nominal_pressure_mpa(t, steps))
        if stage == 'pressure_transverse':
            df['strain_load'] = -df[u_var] / total_height
        else:
            df['strain_load'] = df[u_var] / total_width
        origin = {c: 0.0 for c in df.columns}
        df = pd.concat([pd.DataFrame([origin]), df], ignore_index=True)
        return df, usecase

    df_t, uc_t = _load('fd_pressure', 'uy_top',  'UY', 'pressure_transverse')
    df_r, uc_r = _load('fd_radial',   'ux_left',  'UX', 'pressure_radial')

    if df_t is None and df_r is None:
        print('[pressure_combined] Neither pressure stage output found; skipping.')
        return False

    plots_dir = os.path.join(apdl_runfolder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    if df_t is not None:
        ax.plot(df_t['strain_load'], df_t['sigma_load_MPa'], '-o', linewidth=2,
                markersize=5, color='C0', label='Transverse (sigma_y)')
    if df_r is not None:
        ax.plot(df_r['strain_load'], df_r['sigma_load_MPa'], '-s', linewidth=2,
                markersize=5, color='C1', label='Radial (sigma_x)')

    ref_steps = steps
    eps_vals = []
    if df_t is not None:
        eps_vals.append(df_t['strain_load'].max())
    if df_r is not None:
        eps_vals.append(df_r['strain_load'].max())
    eps_max = max(eps_vals) if eps_vals else 1e-4
    for s in ref_steps:
        c = _KIND_COLOR.get(s['kind'], 'gray')
        ax.axhline(s['pressure_MPa'], color=c, linestyle=':', alpha=0.4, linewidth=1)
        ax.annotate(f" {s['pressure_MPa']:.0f} MPa ({s['kind']})",
                    xy=(eps_max * 1.01, s['pressure_MPa']), fontsize=8, va='center',
                    color=c, annotation_clip=False)

    ax.set_title(f'{cable} - pressure stages combined\n[{_FORM_SHORT[_detect_formulation(apdl_runfolder)]}]', fontsize=12)
    ax.set_xlabel('Compressive strain (-)', fontsize=13)
    ax.set_ylabel('Applied pressure (MPa)', fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11)
    ax.grid(True)
    plt.tight_layout()
    out = os.path.join(plots_dir, f'{cable}_pressure_combined.svg')
    fig.savefig(out)
    plt.close(fig)
    print(f'[pressure_combined] -> {out}')
    return True


# ── Displacement (cyclic-strain) stages ───────────────────────────────────────

def _analyse_displacement(apdl_runfolder, jd, plots_dir, *, stage, usecase, axis):
    """
    Plot fd_good_<usecase>.txt for a displacement-controlled stage.

    axis="Y": load axis is vertical (compaction by UY on top)         → use UY/FY
    axis="X": load axis is radial   (compaction by UX on left wall)   → use UX/FX
    """
    fd_path = _pp_in(apdl_runfolder, f'fd_good_{usecase}.txt')
    if not os.path.exists(fd_path):
        print(f'[{stage}] fd_good_{usecase}.txt not found; skipping.')
        return None

    total_width  = 2.0 * jd['x_cab_m']
    total_height = _total_height_m(apdl_runfolder, jd['n_stacks'])
    df = _read_fd_good(fd_path, total_width, total_height)
    if len(df) <= 1:
        # Only the prepended (0,0) origin row: the file exists but no data row
        # parsed — MAPDL most likely crashed after opening the dump.
        print(f'[{stage}] {os.path.basename(fd_path)} exists but contains no parseable data rows; skipping.')
        return None

    if axis == "Y":
        strain_load   = df['epsilon_y']
        sigma_load    = df['sigma_y_MPa']
        strain_react  = df['epsilon_x']
        sigma_react   = df['sigma_x_MPa']
        load_label    = "Applied stress sigma_y (MPa)"
        strain_label  = "Vertical strain epsilon_y (-)"
        react_label_s = "Lateral reaction sigma_x (MPa)"
        react_label_e = "Lateral strain epsilon_x (-)"
    else:  # X
        strain_load   = df['epsilon_x']
        sigma_load    = df['sigma_x_MPa']
        strain_react  = df['epsilon_y']
        sigma_react   = df['sigma_y_MPa']
        load_label    = "Applied stress sigma_x (MPa)"
        strain_label  = "Radial strain epsilon_x (-)"
        react_label_s = "Vertical reaction sigma_y (MPa)"
        react_label_e = "Vertical strain epsilon_y (-)"

    tag = _stage_tag(apdl_runfolder, stage)
    print(f'\n[{stage}] {usecase}  ({tag})')
    print(f'  Geometry: total_width={total_width*1e3:.2f} mm, total_height={total_height*1e3:.2f} mm')
    print(f'  Data points: {len(df)} (incl. origin)')
    if len(df) > 1:
        print(f'  Peak load stress: {sigma_load.abs().max():.2f} MPa at strain {strain_load.iloc[sigma_load.abs().idxmax()]:.5f}')

    # ── Figure: 2x2 overview ──────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{stage} - {usecase}\n[{tag}]', fontsize=12)

    axs[0, 0].plot(strain_load, sigma_load, '-o', markersize=5, color='C0')
    axs[0, 0].set_xlabel(strain_label)
    axs[0, 0].set_ylabel(load_label)
    axs[0, 0].set_title('Stress vs strain (load axis)')
    axs[0, 0].grid(True)

    axs[0, 1].plot(strain_load, sigma_react, '-o', markersize=5, color='C2')
    axs[0, 1].set_xlabel(strain_label)
    axs[0, 1].set_ylabel(react_label_s)
    axs[0, 1].set_title('Cross-axis reaction vs load-axis strain')
    axs[0, 1].grid(True)

    axs[1, 0].plot(df['Time'], sigma_load, '-o', markersize=5, color='C0')
    axs[1, 0].set_xlabel('APDL time (s)')
    axs[1, 0].set_ylabel(load_label)
    axs[1, 0].set_title('Load schedule')
    axs[1, 0].grid(True)

    axs[1, 1].plot(df['Time'], strain_load, '-o', markersize=5, color='C0')
    axs[1, 1].set_xlabel('APDL time (s)')
    axs[1, 1].set_ylabel(strain_label)
    axs[1, 1].set_title('Compaction history')
    axs[1, 1].grid(True)

    plt.tight_layout()
    out1 = os.path.join(plots_dir, f'{usecase}_subplots.svg')
    fig.savefig(out1)
    plt.close(fig)

    # ── Figure: standalone stress-strain ──────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(strain_load, sigma_load, '-o', linewidth=2, markersize=6, color='C0')
    ax2.set_xlabel(strain_label, fontsize=13)
    ax2.set_ylabel(load_label, fontsize=13)
    ax2.set_title(f'{stage} - {usecase}\n[{tag}]', fontsize=12)
    ax2.tick_params(labelsize=11)
    ax2.grid(True)
    plt.tight_layout()
    out2 = os.path.join(plots_dir, f'{usecase}_stress_strain.svg')
    fig2.savefig(out2)
    plt.close(fig2)

    # ── Text export ───────────────────────────────────────────────────────────
    out_txt = _pp_out(apdl_runfolder, f'{usecase}_stress_strain.txt')
    with open(out_txt, 'w') as f:
        f.write(f'# {stage} stress-strain data - {usecase}\n')
        f.write(f'# Formulation: {_detect_formulation(apdl_runfolder)} | BC: {_detect_stage_bc(apdl_runfolder, stage)}\n')
        f.write(f'# sigma_load:    stress along applied-displacement axis [MPa]\n')
        f.write(f'# sigma_react:   stress on the cross-axis (other wall reaction) [MPa]\n')
        f.write(f'# Geometry: total_width={total_width*1e3:.4f} mm, total_height={total_height*1e3:.4f} mm\n')
        f.write(f'{"Set":>6} {"Time":>12} {"strain_load":>14} {"sigma_load_MPa":>16}'
                f' {"strain_react":>14} {"sigma_react_MPa":>16}\n')
        for i in range(len(df)):
            f.write(f'{df["Set"].iloc[i]:6.0f} {df["Time"].iloc[i]:12.6g}'
                    f' {strain_load.iloc[i]:14.6g} {sigma_load.iloc[i]:16.6g}'
                    f' {strain_react.iloc[i]:14.6g} {sigma_react.iloc[i]:16.6g}\n')

    print(f'  -> {out1}')
    print(f'  -> {out2}')
    print(f'  -> {out_txt}')
    return df


def postprocess_displacement_transverse(apdl_runfolder):
    jd = _load_jd(apdl_runfolder)
    if jd is None:
        return False
    plots_dir = os.path.join(apdl_runfolder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    df = _analyse_displacement(
        apdl_runfolder, jd, plots_dir,
        stage="displacement_transverse",
        usecase=_stage_usecase(jd, "displacement_transverse"),
        axis="Y",
    )
    return df is not None


def postprocess_displacement_radial(apdl_runfolder):
    jd = _load_jd(apdl_runfolder)
    if jd is None:
        return False
    plots_dir = os.path.join(apdl_runfolder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    df = _analyse_displacement(
        apdl_runfolder, jd, plots_dir,
        stage="displacement_radial",
        usecase=_stage_usecase(jd, "displacement_radial"),
        axis="X",
    )
    return df is not None


def postprocess_displacement_combined(apdl_runfolder):
    """Overlay transverse and radial displacement stages on one stress-strain figure."""
    jd = _load_jd(apdl_runfolder)
    if jd is None:
        return False
    total_width  = 2.0 * jd['x_cab_m']
    total_height = _total_height_m(apdl_runfolder, jd['n_stacks'])
    cable        = jd['cable']

    def _load_disp(stage):
        usecase = _stage_usecase(jd, stage)
        fd_path = _pp_in(apdl_runfolder, f'fd_good_{usecase}.txt')
        if not os.path.exists(fd_path):
            return None
        return _read_fd_good(fd_path, total_width, total_height)

    df_t = _load_disp('displacement_transverse')
    df_r = _load_disp('displacement_radial')

    if df_t is None and df_r is None:
        print('[displacement_combined] Neither displacement stage output found; skipping.')
        return False

    plots_dir = os.path.join(apdl_runfolder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    if df_t is not None:
        ax.plot(df_t['epsilon_y'], df_t['sigma_y_MPa'], '-o', linewidth=2,
                markersize=5, color='C0', label='Transverse (sigma_y vs epsilon_y)')
    if df_r is not None:
        ax.plot(df_r['epsilon_x'], df_r['sigma_x_MPa'], '-s', linewidth=2,
                markersize=5, color='C1', label='Radial (sigma_x vs epsilon_x)')

    ax.set_title(f'{cable} - displacement stages combined\n[{_FORM_SHORT[_detect_formulation(apdl_runfolder)]}]', fontsize=12)
    ax.set_xlabel('Compressive strain (-)', fontsize=13)
    ax.set_ylabel('Applied stress (MPa)', fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11)
    ax.grid(True)
    plt.tight_layout()
    out = os.path.join(plots_dir, f'{cable}_displacement_combined.svg')
    fig.savefig(out)
    plt.close(fig)
    print(f'[displacement_combined] -> {out}')
    return True


def postprocess_all_combined(apdl_runfolder):
    """All four stages on one stress-strain figure."""
    jd = _load_jd(apdl_runfolder)
    if jd is None:
        return False
    steps        = jd['steps']
    total_width  = 2.0 * jd['x_cab_m']
    total_height = _total_height_m(apdl_runfolder, jd['n_stacks'])
    cable        = jd['cable']

    # ── load displacement stages ──────────────────────────────────────────────
    def _load_disp(stage):
        usecase = _stage_usecase(jd, stage)
        fd_path = _pp_in(apdl_runfolder, f'fd_good_{usecase}.txt')
        if not os.path.exists(fd_path):
            return None
        return _read_fd_good(fd_path, total_width, total_height)

    df_dt = _load_disp('displacement_transverse')
    df_dr = _load_disp('displacement_radial')

    # ── load pressure stages ──────────────────────────────────────────────────
    def _load_pres(fd_prefix, u_prefix, u_var, stage):
        usecase = _stage_usecase(jd, stage)
        fd_path = _pp_in(apdl_runfolder, f'{fd_prefix}_{usecase}.txt')
        u_path  = _pp_in(apdl_runfolder, f'{u_prefix}_{usecase}.txt')
        if not (os.path.exists(fd_path) and os.path.exists(u_path)):
            return None
        fd     = _read_5col(fd_path, ['LoadStep', 'SubStep', 'Time', 'F_analytical', 'F_reaction'])
        mean_u = _read_node_mean(u_path, u_var)
        df     = fd.merge(mean_u, on=['LoadStep', 'SubStep'], how='inner')
        if df.empty:
            return None
        df['sigma_load_MPa'] = df['Time'].apply(lambda t: nominal_pressure_mpa(t, steps))
        if stage == 'pressure_transverse':
            df['strain_load'] = -df[u_var] / total_height
        else:
            df['strain_load'] = df[u_var] / total_width
        origin = {c: 0.0 for c in df.columns}
        return pd.concat([pd.DataFrame([origin]), df], ignore_index=True)

    df_pt = _load_pres('fd_pressure', 'uy_top',  'UY', 'pressure_transverse')
    df_pr = _load_pres('fd_radial',   'ux_left',  'UX', 'pressure_radial')

    datasets = [df_dt, df_dr, df_pt, df_pr]
    if all(d is None for d in datasets):
        print('[all_combined] No stage outputs found; skipping.')
        return False

    plots_dir = os.path.join(apdl_runfolder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    if df_dt is not None:
        ax.plot(df_dt['epsilon_y'], df_dt['sigma_y_MPa'], '-o', linewidth=2,
                markersize=5, color='C0', label='Disp. transverse (sigma_y)')
    if df_dr is not None:
        ax.plot(df_dr['epsilon_x'], df_dr['sigma_x_MPa'], '-s', linewidth=2,
                markersize=5, color='C1', label='Disp. radial (sigma_x)')
    if df_pt is not None:
        ax.plot(df_pt['strain_load'], df_pt['sigma_load_MPa'], '--^', linewidth=2,
                markersize=5, color='C2', label='Pres. transverse (sigma_y)')
    if df_pr is not None:
        ax.plot(df_pr['strain_load'], df_pr['sigma_load_MPa'], '--D', linewidth=2,
                markersize=5, color='C3', label='Pres. radial (sigma_x)')

    ax.set_title(f'{cable} - all stages\n[{_FORM_SHORT[_detect_formulation(apdl_runfolder)]}]', fontsize=12)
    ax.set_xlabel('Compressive strain (-)', fontsize=13)
    ax.set_ylabel('Applied stress / pressure (MPa)', fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11)
    ax.grid(True)
    plt.tight_layout()
    out = os.path.join(plots_dir, f'{cable}_all_combined.svg')
    fig.savefig(out)
    plt.close(fig)
    print(f'[all_combined] -> {out}')
    return True


# ── Thermal cooldown (SKELETON) ──────────────────────────────────────────────
# Architecture for a future 293 K -> 4.2 K cooldown stage that develops CTE-
# mismatch axial pre-strain on Nb3Sn filaments (drives Ic).  The MAPDL deck
# (0-start-thermal.inp / 5-BC-thermal.inp / 8-PP-thermal.inp) is presently a
# /com NOT IMPLEMENTED stub.  When the physics is filled in, this function
# should read the cooldown .rst / pp dump and produce a Nb3Sn ε_zz figure.

def postprocess_thermal_cooldown(apdl_runfolder):
    """SKELETON: thermal-cooldown postprocess.

    When implemented, this should:
      * read the per-element axial strain field on Nb3Sn (mat=3) elements,
      * compute area-weighted mean / spatial distribution of ε_zz,
      * write <usecase>_thermal_strain.txt + a histogram SVG to plots/,
      * return True on success.

    Until the 5-BC-thermal.inp physics is implemented this no-ops.
    """
    # No-op until 5-BC-thermal.inp + 8-PP-thermal.inp are filled in.
    return False


# ── Orchestrator ──────────────────────────────────────────────────────────────

_ALL_POSTPROCESSORS = [
    postprocess_displacement_transverse,
    postprocess_displacement_radial,
    postprocess_displacement_combined,
    postprocess_pressure_transverse,
    postprocess_pressure_radial,
    postprocess_pressure_combined,
    postprocess_all_combined,
    postprocess_thermal_cooldown,   # SKELETON -- always returns False until implemented
]


def analyse(apdl_runfolder):
    """
    Run every stage's postprocess on one apdl_runfolder.  Each stage silently
    skips itself when its inputs are absent.  Returns True iff any produced
    output.  Never raises on missing files.
    """
    apdl_runfolder = os.path.abspath(apdl_runfolder)
    jd = _load_jd(apdl_runfolder)
    if jd is None:
        return False

    formulation = _detect_formulation(apdl_runfolder)
    cable       = jd.get('cable', '?')
    n_strands   = jd.get('n_strands', '?')
    n_stacks    = jd.get('n_stacks', '?')
    print(f'\n========== POSTPROCESS ({cable}) ==========')
    print(f'  Run folder  : {apdl_runfolder}')
    print(f'  Formulation : {formulation}')
    print(f'  Geometry    : n_strands={n_strands}, n_stacks={n_stacks}')

    results = [bool(fn(apdl_runfolder)) for fn in _ALL_POSTPROCESSORS]

    # Final summary: one row per cablestack stage with BC and pass/skip.
    stage_order = [
        "displacement_transverse",
        "displacement_radial",
        "pressure_transverse",
        "pressure_radial",
        # thermal_cooldown stays out of the table until the deck is implemented.
    ]
    stage_fn = {
        "displacement_transverse": postprocess_displacement_transverse,
        "displacement_radial":     postprocess_displacement_radial,
        "pressure_transverse":     postprocess_pressure_transverse,
        "pressure_radial":         postprocess_pressure_radial,
    }
    fn_to_idx = {fn: i for i, fn in enumerate(_ALL_POSTPROCESSORS)}
    print(f'\n--- Stage summary ({cable}, {_FORM_SHORT[formulation]}) ---')
    print(f'  {"Stage":<26}  {"BC":<22}  {"Status"}')
    for stage in stage_order:
        bc = _detect_stage_bc(apdl_runfolder, stage)
        idx = fn_to_idx.get(stage_fn[stage])
        status = ('OK' if (idx is not None and results[idx]) else 'skipped (no output)')
        print(f'  {stage:<26}  {bc:<22}  {status}')
    print(f'-------------------------------------------------\n')

    return any(results)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) >= 2:
        try:
            apdl_rf = _find_apdl_runfolder(sys.argv[1])
        except FileNotFoundError as exc:
            print(f'Error: {exc}', file=sys.stderr)
            sys.exit(1)
    else:
        try:
            apdl_rf = _find_latest_runfolder()
        except FileNotFoundError as exc:
            print(f'Error: {exc}', file=sys.stderr)
            sys.exit(1)
        print(f'Auto-detected latest run folder: {apdl_rf}')

    ok = analyse(apdl_rf)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
