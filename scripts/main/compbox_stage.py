"""Compression box simulation stage (workflow step 9).

Chains the paper_clean_version box+submodel pipeline onto a completed run
folder (needs the conformal-mesh geometry from step 7):

  1. parent_mag     stage + solve the 3D box magnetic model (CompBox_mag.inp,
                    MAG-only -- the legacy CompBox_mech .rst is not needed by
                    the field chain) -> CompBox_MAG_<cable_type>.rmg
  2. vtu_export     convert_rmg_to_vtu.py: .rmg -> per-loadstep enhanced VTUs
                    (Bx/By/Bz/B_magnitude on the conductor mesh)
  3. field_tables   create_magnetic_heatmaps.py: VTUs + keypoints_nodes_<i>.txt
                    -> nb3sn_combined_data_case_<i>_<t>.inp field tables
  4. submodel       stage + solve the one-turn 2D submodel (one case per
                    stack cross-section) -> strains_out_strand_*_set_*_case_*.out
  5. analysis       complete_analysis.py (strain + Ic correlation CSVs/plots)
                    and ic_calculator.py

MAPDL solves run either in the local Docker MAPDL container (same compose
service as the cablestack stages) or on an SSH-reachable SLURM HPC cluster
(default ETH Euler; override via HPC_* env vars), selected by
compression_box.solver in cable_parameters_user.json ('local' | 'hpc';
the --hpc CLI flag forces 'hpc').

The Nb3Sn modulus in the submodel materials is the fixed 70 GPa standard
(the value the box Mat 11 amplification factor 1.2 is based on); the RVE
homogenised override used by the cablestack stage is deliberately NOT
applied here until the RVE amplification-factor question is settled.

Run-folder layout written by this module:

  <run>/APDL/compbox/
    parent_runfolder/    box MAG deck + loading schedule (+ .rmg after solve)
    vtu/                 enhanced per-loadstep VTUs
    field_tables/        nb3sn_combined_data_case_<i>_<t>.inp
    submodel_runfolder/  decks + geometry + field tables (+ strains after solve)
    results/             heatmaps/ + strain_analysis/ + ic_calculation.log
    compbox_summary.json per-substep status + config audit
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Geometry file families written by conformalRutherfordMesh into
# <run>/APDL/submodel/apdl_runfolder/ and consumed by the submodel decks
# (2-geo.inp / 4-cont.inp) per case i = 1..n_stacks.
GEOMETRY_FAMILIES = (
    "keypoints_{i}.txt",
    "keypoints_nodes_{i}.txt",
    "keypoints_insulation_nodes_{i}.txt",
    "connections_{i}.txt",
    "contacts_strands_{i}.txt",
    "inner_insulation_lines_{i}.txt",
    "outer_insulation_lines_{i}.txt",
)

SUBMODEL_DECKS = (
    "0-start.inp",
    "1-material_properties.inp",
    "2-geo.inp",
    "3-mesh.inp",
    "4-cont.inp",
    "5-BC-submodel.inp",
    "6-PP-submodel.inp",
    "cable_materials_LN2.inp",
)

PARENT_DRIVER = "CompBox_mag.inp"
SUBMODEL_DRIVER = "0-start.inp"

# Fixed Nb3Sn standard for the compbox submodel (see module docstring).
NB3SN_STANDARD_PA = 70e9


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def compbox_base_dir(run_output_dir: Path) -> Path:
    return run_output_dir / "APDL" / "compbox"


def _dirs(run_output_dir: Path) -> Dict[str, Path]:
    base = compbox_base_dir(run_output_dir)
    return {
        "base": base,
        "parent": base / "parent_runfolder",
        "vtu": base / "vtu",
        "field": base / "field_tables",
        "submodel": base / "submodel_runfolder",
        "results": base / "results",
    }


def _templates(workspace_root: Path) -> Dict[str, Path]:
    return {
        "parent": workspace_root / "scripts" / "apdl" / "parentmodel" / "compbox",
        "submodel": workspace_root / "scripts" / "apdl" / "submodel" / "compbox",
        "analysis": workspace_root / "scripts" / "analysis" / "submodel" / "compbox",
    }


def _fresh_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _patch_line(text: str, pattern: str, replacement: str, deck: str) -> str:
    new, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if n == 0:
        raise RuntimeError(f"Pattern {pattern!r} not found in {deck} template.")
    return new


# ---------------------------------------------------------------------------
# Loading schedule generation (single source: the measurement table)
# ---------------------------------------------------------------------------

LOADING_SENTINEL_START = "! <<<COMPBOX_LOADING_BLOCK_START>>>"
LOADING_SENTINEL_END = "! <<<COMPBOX_LOADING_BLOCK_END>>>"


def parse_measurement_table(path: Path) -> List[Dict[str, Optional[float]]]:
    """Parse a BOX9.txt-style measurement table.

    Whitespace-separated columns: peak pressure [MPa], Ic under load [A],
    Ic after unload [A] (NaN on the virgin row). Extra columns (Ic ratios)
    and incomplete trailing rows are ignored.
    """
    rows: List[Dict[str, Optional[float]]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pressure = float(parts[0])
            ic_load = float(parts[1])
        except ValueError:
            continue
        if not (pressure == pressure and ic_load == ic_load):  # NaN guard
            continue
        ic_unload: Optional[float] = None
        if len(parts) >= 3:
            try:
                v = float(parts[2])
                if v == v:
                    ic_unload = v
            except ValueError:
                pass
        rows.append({"pressure_MPa": pressure, "ic_load_A": ic_load,
                     "ic_unload_A": ic_unload})
    if not rows:
        raise RuntimeError(f"No usable rows in measurement table {path}.")
    return rows


def build_loading_steps(rows: List[Dict[str, Optional[float]]],
                        nsteps_cap: int = 0) -> List[Dict[str, object]]:
    """Interleaved LOAD/UNLOAD schedule from the measurement rows.

    Row 1 defines the baseline holding pressure and the virgin Ic. Every row
    becomes a pair: odd step = load at the row's peak pressure with the
    measured under-load Ic, even step = unload to the baseline carrying the
    measured after-unload Ic (virgin Ic when not measured, e.g. row 1).
    """
    baseline_p = float(rows[0]["pressure_MPa"])  # type: ignore[arg-type]
    virgin_ic = float(rows[0]["ic_load_A"])  # type: ignore[arg-type]
    steps: List[Dict[str, object]] = []
    for row in rows:
        unload_ic = row["ic_unload_A"] if row["ic_unload_A"] is not None else virgin_ic
        steps.append({"kind": "load", "pressure_MPa": row["pressure_MPa"],
                      "current_A": row["ic_load_A"]})
        steps.append({"kind": "unload", "pressure_MPa": baseline_p,
                      "current_A": unload_ic})
    if nsteps_cap:
        if nsteps_cap % 2:
            logger.warning(f"[compbox] nsteps={nsteps_cap} is odd - the final "
                           f"load step has no matching unload step.")
        if nsteps_cap > len(steps):
            raise RuntimeError(
                f"compression_box.nsteps={nsteps_cap} exceeds the "
                f"{len(steps)} steps available from the measurement table.")
        steps = steps[:nsteps_cap]
    return steps


def loading_block_apdl(steps: List[Dict[str, object]], source_name: str) -> str:
    lines = [
        f"! Loading schedule generated by compbox_stage.py from {source_name}.",
        "! Interleaved LOAD/UNLOAD pairs: odd steps = measured load peaks,",
        "! even steps = unload to the baseline holding pressure (after-unload",
        "! current when measured, virgin current otherwise). Do not edit here;",
        "! edit the measurement table instead.",
        f"*set,nsteps,{len(steps)}",
        "",
        "*DIM,totalCurr,array,nsteps",
    ]
    for i, s in enumerate(steps, 1):
        lines.append(f"totalCurr({i}) = {s['current_A']:g}")
    lines += ["", "*DIM,pres_pusher,array,nsteps"]
    for i, s in enumerate(steps, 1):
        lines.append(f"pres_pusher({i}) = {s['pressure_MPa']:g}*1e6")
    return "\n".join(lines) + "\n"


def _replace_loading_block(text: str, block: str, deck: str) -> str:
    pattern = re.compile(
        re.escape(LOADING_SENTINEL_START) + r".*?" + re.escape(LOADING_SENTINEL_END),
        re.DOTALL)
    replacement = f"{LOADING_SENTINEL_START}\n{block}{LOADING_SENTINEL_END}"
    new, n = pattern.subn(lambda _m: replacement, text, count=1)
    if n == 0:
        raise RuntimeError(f"Loading sentinel block not found in {deck} template.")
    return new


def resolve_measurement_path(workspace_root: Path, measurement_file: str) -> Path:
    """Bare names resolve against the parentmodel/compbox template dir, so a
    new campaign can either drop its table there or point the config at an
    absolute path."""
    p = Path(measurement_file)
    candidates = [p] if p.is_absolute() else [
        _templates(workspace_root)["parent"] / p,
        workspace_root / p,
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"Measurement table {measurement_file!r} not found "
        f"(searched: {[str(c) for c in candidates]}).")


# ---------------------------------------------------------------------------
# Staging + patching
# ---------------------------------------------------------------------------

def stage_parent_runfolder(workspace_root: Path, run_output_dir: Path,
                           cable_label: str, x_cab_m: float, y_cab_m: float,
                           steps: List[Dict[str, object]],
                           schedule_source: str) -> Path:
    """Copy + patch the box MAG deck into <run>/APDL/compbox/parent_runfolder.

    The loading schedule is written as a generated loading_schedule.inp
    (from the measurement table), replacing the template's box9_loading
    include."""
    tpl = _templates(workspace_root)["parent"]
    rf = _fresh_dir(_dirs(run_output_dir)["parent"])

    (rf / "loading_schedule.inp").write_text(
        loading_block_apdl(steps, schedule_source), encoding="ascii")

    cable_type = f"{cable_label}_submodel"
    text = (tpl / PARENT_DRIVER).read_text(encoding="ascii")
    text = _patch_line(text, r"^cable_type = .*$",
                       f"cable_type = '{cable_type}'", PARENT_DRIVER)
    text = _patch_line(text, r"^wf_cable_width = .*$",
                       f"wf_cable_width = {2.0 * x_cab_m:.6e}", PARENT_DRIVER)
    text = _patch_line(text, r"^wf_cable_height = .*$",
                       f"wf_cable_height = {2.0 * y_cab_m:.6e}", PARENT_DRIVER)
    text = _patch_line(text, r"^/inp,box9_loading,inp\s*$",
                       "/inp,loading_schedule,inp", PARENT_DRIVER)
    (rf / PARENT_DRIVER).write_text(text, encoding="ascii")

    logger.info(
        f"[compbox] Parent runfolder staged: cable_type='{cable_type}', "
        f"width={2 * x_cab_m:.6e} m, height={2 * y_cab_m:.6e} m, "
        f"{len(steps)} loadsteps from {schedule_source}"
    )
    return rf


def stage_submodel_runfolder(workspace_root: Path, run_output_dir: Path,
                             apdl_runfolder: Path, n_cases: int,
                             n_strands: int, cable_label: str,
                             x_cab_m: float, y_cab_m: float, impreg: int,
                             measurement_path: Path,
                             steps: List[Dict[str, object]],
                             require_field_tables: bool = True) -> Path:
    """Stage decks + geometry + field tables flat into submodel_runfolder.

    Flat layout (deck /INPUTs by bare filename) mirrors the proven
    paper_clean_version orchestrator staging.  The loading block inside
    5-BC-submodel.inp is replaced with the schedule generated from the
    measurement table, so box and submodel can never drift apart.
    """
    tpls = _templates(workspace_root)
    rf = _fresh_dir(_dirs(run_output_dir)["submodel"])
    field_dir = _dirs(run_output_dir)["field"]

    # Decks
    for name in SUBMODEL_DECKS:
        shutil.copy2(tpls["submodel"] / name, rf / name)

    # Generated loading schedule into the BC deck's sentinel block.
    bc_deck = rf / "5-BC-submodel.inp"
    bc_text = _replace_loading_block(
        bc_deck.read_text(encoding="ascii"),
        loading_block_apdl(steps, measurement_path.name),
        "5-BC-submodel.inp")
    bc_deck.write_text(bc_text, encoding="ascii")

    # Measurement file for the downstream strain/Ic analysis (read from the
    # data dir by complete_analysis.py, next to the strains_out dumps).
    shutil.copy2(measurement_path, rf / measurement_path.name)

    # Geometry families from the conformal-mesh step
    missing: List[str] = []
    for i in range(1, n_cases + 1):
        for fam in GEOMETRY_FAMILIES:
            name = fam.format(i=i)
            src = apdl_runfolder / name
            if src.is_file():
                shutil.copy2(src, rf / name)
            else:
                missing.append(name)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} geometry files missing from {apdl_runfolder} "
            f"(first: {missing[:4]}). Run step 7 (APDL submodel) first."
        )

    # Field tables
    tables = sorted(field_dir.glob("nb3sn_combined_data_case_*.inp"))
    for t in tables:
        shutil.copy2(t, rf / t.name)
    if not tables:
        msg = (f"No field tables in {field_dir} - the parent MAG solve + "
               f"field_tables substeps must run first.")
        if require_field_tables:
            raise FileNotFoundError(msg)
        logger.warning(f"[compbox] {msg} (staging-only run, continuing)")

    # Patch the driver deck
    text = (rf / SUBMODEL_DRIVER).read_text(encoding="ascii")
    text = _patch_line(text, r"^cases\s*=\s*\S+", f"cases = {n_cases}", SUBMODEL_DRIVER)
    text = _patch_line(text, r"^n_strands\s*=\s*\S+", f"n_strands = {n_strands}", SUBMODEL_DRIVER)
    text = _patch_line(text, r"^usecase\s*=\s*\S+", f"usecase = '{cable_label}'", SUBMODEL_DRIVER)
    text = _patch_line(text, r"^x_cab\s*=\s*\S+", f"x_cab = {x_cab_m:.6e}", SUBMODEL_DRIVER)
    text = _patch_line(text, r"^y_cab\s*=\s*\S+", f"y_cab = {y_cab_m:.6e}", SUBMODEL_DRIVER)
    (rf / SUBMODEL_DRIVER).write_text(text, encoding="ascii")

    # Patch impreg in the materials shim
    shim = rf / "1-material_properties.inp"
    shim_text = _patch_line(shim.read_text(encoding="ascii"),
                            r"^impreg\s*=\s*\S+", f"impreg = {impreg}",
                            "1-material_properties.inp")
    shim.write_text(shim_text, encoding="ascii")

    logger.info(
        f"[compbox] Submodel runfolder staged: cases={n_cases}, "
        f"n_strands={n_strands}, usecase='{cable_label}', "
        f"x_cab={x_cab_m:.6e}, y_cab={y_cab_m:.6e}, impreg={impreg}, "
        f"{len(tables)} field tables."
    )
    return rf


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def _write_jobscript(runfolder: Path, tag: str, driver: str, ntasks: int,
                     mem_per_cpu: str) -> str:
    """SLURM script following the proven paper_clean_version pattern.

    Emits the <tag>_END rc=N marker that hpc_submit.parse_stage_results
    reads back, and clears stale .lock files (rc=100 batch-abort guard).
    """
    name = "jobslurm.sh"
    script = (
        "#!/bin/bash\n"
        "#SBATCH --time=24:00:00\n"
        f"#SBATCH --job-name=compbox_{tag}\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --ntasks={ntasks}\n"
        f"#SBATCH --tasks-per-node={ntasks}\n"
        "#SBATCH --constraint=ethernet\n"
        f"#SBATCH --mem-per-cpu={mem_per_cpu}\n"
        "\n"
        "module purge\n"
        "module load ansys/24.2_research stack/2024-06 mesa-glu/9.0.2 "
        "motif/2.3.8-ehjpbwx\n"
        "\n"
        "rm -f ./*.lock\n"
        f"echo \"{tag}_START\" >> mapdl_run.log\n"
        f"mapdl -np {ntasks} -mpp -i ./{driver} > mapdl_{tag}.log 2>&1\n"
        "rc=$?\n"
        f"echo \"{tag}_END rc=$rc\" >> mapdl_run.log\n"
        f"if [ $rc -eq 0 ]; then touch {tag}.success; fi\n"
        "exit $rc\n"
    )
    (runfolder / name).write_text(script, encoding="ascii", newline="\n")
    return name


def _solve_local(workspace_root: Path, license_detector, runfolder: Path,
                 driver: str, tag: str, ncpu: int, memory_mb: int) -> bool:
    """One docker compose MAPDL run (same service the cablestack stages use)."""
    docker_dir = workspace_root / "scripts" / "apdl" / "docker"
    log_path = runfolder / f"mapdl_{tag}.log"

    env = os.environ.copy()
    env["MAPDL_RUN_DIR"] = str(runfolder.absolute())
    env["MAPDL_INPUT"] = driver
    env["MAPDL_JOBNAME"] = f"compbox_{tag}"
    env["MAPDL_NCPU"] = str(ncpu)
    env["MAPDL_MEMORY"] = str(memory_mb)
    license_server = license_detector.detect()
    env["ANSYSLI_SERVERS"] = license_server
    env["ANSYSLMD_LICENSE_FILE"] = license_server

    project = f"mapdl_{runfolder.parents[2].name}_compbox_{tag}".lower()

    # Stale-lock guard (same failure mode as the HPC jobscripts).
    for lock in runfolder.glob("*.lock"):
        lock.unlink(missing_ok=True)

    try:
        subprocess.run(["docker", "compose", "-p", project, "down"],
                       cwd=str(docker_dir), env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)
        print(f"[compbox] {tag}: running {driver} in Docker MAPDL "
              f"({ncpu} cpus) - logs: {log_path.name}")
        with open(log_path, "w") as log_file:
            result = subprocess.run(
                ["docker", "compose", "-p", project, "up", "--no-color",
                 "--abort-on-container-exit", "--exit-code-from", "mapdl"],
                cwd=str(docker_dir), env=env,
                stdout=log_file, stderr=subprocess.STDOUT,
            )
        ok = result.returncode == 0
        print(f"[compbox] {tag}: {'completed' if ok else f'exited rc={result.returncode}'}"
              f"{'' if ok else f' - check {log_path}'}")
        return ok
    except Exception as e:
        logger.error(f"[compbox] {tag}: Docker MAPDL launch failed: {e}")
        traceback.print_exc()
        return False


def _solve_hpc(runfolder: Path, run_label: str, tag: str, driver: str,
               ntasks: int, mem_per_cpu: str,
               fetch_patterns: List[str]) -> bool:
    _here = str(Path(__file__).parent)
    if _here not in sys.path:
        sys.path.insert(0, _here)
    import hpc_submit  # type: ignore

    jobscript = _write_jobscript(runfolder, tag, driver, ntasks, mem_per_cpu)
    remote_label = f"{run_label}_compbox_{tag}"
    ok = hpc_submit.submit_runfolder(
        runfolder, remote_label, jobscript=jobscript,
        fetch_patterns=fetch_patterns,
    )
    if not ok:
        return False
    results = hpc_submit.parse_stage_results(runfolder / "mapdl_run.log", [tag])
    return results.get(tag, False)


def _run_python_step(script: Path, cwd: Path, env_overrides: Dict[str, str],
                     log_path: Path) -> bool:
    env = os.environ.copy()
    env.update(env_overrides)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[compbox] {script.name} - logs: {log_path}")
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        rc = subprocess.run([sys.executable, str(script)], cwd=str(cwd),
                            env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        logger.error(f"[compbox] {script.name} exited rc={rc}; see {log_path}")
    return rc == 0


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def run_compression_box(workspace_root: Path, run_output_dir: Path,
                        license_detector=None, use_hpc: bool = False,
                        launch: bool = True) -> bool:
    """Run the compression box simulation on a run folder (workflow step 9).

    Reads cable + compression_box config from cable_parameters_user.json.
    launch=False stages the parent + submodel runfolders without solving
    (generate-files-only, like --no-cablestack for the cablestack stage).
    Returns True when every executed substep succeeded.
    """
    main_dir = Path(__file__).parent
    with open(main_dir / "cable_parameters_user.json") as f:
        cfg = json.load(f)

    # Cable resolution from the run-folder name (same logic as
    # copy_cablestack_files), falling back to active_cable.
    folder_parts = run_output_dir.name.split("_", 2)
    cable_label = folder_parts[2] if len(folder_parts) == 3 else None
    if cable_label:
        cable_label = re.sub(r"_apdl_rerun(_\d+)?$", "", cable_label)
    if not (cable_label and cable_label in cfg.get("cables", {})):
        cable_label = cfg["active_cable"]
    cable = cfg["cables"][cable_label]

    cb_cfg = cfg.get("compression_box", {})
    cs_cfg = cfg.get("cablestack", {})

    solver = "hpc" if use_hpc else cb_cfg.get("solver", "local")
    if solver not in ("local", "hpc"):
        raise ValueError(f"compression_box.solver must be 'local' or 'hpc', got {solver!r}")
    measurement_file = cb_cfg.get("measurement_file", "BOX9.txt")
    nsteps_cap = int(cb_cfg.get("nsteps", 0))
    run_analysis = bool(cb_cfg.get("run_analysis", True))
    impreg = int(cb_cfg.get("impreg", cs_cfg.get("impreg", 4)))
    local_ncpu = int(cb_cfg.get("local_ncpu", 4))
    local_memory_mb = int(cb_cfg.get("local_memory_mb", 4000))
    hpc_ntasks = int(cb_cfg.get("hpc_ntasks", 24))

    n_cases = int(cable["n_stacks"])
    n_strands = int(cable["N_Strands"])
    # Same half-size convention as the cablestack patch: the conformal-mesh
    # geometry was built against these dimensions.
    x_cab_margin_mm = cs_cfg.get("x_cab_margin_mm", 0.5)
    y_cab = cable.get("stack_height_mm", cable["cable_height"] + 0.3) / 2.0 * 1e-3
    x_cab = (cable["cable_width"] / 2.0 + x_cab_margin_mm) * 1e-3

    apdl_runfolder = run_output_dir / "APDL" / "submodel" / "apdl_runfolder"
    if not apdl_runfolder.is_dir():
        logger.error(f"[compbox] {apdl_runfolder} not found - step 7 (APDL "
                     f"submodel) must complete before the compression box stage.")
        return False

    dirs = _dirs(run_output_dir)
    for key in ("base", "vtu", "field", "results"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    analysis_dir = _templates(workspace_root)["analysis"]

    # Single source of truth for the loading schedule: the measurement table.
    # Both the parent box deck and the submodel BC deck get the same generated
    # nsteps/totalCurr/pres_pusher block; loading_schedule.json is the audit.
    measurement_path = resolve_measurement_path(workspace_root, measurement_file)
    steps = build_loading_steps(parse_measurement_table(measurement_path), nsteps_cap)
    with open(dirs["base"] / "loading_schedule.json", "w", encoding="utf-8") as f:
        json.dump({"source": str(measurement_path), "nsteps": len(steps),
                   "nsteps_cap": nsteps_cap, "steps": steps}, f, indent=2)

    if launch and solver == "local" and license_detector is None:
        sys.path.insert(0, str(main_dir))
        from license_detector import NetworkProbeLicenseDetector
        license_detector = NetworkProbeLicenseDetector()

    print("=" * 70)
    print(f"Compression box simulation: cable={cable_label}, cases={n_cases}, "
          f"{len(steps)} loadsteps ({measurement_path.name}), "
          f"solver={solver}{'' if launch else ' (stage-only)'}")
    print("=" * 70)

    status: Dict[str, Optional[bool]] = {
        "staging": None, "parent_mag": None, "vtu_export": None,
        "field_tables": None, "submodel": None, "analysis": None,
    }
    run_label = run_output_dir.name

    try:
        # 1. Parent box MAG ------------------------------------------------
        parent_rf = stage_parent_runfolder(
            workspace_root, run_output_dir, cable_label, x_cab, y_cab,
            steps, measurement_path.name)
        if launch:
            if solver == "hpc":
                status["parent_mag"] = _solve_hpc(
                    parent_rf, run_label, "parent_mag", PARENT_DRIVER,
                    hpc_ntasks, "4G",
                    fetch_patterns=["CompBox_MAG_*.rmg", "mapdl_*.log",
                                    "mapdl_run.log", "slurm-*.out", "slurm-*.err"])
            else:
                status["parent_mag"] = _solve_local(
                    workspace_root, license_detector, parent_rf,
                    PARENT_DRIVER, "parent_mag", local_ncpu, local_memory_mb)
            has_rmg = any(parent_rf.glob("CompBox_MAG_*.rmg"))
            if status["parent_mag"] and not has_rmg:
                logger.error("[compbox] parent_mag rc=0 but no .rmg produced.")
                status["parent_mag"] = False
            if not status["parent_mag"]:
                return _finalize(dirs["base"], status, cable_label, solver, launch)

            # 2. VTU export --------------------------------------------------
            status["vtu_export"] = _run_python_step(
                analysis_dir / "convert_rmg_to_vtu.py", analysis_dir,
                {"COMPBOX_BOX_RESULTS": str(parent_rf),
                 "COMPBOX_VTU_OUT": str(dirs["vtu"])},
                dirs["base"] / "vtu_export.log")
            if not status["vtu_export"]:
                return _finalize(dirs["base"], status, cable_label, solver, launch)

            # 3. Field tables ------------------------------------------------
            status["field_tables"] = _run_python_step(
                analysis_dir / "create_magnetic_heatmaps.py", analysis_dir,
                {"COMPBOX_ROOT": str(dirs["base"]),
                 "COMPBOX_VTU_DIR": str(dirs["vtu"]),
                 "COMPBOX_KEYPOINTS_DIR": str(apdl_runfolder),
                 "COMPBOX_FIELD_OUT_DIR": str(dirs["field"]),
                 "COMPBOX_HEATMAP_DIR": str(dirs["results"] / "heatmaps"),
                 "COMPBOX_N_CASES": str(n_cases)},
                dirs["base"] / "field_tables.log")
            if not status["field_tables"]:
                return _finalize(dirs["base"], status, cable_label, solver, launch)

        # 4. Submodel --------------------------------------------------------
        sub_rf = stage_submodel_runfolder(
            workspace_root, run_output_dir, apdl_runfolder, n_cases,
            n_strands, cable_label, x_cab, y_cab, impreg, measurement_path,
            steps, require_field_tables=launch)
        if launch:
            if solver == "hpc":
                status["submodel"] = _solve_hpc(
                    sub_rf, run_label, "submodel", SUBMODEL_DRIVER,
                    hpc_ntasks, "2G",
                    fetch_patterns=["strains_out_strand_*_set_*_case_*.out",
                                    "mapdl_*.log", "mapdl_run.log",
                                    "slurm-*.out", "slurm-*.err"])
            else:
                status["submodel"] = _solve_local(
                    workspace_root, license_detector, sub_rf,
                    SUBMODEL_DRIVER, "submodel", local_ncpu, local_memory_mb)
            n_strain = len(list(sub_rf.glob("strains_out_strand_*_set_*_case_*.out")))
            print(f"[compbox] submodel strain output files: {n_strain}")
            if status["submodel"] and n_strain == 0:
                logger.error("[compbox] submodel rc=0 but no strains_out files.")
                status["submodel"] = False
            if not status["submodel"]:
                return _finalize(dirs["base"], status, cable_label, solver, launch)

            # 5. Analysis ----------------------------------------------------
            if run_analysis:
                strain_dir = dirs["results"] / "strain_analysis"
                ok_strain = _run_python_step(
                    analysis_dir / "complete_analysis.py", analysis_dir,
                    {"COMPBOX_DATA_DIR": str(sub_rf),
                     "COMPBOX_RESULTS_DIR": str(strain_dir),
                     "COMPBOX_PLOT_DIR": str(strain_dir / "Plots"),
                     "COMPBOX_EXPERIMENT_FILE": measurement_path.name},
                    dirs["base"] / "strain_analysis.log")
                ok_ic = _run_python_step(
                    analysis_dir / "ic_calculator.py", analysis_dir, {},
                    dirs["results"] / "ic_calculation.log")
                status["analysis"] = ok_strain and ok_ic

        status["staging"] = True
        return _finalize(dirs["base"], status, cable_label, solver, launch)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"[compbox] {e}")
        status["staging"] = False
        return _finalize(dirs["base"], status, cable_label, solver, launch)


def _finalize(base_dir: Path, status: Dict[str, Optional[bool]],
              cable_label: str, solver: str, launch: bool) -> bool:
    """Write compbox_summary.json and reduce the substep statuses to a bool.

    None entries are substeps that were not reached/executed; a stage-only
    run with no failures counts as success.
    """
    summary = {
        "cable": cable_label,
        "solver": solver,
        "launched": launch,
        "substeps": status,
        "nb3sn_modulus": {
            "value_Pa": NB3SN_STANDARD_PA,
            "source": "standard_70GPa",
            "note": "RVE override deliberately not applied (amplification "
                    "factor unresolved); see compression_box stage docs.",
        },
    }
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
        with open(base_dir / "compbox_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except OSError as e:
        logger.warning(f"[compbox] Could not write compbox_summary.json: {e}")

    executed = {k: v for k, v in status.items() if v is not None}
    ok = all(executed.values()) if executed or not launch else False
    print(f"[compbox] Substep results: {status} -> {'OK' if ok else 'FAILED'}")
    return ok
