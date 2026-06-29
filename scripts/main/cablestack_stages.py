"""Cablestack stage registry + helpers.

Single source of truth for the four-stage cablestack architecture.  main.py
imports the registry and the order/jobslurm helpers;
scripts/analysis/submodel/cablestack/analyse_pressure.py imports
STAGE_USECASE_SUFFIX so the output-filename convention cannot drift between
the solver side and the postprocess side.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


# Cablestack APDL stage registry.
# Each stage = one MAPDL invocation (driver .inp), with a usecase suffix appended
# to the cable label and a postprocess dispatch tag consumed by
# WorkflowRunner.run_cablestack_postprocess.  depends_on is the list of stages
# whose .db must already exist before this stage can restart.
#
# Stage naming convention: <bc_mode>_<direction> where
#   bc_mode   in {displacement, pressure}
#   direction in {transverse (vertical, top surface), radial (horizontal, left wall)}
CABLESTACK_STAGES: Dict[str, Dict[str, object]] = {
    "build": {
        "input_file": "0-start.inp",
        "depends_on": [],
        "usecase_suffix": "",                  # build uses bare cable label as jobname
        "post_tag": "build",
        "description": (
            "Geometry + mesh + contact build only. Writes base.db (post-mesh, "
            "pre-load) which every loading stage RESUMEs. No BC, no SOLVE."
        ),
    },
    "displacement_transverse": {
        "input_file": "00-restart-transverse.inp",
        "depends_on": ["build"],
        "usecase_suffix": "",                  # plain <cable_label>
        "post_tag": "displacement_transverse",
        "description": (
            "Displacement-controlled transverse compaction on an UNDEFORMED cable. "
            "00-restart-transverse.inp RESUMEs base.db, then applies UY ramp on "
            "top wall via 5-BC.inp (cyclic/linear, optionally -free per JSON)."
        ),
    },
    "displacement_radial": {
        "input_file": "00-restart-radial.inp",
        "depends_on": ["build"],
        "usecase_suffix": "_disp_radial",
        "post_tag": "displacement_radial",
        "description": (
            "Displacement-controlled radial compaction on an UNDEFORMED cable. "
            "00-restart-radial.inp RESUMEs base.db, then applies UX ramp on "
            "left wall via 5-BC-displacement-radial.inp (optionally -free)."
        ),
    },
    "pressure_transverse": {
        "input_file": "00-restart-pressure.inp",
        "depends_on": ["build"],
        "usecase_suffix": "_pressure",
        "post_tag": "pressure_transverse",
        "description": (
            "Cyclic SFL pressure on top surface, applied to an UNDEFORMED cable. "
            "00-restart-pressure.inp RESUMEs base.db, then runs 5-BC-pressure + "
            "8-PP-pressure (optionally -free per JSON)."
        ),
    },
    "pressure_radial": {
        "input_file": "00-restart-pressure-radial.inp",
        "depends_on": ["build"],
        "usecase_suffix": "_radial",
        "post_tag": "pressure_radial",
        "description": (
            "Cyclic SFL pressure on left wall, applied to an UNDEFORMED cable. "
            "00-restart-pressure-radial.inp RESUMEs base.db, then runs "
            "5-BC-radial + 8-PP-radial (optionally -free per JSON)."
        ),
    },
    # ---- SKELETON STAGE (not yet implemented) -----------------------------
    # Thermal cooldown 293 K -> 4.2 K to develop the CTE-mismatch axial
    # pre-strain on Nb3Sn filaments before mechanical loading.  Architecture
    # is wired (template files, postprocess stub, dispatch entry) but the
    # physics inside 5-BC-thermal.inp is intentionally a no-op until CTE
    # values + load step are implemented.  Do NOT add to cablestack.stages
    # in cable_parameters_user.json until the physics is filled in.
    "thermal_cooldown": {
        "input_file": "0-start-thermal.inp",
        "depends_on": [],
        "usecase_suffix": "_thermal",
        "post_tag": "thermal_cooldown",
        "description": "SKELETON: cooldown 293 K -> 4.2 K for CTE-mismatch pre-strain on Nb3Sn. Not implemented.",
    },
}


# Derived map consumed by analyse_pressure.py (filename convention
# fd_*_<cable><suffix>.txt).  Import this instead of re-declaring it.
STAGE_USECASE_SUFFIX: Dict[str, str] = {
    name: str(spec["usecase_suffix"]) for name, spec in CABLESTACK_STAGES.items()
}


def resolve_cablestack_stage_order(stages: List[str]) -> List[str]:
    """Return stages in dependency order, auto-including any missing dependencies.

    Unknown stage names are dropped with a warning.  Duplicates are collapsed.
    """
    seen: List[str] = []

    def _visit(name: str) -> None:
        if name in seen:
            return
        if name not in CABLESTACK_STAGES:
            logger.warning(f"Unknown cablestack stage '{name}' — ignoring.")
            return
        for dep in CABLESTACK_STAGES[name]["depends_on"]:  # type: ignore[index]
            _visit(dep)
        seen.append(name)

    for s in stages:
        _visit(s)
    return seen


def write_cablestack_jobslurm(dst_dir: Path, cable_label: str, n_strands: int,
                              ordered_stages: List[str]) -> Path:
    """Generate jobslurm.sh covering only the requested stages, in dependency order.

    The log tags emitted by run_stage (`<tag>_START` / `<tag>_END rc=N`) are exactly
    the stage names — hpc_submit.parse_stage_results greps for that pattern when
    determining per-stage success after fetching mapdl_run.log.

    Failure policy:
      - `build` is fatal: without base.db none of the restart stages can run.
      - Restart stages are non-fatal: they are mutually independent (each
        RESUMEs base.db from an undeformed state), so one failed stage must not
        prevent the others from running.  The first non-zero rc is remembered
        in GLOBAL_RC and becomes the job's exit code, so SLURM still reports
        the job as failed.
      - Each stage writes `<stage>.success` on rc=0 so partial runs are
        machine-detectable, and stale MAPDL lock files are removed before the
        job and after any failed stage (see the file.lock gotcha in CLAUDE.md).
    """
    db_file = "base.db"
    mail_user = os.environ.get("HPC_MAIL_USER", "joep@ethz.ch")

    header = f"""#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --job-name={cable_label}
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --tasks-per-node=16
#SBATCH --constraint=ethernet
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user={mail_user}

# No `set -e`: restart stages are non-fatal by design; rc is captured per stage.
set -uo pipefail

# Post-mesh, pre-load database written by 0-start.inp during the `build` stage.
# All 00-restart-*.inp files RESUME this single file (it is cable-independent
# in name -- the cable label is baked into the contents).
DB_FILE={db_file}

# Clear stale MAPDL lock files and stage sentinels from prior scancels/crashes.
rm -f *.lock file.lock *.success

module purge
module load ansys/24.2_research stack/2024-06 mesa-glu/9.0.2 motif/2.3.8-ehjpbwx

COMBINED_LOG="mapdl_run.log"
GLOBAL_RC=0

echo "[$(date '+%F %T')] JOB_START jobid=${{SLURM_JOB_ID:-unknown}} host=$(hostname)" | tee -a "${{COMBINED_LOG}}"

# Fatal stage: failure aborts the whole job (build only -- no .db, no restarts).
run_stage_fatal () {{
  local tag="$1"
  local inp="$2"

  echo "[$(date '+%F %T')] ${{tag}}_START input=${{inp}}" | tee -a "${{COMBINED_LOG}}"
  mapdl -np 16 -mpp -i "${{inp}}" >> "${{COMBINED_LOG}}" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] ${{tag}}_END rc=${{rc}}" | tee -a "${{COMBINED_LOG}}"

  if [ "${{rc}}" -ne 0 ]; then
    echo "[$(date '+%F %T')] ${{tag}}_FAILED rc=${{rc}} -- aborting job (no .db for restarts)" | tee -a "${{COMBINED_LOG}}"
    exit "${{rc}}"
  fi
  echo "rc=0 $(date '+%F %T')" > "${{tag}}.success"
}}

# Non-fatal stage: failure is logged, GLOBAL_RC is set, other stages continue.
run_stage () {{
  local tag="$1"
  local inp="$2"

  echo "[$(date '+%F %T')] ${{tag}}_START input=${{inp}}" | tee -a "${{COMBINED_LOG}}"
  mapdl -np 16 -mpp -i "${{inp}}" >> "${{COMBINED_LOG}}" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] ${{tag}}_END rc=${{rc}}" | tee -a "${{COMBINED_LOG}}"

  if [ "${{rc}}" -ne 0 ]; then
    echo "[$(date '+%F %T')] ${{tag}}_FAILED rc=${{rc}} -- continuing with next stage" | tee -a "${{COMBINED_LOG}}"
    GLOBAL_RC="${{rc}}"
    rm -f *.lock file.lock
  else
    echo "rc=0 $(date '+%F %T')" > "${{tag}}.success"
  fi
}}

"""

    lines = [header]
    db_guard_emitted = False
    for stage_name in ordered_stages:
        stage = CABLESTACK_STAGES[stage_name]
        inp_file = stage["input_file"]
        if stage["depends_on"] and not db_guard_emitted:
            lines.append(
                "# Guard: verify the prior stage wrote .db before launching this restart.\n"
                "if [ ! -f \"${DB_FILE}\" ]; then\n"
                "  echo \"[$(date '+%F %T')] restart_SKIPPED db='${DB_FILE}' not found\" | tee -a \"${COMBINED_LOG}\"\n"
                "  exit 1\n"
                "fi\n"
            )
            db_guard_emitted = True
        runner = "run_stage_fatal" if stage_name == "build" else "run_stage"
        lines.append(f"{runner} {stage_name} ./{inp_file}\n")

    lines.append(
        "\necho \"[$(date '+%F %T')] JOB_END rc=${GLOBAL_RC}\" | tee -a \"${COMBINED_LOG}\"\n"
        "exit \"${GLOBAL_RC}\"\n"
    )

    text = "\n".join(lines)
    dst = dst_dir / "jobslurm.sh"
    dst.write_text(text, encoding="utf-8", newline="\n")
    return dst
