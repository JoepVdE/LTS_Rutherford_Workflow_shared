"""Build a plane-stress sibling of an existing GPS apdl_runfolder.

For each cable, clones the existing apdl_runfolder/ to apdl_runfolder_ps/, overwrites
all formulation-sensitive templates from scripts/apdl/submodel/cablestack/, patches
0-start.inp to set `formulation = 0` (plane stress), and writes a fresh non-fatal
jobslurm.sh (every stage runs even if a prior one fails).

The cable-specific files (keypoints_*, contacts_*, loading_cycle.json, 0-start.inp's
geometry parameters n_strands/n_stacks/x_cab/y_cab) are preserved.

Usage:
    python build_plane_stress_run.py [CABLE ...]

With no arguments, builds PS siblings for R2D2_HF, R2D2_LF and CD1, each from
the most recently modified run folder of that cable that has an apdl_runfolder.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO_ROOT / "scripts" / "apdl" / "submodel" / "cablestack"
RUNS_ROOT = REPO_ROOT / "data" / "runs"
DEFAULT_CABLES = ["R2D2_HF", "R2D2_LF", "CD1"]

# Templates that change with the new formulation toggle / non-fatal slurm / fresh-start
# refactor — overwrite from source so the cloned PS run picks up bug fixes.
# Note: 0-start.inp is NOT in this list because it carries cable-specific patches
# (n_strands, n_stacks, x_cab, y_cab) that we want to preserve from the source
# apdl_runfolder. The SAVE,base,db line it needs is injected separately by
# ensure_save_base_db().
TEMPLATE_FILES = [
    "2-geo.inp",
    "5-BC-solver-settings.inp",
    "5-BC-cyclic.inp",
    "5-BC-linear.inp",
    "5-BC-displacement-radial.inp",
    "5-BC-pressure.inp",
    "5-BC-radial.inp",
    "6-PP.inp",
    "7-PP.inp",
    "8-PP-pressure.inp",
    "8-PP-radial.inp",
    "00-restart-transverse.inp",
    "00-restart-radial.inp",
    "00-restart-pressure.inp",
    "00-restart-pressure-radial.inp",
    # Free-compression BC variants (new under boundary_type toggle)
    "5-BC-cyclic-free.inp",
    "5-BC-displacement-radial-free.inp",
    "5-BC-pressure-free.inp",
    "5-BC-radial-free.inp",
]

def find_latest_apdl_runfolder(cable: str) -> Optional[Path]:
    """Most recently modified <run>/APDL/submodel/apdl_runfolder for this cable.

    Matches run folders named ..._<CABLE> or ..._<CABLE>_apdl_rerun[_N];
    skips already-built _ps siblings.
    """
    candidates = []
    if not RUNS_ROOT.is_dir():
        return None
    for run in RUNS_ROOT.iterdir():
        name = run.name
        if not run.is_dir() or name.endswith("_ps"):
            continue
        if not (name.endswith(f"_{cable}") or f"_{cable}_apdl_rerun" in name):
            continue
        apdl_rf = run / "APDL" / "submodel" / "apdl_runfolder"
        if (apdl_rf / "0-start.inp").is_file():
            candidates.append(apdl_rf)
    return max(candidates, key=lambda p: p.parents[2].stat().st_mtime, default=None)


def ensure_save_base_db(path: Path) -> None:
    """Ensure 0-start.inp writes base.db right after /inp,4-cont,inp.

    The cloned source apdl_runfolder may pre-date the fresh-start refactor and
    lack the SAVE,base,db line. The four restart decks (00-restart-transverse,
    00-restart-radial, 00-restart-pressure, 00-restart-pressure-radial) RESUME
    from base.db, so it must exist for any restart stage to work.
    """
    text = path.read_text(encoding="utf-8")
    if re.search(r"^\s*SAVE\s*,\s*base\s*,\s*db", text, re.IGNORECASE | re.MULTILINE):
        return  # already present
    # Insert immediately after the /inp,4-cont,inp line.
    inject = (
        "\n"
        "! Build-only deck. All four loading stages live in 00-restart-*.inp files\n"
        "! and RESUME base.db. No BC, no SOLVE here.\n"
        "SAVE,base,db\n"
    )
    new_text, n = re.subn(
        r"(^/inp\s*,\s*4-cont\s*,\s*inp\s*$)",
        r"\1" + inject,
        text,
        count=1,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if n == 0:
        # Couldn't anchor on 4-cont; fall back: prepend just before 5-BC call.
        new_text, n = re.subn(
            r"(^/inp\s*,\s*5-BC[^\n]*$)",
            inject.lstrip("\n") + r"\1",
            text,
            count=1,
            flags=re.MULTILINE | re.IGNORECASE,
        )
    if n == 0:
        raise RuntimeError(f"Could not inject SAVE,base,db into {path} (no 4-cont or 5-BC anchor)")
    path.write_text(new_text, encoding="utf-8", newline="\n")


def patch_fresh_start_usecases(dst: Path, cable: str) -> None:
    """Patch the hardcoded `usecase = 'R2D2_LF_...'` line in each fresh-start template
    to use the actual cable label. Mirrors main.py copy_cablestack_files (lines 1423,
    1484, 1518) which does the same for the GPS pipeline.

    Without this, HF/CD1 PS runs write pp/*.txt files with R2D2_LF embedded in the
    filename — analyse_pressure looks for <cable>_<suffix> and silently skips.
    """
    suffixes = {
        "00-restart-transverse.inp":      "",
        "00-restart-radial.inp":          "_disp_radial",
        "00-restart-pressure.inp":        "_pressure",
        "00-restart-pressure-radial.inp": "_radial",
    }
    for fname, suffix in suffixes.items():
        path = dst / fname
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        new_text, n = re.subn(
            r"^usecase\s*=\s*'[^']*'",
            f"usecase = '{cable}{suffix}'",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if n == 0:
            raise RuntimeError(f"Could not find `usecase = '...'` line in {path}")
        path.write_text(new_text, encoding="utf-8", newline="\n")


def patch_0start(path: Path) -> None:
    """Ensure 0-start.inp has `formulation = 0` (plane stress)."""
    text = path.read_text(encoding="utf-8")
    if re.search(r"^\s*formulation\s*=", text, re.MULTILINE):
        text = re.sub(r"^\s*formulation\s*=\s*\d+.*$",
                      "formulation = 0   ! 0 = plane stress (this run), 1 = generalized plane strain",
                      text, count=1, flags=re.MULTILINE)
    else:
        # Insert after the one_area_impreg / side_support block
        anchor = re.search(r"(one_area_impreg\s*=.*$)", text, re.MULTILINE)
        if not anchor:
            anchor = re.search(r"(n_stacks\s*=.*$)", text, re.MULTILINE)
        if anchor:
            insert_at = anchor.end()
            insertion = (
                "\n\n"
                "! formulation: 1 = generalized plane strain + mixed u-P (default).\n"
                "!              0 = plane stress (KEYOPT(3)=0, no u-P, no GSGDATA/GSBDATA) -- matches Zwick test.\n"
                "formulation = 0"
            )
            text = text[:insert_at] + insertion + text[insert_at:]
        else:
            text = "formulation = 0\n" + text
    path.write_text(text, encoding="utf-8", newline="\n")


def write_jobslurm(dst: Path, cable: str) -> None:
    """Write a non-fatal restart-stage-safe jobslurm.sh.

    Every stage runs and logs its rc; failures do NOT abort the rest of the job.
    Also removes any stale *.lock from prior aborted runs.
    """
    content = f"""#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --job-name={cable}_ps
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --tasks-per-node=16
#SBATCH --constraint=ethernet
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-ps-%j.out
#SBATCH --error=slurm-ps-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joep@ethz.ch

# Plane-stress run for {cable}. Each stage is non-fatal: a failed stage logs its
# rc but does NOT stop subsequent stages from attempting.
set -uo pipefail

DB_FILE=base.db

# Clear any stale lock files from prior scancels/crashes (see CLAUDE.md gotcha).
rm -f *.lock file.lock

module purge
module load ansys/24.2_research stack/2024-06 mesa-glu/9.0.2 motif/2.3.8-ehjpbwx

COMBINED_LOG="mapdl_run_ps.log"

echo "[$(date '+%F %T')] JOB_START jobid=${{SLURM_JOB_ID:-unknown}} host=$(hostname)" | tee -a "${{COMBINED_LOG}}"

run_stage_fatal () {{
  # 0-start failure aborts the job (no .db -> nothing to restart).
  local tag="$1"; local inp="$2"
  echo "[$(date '+%F %T')] ${{tag}}_START input=${{inp}}" | tee -a "${{COMBINED_LOG}}"
  mapdl -np 16 -mpp -i "${{inp}}" >> "${{COMBINED_LOG}}" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] ${{tag}}_END rc=${{rc}}" | tee -a "${{COMBINED_LOG}}"
  if [ "${{rc}}" -ne 0 ]; then
    echo "[$(date '+%F %T')] ${{tag}}_FAILED rc=${{rc}} -- aborting job (no .db for restarts)" | tee -a "${{COMBINED_LOG}}"
    exit "${{rc}}"
  fi
}}

run_stage_continue () {{
  # Failed stage logs rc but the job moves on to the next.
  local tag="$1"; local inp="$2"
  echo "[$(date '+%F %T')] ${{tag}}_START input=${{inp}}" | tee -a "${{COMBINED_LOG}}"
  mapdl -np 16 -mpp -i "${{inp}}" >> "${{COMBINED_LOG}}" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] ${{tag}}_END rc=${{rc}}" | tee -a "${{COMBINED_LOG}}"
  if [ "${{rc}}" -ne 0 ]; then
    echo "[$(date '+%F %T')] ${{tag}}_FAILED rc=${{rc}} -- continuing with next stage" | tee -a "${{COMBINED_LOG}}"
    rm -f *.lock file.lock
  fi
}}

run_stage_fatal build ./0-start.inp

if [ ! -f "${{DB_FILE}}" ]; then
  echo "[$(date '+%F %T')] restart_SKIPPED db='${{DB_FILE}}' not found -- build did not complete" | tee -a "${{COMBINED_LOG}}"
  exit 1
fi

run_stage_continue displacement_transverse ./00-restart-transverse.inp
run_stage_continue displacement_radial    ./00-restart-radial.inp
run_stage_continue pressure_transverse    ./00-restart-pressure.inp
run_stage_continue pressure_radial        ./00-restart-pressure-radial.inp

echo "[$(date '+%F %T')] JOB_END" | tee -a "${{COMBINED_LOG}}"
"""
    dst.write_text(content, encoding="utf-8", newline="\n")


def build_one_cable(cable: str, src_apdl_rf: Path) -> Path:
    """Create <run>_ps/apdl_runfolder/ sibling and return its path."""
    src_run = src_apdl_rf.parents[2]      # 20260504_..._apdl_rerun_51
    ps_run  = RUNS_ROOT / (src_run.name + "_ps")
    ps_apdl = ps_run / "APDL" / "submodel" / "apdl_runfolder"
    if ps_apdl.exists():
        shutil.rmtree(ps_apdl)
    shutil.copytree(src_apdl_rf, ps_apdl)

    # Clean local-only artefacts so they don't get uploaded.
    # pp/ is recreated immediately as an empty dir -- 7-PP / 8-PP-*.inp open
    # output files via the 'pp/<name>' path prefix, which fails if the dir
    # doesn't exist on the cluster (slurm error: I/O code 29).
    for pat in ["pp", "plots"]:
        d = ps_apdl / pat
        if d.is_dir():
            shutil.rmtree(d)
    (ps_apdl / "pp").mkdir(exist_ok=True)
    for pat in ("*.db", "*.dbb", "*.rst", "*.full", "*.esav", "*.emat",
                "*.r0??", "*.rdb", "*.mntr", "*.stat", "*.lock",
                "file.err", "file.log", "file.page",
                "file.PAGE", "file.LOCK", "menust.tmp", "*.tmp", "*.BAT",
                "anstmp", "cleanup*.bat", "cleanup*.sh",
                "mapdl_run*.log", "slurm-*.out", "slurm-*.err",
                # Dead under the 4-stage fresh-start design. Older sources hardcoded
                # usecaserestart='submodel_cable_34_R2D2_LF' here; leaving them
                # behind risks an old jobslurm.sh calling them with wrong cable.
                "00-restart*.inp"):
        for p in ps_apdl.glob(pat):
            try:
                p.unlink()
            except OSError:
                pass

    # Overwrite formulation-sensitive templates from source
    for f in TEMPLATE_FILES:
        src = TEMPLATE_DIR / f
        if src.is_file():
            shutil.copy2(src, ps_apdl / f)

    # Patch 0-start.inp: formulation = 0 + ensure SAVE,base,db is present
    # (older cloned sources may pre-date the fresh-start refactor)
    patch_0start(ps_apdl / "0-start.inp")
    ensure_save_base_db(ps_apdl / "0-start.inp")

    # Patch hardcoded usecase in the 3 fresh-start restart templates per cable.
    patch_fresh_start_usecases(ps_apdl, cable)

    # Make 5-BC.inp link to whichever cyclic/linear the original deck used
    # (the cable-specific 0-start.inp uses /inp,5-BC-cyclic,inp directly, so 5-BC.inp is unused — leave whatever is there)

    # Fresh non-fatal jobslurm.sh
    write_jobslurm(ps_apdl / "jobslurm.sh", cable)

    return ps_apdl


def main() -> None:
    cables = sys.argv[1:] or DEFAULT_CABLES
    built: list[tuple[str, Path]] = []
    for cable in cables:
        src = find_latest_apdl_runfolder(cable)
        if src is None:
            print(f"[skip] {cable}: no run folder with an apdl_runfolder under {RUNS_ROOT}")
            continue
        ps_apdl = build_one_cable(cable, src)
        run_name = ps_apdl.parents[2].name
        print(f"[ok ] {cable}: {ps_apdl}  (run_label={run_name}, source={src.parents[2].name})")
        built.append((cable, ps_apdl))

    print("\nBuilt plane-stress run folders:")
    for cable, p in built:
        print(f"  {cable:<10} -> {p}")
    print("\nNext: upload each to /cluster/scratch/jvanden/cablestack_runs/<run_name> and sbatch jobslurm.sh")


if __name__ == "__main__":
    main()
