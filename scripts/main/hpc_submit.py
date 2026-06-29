"""Upload + sbatch + wait + fetch a cablestack apdl_runfolder on an HPC cluster.

Cluster-agnostic: works with any SSH-reachable SLURM-managed HPC (ETH Euler,
CERN, PSI, your own).  Defaults target ETH Euler; override via HPC_* env vars.

Wrapped by solver.HPCMAPDL.  Caller is responsible for:
  - Generating jobslurm.sh inside local_dir (done by copy_cablestack_files).
  - Generating all *.inp files inside local_dir.

Assumes key-based SSH from this machine to the cluster (Windows OpenSSH
agent loaded) and network access (VPN if off-site).

Env-var overrides:
    HPC_USER          (default 'jvanden')
    HPC_HOST          (default 'euler.ethz.ch')
    HPC_REMOTE_BASE   (default '/cluster/scratch/jvanden/cablestack_runs')

Fetch policy: only the postprocess-relevant outputs come back -- pp/*, mapdl_run.log,
slurm-*.out, slurm-*.err.  Heavy MAPDL artefacts (.db, .rst, .full, .esav, .r0*)
stay on the cluster (Euler /cluster/scratch auto-deletes after 15 days; other
clusters have their own retention policies).
"""
from __future__ import annotations

import fnmatch
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

DEFAULT_USER        = "jvanden"
DEFAULT_HOST        = "euler.ethz.ch"
DEFAULT_REMOTE_BASE = "/cluster/scratch/jvanden/cablestack_runs"
POLL_S              = 60

FETCH_PATTERNS = [
    "pp/*",
    "mapdl_run.log",
    "slurm-*.out",
    "slurm-*.err",
    "*.success",       # per-stage sentinels written by jobslurm.sh on rc=0
]

# Files in apdl_runfolder that must NOT be uploaded.  These are either MAPDL
# session droppings from a prior local Docker run (.db/.dbb/.rst/.full/.esav/
# file.*/menust.tmp) or local-only artefacts (plots/ debug SVGs, pp/ outputs
# from a previous run).  The HPC cluster will write its own .db/.rst and pp/.
EXCLUDE_FILE_PATTERNS = (
    "*.db", "*.dbb", "*.rst", "*.full", "*.esav", "*.emat",
    "*.r0??", "*.rdb", "*.mntr", "*.stat",
    "file.err", "file.log", "file.page", "*.lock",
    "file.PAGE", "file.LOCK", "*.success",
    "menust.tmp", "*.tmp", "*.BAT",
    "anstmp", "cleanup*.bat", "cleanup*.sh",
)
EXCLUDE_DIR_NAMES = ("plots", "pp", "__pycache__")


def _remote() -> str:
    user = os.environ.get("HPC_USER", DEFAULT_USER)
    host = os.environ.get("HPC_HOST", DEFAULT_HOST)
    return f"{user}@{host}"


def _remote_base() -> str:
    return os.environ.get("HPC_REMOTE_BASE", DEFAULT_REMOTE_BASE)


def _sh(cmd, check=True, capture=False):
    logger.info("[HPC] $ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def ssh_ok() -> bool:
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
         "-o", "StrictHostKeyChecking=accept-new", _remote(), "true"],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def _ssh_help_message() -> str:
    return (
        f"\nSSH to {_remote()} failed (non-interactive). Two common causes:\n"
        f"  A. You are off the cluster's network (many academic clusters\n"
        f"     accept SSH only from inside the institution or via VPN).\n"
        f"  B. No SSH key loaded in the Windows OpenSSH agent. Set up:\n"
        f"     1.  ssh-keygen -t ed25519\n"
        f"     2.  type %USERPROFILE%\\.ssh\\id_ed25519.pub | "
        f"ssh {_remote()} \"cat >> .ssh/authorized_keys\"\n"
        f"     3.  ssh-add  %USERPROFILE%\\.ssh\\id_ed25519\n"
    )


def _should_skip(p: Path) -> bool:
    if p.is_dir() and p.name in EXCLUDE_DIR_NAMES:
        return True
    if p.is_file() and any(fnmatch.fnmatch(p.name, pat) for pat in EXCLUDE_FILE_PATTERNS):
        return True
    return False


def upload(local_dir: Path, remote_dir: str) -> None:
    """Stage a filtered copy of local_dir, then scp -r the staging to remote_dir.

    Filtering drops MAPDL session droppings (.db, .dbb, .rst, .full, .esav,
    file.*, menust.tmp) and local-only directories (plots/, pp/, __pycache__/)
    so only the cablestack deck (~3 MB) is transferred regardless of what
    leftover artefacts sit in local_dir from prior Docker runs.

    The remote pp/ directory is recreated empty so APDL *cfopen has a writable
    target for fd_*.txt / uy_top_*.txt / ux_left_*.txt / area_summary.txt.
    """
    _sh(["ssh", _remote(), f"mkdir -p {remote_dir} {remote_dir}/pp"])

    total_files = 0
    total_bytes = 0
    skipped = 0
    with tempfile.TemporaryDirectory(prefix="hpc_upload_") as td:
        staging = Path(td) / "upload"
        staging.mkdir()
        for p in local_dir.iterdir():
            if _should_skip(p):
                skipped += 1
                continue
            if p.is_file():
                shutil.copy2(p, staging / p.name)
                total_files += 1
                total_bytes += p.stat().st_size
            # No nested dirs are expected in apdl_runfolder beyond plots/ and pp/,
            # both of which are excluded.  Anything else is silently ignored.

        logger.info(
            f"[HPC] Upload {local_dir.name} -> {_remote()}:{remote_dir} "
            f"({total_files} files, {total_bytes/1024:.0f} KiB; {skipped} entries skipped)"
        )
        _sh(["scp", "-r", "-q",
             f"{staging.as_posix()}/.",
             f"{_remote()}:{remote_dir}/"])


def submit(remote_dir: str) -> int:
    return submit_jobscript(remote_dir, "jobslurm.sh")


def wait_for(jobid: int, poll_s: int = POLL_S) -> None:
    logger.info(f"[HPC] Poll squeue every {poll_s}s for JOBID {jobid}")
    state_prev = None
    while True:
        r = subprocess.run(
            ["ssh", _remote(), f"squeue -j {jobid} -h -o '%T %M %R'"],
            capture_output=True, text=True,
        )
        line = (r.stdout or "").strip()
        if not line:
            logger.info(f"[HPC] JOBID {jobid} no longer in queue (finished or cancelled)")
            return
        if line != state_prev:
            logger.info(f"[HPC] {jobid}: {line}")
            state_prev = line
        time.sleep(poll_s)


def fetch_postprocess_outputs(local_dir: Path, remote_dir: str) -> None:
    logger.info(f"[HPC] Fetch postprocess outputs {_remote()}:{remote_dir} -> {local_dir}")
    (local_dir / "pp").mkdir(parents=True, exist_ok=True)
    for pat in FETCH_PATTERNS:
        if pat.startswith("pp/"):
            dst = (local_dir / "pp").as_posix() + "/"
        else:
            dst = local_dir.as_posix() + "/"
        rc = subprocess.run(
            ["scp", "-q", f"{_remote()}:{remote_dir}/{pat}", dst],
        ).returncode
        logger.info(f"[HPC]   {'ok  ' if rc == 0 else 'miss'} {pat}")


def parse_stage_results(log_path: Path, ordered_stages: List[str]) -> Dict[str, bool]:
    """Parse mapdl_run.log for <stage>_END rc=N markers emitted by jobslurm.sh."""
    results: Dict[str, bool] = {s: False for s in ordered_stages}
    if not log_path.is_file():
        return results
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for s in ordered_stages:
        m = re.search(rf"{re.escape(s)}_END rc=(\d+)", text)
        if m and int(m.group(1)) == 0:
            results[s] = True
    return results


def fetch_outputs(local_dir: Path, remote_dir: str, patterns: List[str]) -> None:
    """Generic scp fetch of glob patterns from remote_dir into local_dir."""
    logger.info(f"[HPC] Fetch outputs {_remote()}:{remote_dir} -> {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    for pat in patterns:
        rc = subprocess.run(
            ["scp", "-q", f"{_remote()}:{remote_dir}/{pat}",
             local_dir.as_posix() + "/"],
        ).returncode
        logger.info(f"[HPC]   {'ok  ' if rc == 0 else 'miss'} {pat}")


def submit_runfolder(local_dir: Path, run_label: str,
                     jobscript: str = "jobslurm.sh",
                     fetch_patterns: List[str] = ()) -> bool:
    """Generic upload -> sbatch -> wait -> fetch round-trip for one runfolder.

    Used by the compression box stage (one SLURM job per runfolder); the
    cablestack stage keeps its own submit_cablestack wrapper below.  Returns
    False on SSH/upload/sbatch failure; job-level rc must be parsed by the
    caller (e.g. parse_stage_results on a fetched mapdl_run.log).
    """
    if not ssh_ok():
        logger.error(_ssh_help_message())
        return False
    if not (local_dir / jobscript).is_file():
        logger.error(f"[HPC] {jobscript} missing in {local_dir} -- abort")
        return False
    remote_dir = f"{_remote_base()}/{run_label}"
    try:
        upload(local_dir, remote_dir)
        jobid = submit_jobscript(remote_dir, jobscript)
        logger.info(f"[HPC] JOBID {jobid} submitted; waiting...")
        wait_for(jobid)
        fetch_outputs(local_dir, remote_dir, list(fetch_patterns))
        logger.warning("[HPC] Note: ETH Euler /cluster/scratch auto-deletes after 15 days; check your cluster's retention policy.")
    except (subprocess.CalledProcessError, RuntimeError) as e:
        logger.error(f"[HPC] {e}")
        return False
    return True


def submit_jobscript(remote_dir: str, jobscript: str = "jobslurm.sh") -> int:
    logger.info(f"[HPC] sbatch {jobscript} in {remote_dir}")
    r = _sh(["ssh", _remote(),
             f"cd {remote_dir} && sbatch {jobscript}"], capture=True)
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    if out:
        logger.info(f"[HPC] sbatch: {out}")
    if err:
        logger.warning(f"[HPC] sbatch stderr: {err}")
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        raise RuntimeError(f"sbatch returned no JOBID. stdout={out!r} stderr={err!r}")
    return int(m.group(1))


def submit_cablestack(local_dir: Path, run_label: str,
                      ordered_stages: List[str]) -> Dict[str, bool]:
    """End-to-end: ssh check -> upload -> sbatch -> wait -> fetch -> parse rc.

    local_dir: <run>/APDL/submodel/apdl_runfolder/ (jobslurm.sh + *.inp inside).
    run_label: stable remote subdir name (typically the parent run-folder name).
    ordered_stages: stage names in dependency order, must match jobslurm.sh log tags.

    Returns {stage_name: success_bool}.  Empty dict on SSH/upload/sbatch failure.
    """
    if not ssh_ok():
        logger.error(_ssh_help_message())
        return {s: False for s in ordered_stages}
    if not (local_dir / "jobslurm.sh").is_file():
        logger.error(f"[HPC] jobslurm.sh missing in {local_dir} -- abort")
        return {s: False for s in ordered_stages}
    remote_dir = f"{_remote_base()}/{run_label}"
    try:
        upload(local_dir, remote_dir)
        jobid = submit(remote_dir)
        logger.info(f"[HPC] JOBID {jobid} submitted; waiting...")
        wait_for(jobid)
        fetch_postprocess_outputs(local_dir, remote_dir)
        logger.warning("[HPC] Note: ETH Euler /cluster/scratch auto-deletes after 15 days; check your cluster's retention policy.")
    except subprocess.CalledProcessError as e:
        logger.error(f"[HPC] command failed: {e}")
        return {s: False for s in ordered_stages}
    except RuntimeError as e:
        logger.error(f"[HPC] {e}")
        return {s: False for s in ordered_stages}

    return parse_stage_results(local_dir / "mapdl_run.log", ordered_stages)
