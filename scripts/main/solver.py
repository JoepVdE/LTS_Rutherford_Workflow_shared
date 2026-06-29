"""Cablestack MAPDL solver port + adapters.

Two implementations of the same port:

- LocalMAPDL: one `docker compose up` per stage, writes mapdl_<stage>.log per stage.
- HPCMAPDL: uploads runfolder to a SLURM HPC cluster (default ETH Euler;
  override via HPC_* env vars), sbatches jobslurm.sh (one job covers all
  stages), waits, fetches outputs, parses per-stage rc from mapdl_run.log.

Both expose run_stages(runfolder, ordered_stages, on_stage_complete=None)
returning {stage_name: success_bool} for the stages attempted.  Pre-flight gating
(base.db existence for restart stages) lives in LocalMAPDL because the cluster's
jobslurm.sh handles it remote-side.

The on_stage_complete callback fires once per stage with a StageResult, letting
the caller invoke postprocess at the natural granularity for each backend:
LocalMAPDL fires after each container exits; HPCMAPDL fires once per stage
after the batch job completes and per-stage rc is parsed.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol

from license_detector import LicenseDetector

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    name: str
    success: bool
    log_path: Optional[Path] = None


StageCallback = Callable[[StageResult], None]


class CablestackSolver(Protocol):
    """Port: runs an ordered list of cablestack stages against a runfolder."""

    def run_stages(
        self,
        runfolder: Path,
        ordered_stages: List[str],
        on_stage_complete: Optional[StageCallback] = None,
    ) -> Dict[str, bool]: ...


class LocalMAPDL:
    """Adapter: per-stage docker compose invocations on the local machine.

    Stages are grouped into dependency levels; stages within one level are
    mutually independent (the four load stages all RESUME base.db from an
    undeformed state) and run concurrently, up to max_parallel containers.
    Each parallel stage gets its own MAPDL jobname (stage_<name>) so the
    initial scratch/lock files cannot collide before the deck's /filname
    switches to the per-usecase jobname.  Note each container holds one MAPDL
    license seat for its duration -- dial max_parallel down via
    cablestack.max_parallel_stages in cable_parameters_user.json if seats are
    scarce.
    """

    def __init__(self, workspace_root: Path, license_detector: LicenseDetector,
                 stage_registry: Dict[str, Dict[str, object]],
                 max_parallel: int = 4):
        self.workspace_root = workspace_root
        self.license_detector = license_detector
        self.stages = stage_registry
        self.max_parallel = max(1, int(max_parallel))

    def _dependency_levels(self, ordered_stages: List[str]) -> List[List[str]]:
        """Group a dependency-ordered stage list into levels.

        Level N stages depend only on stages in levels < N, so all stages
        within one level can run concurrently.  Dependencies that are not in
        the list (e.g. a restart stage relying on an on-disk base.db from a
        prior run) contribute nothing -- the stage lands in level 0 and the
        base.db pre-flight in _run_one still guards it.
        """
        level_of: Dict[str, int] = {}
        levels: List[List[str]] = []
        for name in ordered_stages:
            spec = self.stages.get(name, {})
            deps = [d for d in spec.get("depends_on", []) if d in level_of]  # type: ignore[union-attr]
            lvl = max((level_of[d] + 1 for d in deps), default=0)
            level_of[name] = lvl
            while len(levels) <= lvl:
                levels.append([])
            levels[lvl].append(name)
        return levels

    def run_stages(
        self,
        runfolder: Path,
        ordered_stages: List[str],
        on_stage_complete: Optional[StageCallback] = None,
    ) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        # Postprocess callbacks (matplotlib etc.) are not thread-safe; results
        # land as stages finish, but callbacks fire one at a time.
        cb_lock = threading.Lock()

        def _finish(name: str, ok: bool, log_path: Optional[Path]) -> None:
            results[name] = ok
            if on_stage_complete is not None:
                with cb_lock:
                    on_stage_complete(StageResult(name=name, success=ok, log_path=log_path))

        for level in self._dependency_levels(ordered_stages):
            if len(level) == 1 or self.max_parallel == 1:
                for name in level:
                    ok, log_path = self._run_one(runfolder, name)
                    _finish(name, ok, log_path)
                continue
            workers = min(self.max_parallel, len(level))
            logger.info(
                f"Running {len(level)} independent cablestack stages in parallel "
                f"(max {workers} containers): {level}"
            )
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self._run_one, runfolder, name): name for name in level}
                for fut in as_completed(futures):
                    name = futures[fut]
                    try:
                        ok, log_path = fut.result()
                    except Exception as e:
                        logger.error(f"[{name}] stage thread failed: {e}")
                        traceback.print_exc()
                        ok, log_path = False, None
                    _finish(name, ok, log_path)

        n_ok = sum(1 for v in results.values() if v)
        logger.info(f"Cablestack stages: {n_ok}/{len(results)} MAPDL rc=0 - {results}")
        return results

    def _run_one(self, runfolder: Path, stage_name: str) -> tuple[bool, Optional[Path]]:
        if stage_name not in self.stages:
            logger.error(f"Unknown cablestack stage: {stage_name!r}")
            return False, None
        stage = self.stages[stage_name]
        input_file = str(stage["input_file"])

        # Pre-flight: restart stages need base.db from the build stage.
        if stage["depends_on"]:
            if not (runfolder / "base.db").is_file():
                logger.warning(
                    f"[{stage_name}] base.db not found in {runfolder.name}; "
                    f"skipping (requires the `build` stage to have completed)."
                )
                return False, None

        docker_dir = self.workspace_root / "scripts" / "apdl" / "docker"
        log_file_path = runfolder / f"mapdl_{stage_name}.log"

        env = os.environ.copy()
        env["MAPDL_RUN_DIR"] = str(runfolder.absolute())
        env["MAPDL_INPUT"] = input_file
        # Per-stage initial jobname: parallel stages share the run dir, so the
        # default jobname 'file' would collide on file.lock before each deck's
        # /filname takes over.  Also keeps per-stage <jobname>.out files apart.
        env["MAPDL_JOBNAME"] = f"stage_{stage_name}"
        license_server = self.license_detector.detect()
        env["ANSYSLI_SERVERS"] = license_server
        env["ANSYSLMD_LICENSE_FILE"] = license_server

        # Unique project name per (run, stage) so parallel cables and parallel
        # stage launches don't share containers.  runfolder.name is always
        # 'apdl_runfolder' for every cable -- a `--cables` parallel launch must
        # disambiguate using parents (run_id + 'cablestack'/'submodel') so two
        # cables don't collide on the same Docker compose project.
        run_id = runfolder.parent.parent.parent.name  # …/<run_id>/APDL/submodel/apdl_runfolder
        project = f"mapdl_{run_id}_{stage_name}".lower()

        try:
            subprocess.run(
                ["docker", "compose", "-p", project, "down"],
                cwd=str(docker_dir),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            logger.info(f"[{stage_name}] Starting MAPDL container ({input_file}). Logs: {log_file_path}")
            print(f"[MAPDL] {stage_name}: running {input_file} - logs: {log_file_path.name}")
            with open(log_file_path, "w") as log_file:
                # --exit-code-from mapdl makes docker compose propagate the
                # container's exit code (e.g. 137 OOM-kill) instead of always
                # returning 0 once orchestration succeeds.
                result = subprocess.run(
                    ["docker", "compose", "-p", project, "up", "--no-color",
                     "--exit-code-from", "mapdl"],
                    cwd=str(docker_dir),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            # MAPDL emits exit 0 on clean runs; exit 8 means "warnings only, no
            # errors" (still wrote outputs).  Both are success for our purposes.
            # Any other non-zero is a real failure (33 = MPI_Abort, 137 = OOM
            # SIGKILL, 1 = unhandled error, etc.).
            mapdl_ok = result.returncode in (0, 8)
            if mapdl_ok:
                if result.returncode == 0:
                    print(f"[MAPDL] {stage_name} completed successfully.")
                    logger.info(f"[{stage_name}] MAPDL completed successfully.")
                else:
                    print(f"[MAPDL] {stage_name} completed with warnings (rc=8).")
                    logger.info(f"[{stage_name}] MAPDL completed with warnings (rc=8).")
            else:
                print(f"[MAPDL] {stage_name} exited with code {result.returncode}. Check {log_file_path}")
                logger.warning(f"[{stage_name}] MAPDL exited with code {result.returncode}.")
            return mapdl_ok, log_file_path
        except Exception as e:
            logger.error(f"[{stage_name}] Failed to run MAPDL container: {e}")
            traceback.print_exc()
            return False, log_file_path


class HPCMAPDL:
    """Adapter: SSH upload + sbatch + wait + fetch on a SLURM HPC cluster.

    Cluster-agnostic; defaults to ETH Euler but works with any SSH-reachable
    SLURM cluster (override HPC_HOST / HPC_USER / HPC_REMOTE_BASE env vars).
    One sbatch job covers all stages (jobslurm.sh sequences them); per-stage rc
    is parsed from mapdl_run.log after the batch completes.
    """

    def run_stages(
        self,
        runfolder: Path,
        ordered_stages: List[str],
        on_stage_complete: Optional[StageCallback] = None,
    ) -> Dict[str, bool]:
        if not ordered_stages:
            logger.info("No cablestack stages selected - SLURM job not launched.")
            return {}

        # main.py runs as a top-level script, so a relative import won't work.
        _here = str(Path(__file__).parent)
        if _here not in sys.path:
            sys.path.insert(0, _here)
        import hpc_submit  # type: ignore

        # <run>/APDL/submodel/apdl_runfolder -> run folder is parents[2]
        run_label = runfolder.parents[2].name
        logger.info(f"[HPC] Cablestack -> remote subdir '{run_label}', stages={ordered_stages}")
        results = hpc_submit.submit_cablestack(
            runfolder, run_label=run_label, ordered_stages=ordered_stages,
        )

        if on_stage_complete is not None:
            log_path = runfolder / "mapdl_run.log"
            for name in ordered_stages:
                on_stage_complete(StageResult(
                    name=name,
                    success=results.get(name, False),
                    log_path=log_path if log_path.is_file() else None,
                ))

        n_ok = sum(1 for v in results.values() if v)
        logger.info(f"[HPC] Cablestack stages: {n_ok}/{len(results)} rc=0 - {results}")
        return results
