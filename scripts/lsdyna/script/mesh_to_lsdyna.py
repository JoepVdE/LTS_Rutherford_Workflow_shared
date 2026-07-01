"""
mesh_to_lsdyna.py
=================
Imports a STEP file from PARTS_DIR into Ansys Mechanical 2025,
generates a solid mesh, and exports the result as an LS-DYNA
keyword (.k) file.

Requirements
------------
    pip install ansys-mechanical-core
    Docker Desktop (if USE_DOCKER = True)

Docker mode
-----------
    Set USE_DOCKER = True.  The script will:
      1. Pull the image if not present locally.
      2. Start the container, mounting PARTS_DIR as /workdir.
      3. Connect via gRPC and run the meshing script.
      4. Stop and remove the container when done.
    The output .k file is written to PARTS_DIR on the host because the
    directory is bind-mounted at /workdir inside the container.

Usage
-----
    python mesh_to_lsdyna.py
"""

from __future__ import annotations

import glob
import io
import os
import subprocess
import sys
import threading
import time


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  —  edit this block only
# ══════════════════════════════════════════════════════════════════════════════

PARTS_DIR = (
    r"c:\Users\vanden_j\OneDrive - ETH Zurich\Desktop\workbench meshing"
    r"\20260410_103406_SMACC_LF\stl_parts"
)

# Output LS-DYNA keyword file path
OUTPUT_K_FILE = os.path.join(PARTS_DIR, "SMACC_LF_mesh.k")

# File extension to search for in PARTS_DIR.
#   "stl"  – surface/faceted import (solid mesh only if parts are watertight)
#   "step" – recommended for solid hex meshing
#   "stp"  – alternative STEP extension
GEOMETRY_EXT = "step"

# Global element size in mm.
#   None  → Ansys automatic sizing  (recommended for a first pass)
#   float → e.g.  ELEMENT_SIZE_MM = 10  enforces a 10 mm global size
ELEMENT_SIZE_MM: float | None = 0.1

# Mesh method applied to all bodies:
#   "HexDominant"  – mostly hex + wedge/tet fill  ← default, works on any solid
#   "Sweep"        – pure hex; only viable for prismatic/sweepable bodies
#   "Automatic"    – per-body, Ansys decides
MESH_METHOD = "Sweep"

# Ansys Mechanical version integer:  2025 R1 = 251  |  2025 R2 = 252
# Used only when USE_DOCKER = False (local installation).
ANSYS_VERSION = 252

# Log file for this run (written next to the output .k file).
# Set to None to disable.
LOG_FILE: str | None = os.path.join(PARTS_DIR, "mesh_to_lsdyna.log")

# ------------------------------------------------------------------------------
# Docker settings  (only used when USE_DOCKER = True)
# ------------------------------------------------------------------------------

# Set True to drive the PSI Gitea container instead of a local Ansys install.
USE_DOCKER = True

# Full image reference.  Tag 25.2 = Ansys 2025 R2.
# Registry chosen by REGISTRY_PREFIX env var; default = PSI Gitea.
# CERN users: set REGISTRY_PREFIX=registry.cern.ch/chart-magnum
DOCKER_IMAGE = os.environ.get("REGISTRY_PREFIX", "gitea.psi.ch/vanden_j") + "/mechanical:25.2"

# Host port to forward the Mechanical gRPC server to.
DOCKER_GRPC_PORT = 10000

# Name for the Docker container.  Override per-cable to avoid collisions.
CONTAINER_NAME = "ansys_mechanical_mesh"

# Path inside the container where PARTS_DIR will be bind-mounted.
# All geometry paths and the output path in the inner script use this prefix.
CONTAINER_WORKDIR = "/workdir"

# Seconds to wait for Mechanical to become ready after container start.
# 300s gives headroom when multiple containers start simultaneously (parallel cable runs).
DOCKER_STARTUP_TIMEOUT = 300

# Ansys FlexLM license server.  Format: port@hostname  (colon-separated for fallback)
# PSI network:  1055@winlic03.psi.ch
# ETH network:  1801@lic-ansys-research.ethz.ch
ANSYS_LICENSE_SERVER = "1801@lic-ansys-research.ethz.ch:1055@winlic03.psi.ch"

# ══════════════════════════════════════════════════════════════════════════════


class _Tee(io.TextIOBase):
    """Write to both the original stream and a log file simultaneously."""

    def __init__(self, stream, log_path: str):
        self._stream = stream
        self._log = open(log_path, "w", encoding="utf-8", errors="replace")

    def write(self, s):
        self._stream.write(s)
        self._stream.flush()
        self._log.write(s)
        self._log.flush()
        return len(s)

    def flush(self):
        self._stream.flush()
        if not self._log.closed:
            self._log.flush()

    def close(self):
        if not self._log.closed:
            self._log.close()
        super().close()

    # Needed so libraries that check isatty() don't crash.
    def isatty(self):
        return False


def find_geometry_files(directory: str, ext: str) -> list[str]:
    pattern = os.path.join(directory, f"*.{ext.lstrip('.')}")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No *.{ext} files found in:\n  {directory}\n"
        )
    return files


def _build_workflow_script(
    server_files: list[str],
    element_size_mm: float | None,
    output_k: str,
) -> str:
    """Build the IronPython script that runs inside Mechanical.

    Following the official PyMechanical pattern, geometry import and the full
    meshing workflow are combined into a *single* run_python_script() call so
    that everything executes inside one writable transaction context.
    """
    out_fwd = output_k.replace("\\", "/")

    lines: list[str] = []
    a = lines.append

    a("import json, os")
    a("")
    # Helper to safely read mesh statistics regardless of API version
    a("def _safe_int(obj, *attrs):")
    a("    for attr in attrs:")
    a("        try:")
    a("            v = getattr(obj, attr, None)")
    a("            if v is not None:")
    a("                return int(v)")
    a("        except Exception:")
    a("            pass")
    a("    return -1")
    a("")

    # -- Unit system
    a("ExtAPI.Application.ActiveUnitSystem = MechanicalUnitSystem.StandardNMM")
    a("")

    # -- Import geometry files
    for p in server_files:
        p_fwd = p.replace("\\", "/")
        a(f"geo_imp = Model.GeometryImportGroup.AddGeometryImport()")
        a(f"geo_imp.Import(r'{p_fwd}')")
    a("")

    # -- Add LS-DYNA analysis
    a("analysis = Model.AddLSDynaAnalysis()")
    a("")

    # -- Configure mesh (no element type override — keep Mechanical's default)
    a("mesh = Model.Mesh")
    a("mesh.PhysicsPreference = MeshPhysicsPreferenceType.Explicit")
    if element_size_mm is not None:
        a(f"mesh.ElementSize = Quantity({element_size_mm!r}, 'mm')")
    a("")

    # -- Collect body IDs for summary reporting only
    a("all_geo_ids = [")
    a("    geo_body.Id")
    a("    for assembly in ExtAPI.DataModel.GeoData.Assemblies")
    a("    for part in assembly.Parts")
    a("    for geo_body in part.Bodies")
    a("]")
    a("")

    # -- Generate mesh
    a("mesh.GenerateMesh()")
    a("")
    # -- Warn if mesh produced nothing
    a("_n_elem = _safe_int(mesh, 'NumberOfElements', 'ElementCount', 'TotalElements')")
    a("if _n_elem == 0:")
    a("    raise Exception('GenerateMesh produced 0 elements.')")
    a("")

    # -- Export LS-DYNA keyword file
    a(f"out_dir = os.path.dirname(r'{out_fwd}')")
    a("if out_dir and not os.path.exists(out_dir):")
    a("    os.makedirs(out_dir)")
    a("")
    a("try:")
    a(f"    analysis.WriteInputFile(r'{out_fwd}')")
    a("except Exception:")
    a(f"    analysis.Solution.WriteInputFile(r'{out_fwd}')")
    a("")

    # -- Return summary as JSON
    a("result = json.dumps({")
    a("    'bodies':   len(all_geo_ids),")
    a("    'nodes':    _safe_int(mesh, 'NumberOfNodes', 'NodeCount', 'TotalNodes'),")
    a("    'elements': _safe_int(mesh, 'NumberOfElements', 'ElementCount', 'TotalElements'),")
    a("})")
    a("result")

    return "\n".join(lines)


def run_meshing_workflow(
    mechanical,
    host_files: list[str],
    element_size_mm: float | None,
    output_k: str,
) -> None:
    """Upload geometry to the project directory, then mesh and export.

    Follows the official PyMechanical remote-session pattern:
      1. Upload each file to the Mechanical project directory.
      2. Run the entire import + mesh + export in ONE run_python_script()
         call so that it executes inside a single writable context.
    """
    import json as _json

    # -- Get the project directory on the server
    project_dir = mechanical.run_python_script(
        "ExtAPI.DataModel.Project.ProjectDirectory"
    )
    project_dir = project_dir.replace("\\", "/").rstrip("/")
    print(f"  Server project directory: {project_dir}")

    # -- Upload every geometry file into the project directory
    server_files: list[str] = []
    for i, hf in enumerate(host_files):
        basename = os.path.basename(hf)
        print(f"  Uploading {i+1}/{len(host_files)}: {basename}")
        mechanical.upload(hf, file_location_destination=project_dir)
        server_files.append(project_dir + "/" + basename)

    # -- Build and send the single workflow script
    script = _build_workflow_script(
        server_files, element_size_mm, output_k,
    )
    print("  Running meshing workflow (single script) ...")
    try:
        result_str = mechanical.run_python_script(script)
    except Exception as exc:
        print(f"  ERROR in meshing workflow:\n    {exc}")
        raise

    # -- Report results
    try:
        info = _json.loads(result_str)
        print(f"    Bodies:   {info.get('bodies')}")
        print(f"    Nodes:    {info.get('nodes')}")
        print(f"    Elements: {info.get('elements')}")
    except Exception:
        print(f"    Raw result: {result_str}")

    print(f"\n  Export complete: {output_k}")


def _stream_docker_logs(container_id: str, stop_event: threading.Event) -> None:
    """Follow docker logs and print each line prefixed with [docker]."""
    try:
        proc = subprocess.Popen(
            ["docker", "logs", "--follow", container_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            if stop_event.is_set():
                break
            print("[docker] " + line, end="", flush=True)
        proc.wait()
    except Exception:  # noqa: BLE001
        pass


def _wait_for_mechanical(mech_module, host: str, port: int, timeout: int):
    """Poll until Mechanical gRPC is accepting connections or timeout."""
    deadline = time.monotonic() + timeout
    last_exc = None
    while time.monotonic() < deadline:
        try:
            m = mech_module.Mechanical(ip=host, port=port, cleanup_on_exit=False)
            return m
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(3)
    raise RuntimeError(
        f"Mechanical did not become ready within {timeout}s.\n"
        f"Last error: {last_exc}"
    )


def main() -> None:
    # ── Tee stdout/stderr to log file ────────────────────────────────────────
    _tee = None
    if LOG_FILE:
        import datetime
        _tee = _Tee(sys.stdout, LOG_FILE)
        sys.stdout = _tee
        sys.stderr = _tee
        print(f"=== mesh_to_lsdyna run started {datetime.datetime.now().isoformat()} ===")
        print(f"Log: {LOG_FILE}")

    # ── Discover geometry files ───────────────────────────────────────────────
    host_files = find_geometry_files(PARTS_DIR, GEOMETRY_EXT)
    print(f"Found {len(host_files)} *.{GEOMETRY_EXT} file(s) in:")
    print(f"  {PARTS_DIR}")

    # ── Sanity-check the package ──────────────────────────────────────────────
    try:
        import ansys.mechanical.core as mech
    except ImportError:
        sys.exit(
            "ERROR: ansys-mechanical-core is not installed.\n"
            "Install with:  pip install ansys-mechanical-core"
        )

    # Relative path from PARTS_DIR to OUTPUT_K_FILE — preserves subdirs (e.g. LSDYNA/)
    # inside the container's bind-mounted workdir.
    output_k_rel = os.path.relpath(OUTPUT_K_FILE, PARTS_DIR).replace("\\", "/")

    if USE_DOCKER:
        # ── Docker path ───────────────────────────────────────────────────────
        # Normalise host path for Docker (needs forward slashes on Windows)
        host_mount = PARTS_DIR.replace("\\", "/")

        # Remove any leftover container from a previous aborted run.
        subprocess.run(
            ["docker", "rm", "-f", CONTAINER_NAME],
            capture_output=True,
        )

        # Pull image first so progress is visible (images can be ~10 GB).
        print(f"Pulling image (may take a while on first run): {DOCKER_IMAGE}")
        pull = subprocess.run(["docker", "pull", DOCKER_IMAGE])
        if pull.returncode != 0:
            sys.exit(
                f"ERROR: docker pull failed for {DOCKER_IMAGE}\n"
                "Tip: make sure you are logged in:\n"
                f"     docker login {DOCKER_IMAGE.split('/')[0]}"
            )

        # Start the container detached. Image is already local so this is fast.
        docker_cmd = [
            "docker", "run",
            "--detach",
            "--name", CONTAINER_NAME,
            "-e", "WB1_STANDALONE=1",
            "-e", f"ANSYSLMD_LICENSE_FILE={ANSYS_LICENSE_SERVER}",
            "-e", f"ANSYS_LICENSE_SERVER={ANSYS_LICENSE_SERVER}",
            "-v", f"{host_mount}:{CONTAINER_WORKDIR}",
            "-p", f"{DOCKER_GRPC_PORT}:10000",
            DOCKER_IMAGE,
        ]
        print(f"\nStarting container...")
        result = subprocess.run(docker_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(
                f"ERROR: docker run failed:\n{result.stderr.strip()}"
            )
        container_id = result.stdout.strip()
        print(f"Container started: {container_id[:12]}")
        print(f"Waiting up to {DOCKER_STARTUP_TIMEOUT}s for Mechanical gRPC...")
        print("(container output prefixed with [docker])\n")

        # Stream container logs to the terminal in a background thread.
        stop_logs = threading.Event()
        log_thread = threading.Thread(
            target=_stream_docker_logs,
            args=(container_id, stop_logs),
            daemon=True,
        )
        log_thread.start()

        try:
            mechanical = _wait_for_mechanical(
                mech, "localhost", DOCKER_GRPC_PORT, DOCKER_STARTUP_TIMEOUT
            )
            print("\n[host] Connected to Mechanical gRPC.\n")

            # Output goes to the bind-mounted workdir so it appears on the host.
            inner_output_k = CONTAINER_WORKDIR + "/" + output_k_rel

            run_meshing_workflow(
                mechanical, host_files, ELEMENT_SIZE_MM, inner_output_k
            )

            # Download the .k file from server to host if needed
            # (it may already be on the bind mount, but ensure we have it)
            mechanical.exit()
        finally:
            stop_logs.set()
            log_thread.join(timeout=5)
            subprocess.run(["docker", "stop", container_id], capture_output=True)
            subprocess.run(["docker", "rm",   container_id], capture_output=True)
            print(f"\n[host] Container {container_id[:12]} stopped and removed.")

    else:
        # ── Local Mechanical path ─────────────────────────────────────────────
        print(f"Launching Ansys Mechanical (version {ANSYS_VERSION})...")
        mechanical = mech.launch_mechanical(version=ANSYS_VERSION, cleanup_on_exit=True)
        try:
            run_meshing_workflow(
                mechanical, host_files, ELEMENT_SIZE_MM, MESH_METHOD, OUTPUT_K_FILE
            )
        finally:
            mechanical.exit()

    print(f"\nAll done.  Output file:\n  {OUTPUT_K_FILE}")

    if _tee is not None:
        import datetime
        print(f"=== mesh_to_lsdyna run finished {datetime.datetime.now().isoformat()} ===")
        sys.stdout = _tee._stream
        sys.stderr = _tee._stream
        _tee.close()


if __name__ == "__main__":
    main()
