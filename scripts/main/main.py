"""
Main script for LTS Rutherford Workflow
Runs cable parameter setup and calculations

Usage:
    python main.py [OPTIONS]

Arguments:
    -c, --cable         Cable configuration to use (default: R2D2_LF)
                        Available: R2D2_LF, R2D2_HF, CD1  (see cable_parameters_user.json)
    --cables            Run multiple cables (e.g. --cables R2D2_LF R2D2_HF CD1)
    -t, --time          Simulation termination time in milliseconds (default: 0.0001)
    --min-mesh-size     Global element size passed to Ansys Mechanical (ELEMENT_SIZE_MM) for
                        meshing the cable STEP geometry into mesh.k, in mm
                        (default: auto = π × D_strand / 20; e.g. ~0.133 mm for D=0.85 mm)
    --max-mesh-size     Reserved / not currently used by the mesher
    --list-cables       List all available cable configurations and exit
    --reuse-folder      Reuse the most recent run folder instead of creating a new one
    --quick-run         Skip geometry + meshing; only run mesh conversion + LS-DYNA
                        (uses latest existing run folder for the selected cable)
    --apdl-only         Copy latest run → new apdl_rerun folder and run d3plot→APDL + cablestack
    --no-cablestack     Generate cablestack .inp files but do NOT launch MAPDL Docker (0-start.inp not run)
    --hpc               Run the cablestack APDL stages on an SSH-reachable SLURM HPC cluster
                        (upload+sbatch+wait+fetch). Default target is ETH Euler; override via
                        HPC_HOST / HPC_USER / HPC_REMOTE_BASE env vars. Honours cablestack.stages from JSON.
    --debug-plots       Emit per-pair conformal mesh / outer-node SVGs (slow; off by default)

Examples:
    python main.py                                      # Full run with R2D2_LF defaults
    python main.py -c R2D2_HF -t 0.0002                # R2D2_HF cable, 2 ms simulation
    python main.py --quick-run                          # Re-run LS-DYNA on latest mesh
    python main.py --quick-run -c R2D2_LF              # Re-run LS-DYNA on latest R2D2_LF mesh
    python main.py --apdl-only                         # d3plot→APDL + cablestack on latest R2D2_LF run
    python main.py --apdl-only -c R2D2_HF             # Same, for R2D2_HF cable
    python main.py --apdl-only --no-cablestack         # Generate .inp files only, skip MAPDL Docker
    python main.py --list-cables                        # Show available cables
    python main.py --cables R2D2_LF R2D2_HF CD1        # Run all three cables in parallel
"""

import argparse
import json
import logging
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from license_detector import LicenseDetector
    from solver import CablestackSolver

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Cablestack stage registry, dependency resolver and jobslurm generator live
# in cablestack_stages.py — shared with scripts/analysis/submodel/cablestack/
# analyse_pressure.py so the stage→usecase-suffix filename convention has a
# single source of truth.
from cablestack_stages import (  # noqa: E402
    CABLESTACK_STAGES,
    resolve_cablestack_stage_order,
    write_cablestack_jobslurm,
)


class WorkflowRunner:
    """Handles the LTS Rutherford Workflow execution"""
    
    def __init__(self, workspace_root: Path,
                 license_detector: Optional["LicenseDetector"] = None):
        self.workspace_root = workspace_root
        self.current_dir = Path(__file__).parent
        self.run_id = None
        self.run_output_dir = None
        if license_detector is None:
            from license_detector import NetworkProbeLicenseDetector
            license_detector = NetworkProbeLicenseDetector()
        self.license_detector = license_detector
        
    def get_tool_versions(self) -> Dict[str, str]:
        """Collect versions of all tools used in the workflow"""
        import platform
        import subprocess
        
        versions = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": "Not available",
            "freecad": "Not detected",
            "paraview": "Not detected",
        }
        
        # Get numpy version if available
        try:
            import numpy as np
            versions["numpy"] = np.__version__
        except ImportError:
            pass
        
        # Auto-detect tool versions from directory names
        versions["freecad"] = self._detect_tool_version("tools/freecad", "FreeCAD", separator="_")
        versions["paraview"] = self._detect_tool_version("tools/paraview", "ParaView", separator="-")
        
        return versions
    
    def _detect_tool_version(self, tool_dir: str, prefix: str, separator: str) -> str:
        """Helper to detect tool version from directory name"""
        tool_path = self.workspace_root / tool_dir
        if not tool_path.exists():
            return "Not found"
            
        try:
            for item in tool_path.iterdir():
                if item.name.startswith(prefix):
                    parts = item.name.split(separator)
                    if len(parts) > 1:
                        version_part = parts[1].split("-")[0] if separator == "_" else parts[1]
                        return version_part
        except Exception as e:
            return f"Detection failed: {e}"
            
        return "Not detected"
    
    def generate_run_id(self, cable_name: str) -> str:
        """Generate a unique run ID"""
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{cable_name}"

    def _update_metadata_step(self, run_output_dir: Path, step: str, status: str) -> None:
        """Write a single workflow_steps entry to metadata.json (best-effort)."""
        metadata_file = run_output_dir / "metadata.json"
        if not metadata_file.exists():
            return
        try:
            with open(metadata_file) as _f:
                meta = json.load(_f)
            meta.setdefault("workflow_steps", {})[step] = status
            with open(metadata_file, "w") as _f:
                json.dump(meta, _f, indent=4)
        except Exception as _e:
            logger.warning(f"Could not update metadata step '{step}': {_e}")
    
    def find_latest_run_folder(
        self,
        cable_name: str,
        require_d3plot: bool = False,
    ) -> Optional[Path]:
        """Find the most recent run folder for the given cable"""
        runs_dir = self.workspace_root / "data" / "runs"
        if not runs_dir.exists():
            return None

        # Find all folders matching the cable name pattern
        matching_folders = []
        for folder in runs_dir.iterdir():
            if folder.is_dir() and folder.name.endswith(f"_{cable_name}"):
                matching_folders.append(folder)

        if not matching_folders:
            return None

        # Apply optional filters
        if require_d3plot:
            def _lsdyna_done(folder: Path) -> bool:
                meta_path = folder / "metadata.json"
                if meta_path.exists():
                    try:
                        ws = json.load(open(meta_path)).get("workflow_steps", {})
                        if ws.get("5_lsdyna_simulation") == "completed":
                            return True
                    except Exception:
                        pass
                return (folder / "LSDYNA" / "d3plot").exists()
            matching_folders = [f for f in matching_folders if _lsdyna_done(f)]

        if not matching_folders:
            return None

        # Return the most recently named folder
        latest_folder = max(matching_folders, key=lambda f: f.name)
        return latest_folder
    
    @staticmethod
    def _warn_geometry_stability(cable_name: str, cable: dict) -> None:
        """Pre-flight: log a warning when the cable geometry matches a
        failure pattern observed in the 2026-06 test matrix.

        Thresholds (empirical, conservative — false-positive cost is one log
        line; false-negative cost is wasted LS-DYNA time + a build crash):

          - D_Strand < 0.65 mm     -> APDL ASBA Boolean fails (TEST_A: 0.5 mm)
          - cable_height / (2*D)   -> Y compaction; < 0.85 risks mesh degeneracy
                                      (TEST_B_HEAVY segfault at 0.82)
          - cable_width / W_ruther -> X compaction; < 0.70 risks tight keystone
                                      Boolean failure (TEST_E_HEAVY at 0.65)

        Two flags simultaneously -> strong warning.
        """
        import math
        try:
            D    = float(cable['D_Strand'])
            N    = int(cable['N_Strands'])
            w    = float(cable['cable_width'])
            h    = float(cable['cable_height'])
        except (KeyError, TypeError, ValueError):
            return  # config malformed; downstream code will surface it

        # Same W_rutherford formula as calc_cable_params_sim.
        pitch_min     = math.pi * D * (N - 2) / (2 * N) if N > 2 else 0.0
        W_rutherford  = N * pitch_min / 2.0 + D
        h_ratio       = h / (2.0 * D) if D > 0 else float('nan')
        w_ratio       = w / W_rutherford if W_rutherford > 0 else float('nan')

        warns = []
        if D < 0.65:
            warns.append(
                f"D_Strand={D:.2f} mm < 0.65 mm: for small strands the middle "
                f"and outer circular layers of the conformal strand mesh "
                f"collapse to ~3.5 um radial separation at the same angular "
                f"position (verified for D=0.5 mm). This makes the strand-area "
                f"ASBA boolean in 2-geo.inp degenerate; btol changes (1e-7 to "
                f"5e-6) do not help. Root fix needs minimum-radial-separation "
                f"in meshMapping.map_circumferential_layer_to_bspline."
            )
        if h_ratio < 0.85:
            warns.append(
                f"cable_height/(2*D_Strand)={h_ratio:.2f} < 0.85: very tight "
                f"Y compaction. TEST_B_HEAVY (0.82) seg-faulted in MAPDL "
                f"meshing; suspected mesh degeneracy."
            )
        if w_ratio < 0.70:
            warns.append(
                f"cable_width/W_rutherford={w_ratio:.2f} < 0.70: narrow "
                f"keystone (high X compaction). TEST_E_HEAVY (0.65) hit "
                f"ASBA Boolean failure."
            )

        if not warns:
            return

        prefix = "STABILITY WARNING (strong)" if len(warns) >= 2 else "STABILITY WARNING"
        logger.warning(
            f"[{cable_name}] {prefix}: geometry matches a failure pattern from "
            f"the 2026-06 test matrix. Workflow will still run, but expect a "
            f"crash in MAPDL build (or earlier). Reasons:"
        )
        for w_msg in warns:
            logger.warning(f"  - {w_msg}")
        logger.warning(
            "  See data/runs/_TEST_MATRIX_REPORT_2026-06-25.md for full context."
        )

    def setup_cable_config(self, selected_cable: str) -> str:
        """Update cable configuration and return cable name"""
        cable_params_json = self.current_dir / 'cable_parameters_user.json'

        # Read and update config
        with open(cable_params_json, 'r') as f:
            cable_config = json.load(f)

        # Validate cable exists
        if selected_cable not in cable_config.get('cables', {}):
            available = list(cable_config.get('cables', {}).keys())
            raise ValueError(f"Cable '{selected_cable}' not found. Available: {available}")

        # Pre-flight: warn on geometries empirically known to crash the APDL
        # build stage so the user notices before LS-DYNA spends ~30 min.
        # See data/runs/_TEST_MATRIX_REPORT_2026-06-25.md for the source data.
        self._warn_geometry_stability(selected_cable, cable_config['cables'][selected_cable])

        cable_config['active_cable'] = selected_cable

        # When --cables dispatches parallel subprocesses, each one calls this
        # function with its own ACTIVE_CABLE env var.  The JSON's active_cable
        # field is then redundant -- calc_cable_params_sim prefers the env var.
        # On Windows, racing os.replace calls from parallel subprocesses hit
        # PermissionError [WinError 5].  Skip the write when ACTIVE_CABLE is
        # already set; otherwise do the atomic write with a brief retry on
        # transient Windows file-locking errors.
        import os as _os
        if _os.environ.get('ACTIVE_CABLE'):
            logger.info(f"Selected cable: {selected_cable} (ACTIVE_CABLE set; skip JSON rewrite)")
            return selected_cable

        import tempfile as _tempfile, time as _time
        _dir = cable_params_json.parent
        with _tempfile.NamedTemporaryFile('w', dir=_dir, suffix='.tmp', delete=False) as _tf:
            json.dump(cable_config, _tf, indent=4)
            _tmp_path = _tf.name
        for _attempt in range(5):
            try:
                _os.replace(_tmp_path, cable_params_json)
                break
            except PermissionError:
                if _attempt == 4:
                    raise
                _time.sleep(0.2 * (2 ** _attempt))

        logger.info(f"Selected cable: {selected_cable}")
        return selected_cable
    
    def setup_directories(self, run_id: str) -> Path:
        """Create and setup run directories"""
        run_output_dir = self.workspace_root / "data" / "runs" / run_id
        run_output_dir.mkdir(parents=True, exist_ok=True)
        
        lsdyna_dir = run_output_dir / "LSDYNA"
        lsdyna_dir.mkdir(exist_ok=True)
        
        logger.info(f"Output directory: {run_output_dir}")
        return run_output_dir
    
    def generate_metadata(self, run_id: str, run_output_dir: Path, cable_params_file: Path) -> Path:
        """Generate metadata for the current run"""
        # Import here to avoid circular imports
        sys.path.insert(0, str(self.current_dir))
        try:
            import calc_cable_params_sim
            
            metadata = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "cable_name": calc_cable_params_sim.cable_name,
                "tool_versions": self.get_tool_versions(),
                "files": {
                    "cable_parameters": str(cable_params_file.absolute()),
                    "metadata": str(run_output_dir / "metadata.json")
                },
                "workflow_steps": {
                    "1_cable_parameters": "completed",
                    "2_freecad_geometry": "pending",
                    "3_mesh_conversion": "pending",
                    "4_apdl_simulation": "pending",
                    "5_lsdyna_simulation": "pending",
                    "6_analysis": "pending"
                },
                "parameters": {
                    "cable_width_mm": calc_cable_params_sim.cable_width,
                    "cable_height_mm": calc_cable_params_sim.cable_height,
                    "n_strands": calc_cable_params_sim.N_Strands,
                    "transposition_pitch_mm": calc_cable_params_sim.T_pitch
                }
            }
            
            metadata_file = run_output_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Metadata exported to: {metadata_file}")
            return metadata_file
        finally:
            # Clean up sys.path
            if str(self.current_dir) in sys.path:
                sys.path.remove(str(self.current_dir))
    
    def run_cable_parameters(self, run_output_dir: Path, termination_time: float) -> Path:
        """Run cable parameter calculations"""
        logger.info("Running cable parameter calculations...")
        
        sys.path.insert(0, str(self.current_dir))
        try:
            import calc_cable_params_sim
            import importlib
            importlib.reload(calc_cable_params_sim)
            
            # Set time from main script arg and compute velocities
            calc_cable_params_sim.time = termination_time
            calc_cable_params_sim.velocity_y = calc_cable_params_sim.distance_y / termination_time
            calc_cable_params_sim.velocity_x = calc_cable_params_sim.distance_x / termination_time
            calc_cable_params_sim.velocity_z = calc_cable_params_sim.distance_z / termination_time
            
            cable_params_file = calc_cable_params_sim.export_parameters_to_json(str(run_output_dir))
            return Path(cable_params_file)
        finally:
            if str(self.current_dir) in sys.path:
                sys.path.remove(str(self.current_dir))
    
    def run_geometry_generation(self, run_output_dir: Path, metadata_file: Path) -> bool:
        """Generate STEP geometry using FreeCAD"""
        logger.info("Generating STEP geometry with FreeCAD...")
        
        setup_step_dir = self.workspace_root / "scripts" / "setup_step"
        sys.path.insert(0, str(setup_step_dir))
        
        try:
            import generate_step
            import importlib
            importlib.reload(generate_step)
            success = generate_step.generate_step_geometry_rutherford(str(run_output_dir))
            
            if success:
                # Show updated workflow status
                with open(metadata_file, 'r') as f:
                    updated_metadata = json.load(f)
                
                logger.info("Updated workflow status:")
                for step, status in updated_metadata['workflow_steps'].items():
                    status_symbol = "✓" if status == "completed" else "⏳"
                    logger.info(f"  {status_symbol} {step}: {status}")
            else:
                logger.warning("STEP geometry generation failed, but workflow continues...")
                
            return success
            
        except Exception as e:
            logger.warning(f"Error generating STEP geometry: {e}")
            return False
        finally:
            if str(setup_step_dir) in sys.path:
                sys.path.remove(str(setup_step_dir))
    
    def run_lsdyna_setup(self, run_output_dir: Path, termination_time: float,
                        min_mesh_size: Optional[float] = None,
                        max_mesh_size: Optional[float] = None) -> bool:
        """Setup LS-DYNA mesh using Ansys Mechanical via mesh_to_lsdyna"""
        logger.info("Setting up LS-DYNA mesh with Ansys Mechanical...")
        lsdyna_script_dir = self.workspace_root / "scripts" / "lsdyna" / "script"
        sys.path.insert(0, str(lsdyna_script_dir))

        try:
            import math

            # Load metadata to get cable parameters
            metadata_file = run_output_dir / "metadata.json"
            with open(metadata_file) as f:
                metadata = json.load(f)

            # Auto-calculate element size from strand diameter if not provided
            if min_mesh_size is None:
                # Prefer cable_parameters.json (has the geometry-corrected D_Strand),
                # fall back to metadata parameters, then to R2D2_LF default.
                cable_params_file = run_output_dir / "cable_parameters.json"
                if cable_params_file.exists():
                    with open(cable_params_file) as _f:
                        _cp = json.load(_f)
                    strand_diameter = _cp.get('D_Strand_base', _cp.get('D_Strand', metadata['parameters'].get('strand_diameter', 0.85)))
                else:
                    strand_diameter = metadata['parameters'].get('strand_diameter', 0.85)
                min_mesh_size = (math.pi * strand_diameter) / 20
                logger.info(
                    f"Auto-calculated element size: {min_mesh_size:.4f} mm "
                    f"(strand_diameter={strand_diameter} mm)"
                )

            # Prepare output directory
            lsdyna_dir = run_output_dir / "LSDYNA"
            lsdyna_dir.mkdir(exist_ok=True)
            output_k = str(lsdyna_dir / "mesh.k")

            # Configure mesh_to_lsdyna and run
            import mesh_to_lsdyna
            mesh_to_lsdyna.PARTS_DIR = str(run_output_dir)
            mesh_to_lsdyna.OUTPUT_K_FILE = output_k
            mesh_to_lsdyna.GEOMETRY_EXT = "step"
            mesh_to_lsdyna.ELEMENT_SIZE_MM = min_mesh_size
            mesh_to_lsdyna.LOG_FILE = str(lsdyna_dir / "mesh_to_lsdyna.log")
            # Unique container name and port per run folder to avoid collisions
            # when multiple cables run in parallel.  Port pool was 10000-10999
            # (1000 ports); birthday-paradox gives ~1% collision for 3 cables
            # in parallel and we hit it.  Widen to 10000-49999 (40k ports).
            _run_hash = abs(hash(run_output_dir.name)) % 40000
            mesh_to_lsdyna.CONTAINER_NAME = f"ansys_mechanical_{run_output_dir.name}".lower()
            mesh_to_lsdyna.DOCKER_GRPC_PORT = 10000 + _run_hash

            try:
                mesh_to_lsdyna.main()
            except SystemExit as e:
                if e.code != 0:
                    logger.error(f"mesh_to_lsdyna exited with error code {e.code}")
                    return False
            except Exception as e:
                # Only fail if the output file wasn't produced
                if not Path(output_k).exists():
                    logger.error(f"mesh_to_lsdyna raised an exception and no mesh.k produced: {e}")
                    traceback.print_exc()
                    return False
                logger.warning(f"mesh_to_lsdyna raised a non-fatal exception (mesh.k exists): {e}")

            if not Path(output_k).exists():
                logger.error(f"Expected mesh file not found: {output_k}")
                return False

            # Update metadata
            with open(metadata_file) as f:
                metadata = json.load(f)
            metadata.setdefault('workflow_steps', {})['5_lsdyna_simulation'] = 'mesh_completed'
            metadata.setdefault('files', {})['lsdyna_mesh'] = os.path.abspath(output_k)
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Mesh saved: {output_k}")

            return True

        except Exception as e:
            logger.error(f"LS-DYNA setup failed: {e}")
            traceback.print_exc()
            return False
        finally:
            if str(lsdyna_script_dir) in sys.path:
                sys.path.remove(str(lsdyna_script_dir))
    
    def run_mesh_conversion(self, run_output_dir: Path, termination_time: float) -> bool:
        """Convert mesh to processed input format"""
        logger.info("Converting mesh to input file...")
        
        mesh_file = run_output_dir / "LSDYNA" / "mesh.k"
        if not mesh_file.exists():
            logger.error(f"Mesh file not found: {mesh_file}")
            return False
        
        meshconverter_dir = self.workspace_root / "scripts" / "meshconverter"
        sys.path.insert(0, str(meshconverter_dir))
        
        try:
            from inputfile_generator import MeshParser, MeshProcessor, InputFileWriter
            
            # Parse mesh
            parser = MeshParser(str(mesh_file))
            nodes, elements = parser.parse()
            processor = MeshProcessor(nodes, elements)
            processor.analyze()
            
            # Calculate velocities
            velocities = self._calculate_plate_velocities(termination_time)
            
            # Generate input file
            output_input = run_output_dir / "LSDYNA" / "processed_input.k"

            # Load wire material properties from JSON
            cable_params_file = self.current_dir / 'cable_parameters_user.json'
            with open(cable_params_file, 'r') as _f:
                _cable_cfg = json.load(_f)
            _wire_mat = _cable_cfg.get('wire_material', {})
            wire_sigy = _wire_mat.get('sigy_MPa', 20)
            wire_etan = _wire_mat.get('etan_MPa', 5000)

            writer = InputFileWriter(
                str(meshconverter_dir),
                output_file=str(output_input),
                end_time=termination_time,
                plate_velocity_y=velocities['y'],
                plate_velocity_x=velocities['x'],
                wire_sigy=wire_sigy,
                wire_etan=wire_etan,
            )
            
            writer.write(
                parts=processor.parts,
                mesh_file=str(mesh_file),
                segment_connections=processor.segment_connections,
                all_segments=processor.all_segments,
                nodes=nodes,
                part_nodes=processor.part_nodes,
                part_info=processor.part_info,
                face_nodes=processor.face_nodes,
                z_tol=processor.z_tol,
            )
            
            logger.info(f"Converted mesh to input file: {output_input}")
            return True
            
        except Exception as e:
            logger.error(f"Mesh conversion failed: {e}")
            traceback.print_exc()
            return False
        finally:
            if str(meshconverter_dir) in sys.path:
                sys.path.remove(str(meshconverter_dir))
    
    def _calculate_plate_velocities(self, termination_time: float) -> Dict[str, float]:
        """Calculate plate velocities from cable parameters"""
        sys.path.insert(0, str(self.current_dir))
        try:
            import calc_cable_params_sim
            
            velocity_y = calc_cable_params_sim.distance_y / termination_time
            velocity_x = calc_cable_params_sim.distance_x / termination_time
            velocity_z = calc_cable_params_sim.distance_z / termination_time
            
            # Update module-level values for consistency
            calc_cable_params_sim.time = termination_time
            calc_cable_params_sim.velocity_y = velocity_y
            calc_cable_params_sim.velocity_x = velocity_x
            calc_cable_params_sim.velocity_z = velocity_z
            
            logger.info(f"Computed plate velocities:")
            logger.info(f"  Velocity Y: {velocity_y}")
            logger.info(f"  Velocity X: {velocity_x}")
            logger.info(f"  Velocity Z: {velocity_z}")
            
            return {'x': velocity_x, 'y': velocity_y, 'z': velocity_z}
            
        except Exception as e:
            logger.error(f"Failed to compute plate velocities: {e}")
            raise RuntimeError(f"Failed to compute plate velocities from calc_cable_params_sim: {e}") from e
        finally:
            if str(self.current_dir) in sys.path:
                sys.path.remove(str(self.current_dir))
    
    def run_lsdyna_simulation(self, run_output_dir: Path) -> bool:
        """Run LS-DYNA simulation in a detached Docker container."""
        logger.info("Starting LS-DYNA Docker container...")
        
        import subprocess
        
        lsdyna_run_dir = run_output_dir / "LSDYNA"
        input_file = lsdyna_run_dir / "processed_input.k"
        
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return False
            
        docker_dir = self.workspace_root / "scripts" / "lsdyna" / "docker"
        log_file_path = lsdyna_run_dir / "lsdyna_container.log"
        
        env = os.environ.copy()
        # Pass the newly created run directory to docker-compose so it won't use ./run-lsdyna
        env["LSDYNA_RUN_DIR"] = str(lsdyna_run_dir.absolute())
        
        # Detect reachable license server to avoid FlexLM timeout on unreachable servers
        license_server = self.license_detector.detect()
        env["ANSYSLI_SERVERS"] = license_server
        env["ANSYSLMD_LICENSE_FILE"] = license_server
        
        # Unique project name per run so parallel cables don't share containers.
        project = f"lsdyna_{run_output_dir.name}".lower()

        try:
            # Stop any pre-existing container for this project
            subprocess.run(
                ["docker", "compose", "-p", project, "down"],
                cwd=str(docker_dir),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
            
            logger.info(f"Running LS-DYNA in background. Logs saved to: {log_file_path}")
            
            # Run docker compose up in background & pipe logs to file
            log_file = open(log_file_path, "w")
            
            # Using start_new_session avoids killing the container when Python exits 
            subprocess.Popen(
                ["docker", "compose", "-p", project, "up", "--no-color"],
                cwd=str(docker_dir),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to start LS-DYNA Docker container: {e}")
            traceback.print_exc()
            return False

    def _wait_for_lsdyna_completion(self, log_file_path: Path, termination_time: float, poll_interval: float = 5.0) -> bool:
        """Block until LS-DYNA finishes by tailing the container log.

        Prints a live status line every poll_interval seconds showing LS-DYNA
        progress (from LSDYNA/mes0000).

        Returns True on normal termination, False if the container exited with an error.
        """
        import re
        import sys
        import time

        mes_path = log_file_path.parent / "mes0000"
        run_output_dir = log_file_path.parent.parent

        # Short cable label: YYYYMMDD_HHMMSS_<CABLE>[_apdl_rerun...] -> <CABLE>[_apdl_rerun...]
        parts = run_output_dir.name.split("_")
        cable_label = "_".join(parts[2:]) if len(parts) > 2 else run_output_dir.name

        def _lsdyna_pct() -> str:
            if not mes_path.exists():
                return "  ---  "
            try:
                matches = re.findall(
                    r"^\s*\d+\s+t\s+([\d.E+\-]+)\s+dt",
                    mes_path.read_text(errors="replace"),
                    re.MULTILINE,
                )
                if not matches:
                    return "  ---  "
                pct = min(100.0, float(matches[-1]) / termination_time * 100.0)
                return f"{pct:5.1f}%"
            except Exception:
                return "  err  "

        logger.info(f"Waiting for LS-DYNA to finish. Monitoring: {log_file_path}")
        # Wait for the log file to appear (Docker may take a moment to create it)
        for _ in range(60):
            if log_file_path.exists():
                break
            time.sleep(1.0)
        else:
            logger.error("Log file never appeared - cannot monitor LS-DYNA completion.")
            return False

        with open(log_file_path, "r", errors="replace") as fh:
            while True:
                chunk = fh.read(4096)
                if chunk:
                    # Check for clean exit
                    if "N o r m a l    t e r m i n a t i o n" in chunk:
                        sys.stdout.write(f"\r[{cable_label}] LS-DYNA: 100.0%\n")
                        sys.stdout.flush()
                        logger.info("LS-DYNA reported Normal termination.")
                        return True
                    # Check for Docker exit codes
                    if "exited with code 0" in chunk:
                        sys.stdout.write(f"\r[{cable_label}] LS-DYNA: 100.0%\n")
                        sys.stdout.flush()
                        logger.info("LS-DYNA container exited with code 0.")
                        return True
                    if "exited with code" in chunk:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        logger.error("LS-DYNA container exited with a non-zero code.")
                        return False
                else:
                    sys.stdout.write(
                        f"\r[{cable_label}] LS-DYNA: {_lsdyna_pct()}    "
                    )
                    sys.stdout.flush()
                    time.sleep(poll_interval)

    def _find_pvpython(self) -> Optional[Path]:
        """Locate a pvpython binary across platforms.

        Resolution order:
          1. PVPYTHON_EXE env var (verbatim path).
          2. Bundled ParaView under tools/paraview/ParaView*/bin/pvpython[.exe].
          3. System PATH lookup -- what the supplied Dockerfile and any
             Linux/macOS install rely on.
        """
        import shutil as _shutil
        env_exe = os.environ.get("PVPYTHON_EXE")
        if env_exe and Path(env_exe).is_file():
            return Path(env_exe)

        pv_root = self.workspace_root / "tools" / "paraview"
        if pv_root.is_dir():
            for item in pv_root.iterdir():
                if item.is_dir() and item.name.startswith("ParaView"):
                    for name in ("pvpython.exe", "pvpython"):
                        candidate = item / "bin" / name
                        if candidate.is_file():
                            return candidate

        for name in ("pvpython", "pvpython.exe"):
            path = _shutil.which(name)
            if path:
                return Path(path)

        return None

    def run_paraview_extraction(self, run_output_dir: Path) -> bool:
        """Run the ParaView extraction script (pvpython) on the finished LS-DYNA results."""
        import shutil
        import subprocess

        pvpython = self._find_pvpython()
        script = self.workspace_root / "scripts" / "paraview" / "extract_coordinates_stack_sort.py"
        runs_dir = self.workspace_root / "data" / "runs"

        if pvpython is None:
            logger.error(
                "pvpython not found.  Install ParaView (apt: paraview / brew: paraview), "
                "or set PVPYTHON_EXE to point at the binary, or place a bundled ParaView "
                "under tools/paraview/ParaView*/bin/pvpython[.exe]."
            )
            return False
        if not script.exists():
            logger.error(f"ParaView script not found: {script}")
            return False

        # The script uses os.getcwd() + subfolder to locate the LSDYNA directory
        subfolder = run_output_dir.name
        logger.info(f"Running ParaView extraction for '{subfolder}'...")

        try:
            result = subprocess.run(
                [str(pvpython), str(script), subfolder],
                cwd=str(runs_dir),
                check=True,
            )
            logger.info("ParaView extraction completed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"ParaView extraction failed with return code {e.returncode}")
            return False
        except Exception as e:
            logger.error(f"Failed to run ParaView extraction: {e}")
            traceback.print_exc()
            return False

    def run_apdl_submodel(self, run_output_dir: Path, debug_plots: bool = False) -> bool:
        """Generate APDL submodel input files using conformalRutherfordMesh."""
        logger.info("Generating APDL submodel input files...")

        stack_dir = run_output_dir / "stack"
        if not stack_dir.exists():
            logger.info("Stack CSV directory not found — running ParaView extraction first...")
            pv_ok = self.run_paraview_extraction(run_output_dir)
            if not pv_ok or not stack_dir.exists():
                logger.error(f"ParaView extraction did not produce stack directory: {stack_dir}")
                return False

        # Read n_stacks and n_parts by deriving the cable name from the folder name
        # Folder format: YYYYMMDD_HHMMSS_<CABLE_NAME>[_apdl_rerun[_N]]
        import re as _re
        folder_parts = run_output_dir.name.split("_", 2)
        cable_name_from_folder = folder_parts[2] if len(folder_parts) == 3 else None
        # Strip _apdl_rerun[_<digits>] suffix if present
        if cable_name_from_folder:
            cable_name_from_folder = _re.sub(r'_apdl_rerun(_\d+)?$', '', cable_name_from_folder)

        user_json = self.current_dir / "cable_parameters_user.json"
        with open(user_json) as f:
            cfg = json.load(f)

        if cable_name_from_folder and cable_name_from_folder in cfg.get("cables", {}):
            cable = cfg["cables"][cable_name_from_folder]
            logger.info(f"Cable detected from folder name: {cable_name_from_folder}")
        else:
            # Fall back to active cable
            active = cfg["active_cable"]
            cable = cfg["cables"][active]
            logger.warning(
                f"Could not derive cable from folder name '{run_output_dir.name}'; "
                f"falling back to active cable '{active}'"
            )

        n_parts = cable["N_Strands"]
        n_stacks = cable.get("n_stacks", 10)
        stack_height_m = cable.get("stack_height_mm", cable["cable_height"] + 0.3) * 1e-3

        apdl_submodel_dir = run_output_dir / "APDL" / "submodel" / "apdl_runfolder"
        apdl_submodel_dir.mkdir(parents=True, exist_ok=True)

        import pickle as _pickle
        import re as _re2

        d3plot_pkg = self.workspace_root / "scripts" / "d3plottoapdl_package"
        sys.path.insert(0, str(d3plot_pkg))
        try:
            import conformalRutherfordMesh as crm
            import importlib
            importlib.reload(crm)
            all_stacks_mappers, all_insulation_polygons = crm.run(
                stack_dir=stack_dir,
                n_stacks=n_stacks,
                n_parts=n_parts,
                stack_height=stack_height_m,
                output_dir=apdl_submodel_dir,
                debug_plots=debug_plots,
                wire=cable.get("wire"),
                diameter_mm=cable.get("D_Strand"),
            )
            logger.info(f"APDL submodel files written to: {apdl_submodel_dir}")

            # Serialise plot data (mapper arrays only) for the combined multi-cable overview
            def _serialise_mappers(stacks_mappers):
                out = {}
                for sn, mappers in stacks_mappers.items():
                    out[sn] = []
                    for m in mappers:
                        if m is None:
                            out[sn].append(None)
                        else:
                            out[sn].append({
                                "nodes": m.mapped_nodes.tolist(),
                                "circ_div": m.mesh.circumferential_divisions,
                            })
                return out

            cable_label = cable_name_from_folder or cfg.get("active_cable", "unknown")
            pkl_data = {
                "cable_label": cable_label,
                "stack_height_mm": stack_height_m * 1e3,
                "stacks": {
                    sn: {
                        "mappers": _serialise_mappers({sn: all_stacks_mappers[sn]})[sn],
                        "insulation": list(all_insulation_polygons.get(sn, [])),
                    }
                    for sn in all_stacks_mappers
                },
            }
            plots_dir = apdl_submodel_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            pkl_file = plots_dir / "all_stacks_data.pkl"
            with open(pkl_file, "wb") as _pf:
                _pickle.dump(pkl_data, _pf)

            # Rebuild combined multi-cable figure from all run folders that have plot data
            runs_dir = self.workspace_root / "data" / "runs"
            cables_data: dict = {}
            for folder in sorted(runs_dir.iterdir()):
                if not folder.is_dir():
                    continue
                candidate = folder / "APDL" / "submodel" / "plots" / "all_stacks_data.pkl"
                if not candidate.exists():
                    continue
                try:
                    with open(candidate, "rb") as _pf:
                        data = _pickle.load(_pf)
                    lbl = data.get("cable_label", folder.name)
                    cables_data[lbl] = data
                except Exception as _e:
                    logger.warning(f"Could not load plot data from {candidate}: {_e}")

            if cables_data:
                combined_svg = runs_dir / "overview_all_cables.svg"
                crm.plot_multi_cable_overview(cables_data, str(combined_svg))
                logger.info(f"Multi-cable overview written to: {combined_svg}")

            return True
        except Exception as e:
            logger.error(f"APDL submodel generation failed: {e}")
            traceback.print_exc()
            return False
        finally:
            if str(d3plot_pkg) in sys.path:
                sys.path.remove(str(d3plot_pkg))

    def run_cablestack_stages(self, dst_dir: Path, stages: List[str],
                              solver: Optional["CablestackSolver"] = None) -> Dict[str, bool]:
        """Run a list of cablestack stages in dependency order via the given solver.

        Auto-includes missing dependencies (see resolve_cablestack_stage_order).
        Dependency gating is on-disk, not in-process: each stage launches as long
        as its required .db file exists on disk.  This lets restart stages still
        run when the build stage exited non-zero but wrote enough of a .db for
        /POST1 to read back what *did* converge.

        Postprocess fires after each stage completes; the postprocess_* functions
        silently skip themselves when their fd_*.txt inputs are absent, so a
        fully-failed stage costs nothing here.

        Returns a dict {stage_name: success_bool} covering every stage attempted.
        """
        ordered = resolve_cablestack_stage_order(stages)
        if not ordered:
            logger.info("No cablestack stages selected — MAPDL not launched.")
            return {}

        if solver is None:
            from solver import LocalMAPDL
            try:
                with open(self.current_dir / "cable_parameters_user.json") as _f:
                    _max_parallel = int(json.load(_f).get("cablestack", {}).get("max_parallel_stages", 4))
            except Exception:
                _max_parallel = 4
            solver = LocalMAPDL(
                workspace_root=self.workspace_root,
                license_detector=self.license_detector,
                stage_registry=CABLESTACK_STAGES,
                max_parallel=_max_parallel,
            )

        def _on_stage_complete(result):
            self.run_cablestack_postprocess(dst_dir, stage_name=result.name)

        return solver.run_stages(dst_dir, ordered, on_stage_complete=_on_stage_complete)

    def copy_cablestack_files(self, run_output_dir: Path,
                              launch_apdl: bool = True, use_hpc: bool = False) -> bool:
        """Copy cablestack APDL template files to the run folder and update parameters.

        Reads n_strands, n_stacks, cable dimensions from cable_parameters_user.json,
        computes x_cab and y_cab (half-sizes in metres, with configurable margin on x),
        sets the impregnation material type, selects the BC file, and updates usecase.
        Nb3Sn modulus is the 70 GPa standard (the value the 1.2 current-amplification
        factor is calibrated against); stamped as nb3sn_modulus in loading_cycle.json
        and metadata.json.
        """
        logger.info("Copying and configuring cablestack APDL files...")

        import re as _re

        # --- resolve cable config ---
        user_json = self.current_dir / "cable_parameters_user.json"
        with open(user_json) as f:
            cfg = json.load(f)

        # Determine cable from folder name (same logic as run_apdl_submodel)
        folder_parts = run_output_dir.name.split("_", 2)
        cable_name_from_folder = folder_parts[2] if len(folder_parts) == 3 else None
        if cable_name_from_folder:
            cable_name_from_folder = _re.sub(r'_apdl_rerun(_\d+)?$', '', cable_name_from_folder)

        if cable_name_from_folder and cable_name_from_folder in cfg.get("cables", {}):
            cable = cfg["cables"][cable_name_from_folder]
            cable_label = cable_name_from_folder
        else:
            cable_label = cfg["active_cable"]
            cable = cfg["cables"][cable_label]

        cablestack_cfg = cfg.get("cablestack", {})
        impreg = cablestack_cfg.get("impreg", 4)
        bc_type = cablestack_cfg.get("bc_type", "cyclic")
        # boundary_type: 'constrained' (default) keeps sidewalls fixed in the
        # load-perpendicular direction; 'free' drops those constraints so the
        # cable can bulge under Poisson effect. Applies to all 4 load stages.
        boundary_type = cablestack_cfg.get("boundary_type", "constrained")
        if boundary_type not in ("constrained", "free"):
            raise ValueError(f"cablestack.boundary_type must be 'constrained' or 'free', got {boundary_type!r}")
        x_cab_margin_mm = cablestack_cfg.get("x_cab_margin_mm", 0.5)
        # Mesh sizes scale linearly with wire diameter: the JSON value is the reference
        # at D_Strand = 0.85 mm, and the applied size is value * (D_Strand / 0.85).
        # Default reference is 50 µm when the key is omitted.
        d_strand_mm = cable.get("D_Strand", 0.85)
        scale = d_strand_mm / 0.85
        # Per-cable override (cables.<NAME>.mesh_size_um / strand_mesh_size_um)
        # takes precedence over the global cablestack.* value.
        _mesh_ref = cable.get("mesh_size_um", cablestack_cfg.get("mesh_size_um"))
        _strand_ref = cable.get("strand_mesh_size_um", cablestack_cfg.get("strand_mesh_size_um"))
        if _mesh_ref is None:
            _mesh_ref = 50
        if _strand_ref is None:
            _strand_ref = 50
        mesh_size_m = _mesh_ref * scale * 1e-6
        strand_mesh_size_m = _strand_ref * scale * 1e-6
        logger.info(f"Mesh sizes: s_ae={mesh_size_m*1e6:.1f} µm, s_ae_str={strand_mesh_size_m*1e6:.1f} µm "
                    f"(refs={_mesh_ref}/{_strand_ref} µm at D=0.85 mm, D_Strand={d_strand_mm} mm, scale={scale:.3f})")

        n_strands = cable["N_Strands"]
        n_stacks = cable.get("n_stacks", 10)
        # y_cab: half of stack_height_mm (compressed height including insulation), in metres
        y_cab = cable.get("stack_height_mm", cable["cable_height"] + 0.3) / 2.0 * 1e-3
        # x_cab: half of cable_width + margin, in metres
        x_cab = (cable["cable_width"] / 2.0 + x_cab_margin_mm) * 1e-3

        # --- source and destination ---
        src_dir = self.workspace_root / "scripts" / "apdl" / "submodel" / "cablestack"
        dst_dir = run_output_dir / "APDL" / "submodel" / "apdl_runfolder"
        dst_dir.mkdir(parents=True, exist_ok=True)
        # Subfolder for ANSYS-written postprocessing output files
        # (fd_good_*.txt, fd_pressure_*.txt, uy_top_*.txt, ux_left_*.txt,
        #  fd_radial_*.txt, area_summary.txt). APDL *cfopen writes here via
        # the 'pp/<name>' path prefix in 7-PP.inp / 8-PP-*.inp.
        (dst_dir / "pp").mkdir(exist_ok=True)

        # Files to copy (all .inp files from cablestack template)
        for src_file in src_dir.glob("*.inp"):
            dst_file = dst_dir / src_file.name
            shutil.copy2(src_file, dst_file)

        # --- select BC file for displacement_transverse: copy the chosen variant as 5-BC.inp ---
        # bc_type ('cyclic'|'linear') x boundary_type ('constrained'|'free').
        # No 5-BC-linear-free.inp exists; refuse that combo with a clear error
        # rather than silently mis-applying.
        if boundary_type == "free":
            bc_source_name = f"5-BC-{bc_type}-free.inp"
        else:
            bc_source_name = f"5-BC-{bc_type}.inp"
        bc_source = dst_dir / bc_source_name
        bc_dest = dst_dir / "5-BC.inp"
        if bc_source.exists():
            shutil.copy2(bc_source, bc_dest)
            logger.info(f"BC file (transverse, {boundary_type}): {bc_source_name} -> 5-BC.inp")
        else:
            raise FileNotFoundError(
                f"BC template '{bc_source_name}' not found for bc_type={bc_type!r} + "
                f"boundary_type={boundary_type!r}. Linear-free is not implemented; "
                f"use bc_type='cyclic' or boundary_type='constrained'."
            )

        # --- apply boundary_type to the radial / pressure stages by overwriting
        # canonical 5-BC-*.inp files with their -free variants. Constrained mode
        # leaves the canonicals as the (already-correct) constrained versions.
        if boundary_type == "free":
            for canonical in ("5-BC-displacement-radial.inp", "5-BC-pressure.inp", "5-BC-radial.inp"):
                free_variant = dst_dir / canonical.replace(".inp", "-free.inp")
                target = dst_dir / canonical
                if not free_variant.exists():
                    raise FileNotFoundError(
                        f"boundary_type='free' but {free_variant.name} not found in templates"
                    )
                shutil.copy2(free_variant, target)
                logger.info(f"BC file (free override): {free_variant.name} -> {canonical}")

        # --- patch 0-start.inp ---
        start_file = dst_dir / "0-start.inp"
        if start_file.exists():
            text = start_file.read_text()
            text = _re.sub(r'^n_strands\s*=\s*\S+', f'n_strands = {n_strands}', text, flags=_re.MULTILINE)
            text = _re.sub(r'^n_stacks\s*=\s*\S+', f'n_stacks = {n_stacks}', text, flags=_re.MULTILINE)
            text = _re.sub(r'^x_cab\s*=\s*\S+', f'x_cab = {x_cab:.6e}', text, flags=_re.MULTILINE)
            text = _re.sub(r'^y_cab\s*=\s*\S+', f'y_cab = {y_cab:.6e}', text, flags=_re.MULTILINE)
            text = _re.sub(r"^usecase\s*=\s*'[^']*'", f"usecase = '{cable_label}'", text, flags=_re.MULTILINE)
            start_file.write_text(text)
            logger.info(f"Updated 0-start.inp: n_strands={n_strands}, n_stacks={n_stacks}, "
                         f"x_cab={x_cab:.6e}, y_cab={y_cab:.6e}, usecase='{cable_label}'")

        # --- patch 00-restart-transverse.inp (displacement_transverse stage) ---
        # RESUME from base.db restores n_strands, n_stacks, x_cab, y_cab.
        # Only the usecase needs patching for per-cable jobname / output naming.
        restart_trans_file = dst_dir / "00-restart-transverse.inp"
        if restart_trans_file.exists():
            text = restart_trans_file.read_text()
            text = _re.sub(r"^usecase\s*=\s*'[^']*'", f"usecase = '{cable_label}'", text, flags=_re.MULTILINE)
            restart_trans_file.write_text(text)
            logger.info(f"Updated 00-restart-transverse.inp: usecase='{cable_label}'")

        # --- patch 00-restart-pressure.inp (pressure_transverse stage) ---
        db_basename = f'submodel_cable_{n_strands}_{cable_label}'
        start_pres_file = dst_dir / "00-restart-pressure.inp"
        if start_pres_file.exists():
            text = start_pres_file.read_text()
            text = _re.sub(r"^usecase\s*=\s*'[^']*'", f"usecase = '{cable_label}_pressure'", text, flags=_re.MULTILINE)
            start_pres_file.write_text(text)
            logger.info(f"Updated 00-restart-pressure.inp: usecase='{cable_label}_pressure'")

        # --- patch 5-BC-pressure.inp (pressure cycle block between sentinel comments) ---
        pres_cfg = cablestack_cfg.get("pressure", {})
        gauge_length_m = float(pres_cfg.get("gauge_length_mm", 15.0)) * 1e-3
        peak_force_N = float(pres_cfg.get("peak_force_N", 45000))
        min_force_N = float(pres_cfg.get("min_force_N", 200))
        ramp_pressures_mpa = list(pres_cfg.get("ramp_pressures_MPa", [50.0, 100.0, 150.0]))
        area_m2 = 2.0 * x_cab * gauge_length_m
        p_min_pa = min_force_N / area_m2
        p_peak_pa = peak_force_N / area_m2
        # Cycle: ramp[0], unload, ramp[1], unload, ..., ramp[-1], peak, unload
        abs_pressures: list = []
        _comments: list = []
        for _i, _p_mpa in enumerate(ramp_pressures_mpa):
            abs_pressures.append(float(_p_mpa) * 1e6)
            _comments.append(f"ramp {_i+1}: {_p_mpa:.0f} MPa")
            if _i < len(ramp_pressures_mpa) - 1:
                abs_pressures.append(p_min_pa)
                _comments.append(f"unload: {min_force_N:.0f} N = {p_min_pa:.4E} Pa")
        abs_pressures.append(p_peak_pa)
        _comments.append(f"peak: {peak_force_N:.0f} N = {p_peak_pa:.4E} Pa")
        abs_pressures.append(p_min_pa)
        _comments.append(f"unload: {min_force_N:.0f} N = {p_min_pa:.4E} Pa")
        n_pres_steps = len(abs_pressures)
        array_lines = [
            f"presArray({i+1}) = {v:.6E}   ! {_comments[i]}"
            for i, v in enumerate(abs_pressures)
        ]
        new_block = (
            f"! area = 2*x_cab*gauge_L = 2*{x_cab:.4e}*{gauge_length_m:.4e} = {area_m2:.4e} m2\n"
            f"n_timesteps = {n_pres_steps}\n"
            f"*dim,presArray,array,n_timesteps\n"
            + "\n".join(array_lines)
        )
        _s_start = '! <<<PRESSURE_CYCLE_BLOCK_START>>>'
        _s_end   = '! <<<PRESSURE_CYCLE_BLOCK_END>>>'

        def _patch_pressure_block(filename: str) -> None:
            f = dst_dir / filename
            if not f.exists():
                return
            text = f.read_text()
            if _s_start in text and _s_end in text:
                _before = text[:text.index(_s_start) + len(_s_start)]
                _after  = text[text.index(_s_end):]
                text = _before + '\n' + new_block + '\n' + _after
                f.write_text(text)

        # Patch the pressure cycle block in every file that has the sentinels --
        # both canonical and -free variants, both 5-BC-pressure (transverse) and
        # 5-BC-radial (radial). Keeps the JSON-driven cycle consistent regardless
        # of which one ends up being the active canonical after boundary_type
        # selection above.
        for f in ("5-BC-pressure.inp", "5-BC-pressure-free.inp",
                  "5-BC-radial.inp",   "5-BC-radial-free.inp"):
            _patch_pressure_block(f)
        logger.info(
            f"Patched pressure cycle in 5-BC-{{pressure,radial}}{{,-free}}.inp: "
            f"{n_pres_steps} steps, ramps={ramp_pressures_mpa} MPa, "
            f"peak={p_peak_pa/1e6:.2f} MPa ({peak_force_N:.0f} N), "
            f"unload={p_min_pa/1e6:.4f} MPa ({min_force_N:.0f} N), "
            f"area={area_m2*1e6:.2f} mm2 (gauge={gauge_length_m*1e3:.1f} mm)"
        )

        # --- patch 00-restart-pressure-radial.inp (pressure_radial stage) ---
        start_pres_radial_file = dst_dir / "00-restart-pressure-radial.inp"
        if start_pres_radial_file.exists():
            text = start_pres_radial_file.read_text()
            text = _re.sub(r"^usecase\s*=\s*'[^']*'", f"usecase = '{cable_label}_radial'", text, flags=_re.MULTILINE)
            start_pres_radial_file.write_text(text)
            logger.info(f"Updated 00-restart-pressure-radial.inp: usecase='{cable_label}_radial'")

        # --- patch 00-restart-radial.inp (displacement_radial stage) ---
        start_radial_file = dst_dir / "00-restart-radial.inp"
        if start_radial_file.exists():
            text = start_radial_file.read_text()
            text = _re.sub(r"^usecase\s*=\s*'[^']*'", f"usecase = '{cable_label}_disp_radial'", text, flags=_re.MULTILINE)
            start_radial_file.write_text(text)
            logger.info(f"Updated 00-restart-radial.inp: usecase='{cable_label}_disp_radial'")

        # Nb3Sn modulus: 70 GPa standard (the value the 1.2 current-amplification
        # factor in the compbox is calibrated against). Stamped in loading_cycle.json
        # and metadata.json for the audit trail.
        nb3sn_e_pa = 70.0e9
        nb3sn_e_source = "fallback"
        nb3sn_rve_exy: Optional[Dict] = None

        # --- write loading_cycle.json (machine-readable record of the patched cycle) ---
        def _step_kind(label: str) -> str:
            low = label.lower()
            if low.startswith("ramp"):
                return "ramp"
            if low.startswith("peak"):
                return "peak"
            if low.startswith("unload"):
                return "unload"
            return "other"
        loading_cycle = {
            "schema_version": 2,
            "description": (
                "Cablestack loading cycle for the cyclic pressure stages "
                "(pressure_transverse via 5-BC-pressure.inp on lset_top, and "
                "pressure_radial via 5-BC-radial.inp on lset_left). Pressures are "
                "absolute [Pa]. APDL load step i runs TIME from i-1 to i; within "
                "the step the applied pressure ramps linearly from steps[i-2].pressure_Pa "
                "(0 for the first step) to steps[i-1].pressure_Pa. Substeps are produced "
                "by AUTOTS and reported by Time(s) in fd_pressure_*.txt / fd_radial_*.txt. "
                "The displacement-controlled stages (displacement_transverse, displacement_radial) "
                "use a separate cyclic-strain ramp defined inside their BC files."
            ),
            "cable": cable_label,
            "n_strands": n_strands,
            "n_stacks": n_stacks,
            "x_cab_m": x_cab,
            "gauge_length_m": gauge_length_m,
            "loaded_area_m2": area_m2,
            "area_formula": "2 * x_cab_m * gauge_length_m",
            "peak_force_N": peak_force_N,
            "min_force_N": min_force_N,
            "ramp_pressures_MPa": [float(p) for p in ramp_pressures_mpa],
            "p_min_Pa": p_min_pa,
            "p_peak_Pa": p_peak_pa,
            # Audit trail: where the Nb3Sn isotropic modulus in
            # 1-material_properties.inp came from ('rve' or 'fallback').
            "nb3sn_modulus": {
                "value_Pa": nb3sn_e_pa,
                "source": nb3sn_e_source,
                "rve": nb3sn_rve_exy,
            },
            "n_steps": n_pres_steps,
            "steps": [
                {
                    "index": i + 1,
                    "time": float(i + 1),
                    "pressure_Pa": abs_pressures[i],
                    "pressure_MPa": abs_pressures[i] / 1e6,
                    "kind": _step_kind(_comments[i]),
                    "label": _comments[i],
                }
                for i in range(n_pres_steps)
            ],
            "stages": {
                name: {
                    "input_file": str(spec["input_file"]),
                    "usecase": cable_label + str(spec["usecase_suffix"]),
                    "depends_on": list(spec["depends_on"]),       # type: ignore[arg-type]
                    "post_tag": str(spec["post_tag"]),
                }
                for name, spec in CABLESTACK_STAGES.items()
            },
            # Back-compat short names kept for any downstream consumer that
            # still reads the v1 schema.
            "usecases": {
                "pressure": f"{cable_label}_pressure",
                "radial":   f"{cable_label}_radial",
            },
            "outputs": {
                "displacement_transverse": [f"fd_good_{cable_label}.txt"],
                "displacement_radial":     [f"fd_good_{cable_label}_disp_radial.txt"],
                "pressure_transverse":     [f"fd_pressure_{cable_label}_pressure.txt",
                                            f"uy_top_{cable_label}_pressure.txt"],
                "pressure_radial":         [f"fd_radial_{cable_label}_radial.txt",
                                            f"ux_left_{cable_label}_radial.txt"],
            },
        }
        lc_file = dst_dir / "loading_cycle.json"
        lc_file.write_text(json.dumps(loading_cycle, indent=2) + "\n")
        logger.info(f"Wrote {lc_file.name}: {n_pres_steps} steps, area={area_m2*1e6:.2f} mm2")

        # --- patch 1-material_properties.inp (impreg selection + Nb3Sn modulus) ---
        mat_file = dst_dir / "1-material_properties.inp"
        if mat_file.exists():
            text = mat_file.read_text()
            text = _re.sub(r'^impreg\s*=\s*\S+', f'impreg = {impreg}', text, flags=_re.MULTILINE)

            new_line = (
                f"mp, ex, 3, {nb3sn_e_pa:.4E} ! 70 GPa standard "
                f"(value the 1.2 compbox current-amplification factor is calibrated against)"
            )
            text, n_subs = _re.subn(
                r'^mp,\s*ex,\s*3,\s*\S+.*$',
                new_line.replace('\\', '\\\\'),
                text,
                count=1,
                flags=_re.MULTILINE,
            )
            if n_subs == 1:
                logger.info(f"Nb3Sn modulus in deck: E = {nb3sn_e_pa/1e9:.2f} GPa")
            else:
                logger.warning("Could not locate 'mp, ex, 3, ...' line in 1-material_properties.inp")

            mat_file.write_text(text)
            logger.info(f"Updated 1-material_properties.inp: impreg={impreg}")

        # --- stamp the modulus decision into metadata.json (audit trail) ---
        meta_path = run_output_dir / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                meta["nb3sn_modulus"] = {
                    "value_Pa": nb3sn_e_pa,
                    "source": nb3sn_e_source,
                    "rve": nb3sn_rve_exy,
                }
                meta_path.write_text(json.dumps(meta, indent=4))
            except Exception as _e:
                logger.warning(f"Could not stamp nb3sn_modulus into metadata.json: {_e}")

        # --- patch 3-mesh.inp (element sizes) ---
        mesh_file = dst_dir / "3-mesh.inp"
        if mesh_file.exists():
            text = mesh_file.read_text()
            text = _re.sub(r'^s_ae\s*=\s*\S+.*$', f's_ae     = {mesh_size_m:.6e}   ! impregnation + insulation element size [m]', text, flags=_re.MULTILINE)
            text = _re.sub(r'^s_ae_str\s*=\s*\S+.*$', f's_ae_str = {strand_mesh_size_m:.6e}   ! strand area element size [m]', text, flags=_re.MULTILINE)
            mesh_file.write_text(text)
            logger.info(f"Updated 3-mesh.inp: s_ae={mesh_size_m:.2e} m ({mesh_size_m*1e6:.0f} um), "
                        f"s_ae_str={strand_mesh_size_m:.2e} m ({strand_mesh_size_m*1e6:.0f} um)")

        # --- generate jobslurm.sh covering only the configured stages ---
        # The stage list always determines what jobslurm.sh runs on the cluster,
        # even when launch_apdl=False (so the deck is consistent if someone
        # picks it up and sbatches it manually on the HPC cluster).
        cs_stages_cfg = list(cablestack_cfg.get("stages", list(CABLESTACK_STAGES.keys())))
        ordered_stages = resolve_cablestack_stage_order(cs_stages_cfg)
        if ordered_stages:
            slurm_dst = write_cablestack_jobslurm(dst_dir, cable_label, n_strands, ordered_stages)
            logger.info(
                f"Written {slurm_dst.name}: job-name={cable_label}, stages={ordered_stages}, "
                f"DB_FILE=submodel_cable_{n_strands}_{cable_label}.db"
            )
        else:
            logger.info("No cablestack stages configured — jobslurm.sh not written.")

        logger.info(f"Cablestack files written to: {dst_dir}")
        if launch_apdl:
            from solver import HPCMAPDL, LocalMAPDL
            if use_hpc:
                solver = HPCMAPDL()
            else:
                solver = LocalMAPDL(
                    workspace_root=self.workspace_root,
                    license_detector=self.license_detector,
                    stage_registry=CABLESTACK_STAGES,
                    # Independent load stages run concurrently (one MAPDL
                    # container = one license seat each).
                    max_parallel=int(cablestack_cfg.get("max_parallel_stages", 4)),
                )
            stage_results = self.run_cablestack_stages(dst_dir, cs_stages_cfg, solver=solver)
            # Propagate failure: if any requested stage failed, the cablestack
            # step is not "completed" and main.py must not stamp it as such.
            if stage_results and not all(stage_results.values()):
                failed = [k for k, v in stage_results.items() if not v]
                logger.warning(f"Cablestack stages failed: {failed}")
                return False
        return True

    def run_cablestack_postprocess(self, dst_dir: Path, stage_name: Optional[str] = None) -> bool:
        """Postprocess one (or all) cablestack stages.

        When stage_name is provided, only that stage's outputs are processed; when
        omitted, every stage with output files present is processed.  Best-effort:
        missing inputs log a notice; exceptions are caught so a postprocessing
        failure cannot mask a successful MAPDL solve.
        """
        analysis_dir = self.workspace_root / "scripts" / "analysis" / "submodel" / "cablestack"
        added = str(analysis_dir) not in sys.path
        if added:
            sys.path.insert(0, str(analysis_dir))
        try:
            import analyse_pressure  # type: ignore

            # Dispatch table: stage post_tag → analyse_pressure function name.
            dispatch = {
                "displacement_transverse": "postprocess_displacement_transverse",
                "displacement_radial":     "postprocess_displacement_radial",
                "pressure_transverse":     "postprocess_pressure_transverse",
                "pressure_radial":         "postprocess_pressure_radial",
                "thermal_cooldown":        "postprocess_thermal_cooldown",  # SKELETON
            }

            if stage_name is not None:
                tag = str(CABLESTACK_STAGES[stage_name]["post_tag"])
                if tag not in dispatch:
                    # `build` (and any future no-output stage) has no postprocess.
                    print(f"[postprocess] {stage_name}: no postprocess function -- skipping.")
                    return True
                fn_name = dispatch[tag]
                print(f"[postprocess] {stage_name}: {fn_name}")
                ok = bool(getattr(analyse_pressure, fn_name)(str(dst_dir)))
            else:
                # No stage specified — run every stage's postprocess; each skips
                # itself silently when its inputs are missing.
                print(f"[postprocess] Running all stage postprocessors on {dst_dir.name}")
                results = [bool(getattr(analyse_pressure, fn)(str(dst_dir))) for fn in dispatch.values()]
                ok = any(results)

            if ok:
                logger.info("Cablestack postprocessing completed.")
            else:
                logger.info("Cablestack postprocessing produced no output (no matching dumps found).")
            return ok
        except Exception as e:
            logger.warning(f"Cablestack postprocessing failed: {e}")
            traceback.print_exc()
            return False
        finally:
            if added and str(analysis_dir) in sys.path:
                sys.path.remove(str(analysis_dir))

    def _compbox_enabled(self, force: bool = False) -> bool:
        """Whether the compression box stage should run (opt-in via
        compression_box.enabled in cable_parameters_user.json, or forced
        with the --compbox CLI flag)."""
        if force:
            return True
        try:
            with open(self.current_dir / "cable_parameters_user.json") as f:
                return bool(json.load(f).get("compression_box", {}).get("enabled", False))
        except Exception:
            return False

    def run_compression_box(self, run_output_dir: Path, use_hpc: bool = False,
                            launch: bool = True) -> bool:
        """Workflow step 9: compression box simulation.

        Parent MAG box solve -> .rmg->VTU conversion -> field-table
        interpolation -> one-turn submodel solve -> strain/Ic analysis.
        Delegates to compbox_stage.run_compression_box and records the
        outcome under workflow_steps['9_compression_box'].
        """
        try:
            import compbox_stage
            ok = compbox_stage.run_compression_box(
                self.workspace_root, run_output_dir,
                license_detector=self.license_detector,
                use_hpc=use_hpc, launch=launch,
            )
            state = ("completed" if launch else "staged") if ok else "failed"
            self._update_metadata_step(run_output_dir, "9_compression_box", state)
            return ok
        except Exception as e:
            logger.error(f"Compression box stage failed: {e}")
            traceback.print_exc()
            self._update_metadata_step(run_output_dir, "9_compression_box", "failed")
            return False

    def run_workflow(self, selected_cable: str, termination_time: float = 1.0,
                    min_mesh_size: Optional[float] = None,
                    max_mesh_size: Optional[float] = None,
                    create_new_folder: bool = True, quick_run: bool = False,
                    debug_plots: bool = False,
                    run_cablestack: bool = True, use_hpc: bool = False,
                    run_compbox: bool = False) -> Path:
        """Run the complete workflow"""
        print("=" * 70)
        print("LTS Rutherford Workflow - Main Script")
        print("=" * 70)
        
        try:
            # Setup cable configuration
            cable_name = self.setup_cable_config(selected_cable)
            
            if quick_run:
                # Quick run mode: always use latest folder, skip early steps
                print("QUICK RUN MODE: Using latest directory, skipping PyPrimeMesh")
                print("-" * 50)
                
                existing_folder = self.find_latest_run_folder(cable_name)
                if not existing_folder:
                    raise RuntimeError(f"No existing run folder found for cable '{cable_name}'. Cannot run in quick mode.")
                
                self.run_output_dir = existing_folder
                self.run_id = existing_folder.name
                logger.info(f"Using latest run folder for quick run: {self.run_output_dir}")
                logger.info(f"Run ID: {self.run_id}")

                # Quick run: only mesh conversion + LS-DYNA
                print("Running mesh conversion...")
                conversion_success = self.run_mesh_conversion(self.run_output_dir, termination_time)
                if conversion_success:
                    print("Running LS-DYNA simulation...")
                    sim_started = self.run_lsdyna_simulation(self.run_output_dir)
                    if sim_started:
                        log_file = self.run_output_dir / "LSDYNA" / "lsdyna_container.log"
                        lsdyna_ok = self._wait_for_lsdyna_completion(log_file, termination_time)
                        if lsdyna_ok:
                            self._update_metadata_step(self.run_output_dir, "5_lsdyna_simulation", "completed")
                            pv_ok = self.run_paraview_extraction(self.run_output_dir)
                            if pv_ok:
                                self._update_metadata_step(self.run_output_dir, "6_paraview_extraction", "completed")
                            apdl_ok = self.run_apdl_submodel(self.run_output_dir, debug_plots=debug_plots)
                            if apdl_ok:
                                self._update_metadata_step(self.run_output_dir, "7_apdl_submodel", "completed")
                            cs_ok = self.copy_cablestack_files(self.run_output_dir, launch_apdl=run_cablestack, use_hpc=use_hpc)
                            if cs_ok:
                                self._update_metadata_step(self.run_output_dir, "8_cablestack", "completed")
                            else:
                                self._update_metadata_step(self.run_output_dir, "8_cablestack", "failed")
                            if self._compbox_enabled(run_compbox):
                                self.run_compression_box(self.run_output_dir, use_hpc=use_hpc)
                        else:
                            self._update_metadata_step(self.run_output_dir, "5_lsdyna_simulation", "failed")
                else:
                    logger.error("Mesh conversion failed in quick run mode")
                    raise RuntimeError("Mesh conversion failed")
            else:
                # Normal workflow mode
                # Check for existing run folder if reuse flag is set
                if not create_new_folder:
                    existing_folder = self.find_latest_run_folder(cable_name)
                    if existing_folder:
                        self.run_output_dir = existing_folder
                        self.run_id = existing_folder.name
                        logger.info(f"Reusing existing run folder: {self.run_output_dir}")
                        logger.info(f"Run ID: {self.run_id}")
                    else:
                        logger.info("No existing run folder found. Creating new one.")
                        self.run_id = self.generate_run_id(cable_name)
                        self.run_output_dir = self.setup_directories(self.run_id)
                        logger.info(f"Run ID: {self.run_id}")
                else:
                    # Generate run ID and setup directories (default behavior)
                    self.run_id = self.generate_run_id(cable_name)
                    self.run_output_dir = self.setup_directories(self.run_id)
                    logger.info(f"Run ID: {self.run_id}")
                
                # Run full workflow steps
                cable_params_file = self.run_cable_parameters(self.run_output_dir, termination_time)
                metadata_file = self.generate_metadata(self.run_id, self.run_output_dir, cable_params_file)

                self.run_geometry_generation(self.run_output_dir, metadata_file)

                lsdyna_success = self.run_lsdyna_setup(
                    self.run_output_dir, termination_time, min_mesh_size, max_mesh_size
                )
                
                if lsdyna_success:
                    conversion_success = self.run_mesh_conversion(self.run_output_dir, termination_time)
                    if conversion_success:
                        sim_started = self.run_lsdyna_simulation(self.run_output_dir)
                        if sim_started:
                            log_file = self.run_output_dir / "LSDYNA" / "lsdyna_container.log"
                            lsdyna_ok = self._wait_for_lsdyna_completion(log_file, termination_time)
                            if lsdyna_ok:
                                self._update_metadata_step(self.run_output_dir, "5_lsdyna_simulation", "completed")
                                pv_ok = self.run_paraview_extraction(self.run_output_dir)
                                if pv_ok:
                                    self._update_metadata_step(self.run_output_dir, "6_paraview_extraction", "completed")
                                apdl_ok = self.run_apdl_submodel(self.run_output_dir, debug_plots=debug_plots)
                                if apdl_ok:
                                    self._update_metadata_step(self.run_output_dir, "7_apdl_submodel", "completed")
                                cs_ok = self.copy_cablestack_files(self.run_output_dir, launch_apdl=run_cablestack, use_hpc=use_hpc)
                                if cs_ok:
                                    self._update_metadata_step(self.run_output_dir, "8_cablestack", "completed")
                                else:
                                    self._update_metadata_step(self.run_output_dir, "8_cablestack", "failed")
                                if self._compbox_enabled(run_compbox):
                                    self.run_compression_box(self.run_output_dir, use_hpc=use_hpc)
                            else:
                                self._update_metadata_step(self.run_output_dir, "5_lsdyna_simulation", "failed")
                                logger.error("LS-DYNA did not complete normally; skipping ParaView extraction.")
            
            print("=" * 70)
            print("Main script completed successfully!")
            print(f"All outputs saved to: {self.run_output_dir}")
            print("=" * 70)
            
            return self.run_output_dir
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            raise


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='LTS Rutherford Workflow')
    parser.add_argument(
        '--cable', '-c',
        type=str,
        default='R2D2_LF',
        help='Cable configuration to use (default: R2D2_LF)'
    )
    parser.add_argument(
        '--cables',
        nargs='+',
        metavar='CABLE',
        help='Run for multiple cables in sequence (e.g. --cables R2D2_LF R2D2_HF CD1); continues on failure'
    )
    parser.add_argument(
        '--time', '-t',
        type=float,
        default=0.0001,
        help='Simulation termination time in milliseconds (default: 0.0001)'
    )
    parser.add_argument(
        '--min-mesh-size',
        type=float,
        default=None,
        help='Minimum mesh size in mm (default: auto-calculated)'
    )
    parser.add_argument(
        '--max-mesh-size',
        type=float,
        default=None,
        help='Maximum mesh size in mm (default: auto-calculated)'
    )
    parser.add_argument(
        '--list-cables',
        action='store_true',
        help='List available cable configurations and exit'
    )
    parser.add_argument(
        '--reuse-folder',
        action='store_true',
        help='Reuse the most recent run folder instead of creating a new one'
    )
    parser.add_argument(
        '--quick-run',
        action='store_true',
        help='Run in latest directory, skip PyPrimeMesh and only do mesh conversion + LS-DYNA'
    )
    parser.add_argument(
        '--apdl-only',
        action='store_true',
        help='Only run d3plot-to-APDL submodel generation on the latest run folder for the selected cable'
    )
    parser.add_argument(
        '--debug-plots',
        action='store_true',
        help='Generate per-pair conformal mesh and outer-nodes plots (slow; off by default)'
    )
    parser.add_argument(
        '--no-cablestack',
        action='store_true',
        help='Skip launching MAPDL via Docker after writing cablestack files (0-start.inp is not run)'
    )
    parser.add_argument(
        '--hpc',
        action='store_true',
        help='Run the cablestack APDL stages on an SSH-reachable SLURM HPC cluster '
             '(upload + sbatch + wait + fetch postprocess outputs). Default target is '
             'ETH Euler; override via HPC_HOST / HPC_USER / HPC_REMOTE_BASE env vars. '
             'Requires SSH key access to the chosen cluster.'
    )
    parser.add_argument(
        '--compbox',
        action='store_true',
        help='Run the compression box simulation (step 9: parent MAG box + '
             'one-turn submodel + strain/Ic analysis) after the cablestack '
             'stage, even if compression_box.enabled is false in the config.'
    )
    parser.add_argument(
        '--compbox-only',
        action='store_true',
        help='Run only the compression box simulation on the latest run '
             'folder for the selected cable (needs the step-7 conformal-mesh '
             'geometry in APDL/submodel/apdl_runfolder). Combine with --hpc '
             'to solve on the cluster.'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable the run cache.  By default, if a prior run with identical '
             'cable params + cablestack config + templates already exists in '
             'data/cache/index.json, the pipeline returns that folder instead of '
             'redoing the work; if only the LS-DYNA-level inputs match, the '
             'cached d3plot is reused and only APDL submodel + cablestack are '
             're-run.  --no-cache forces a fresh run from STEP geometry onward.'
    )

    return parser.parse_args()


def list_available_cables():
    """List available cable configurations"""
    current_dir = Path(__file__).parent
    cable_params_json = current_dir / 'cable_parameters_user.json'
    
    try:
        with open(cable_params_json, 'r') as f:
            config = json.load(f)
        
        print("Available cable configurations:")
        for cable_name, params in config.get('cables', {}).items():
            print(f"  {cable_name}: {params.get('cable_name', cable_name)}")
            
    except FileNotFoundError:
        print("Cable configuration file not found")
    except json.JSONDecodeError:
        print("Invalid cable configuration file")


def main():
    """Main entry point"""
    args = parse_arguments()
    
    if args.list_cables:
        list_available_cables()
        return

    # Setup workspace
    current_dir = Path(__file__).parent
    workspace_root = current_dir.parent.parent
    runner = WorkflowRunner(workspace_root)

    if args.compbox_only:
        cable_name = runner.setup_cable_config(args.cable)
        target_folder = runner.find_latest_run_folder(cable_name)
        if target_folder is None:
            logger.error(f"No existing run folder found for cable '{cable_name}'.")
            sys.exit(1)
        runner.run_output_dir = target_folder
        runner.run_id = target_folder.name
        logger.info(f"Running compression box simulation on: {target_folder}")
        ok = runner.run_compression_box(target_folder, use_hpc=args.hpc)
        sys.exit(0 if ok else 1)

    # --- Run cache (opt-out via --no-cache) ------------------------------
    # Skip the lookup for modes that already imply reuse / partial pipeline
    # (--apdl-only is user-driven reuse; --quick-run / --reuse-folder reuse
    # the latest folder by design; --cables delegates to subprocesses which
    # each do their own cache check; --no-cache is the explicit opt-out).
    lsdyna_fp = None
    cablestack_fp = None
    if not (args.no_cache or args.apdl_only or args.quick_run
            or args.reuse_folder or args.cables):
        try:
            import cache as _cache
            lsdyna_fp, cablestack_fp, hit = _cache.check(
                workspace_root=workspace_root,
                cable_name=args.cable,
                termination_ms=args.time,
                min_mesh_size_mm=args.min_mesh_size,
                max_mesh_size_mm=args.max_mesh_size,
            )
        except Exception as e:
            logger.warning(f"Cache lookup failed ({e}); proceeding without cache.")
            hit = None

        if hit is not None and hit.level == "cablestack":
            print("=" * 70)
            print("Cache hit (full pipeline): identical inputs already solved.")
            print(f"  Run folder: {hit.folder}")
            print("  Pass --no-cache to force a fresh run.")
            print("=" * 70)
            return

        if hit is not None and hit.level == "lsdyna":
            print("=" * 70)
            print("Cache hit (LS-DYNA): reusing d3plot + per-stack CSVs from")
            print(f"  {hit.folder.name}")
            print("  Re-running only APDL submodel + cablestack stages.")
            print("  Pass --no-cache to force a fresh LS-DYNA run.")
            print("=" * 70)
            cable_name = runner.setup_cable_config(args.cable)
            runs_dir = hit.folder.parent
            base_dest_name = hit.folder.name + "_apdl_rerun"
            dest_name = base_dest_name
            counter = 2
            while (runs_dir / dest_name).exists():
                dest_name = f"{base_dest_name}_{counter}"
                counter += 1
            dest_folder = runs_dir / dest_name
            logger.info(f"Copying '{hit.folder.name}' -> '{dest_name}'...")
            shutil.copytree(hit.folder, dest_folder)
            apdl_dir = dest_folder / "APDL"
            if apdl_dir.exists():
                shutil.rmtree(apdl_dir)
            success = runner.run_apdl_submodel(dest_folder, debug_plots=args.debug_plots)
            if success:
                runner._update_metadata_step(dest_folder, "7_apdl_submodel", "completed")
                cs_ok = runner.copy_cablestack_files(
                    dest_folder,
                    launch_apdl=not args.no_cablestack, use_hpc=args.hpc,
                )
                if cs_ok:
                    runner._update_metadata_step(dest_folder, "8_cablestack", "completed")
                    try:
                        _cache.register(workspace_root, "cablestack",
                                        cablestack_fp, dest_folder, args.cable)
                    except Exception as e:
                        logger.warning(f"Cablestack cache register failed: {e}")
                if runner._compbox_enabled(args.compbox):
                    runner.run_compression_box(dest_folder, use_hpc=args.hpc)
            else:
                runner._update_metadata_step(dest_folder, "7_apdl_submodel", "failed")
            sys.exit(0 if success else 1)

    if args.apdl_only and not args.cables:
        cable_name = runner.setup_cable_config(args.cable)
        source_folder = runner.find_latest_run_folder(
            cable_name,
            require_d3plot=True,
        )
        if source_folder is None:
            logger.error(
                f"No completed run folder found for cable '{cable_name}' "
                f"(require_d3plot=True)."
            )
            sys.exit(1)

        # Build a unique destination name: <original>_apdl_rerun[_N]
        runs_dir = source_folder.parent
        base_dest_name = source_folder.name + "_apdl_rerun"
        dest_name = base_dest_name
        counter = 2
        while (runs_dir / dest_name).exists():
            dest_name = f"{base_dest_name}_{counter}"
            counter += 1
        dest_folder = runs_dir / dest_name

        logger.info(f"Copying '{source_folder.name}' → '{dest_name}'...")
        shutil.copytree(source_folder, dest_folder)
        logger.info(f"Copy complete: {dest_folder}")

        # Remove existing APDL subfolder so we start fresh
        apdl_dir = dest_folder / "APDL"
        if apdl_dir.exists():
            shutil.rmtree(apdl_dir)
            logger.info("Removed existing APDL subfolder from copy.")

        logger.info(f"Running APDL submodel on: {dest_folder}")
        success = runner.run_apdl_submodel(dest_folder, debug_plots=args.debug_plots)
        cs_ok = True
        if success:
            runner._update_metadata_step(dest_folder, "7_apdl_submodel", "completed")
            cs_ok = runner.copy_cablestack_files(dest_folder, launch_apdl=not args.no_cablestack, use_hpc=args.hpc)
            if cs_ok:
                runner._update_metadata_step(dest_folder, "8_cablestack", "completed")
            else:
                runner._update_metadata_step(dest_folder, "8_cablestack", "failed")
            if runner._compbox_enabled(args.compbox):
                runner.run_compression_box(dest_folder, use_hpc=args.hpc)
        else:
            runner._update_metadata_step(dest_folder, "7_apdl_submodel", "failed")
        sys.exit(0 if (success and cs_ok) else 1)

    # Run workflow (single cable, or multiple cables in parallel via --cables)
    cable_list = args.cables if args.cables else [args.cable]
    failed_cables = []

    def _run_cable_subprocess(cable_arg):
        """Spawn an isolated child process for one cable so module-level state
        (calc_cable_params_sim, etc.) and the shared cable_parameters_user.json
        write are fully isolated between parallel cables."""
        import subprocess as _sp
        log_path = workspace_root / "data" / "runs" / f"_parallel_{cable_arg}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--cable", cable_arg]
        if args.time != 0.0001:
            cmd += ["--time", str(args.time)]
        if args.min_mesh_size is not None:
            cmd += ["--min-mesh-size", str(args.min_mesh_size)]
        if args.max_mesh_size is not None:
            cmd += ["--max-mesh-size", str(args.max_mesh_size)]
        if args.reuse_folder:
            cmd.append("--reuse-folder")
        if args.quick_run:
            cmd.append("--quick-run")
        if args.debug_plots:
            cmd.append("--debug-plots")
        if args.no_cablestack:
            cmd.append("--no-cablestack")
        if getattr(args, 'hpc', False):
            cmd.append("--hpc")
        if getattr(args, 'apdl_only', False):
            cmd.append("--apdl-only")
        if getattr(args, 'compbox', False):
            cmd.append("--compbox")
        if getattr(args, 'lsdyna_only', False):
            cmd.append("--lsdyna-only")
        if getattr(args, 'no_cache', False):
            cmd.append("--no-cache")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["ACTIVE_CABLE"] = cable_arg
        print(f"[{cable_arg}] log: {log_path}")
        with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
            proc = _sp.Popen(cmd, env=env, stdout=lf, stderr=lf)
            rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"subprocess exited with code {rc}")

    if not args.cables:
        # Single cable — run directly in this process (no subprocess wrapping)
        try:
            runner.run_workflow(
                args.cable,
                termination_time=args.time,
                min_mesh_size=args.min_mesh_size,
                max_mesh_size=args.max_mesh_size,
                create_new_folder=not args.reuse_folder,
                quick_run=args.quick_run,
                debug_plots=args.debug_plots,
                run_cablestack=not args.no_cablestack,
                use_hpc=args.hpc,
                run_compbox=args.compbox,
            )
        except Exception as e:
            logger.error(f"Cable '{args.cable}' workflow failed: {e}")
            sys.exit(1)

        # Inspect workflow_steps: any 'failed' step (e.g. 8_cablestack from a
        # MAPDL crash) must surface as a non-zero subprocess exit so the parent
        # --cables dispatcher records the failure rather than reporting success.
        if runner.run_output_dir is not None:
            try:
                meta_path = runner.run_output_dir / "metadata.json"
                if meta_path.is_file():
                    _steps = json.loads(meta_path.read_text(encoding="utf-8")).get("workflow_steps", {})
                    _failed = [k for k, v in _steps.items() if v == "failed"]
                    if _failed:
                        logger.error(f"Cable '{args.cable}' had failed workflow steps: {_failed}")
                        sys.exit(1)
            except Exception as _e:
                logger.warning(f"Could not read metadata for failure check: {_e}")

        # Register cache fingerprints on success.  Skip when fingerprints
        # weren't computed (e.g. --no-cache, --quick-run) -- the cached hit
        # would have shortcut the run before we got here, so no point.
        if lsdyna_fp is not None and runner.run_output_dir is not None:
            try:
                import cache as _cache
                meta_path = runner.run_output_dir / "metadata.json"
                steps = {}
                if meta_path.is_file():
                    steps = json.loads(meta_path.read_text(encoding="utf-8")).get("workflow_steps", {})
                if (steps.get("5_lsdyna_simulation") == "completed"
                        and steps.get("6_paraview_extraction") == "completed"):
                    _cache.register(workspace_root, "lsdyna", lsdyna_fp,
                                    runner.run_output_dir, args.cable)
                if all(steps.get(k) == "completed" for k in (
                        "5_lsdyna_simulation", "6_paraview_extraction",
                        "7_apdl_submodel", "8_cablestack")):
                    _cache.register(workspace_root, "cablestack", cablestack_fp,
                                    runner.run_output_dir, args.cable)
            except Exception as e:
                logger.warning(f"Cache register failed: {e}")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"Running {len(cable_list)} cables in parallel: {cable_list}")
        futures = {}
        with ThreadPoolExecutor(max_workers=len(cable_list)) as pool:
            for cable_arg in cable_list:
                futures[pool.submit(_run_cable_subprocess, cable_arg)] = cable_arg
            for fut in as_completed(futures):
                cable_arg = futures[fut]
                exc = fut.exception()
                if exc:
                    logger.error(f"Cable '{cable_arg}' workflow failed: {exc}")
                    failed_cables.append(cable_arg)
                else:
                    logger.info(f"Cable '{cable_arg}' completed successfully.")

        print(f"\n{'='*70}")
        if failed_cables:
            print(f"Completed with failures: {failed_cables}")
        else:
            print("All cables completed successfully.")
        print('='*70)

    sys.exit(1 if failed_cables else 0)


if __name__ == "__main__":
    main()
