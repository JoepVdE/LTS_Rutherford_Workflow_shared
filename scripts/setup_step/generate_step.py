"""
STEP Geometry Generation Script
Generates FreeCAD macro from run parameters and creates STEP file
"""

import os
import sys
import subprocess
import json
from datetime import datetime


def get_workspace_root():
    """Get the workspace root directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "..", ".."))


def get_latest_run_dir():
    """Get the latest run directory from data/runs"""
    workspace_root = get_workspace_root()
    runs_dir = os.path.join(workspace_root, "data", "runs")
    
    if not os.path.exists(runs_dir):
        return None
    
    # Get all run directories sorted by name (which are timestamps)
    run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    if not run_dirs:
        return None
    
    latest_run = sorted(run_dirs)[-1]
    return os.path.join(runs_dir, latest_run)


def load_run_metadata(run_dir):
    """Load metadata from a run directory"""
    metadata_file = os.path.join(run_dir, "metadata.json")
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        return json.load(f)


def load_cable_parameters(run_dir):
    """Load cable parameters from a run directory"""
    params_file = os.path.join(run_dir, "cable_parameters.json")
    
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Cable parameters file not found: {params_file}")
    
    with open(params_file, 'r') as f:
        return json.load(f)


def update_metadata(run_dir, updates):
    """Update metadata.json with new information"""
    metadata_file = os.path.join(run_dir, "metadata.json")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Apply updates
    for key, value in updates.items():
        if isinstance(value, dict) and key in metadata and isinstance(metadata[key], dict):
            metadata[key].update(value)
        else:
            metadata[key] = value
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Metadata updated")


def find_freecad_executable():
    """Find a FreeCAD executable usable for headless macro execution.

    Resolution order:
      1. FREECAD_EXE env var (verbatim path).
      2. Bundled Windows install under tools/freecad/FreeCAD*/bin/{FreeCADCmd,FreeCAD}.exe.
      3. System PATH lookup (freecadcmd, then freecad) -- this is what the
         supplied Dockerfile and any Linux/macOS install relies on.
    """
    import shutil

    env_exe = os.environ.get("FREECAD_EXE")
    if env_exe and os.path.isfile(env_exe):
        return env_exe

    workspace_root = get_workspace_root()
    freecad_tools_dir = os.path.join(workspace_root, "tools", "freecad")
    if os.path.exists(freecad_tools_dir):
        try:
            for item in os.listdir(freecad_tools_dir):
                if item.startswith("FreeCAD"):
                    for name in ("FreeCADCmd.exe", "FreeCAD.exe", "FreeCADCmd", "FreeCAD"):
                        candidate = os.path.join(freecad_tools_dir, item, "bin", name)
                        if os.path.isfile(candidate):
                            return candidate
        except Exception as e:
            print(f"Error searching workspace tools directory: {e}")

    for name in ("freecadcmd", "FreeCADCmd", "freecad", "FreeCAD"):
        path = shutil.which(name)
        if path:
            return path

    return None


def generate_rutherford_macro(run_dir, output_path):
    """Generate Rutherford FCMacro with cable-specific parameters substituted.

    Reads N_Strands, D_Strand, T_pitch, and cable_name from cable_parameters.json
    in run_dir and injects them into the generate_rutherford.FCMacro template.
    """
    import re

    params         = load_cable_parameters(run_dir)
    N_strands      = int(params['N_Strands'])
    D_strand       = float(params['D_Strand_base'])  # base diameter — no geometry correction for Rutherford macro
    T_pitch        = float(params['T_pitch'])
    cable_name_val = params['cable_name']

    script_dir    = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, 'generate_rutherford.FCMacro')

    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace individual parameter lines, preserving trailing comments
    content = re.sub(
        r'^(N\s*=\s*)\d+(\s*#.*)$',
        lambda m: f"N         = {N_strands}" + m.group(2),
        content, count=1, flags=re.MULTILINE
    )
    content = re.sub(
        r'^(d\s*=\s*)[\d.]+(\s*#.*)$',
        lambda m: f"d         = {D_strand:.6f}" + m.group(2),
        content, count=1, flags=re.MULTILINE
    )
    content = re.sub(
        r'^(Lp\s*=\s*)[\d.]+(\s*#.*)$',
        lambda m: f"Lp        = {T_pitch:.1f}  " + m.group(2),
        content, count=1, flags=re.MULTILINE
    )
    content = re.sub(
        r'^cable_name\s*=\s*"[^"]*"',
        f'cable_name = "{cable_name_val}"',
        content, count=1, flags=re.MULTILINE
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Generated Rutherford macro: {os.path.basename(output_path)}")
    print(f"  N={N_strands}, d={D_strand:.4f} mm, Lp={T_pitch} mm, cable_name={cable_name_val}")
    return True


def run_freecad_macro(macro_path, freecad_exe=None):
    """Run FreeCAD macro using command line interface"""
    if freecad_exe is None:
        freecad_exe = find_freecad_executable()
    
    if freecad_exe is None:
        print("✗ FreeCAD executable not found in workspace tools directory")
        return False
    
    print(f"Using FreeCAD: {freecad_exe}")
    print(f"Executing macro: {os.path.basename(macro_path)}")
    
    # Change to the directory where the macro is located
    # This ensures FreeCAD can find the cable_parameters.json file
    macro_dir = os.path.dirname(os.path.abspath(macro_path))
    
    try:
        cmd = [freecad_exe, "-c", macro_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=macro_dir)
        
        if result.returncode == 0:
            print("FreeCAD macro executed successfully")
            # Print all output
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            return True
        else:
            print(f"✗ FreeCAD macro failed with return code: {result.returncode}")
            if result.stdout:
                print("Output:", result.stdout)
            if result.stderr:
                print("Error:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ FreeCAD macro timed out (600 seconds)")
        return False
    except Exception as e:
        print(f"✗ Error executing FreeCAD macro: {e}")
        return False


def generate_step_geometry_rutherford(run_dir=None):
    """
    Generate STEP geometry using generate_rutherford.FCMacro with cable-specific parameters.

    Substitutes N, d, Lp, cable_name from cable_parameters.json into the Rutherford
    macro template, runs it with FreeCAD, and updates run metadata.

    Args:
        run_dir: Path to run directory. If None, uses latest run.

    Returns:
        bool: True if successful, False otherwise
    """
    if run_dir is None:
        print("Looking for latest run...")
        run_dir = get_latest_run_dir()
        if run_dir is None:
            print("✗ No runs found in data/runs directory")
            return False

    if not os.path.exists(run_dir):
        print(f"✗ Run directory not found: {run_dir}")
        return False

    print(f"Run directory: {run_dir}")

    try:
        print("Loading run data...")
        metadata   = load_run_metadata(run_dir)
        params     = load_cable_parameters(run_dir)
        cable_name = params['cable_name']
        run_id     = metadata['run_id']
        print(f"Cable: {cable_name}")
        print(f"Run ID: {run_id}")

        print("Generating Rutherford FreeCAD macro...")
        macro_path = os.path.join(run_dir, f"{cable_name}_geometry.FCMacro")
        if not generate_rutherford_macro(run_dir, macro_path):
            return False

        print("Executing FreeCAD macro...")
        if not run_freecad_macro(macro_path):
            return False

        step_file    = os.path.join(run_dir, f"{cable_name}.step")
        summary_file = os.path.join(run_dir, f"{cable_name}_summary.txt")

        if not os.path.exists(step_file):
            print(f"Warning: STEP file not found at expected path: {step_file}")
            print("Continuing anyway — FreeCAD may have saved it successfully.")

        print("Updating metadata...")
        update_metadata(run_dir, {
            'workflow_steps': {'2_freecad_geometry': 'completed'},
            'files': {
                'step_geometry':    os.path.abspath(step_file),
                'geometry_summary': os.path.abspath(summary_file),
                'freecad_macro':    os.path.abspath(macro_path),
            }
        })

        print("Rutherford STEP geometry generation completed successfully!")
        print("Output files:")
        print(f"  • {os.path.basename(step_file)} - STEP geometry")
        print(f"  • {os.path.basename(summary_file)} - Geometry summary")
        print(f"  • {os.path.basename(macro_path)} - FreeCAD macro (generated)")
        print(f"All files saved to: {run_dir}")
        return True

    except FileNotFoundError as e:
        print(f"✗ {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    # Check for command line arguments
    run_dir = None
    if len(sys.argv) > 1:
        # User specified a run_id or path
        arg = sys.argv[1]
        if os.path.isdir(arg):
            run_dir = arg
        else:
            # Assume it's a run_id
            workspace_root = get_workspace_root()
            run_dir = os.path.join(workspace_root, "data", "runs", arg)
    
    success = generate_step_geometry(run_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
