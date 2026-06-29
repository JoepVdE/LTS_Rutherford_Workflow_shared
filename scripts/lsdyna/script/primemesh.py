"""
LS-DYNA Meshing Script
Reads metadata and meshes geometry with PyPrimeMesh
"""

import os
import sys
import json

# Check for required packages
try:
    import ansys.meshing.prime as prime
except ImportError as e:
    print("✗ ansys-meshing-prime not found")
    print("  Install with: pip install ansys-meshing-prime")
    print(f"  Error: {e}")
    sys.exit(1)


def load_metadata(run_dir):
    metadata_file = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    with open(metadata_file, 'r') as f:
        return json.load(f)


def mesh_with_pyprimemesh(geometry_file, lsdyna_dir, n_strands, min_size=0.1, max_size=0.15, use_stl_parts=True):
    print("=" * 70)
    print("Meshing with PyPrimeMesh")
    print("=" * 70)
    print()
    if use_stl_parts:
        print(f"Input STL parts directory: {geometry_file}")
    else:
        print(f"Input geometry file: {geometry_file}")
    print(f"Output directory: {lsdyna_dir}")
    print()
    print("Launching Ansys Prime Server...")
    prime_root = os.environ.get('AWP_ROOT251', r'C:\Program Files\ANSYS Inc\v251')
    prime_root = os.path.join(prime_root, 'meshing', 'Prime')
    if not os.path.exists(prime_root):
        raise FileNotFoundError(f"Ansys Prime Server not found at: {prime_root}")
    print(f"Prime Server root: {prime_root}")
    with prime.launch_prime(prime_root=prime_root) as prime_client:
        model = prime_client.model
        print("✓ Prime Server connected")
        print()
        file_io = prime.FileIO(model)
        if use_stl_parts:
            stl_files = sorted([os.path.join(geometry_file, f) for f in os.listdir(geometry_file) if f.endswith('.stl')])
            print(f"Importing {len(stl_files)} individual STL parts...")
            for i, stl_file in enumerate(stl_files):
                params = prime.ImportCadParams(model, append=(i > 0))
                file_io.import_cad(stl_file, params)
                print(f"  Imported: {os.path.basename(stl_file)}")
            print(f"✓ Imported {len(model.parts)} parts")
        else:
            print("Importing geometry...")
            params = prime.ImportCadParams(model)
            file_io.import_cad(geometry_file, params)
            print(f"✓ Imported {len(model.parts)} parts")
        print()
        mesh_util = prime.lucid.Mesh(model=model)
        parts = model.parts
        print("Parts in model:")
        for i, part in enumerate(parts):
            part_summary = part.get_summary(prime.PartSummaryParams(model, print_mesh=False))
            print(f"  Part {i+1}: {part.name}")
            print(f"    - ID: {part.id}")
            print(f"    - Topo faces: {part_summary.n_topo_faces}")
            print(f"    - Topo volumes: {part_summary.n_topo_volumes}")
        print()
        print("Setting up mesh sizing...")
        print(f"  Mesh size: min={min_size} mm, max={max_size} mm")
        print()
        print("Generating surface mesh...")
        mesh_util.surface_mesh(min_size=min_size, max_size=max_size)
        print("✓ Surface mesh generated")
        print()
        print("Generating volume mesh (tetrahedral)...")
        mesh_util.volume_mesh(volume_fill_type=prime.VolumeFillType.TET)
        print("✓ Volume mesh generated")
        print()
        print("Mesh statistics:")
        for part in model.parts:
            part_summary = part.get_summary(prime.PartSummaryParams(model))
            print(f"  {part.name}:")
            print(f"    Nodes: {part_summary.n_nodes}")
            print(f"    Tet cells: {part_summary.n_tet_cells}")
            print(f"    Tri faces: {part_summary.n_tri_faces}")
        print()
        mesh_file = os.path.join(lsdyna_dir, "mesh.k")
        print(f"Exporting mesh to: {mesh_file}")
        file_io = prime.FileIO(model)
        export_params = prime.ExportLSDynaKeywordFileParams(model=model)
        file_io.export_lsdyna_keyword_file(mesh_file, export_params)
        print("✓ Mesh exported to LS-DYNA format")
        # mesh_file_cdb = os.path.join(lsdyna_dir, "mesh.cdb")
        # print(f"Exporting mesh to: {mesh_file_cdb}")   
        # export_params_cdb = prime.ExportMapdlCdbParams(
        #     model=model,
        #     write_cells=True,
        #     separate_blocks_format_type=prime.SeparateBlocksFormatType.STANDARD
        # )
        # file_io.export_mapdl_cdb(mesh_file_cdb, export_params_cdb)
        # print("✓ Mesh exported to CBD format")
    # print("=" * 70)
    print("✓ Meshing completed successfully")
    # print("=" * 70)
    print()
    return mesh_file

def update_metadata(run_dir, updates):
    metadata_file = os.path.join(run_dir, "metadata.json")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    for key, value in updates.items():
        if isinstance(value, dict) and key in metadata and isinstance(metadata[key], dict):
            metadata[key].update(value)
        else:
            metadata[key] = value
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    print("✓ Metadata updated")

def setup_lsdyna(run_dir=None, termination_time=1.0, min_mesh_size=None, max_mesh_size=None):
    print("=" * 70)
    print("LS-DYNA Simulation Setup")
    print("=" * 70)
    print()
    if run_dir is None:
        print("✗ No run directory specified")
        return False
    if not os.path.exists(run_dir):
        print(f"✗ Run directory not found: {run_dir}")
        return False
    print(f"Run directory: {run_dir}")
    print()
    try:
        print("Loading run metadata...")
        metadata = load_metadata(run_dir)
        cable_name = metadata['cable_name']
        run_id = metadata['run_id']
        n_strands = metadata['parameters']['n_strands']
        print(f"Cable: {cable_name}")
        print(f"Run ID: {run_id}")
        print(f"Number of strands: {n_strands}")
        print()
        stl_quality_preference = ['standard', 'coarse', 'fine']
        geometry_file = None
        use_stl_parts = False
        selected_quality = None
        for quality in stl_quality_preference:
            stl_parts_key = f'stl_parts_{quality}'
            stl_combined_key = f'stl_combined_{quality}'
            stl_parts_dir = metadata['files'].get(stl_parts_key)
            if not stl_parts_dir and quality == 'standard':
                possible_dir = os.path.join(run_dir, 'stl_parts')
                if os.path.exists(possible_dir):
                    from pathlib import Path
                    abs_dir = str(Path(possible_dir).resolve())
                    print(f"⚠ stl_parts_standard missing from metadata, auto-adding: {abs_dir}")
                    update_metadata(run_dir, {'files': {'stl_parts_standard': abs_dir}})
                    stl_parts_dir = abs_dir
                    metadata = load_metadata(run_dir)
            if stl_parts_dir and os.path.exists(stl_parts_dir):
                stl_files = [f for f in os.listdir(stl_parts_dir) if f.endswith('.stl')]
                if stl_files:
                    geometry_file = stl_parts_dir
                    use_stl_parts = True
                    selected_quality = quality
                    print(f"Using {quality.upper()} quality STL parts from: {os.path.basename(stl_parts_dir)}")
                    print(f"  {len(stl_files)} parts found")
                    break
        if geometry_file is None:
            for quality in stl_quality_preference:
                stl_combined_key = f'stl_combined_{quality}'
                stl_combined = metadata['files'].get(stl_combined_key)
                if stl_combined and os.path.exists(stl_combined):
                    geometry_file = stl_combined
                    selected_quality = quality
                    print(f"Using {quality.upper()} quality combined STL: {os.path.basename(geometry_file)}")
                    break
        if geometry_file is None:
            step_file = metadata['files'].get('step_geometry')
            if step_file and os.path.exists(step_file):
                geometry_file = step_file
                print(f"Using STEP geometry: {os.path.basename(geometry_file)}")
            else:
                print(f"✗ No geometry files found (STL or STEP)")
                return False
        print()
        lsdyna_dir = os.path.join(run_dir, "LSDYNA")
        if not os.path.exists(lsdyna_dir):
            os.makedirs(lsdyna_dir)
            print(f"✓ Created LSDYNA directory: {lsdyna_dir}")
            print()
        if min_mesh_size is None or max_mesh_size is None:
            import math
            strand_diameter = metadata['parameters'].get('strand_diameter', 0.7252)
            circumference = math.pi * strand_diameter
            target_elements_around = 40
            if min_mesh_size is None:
                min_mesh_size = circumference / target_elements_around
            if max_mesh_size is None:
                max_mesh_size = min_mesh_size * 1.5
            print(f"Auto-calculated mesh sizes for strand diameter {strand_diameter} mm:")
            print(f"  Target elements around circumference: {target_elements_around}")
            print(f"  Min mesh size: {min_mesh_size:.4f} mm")
            print(f"  Max mesh size: {max_mesh_size:.4f} mm")
            print()
        mesh_file = mesh_with_pyprimemesh(geometry_file, lsdyna_dir, n_strands, min_mesh_size, max_mesh_size, use_stl_parts)
        print("Updating metadata...")
        update_metadata(run_dir, {
            'workflow_steps': {'5_lsdyna_simulation': 'mesh_completed'},
            'files': {
                'lsdyna_mesh': os.path.abspath(mesh_file)
            }
        })
        # print()
        # print("=" * 70)
        # print("✓ LS-DYNA meshing completed successfully!")
        print()
        print("Output files:")
        print(f"  • mesh.k - Meshed geometry with {n_strands} parts")
        # print(f"  • mesh.cdb - ANSYS CDB format")
        print()
        print(f"All files saved to: {lsdyna_dir}")
        # print()
        # print("Note: Material definitions and boundary conditions can be added manually")
        # print("      or using LS-DYNA Pre/Post.")
        print("=" * 70)
        return True
    except Exception as e:
        print(f"✗ Error during LS-DYNA setup: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Set up LS-DYNA simulation with meshing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python primemesh.py ../../data/runs/20240101_120000_R2D2_LF --termination-time 1.0'
    )
    parser.add_argument('run_dir', help='Path to the run directory containing metadata.json and geometry files')
    parser.add_argument('--termination-time', type=float, default=1.0, 
                        help='Simulation termination time in seconds (default: 1.0)')
    parser.add_argument('--min-mesh-size', type=float, default=None,
                        help='Minimum mesh element size in mm (default: auto-calculated from strand diameter)')
    parser.add_argument('--max-mesh-size', type=float, default=None,
                        help='Maximum mesh element size in mm (default: 1.5 × min size)')
    args = parser.parse_args()
    success = setup_lsdyna(
        args.run_dir,
        termination_time=args.termination_time,
        min_mesh_size=args.min_mesh_size,
        max_mesh_size=args.max_mesh_size
    )
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
