"""
Magnetic Field Heatmap Generator

This script reads VTU data and creates heatmaps for magnetic field components:
- Bx (X-component of magnetic field)
- By (Y-component of magnetic field) 
- Bz (Z-component of magnetic field)
- B_magnitude (Total magnetic field magnitude)

Author: Joep
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.interpolate import griddata
import os
import re
import glob
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy import stats

# Configuration — paths resolved against the new paper_clean_version/ layout.
# Override via env vars if invoking from a different root.
# Informational only - the actual lookup globs '*enhanced_loadstep_*.vtu' in
# find_enhanced_vtu_files(), so SMACC/CD1 VTU base names both match.
VTU_FILE_PATTERN = '*enhanced_loadstep_*.vtu'

script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("COMPBOX_ROOT", os.path.dirname(script_dir))
BASE_DIR = os.environ.get("COMPBOX_VTU_DIR",
                          os.path.join(ROOT, "box_simulation", "results"))
KEYPOINTS_BASE_DIR = os.environ.get("COMPBOX_KEYPOINTS_DIR",
                                    os.path.join(ROOT, "submodel_simulation", "geometry"))
INPUT_FOR_SUBMODEL_DIR = os.environ.get("COMPBOX_FIELD_OUT_DIR",
                                        os.path.join(ROOT, "submodel_simulation", "input_for_submodel"))
BASE_OUTPUT_DIR = os.environ.get("COMPBOX_HEATMAP_DIR",
                                 os.path.join(ROOT, "results", "magnetic_heatmaps"))

# Cases follow the geometry files actually staged: one case per
# keypoints_nodes_<i>.txt in KEYPOINTS_BASE_DIR (6 for SMACC_HF, 10 for the
# old CD1 set). Override with COMPBOX_N_CASES if you need a subset.
def _detect_n_cases():
    env = os.environ.get("COMPBOX_N_CASES")
    if env:
        return int(env)
    n = 0
    while os.path.isfile(os.path.join(KEYPOINTS_BASE_DIR, f"keypoints_nodes_{n + 1}.txt")):
        n += 1
    if n == 0:
        raise FileNotFoundError(
            f"No keypoints_nodes_<i>.txt files found in {KEYPOINTS_BASE_DIR} - "
            "stage the submodel geometry before generating field tables.")
    return n

CASE_RANGE = range(1, _detect_n_cases() + 1)

def get_keypoints_file(i_case):
    """Get keypoints file path for given case number."""
    return os.path.join(KEYPOINTS_BASE_DIR, f'keypoints_nodes_{i_case}.txt')

def get_case_output_dir(i_case):
    """Get output directory for given case number."""
    return f'{BASE_OUTPUT_DIR}_case_{i_case}'

def parse_keypoints_file(keypoints_file):
    """Parse the keypoints file to extract polygon definitions."""
    print(f"Parsing keypoints file: {keypoints_file}")
    
    polygons = {}
    current_area = None
    keypoints = {}
    
    try:
        with open(keypoints_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            # Parse area definition
            if line.startswith('NUMSTR') and 'area' in line:
                # Extract area number (e.g., "NUMSTR, area, 10000.0" -> 10000)
                match = re.search(r'area,\s*(\d+)', line)
                if match:
                    current_area = int(match.group(1))
                    if current_area not in polygons:
                        polygons[current_area] = []
            
            # Parse keypoints (e.g., "k,10001.0,1.5400049993e-04,3.3560840942e-04,0.0")
            elif line.startswith('k,'):
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        point_id = int(float(parts[1]))
                        x_coord = float(parts[2])
                        z_coord = float(parts[3])  # Using Z as the second coordinate
                        keypoints[point_id] = (x_coord, z_coord)
                    except (ValueError, IndexError):
                        continue
            
            # Parse FLST section to group keypoints into polygons
            elif line.startswith('FLST') and current_area is not None:
                # Start collecting points for current polygon
                current_polygon_points = []
            
            elif line.startswith('FITEM') and current_area is not None:
                # Extract point ID from FITEM line
                match = re.search(r'FITEM,\d+,(\d+)', line)
                if match:
                    point_id = int(match.group(1))
                    if point_id in keypoints:
                        current_polygon_points.append(keypoints[point_id])
            
            # End of polygon definition (A,P51X line)
            elif line.startswith('A,P51X') and current_area is not None:
                if len(current_polygon_points) >= 3:  # Valid polygon needs at least 3 points
                    polygons[current_area].append(current_polygon_points.copy())
                current_polygon_points = []
        
        print(f"Parsed {len(polygons)} area groups with polygons:")
        for area_id, polys in polygons.items():
            print(f"  Area {area_id}: {len(polys)} polygons")
            
    except FileNotFoundError:
        print(f"Warning: Keypoints file not found: {keypoints_file}")
        polygons = {}
    except Exception as e:
        print(f"Error parsing keypoints file: {e}")
        polygons = {}
    
    return polygons

def calculate_polygon_area(polygon_points):
    """
    Calculate the area of a polygon using the Shoelace formula.
    
    Args:
        polygon_points: List of tuples [(x1, y1), (x2, y2), ..., (xn, yn)]
    
    Returns:
        float: Area of the polygon
    """
    if len(polygon_points) < 3:
        return 0.0
    
    # Ensure polygon is closed by adding first point at the end if not already there
    if polygon_points[0] != polygon_points[-1]:
        polygon_points = polygon_points + [polygon_points[0]]
    
    n = len(polygon_points) - 1  # Exclude the repeated last point
    area = 0.0
    
    # Shoelace formula: Area = 0.5 * |Σ(x_i * y_{i+1} - x_{i+1} * y_i)|
    for i in range(n):
        x_i, y_i = polygon_points[i]
        x_next, y_next = polygon_points[i + 1]
        area += (x_i * y_next - x_next * y_i)
    
    return abs(area) / 2.0

def calculate_all_polygon_areas(polygons):
    """
    Calculate areas for all polygons and print results.
    
    Args:
        polygons: Dictionary of area_id -> list of polygon_points
    
    Returns:
        dict: Dictionary of area_id -> list of areas
    """
    print("\n" + "="*60)
    print("POLYGON AREA CALCULATIONS")
    print("="*60)
    
    all_areas = {}
    total_area = 0.0
    total_polygons = 0
    
    for area_id, area_polygons in polygons.items():
        areas = []
        area_total = 0.0
        
        print(f"\nArea Group {area_id}:")
        print(f"  Number of polygons: {len(area_polygons)}")
        
        for i, polygon_points in enumerate(area_polygons):
            area = calculate_polygon_area(polygon_points)
            areas.append(area)
            area_total += area
            total_area += area
            total_polygons += 1
            
            print(f"  Polygon {i+1}: {area:.6e} m²")
        
        all_areas[area_id] = areas
        print(f"  Group {area_id} total area: {area_total:.6e} m²")
        print(f"  Group {area_id} average area: {area_total/len(area_polygons):.6e} m²")
    
    print(f"\n" + "-"*40)
    print(f"SUMMARY:")
    print(f"  Total polygons: {total_polygons}")
    print(f"  Total area: {total_area:.6e} m²")
    print(f"  Average polygon area: {total_area/total_polygons if total_polygons > 0 else 0:.6e} m²")
    print("="*60)
    
    return all_areas

def calculate_nb3sn_interfilamentary_areas(all_areas):
    """
    Calculate Nb3Sn interfilamentary areas by subtracting the first polygon area 
    from the second polygon area for each group.
    
    Args:
        all_areas: Dictionary of area_id -> list of areas
    
    Returns:
        dict: Dictionary of area_id -> interfilamentary_area
    """
    print("\n" + "="*60)
    print("CALCULATING Nb3Sn INTERFILAMENTARY AREAS")
    print("="*60)
    
    interfilamentary_areas = {}
    
    for area_id, areas in sorted(all_areas.items()):
        if len(areas) >= 3:  # Ensure we have at least 3 polygons
            area_1 = areas[0]  # First polygon (index 0)
            area_2 = areas[1]  # Second polygon (index 1)
            area_3 = areas[2]  # Third polygon (index 2)
            
            # Calculate interfilamentary area (second - first)
            interfilamentary_area = area_2 - area_1
            interfilamentary_areas[area_id] = interfilamentary_area
            
            # Convert to mm² for display
            interfilamentary_area_mm2 = interfilamentary_area * 1e6
            
            print(f"{area_id:<10} {area_1:<15.6e} {area_2:<15.6e} {area_3:<15.6e} {interfilamentary_area:<18.6e} {interfilamentary_area_mm2:<20.6f}")
            
        else:
            print(f"{area_id:<10} {'ERROR: Not enough polygons'}")
    
    # Calculate summary statistics
    if interfilamentary_areas:
        interf_values = list(interfilamentary_areas.values())
        interf_values_mm2 = [v * 1e6 for v in interf_values]
        
        print("\n" + "-" * 40)
        print("SUMMARY STATISTICS:")
        print(f"  Total groups processed: {len(interfilamentary_areas)}")
        print(f"  Mean interfilamentary area: {np.mean(interf_values):.6e} m² ({np.mean(interf_values_mm2):.6f} mm²)")
        print(f"  Std interfilamentary area: {np.std(interf_values):.6e} m² ({np.std(interf_values_mm2):.6f} mm²)")
        print(f"  Min interfilamentary area: {np.min(interf_values):.6e} m² ({np.min(interf_values_mm2):.6f} mm²)")
        print(f"  Max interfilamentary area: {np.max(interf_values):.6e} m² ({np.max(interf_values_mm2):.6f} mm²)")
        print(f"  Total interfilamentary area: {np.sum(interf_values):.6e} m² ({np.sum(interf_values_mm2):.6f} mm²)")
    
    print("="*60)
    
    return interfilamentary_areas

def visualize_polygon_areas(all_areas, output_dir=''):
    """
    Create comprehensive visualizations of polygon areas including:
    - Histogram with statistics
    - Box plot by area groups
    - Scatter plot of areas by group
    - Statistical summary plots
    """
    print("\n" + "="*60)
    print("CREATING AREA VISUALIZATIONS")
    print("="*60)
    
    # Prepare data
    all_area_values = []
    group_labels = []
    group_areas = {}
    
    for area_id, areas in all_areas.items():
        all_area_values.extend(areas)
        group_labels.extend([area_id] * len(areas))
        group_areas[area_id] = areas
    
    all_area_values = np.array(all_area_values)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Histogram with statistics
    ax1 = plt.subplot(2, 3, 1)
    n_bins = min(20, len(all_area_values) // 3)  # Adaptive number of bins
    n, bins, patches = plt.hist(all_area_values * 1e6, bins=n_bins, alpha=0.7, 
                               color='skyblue', edgecolor='black', linewidth=0.5)
    
    # Add statistics to histogram
    mean_area = np.mean(all_area_values) * 1e6
    median_area = np.median(all_area_values) * 1e6
    std_area = np.std(all_area_values) * 1e6
    
    plt.axvline(mean_area, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_area:.3f}')
    plt.axvline(median_area, color='green', linestyle='--', linewidth=2, label=f'Median: {median_area:.3f}')
    plt.xlabel('Area (mm²)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Histogram of Polygon Areas', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = f'Count: {len(all_area_values)}\nMean: {mean_area:.3f} mm²\nStd: {std_area:.3f} mm²\nMin: {np.min(all_area_values)*1e6:.3f} mm²\nMax: {np.max(all_area_values)*1e6:.3f} mm²'
    plt.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Box plot by area groups
    ax2 = plt.subplot(2, 3, 2)
    group_ids = sorted(group_areas.keys())
    box_data = [np.array(group_areas[gid]) * 1e6 for gid in group_ids]
    
    bp = plt.boxplot(box_data, labels=[str(gid) for gid in group_ids], patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
    for box, color in zip(bp['boxes'], colors):
        box.set_facecolor(color)
        box.set_alpha(0.7)
    
    plt.xlabel('Area Group ID', fontsize=12)
    plt.ylabel('Area (mm²)', fontsize=12)
    plt.title('Box Plot of Areas by Group', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 3. Scatter plot of areas by group
    ax3 = plt.subplot(2, 3, 3)
    for i, (area_id, areas) in enumerate(group_areas.items()):
        x_positions = [area_id] * len(areas)
        y_values = np.array(areas) * 1e6
        plt.scatter(x_positions, y_values, alpha=0.6, s=50, label=f'Group {area_id}')
    
    plt.xlabel('Area Group ID', fontsize=12)
    plt.ylabel('Area (mm²)', fontsize=12)
    plt.title('Scatter Plot of Individual Polygon Areas', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    ax4 = plt.subplot(2, 3, 4)
    sorted_areas = np.sort(all_area_values * 1e6)
    cumulative = np.arange(1, len(sorted_areas) + 1) / len(sorted_areas)
    plt.plot(sorted_areas, cumulative, 'b-', linewidth=2)
    plt.fill_between(sorted_areas, cumulative, alpha=0.3)
    
    plt.xlabel('Area (mm²)', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('Cumulative Distribution of Polygon Areas', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add percentile lines
    percentiles = [25, 50, 75, 90, 95]
    for p in percentiles:
        value = np.percentile(sorted_areas, p)
        plt.axvline(value, color='red', linestyle=':', alpha=0.7, 
                   label=f'{p}th: {value:.3f}' if p in [25, 50, 75] else None)
    
    plt.legend()
    
    # 5. Area group statistics comparison
    ax5 = plt.subplot(2, 3, 5)
    group_means = [np.mean(group_areas[gid]) * 1e6 for gid in group_ids]
    group_stds = [np.std(group_areas[gid]) * 1e6 for gid in group_ids]
    
    x_pos = np.arange(len(group_ids))
    bars = plt.bar(x_pos, group_means, yerr=group_stds, capsize=5, 
                  alpha=0.7, color='lightcoral', edgecolor='black')
    
    plt.xlabel('Area Group ID', fontsize=12)
    plt.ylabel('Mean Area (mm²)', fontsize=12)
    plt.title('Mean Area by Group (with Standard Deviation)', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, [str(gid) for gid in group_ids], rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean_val, std_val in zip(bars, group_means, group_stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.001,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # 6. Normal distribution comparison
    ax6 = plt.subplot(2, 3, 6)
    
    # Create histogram
    plt.hist(all_area_values * 1e6, bins=n_bins, density=True, alpha=0.7, 
             color='lightblue', edgecolor='black', label='Actual Data')
    

    
    # Save the comprehensive visualization
    filename = 'polygon_areas_comprehensive_analysis.svg'
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive area analysis: {filepath}")
    plt.close()
    
    # Create detailed statistical summary
    # create_area_statistics_summary(all_areas, output_dir)


def load_and_filter_data(vtu_file_path):
    """Load VTU file and extract magnetic field data."""
    print(f"Loading VTU file: {vtu_file_path}")
    
    # Load the mesh
    mesh = pv.read(vtu_file_path)
    print(f"Loaded mesh with {mesh.n_points} points and {mesh.n_cells} cells")
    
    # Filter for hexahedral elements only
    print("Filtering for hexahedral elements...")
    hex_mesh = mesh.extract_cells_by_type([pv.CellType.QUADRATIC_HEXAHEDRON, pv.CellType.HEXAHEDRON])
    print(f"Filtered mesh has {hex_mesh.n_points} points and {hex_mesh.n_cells} cells")
    
    # Create a slice through the data (Y-normal at origin)
    print("Creating slice through the data...")
    slice_mesh = hex_mesh.slice(normal=[0, 1, 0], origin=[0, 0, 0])
    print(f"Slice has {slice_mesh.n_points} points and {slice_mesh.n_cells} cells")
    
    return slice_mesh

def extract_magnetic_data(slice_mesh):
    """Extract coordinates and magnetic field data from the slice."""
    print("Extracting magnetic field data...")
    
    # Get coordinates
    points = slice_mesh.points
    # Try swapping X and Z coordinates to match keypoints coordinate system
    x_coords = points[:, 2]  # Using Z from VTU as X in plot
    z_coords = points[:, 0]  # Using X from VTU as Z in plot
    
    print(f"Coordinate ranges after swap:")
    print(f"  X (was Z): [{np.min(x_coords):.6f}, {np.max(x_coords):.6f}]")
    print(f"  Z (was X): [{np.min(z_coords):.6f}, {np.max(z_coords):.6f}]")
    
    # Extract magnetic field components
    try:
        bx = slice_mesh.point_data['Bx']
        by = slice_mesh.point_data['By'] 
        bz = slice_mesh.point_data['Bz']
        b_mag = slice_mesh.point_data['B_magnitude']
        
        print(f"Successfully extracted magnetic field data:")
        print(f"  Bx range: [{np.min(bx):.6f}, {np.max(bx):.6f}]")
        print(f"  By range: [{np.min(by):.6f}, {np.max(by):.6f}]")
        print(f"  Bz range: [{np.min(bz):.6f}, {np.max(bz):.6f}]")
        print(f"  B_magnitude range: [{np.min(b_mag):.6f}, {np.max(b_mag):.6f}]")
        
        return x_coords, z_coords, bx, by, bz, b_mag
        
    except KeyError as e:
        print(f"Error: Missing magnetic field data in VTU file: {e}")
        print(f"Available point data arrays: {list(slice_mesh.point_data.keys())}")
        raise

def create_regular_grid(x_coords, z_coords, nx=100, nz=100):
    """Create a regular grid for interpolation."""
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    
    # Add small padding to avoid edge effects
    x_padding = 0#(x_max - x_min) * 0.01
    z_padding = 0#(z_max - z_min) * 0.01
    
    x_grid = np.linspace(x_min - x_padding, x_max + x_padding, nx)
    z_grid = np.linspace(z_min - z_padding, z_max + z_padding, nz)
    
    X_grid, Z_grid = np.meshgrid(x_grid, z_grid)
    
    print(f"Created regular grid: {nx}x{nz}")
    print(f"  X range: [{x_min:.6f}, {x_max:.6f}] with padding")
    print(f"  Z range: [{z_min:.6f}, {z_max:.6f}] with padding")
    
    return X_grid, Z_grid

def interpolate_data(x_coords, z_coords, field_data, X_grid, Z_grid):
    """Interpolate scattered data onto regular grid."""
    print("Interpolating data onto regular grid...")
    
    # Remove any NaN or infinite values
    mask = np.isfinite(field_data) & np.isfinite(x_coords) & np.isfinite(z_coords)
    
    if np.sum(mask) == 0:
        print("Warning: No valid data points found for interpolation")
        return np.zeros_like(X_grid)
    
    # Interpolate using linear method
    field_interpolated = griddata(
        (x_coords[mask], z_coords[mask]), 
        field_data[mask], 
        (X_grid, Z_grid), 
        method='linear',
        fill_value=np.nan
    )
    
    return field_interpolated

def save_polygon_centers_once(polygons, X_grid, Z_grid):
    """Calculate polygon centers and return them for APDL file generation."""
    if not polygons:
        return []
    
    # Calculate the same offset used in add_polygon_overlays
    x_min, x_max = X_grid.min(), X_grid.max()
    z_min, z_max = Z_grid.min(), Z_grid.max()
    
    # Collect all polygon points to calculate offset
    all_polygon_points = []
    for area_id, area_polygons in polygons.items():
        for polygon_points in area_polygons:
            all_polygon_points.extend(polygon_points)
    
    centers = []
    
    if all_polygon_points:
        # Calculate offset to center polygons on slice
        poly_x_coords_raw = [p[0] for p in all_polygon_points]
        poly_z_coords_raw = [p[1] for p in all_polygon_points]
        
        poly_x_center = (min(poly_x_coords_raw) + max(poly_x_coords_raw)) / 2
        poly_z_center = (min(poly_z_coords_raw) + max(poly_z_coords_raw)) / 2
        
        slice_x_center = (x_min + x_max) / 2
        slice_z_center = (z_min + z_max) / 2
        
        offset_x = slice_x_center - poly_x_center
        offset_z = slice_z_center - poly_z_center
        
        # Calculate centers without saving individual CSV
        for area_id, area_polygons in polygons.items():
            for poly_idx, polygon_points in enumerate(area_polygons, start=1):
                if len(polygon_points) < 1:
                    continue
                if poly_idx == 1:  # Only save polygon 1 centers
                    # Apply same offset used in plotting
                    swapped_pts = [(p[0] + offset_x, p[1] + offset_z) for p in polygon_points]
                    cx = sum(pt[0] for pt in swapped_pts) / len(swapped_pts)
                    cz = sum(pt[1] for pt in swapped_pts) / len(swapped_pts)
                    centers.append((area_id, cx, cz))
        
        print(f"Calculated {len(centers)} polygon centers for APDL file generation")

    return centers

def add_polygon_overlays(ax, polygons, X_grid, Z_grid):
    """Add polygon overlays to the plot."""
    if not polygons:
        return
    
    # Define colors for different area types - all black
    area_colors = {
        10000: 'black',    # First area type
        10100: 'black',    # Second area type  
        10200: 'black',    # Third area type
        10300: 'black',    # Fourth area type
        10400: 'black',    # Fifth area type
        10500: 'black',    # Additional area types
        10600: 'black',
        10700: 'black',
        10800: 'black',
        10900: 'black',
        11000: 'black',
        11100: 'black',
        11200: 'black',
        11300: 'black',
        11400: 'black',
        11500: 'black',
        11600: 'black',
        11700: 'black',
        11800: 'black',
        11900: 'black',
        12000: 'black',
    }
    
    # Create mapping from area_id to strand number (1-21)
    area_to_strand = {
        10000: 1, 10100: 2, 10200: 3, 10300: 4, 10400: 5, 10500: 6, 10600: 7,
        10700: 8, 10800: 9, 10900: 10, 11000: 11, 11100: 12, 11200: 13, 11300: 14,
        11400: 15, 11500: 16, 11600: 17, 11700: 18, 11800: 19, 11900: 20, 12000: 21
    }
    
    # Plot extents
    x_min, x_max = X_grid.min(), X_grid.max()
    z_min, z_max = Z_grid.min(), Z_grid.max()
    
    # print(f"Plot bounds: X[{x_min:.6f}, {x_max:.6f}], Z[{z_min:.6f}, {z_max:.6f}]")
    
    patches = []
    area_labels = set()
    all_polygon_points = []
    strand_centers = []  # Store centers for numbering
    
    # Collect all polygon points to check their ranges
    for area_id, area_polygons in polygons.items():
        for polygon_points in area_polygons:
            all_polygon_points.extend(polygon_points)
    
    if all_polygon_points:
        # Swap coordinates: original X becomes Z, original Z becomes X
        poly_x_coords_raw = [p[0] for p in all_polygon_points]  # Use original Z as X
        poly_z_coords_raw = [p[1] for p in all_polygon_points]  # Use original X as Z
        
        # Calculate polygon center
        poly_x_center = (min(poly_x_coords_raw) + max(poly_x_coords_raw)) / 2
        poly_z_center = (min(poly_z_coords_raw) + max(poly_z_coords_raw)) / 2
        
        # Calculate slice center  
        slice_x_center = (x_min + x_max) / 2
        slice_z_center = (z_min + z_max) / 2
        
        # Calculate offset to center polygons on slice
        offset_x = slice_x_center - poly_x_center
        offset_z = slice_z_center - poly_z_center
        
        poly_x_coords = [p + offset_x for p in poly_x_coords_raw]
        poly_z_coords = [p + offset_z for p in poly_z_coords_raw]
        
        # Compute and list centers for each polygon after applying the computed offset
        offset_polygon_centers = []  # list of tuples: (area_id, poly_index, center_x, center_z)

        for area_id, area_polygons in polygons.items():
            for poly_idx, polygon_points in enumerate(area_polygons, start=1):
                if len(polygon_points) < 1:
                    continue
                # Apply same swap/offset used later when plotting
                swapped_pts = [(p[0] + offset_x, p[1] + offset_z) for p in polygon_points]
                cx = sum(pt[0] for pt in swapped_pts) / len(swapped_pts)
                cz = sum(pt[1] for pt in swapped_pts) / len(swapped_pts)
                offset_polygon_centers.append((area_id, poly_idx, cx, cz))
                # print(f"Offset polygon center - Area {area_id} Polygon {poly_idx}: X={cx:.6f}, Z={cz:.6f}")

        # Optionally expose centers for other code in this function
        all_offset_polygon_centers = offset_polygon_centers
        
        
        # print(f"Polygon center: X={poly_x_center:.6f}, Z={poly_z_center:.6f}")
        # print(f"Slice center: X={slice_x_center:.6f}, Z={slice_z_center:.6f}")
        # print(f"Applied offset: X={offset_x:.6f}, Z={offset_z:.6f}")
        # print(f"Polygon ranges (after centering): X[{min(poly_x_coords):.6f}, {max(poly_x_coords):.6f}], Z[{min(poly_z_coords):.6f}, {max(poly_z_coords):.6f}]")
    
    for area_id, area_polygons in polygons.items():
        color = area_colors.get(area_id, 'white')
        strand_number = area_to_strand.get(area_id, 0)
        
        # Only use the first polygon (polygon 1) for labeling to avoid duplicate numbers
        for poly_idx, polygon_points in enumerate(area_polygons):
            if len(polygon_points) >= 3:
                # Swap coordinates for each polygon point: original X becomes Z, original Z becomes X
                # Apply offset to center polygons on slice plane
                swapped_polygon_points = [(p[0] + offset_x, p[1] + offset_z) for p in polygon_points]
                
                # Check if any part of polygon overlaps with plot bounds (more lenient filtering)
                polygon_x = [p[0] for p in swapped_polygon_points]  # Now using swapped coordinates
                polygon_z = [p[1] for p in swapped_polygon_points]
                
                # Check for overlap (not strict containment)
                x_overlap = not (max(polygon_x) < x_min or min(polygon_x) > x_max)
                z_overlap = not (max(polygon_z) < z_min or min(polygon_z) > z_max)
                
                if x_overlap and z_overlap:
                    polygon = Polygon(swapped_polygon_points, closed=True, 
                                    fill=False, edgecolor=color, linewidth=1.5, alpha=0.8)
                    patches.append(polygon)
                    area_labels.add((area_id, color))
                    
                    # Calculate center for strand numbering (only for first polygon to avoid duplicates)
                    if poly_idx == 0 and strand_number > 0:
                        center_x = sum(p[0] for p in swapped_polygon_points) / len(swapped_polygon_points)
                        center_z = sum(p[1] for p in swapped_polygon_points) / len(swapped_polygon_points)
                        strand_centers.append((center_x, center_z, strand_number))
    
    # Add patches to plot
    for patch in patches:
        ax.add_patch(patch)
    
    # Add strand number labels
    for center_x, center_z, strand_number in strand_centers:
        ax.text(center_x, center_z, str(strand_number), 
               horizontalalignment='center', verticalalignment='center',
               fontsize=8, fontweight='bold', color='white',
               bbox=dict(boxstyle='circle,pad=0.2', facecolor='red', alpha=0.8, edgecolor='black'))
    
    print(f"Added {len(patches)} polygon overlays for {len(area_labels)} area types")
    print(f"Added {len(strand_centers)} strand number labels")

def add_polygon_overlays_simple(ax, polygons, X_grid, Z_grid, fontsize=14.4):
    """Add polygon overlays with larger strand numbers for simplified heatmaps."""
    # Calculate plot bounds
    x_min, x_max = X_grid.min(), X_grid.max()
    z_min, z_max = Z_grid.min(), Z_grid.max()
    
    # Use black color for all polygons
    polygon_color = 'black'
    
    # Strand number mapping
    area_to_strand = {
        10000: 1, 10100: 2, 10200: 3, 10300: 4, 10400: 5, 10500: 6, 10600: 7,
        10700: 8, 10800: 9, 10900: 10, 11000: 11, 11100: 12, 11200: 13, 11300: 14,
        11400: 15, 11500: 16, 11600: 17, 11700: 18, 11800: 19, 11900: 20, 12000: 21
    }
    
    patches = []
    strand_centers = []
    area_labels = set()
    
    # Calculate polygon center and offset (same logic as original)
    all_x_coords = []
    all_z_coords = []
    
    for area_id, area_polygons in polygons.items():
        for polygon_points in area_polygons:
            for point in polygon_points:
                all_x_coords.append(point[0])
                all_z_coords.append(point[1])
    
    # Initialize offset values
    offset_x = 0.0
    offset_z = 0.0
    
    if all_x_coords and all_z_coords:
        poly_x_coords_raw = all_x_coords
        poly_z_coords_raw = all_z_coords
        
        poly_x_center = (min(poly_x_coords_raw) + max(poly_x_coords_raw)) / 2
        poly_z_center = (min(poly_z_coords_raw) + max(poly_z_coords_raw)) / 2
        
        slice_x_center = (x_min + x_max) / 2
        slice_z_center = (z_min + z_max) / 2
        
        offset_x = slice_x_center - poly_x_center
        offset_z = slice_z_center - poly_z_center
        
        poly_x_coords = [p + offset_x for p in poly_x_coords_raw]
        poly_z_coords = [p + offset_z for p in poly_z_coords_raw]
    
    for area_id, area_polygons in polygons.items():
        strand_number = area_to_strand.get(area_id, 0)
        
        for poly_idx, polygon_points in enumerate(area_polygons):
            if len(polygon_points) >= 3:
                swapped_polygon_points = [(p[0] + offset_x, p[1] + offset_z) for p in polygon_points]
                
                polygon_x = [p[0] for p in swapped_polygon_points]
                polygon_z = [p[1] for p in swapped_polygon_points]
                
                x_overlap = not (max(polygon_x) < x_min or min(polygon_x) > x_max)
                z_overlap = not (max(polygon_z) < z_min or min(polygon_z) > z_max)
                
                if x_overlap and z_overlap:
                    polygon = Polygon(swapped_polygon_points, closed=True, 
                                    fill=False, edgecolor=polygon_color, linewidth=1.5, alpha=0.8)
                    patches.append(polygon)
                    area_labels.add((area_id, polygon_color))
                    
                    if poly_idx == 0 and strand_number > 0:
                        center_x = sum(p[0] for p in swapped_polygon_points) / len(swapped_polygon_points)
                        center_z = sum(p[1] for p in swapped_polygon_points) / len(swapped_polygon_points)
                        strand_centers.append((center_x, center_z, strand_number))
    
    # Add patches to plot
    for patch in patches:
        ax.add_patch(patch)
    
    # Add strand number labels with larger font size
    for center_x, center_z, strand_number in strand_centers:
        ax.text(center_x, center_z, str(strand_number), 
               horizontalalignment='center', verticalalignment='center',
               fontsize=fontsize, fontweight='bold', color='white',
               bbox=dict(boxstyle='circle,pad=0.2', facecolor='red', alpha=0.8, edgecolor='black'))

def create_simple_heatmap(X_grid, Z_grid, field_data, output_dir='', polygons=None, loadstep=0, i_case=1):
    """Create and save a simplified heatmap for B_magnitude with horizontal colorbar below."""
    print(f"Creating simplified heatmap for B_magnitude (Case {i_case}, Loadstep {loadstep})...")
    
    # Calculate aspect ratio based on data dimensions
    x_range = X_grid.max() - X_grid.min()
    z_range = Z_grid.max() - Z_grid.min()
    aspect_ratio = z_range / x_range
    
    # Adjust figure size to maintain equal spacing and add space for horizontal colorbar below
    # Reduced by 20%
    base_width = 8  # 10 * 0.8 = 8
    fig_height = base_width * aspect_ratio + 1.5  # +1.5 for horizontal colorbar space below
    
    fig, ax = plt.subplots(figsize=(base_width, fig_height))
    
    # Create heatmap with equal aspect ratio
    im = ax.imshow(field_data, extent=[X_grid.min(), X_grid.max(), Z_grid.min(), Z_grid.max()], 
                   origin='lower', aspect='equal', cmap='RdBu_r', interpolation='bilinear')
    
    # Add polygon overlays with larger strand numbers if provided
    if polygons:
        add_polygon_overlays_simple(ax, polygons, X_grid, Z_grid, fontsize=14.4)
    
    # Labels and simplified title
    ax.set_xlabel('X Position (m)', fontsize=14.4)
    ax.set_ylabel('Z Position (m)', fontsize=14.4)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Add horizontal colorbar below with increased padding
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.25, shrink=0.56)  # Increased pad to 0.25 for more spacing
    cbar.set_label('B (T)', fontsize=14.4)
    cbar.ax.tick_params(labelsize=12)  # Increase colorbar tick labels by 20% (10 * 1.2 = 12)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot with _simple suffix
    filename = f'b_magnitude_simple_case_{i_case}_loadstep_{loadstep:02d}.svg'
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved simplified heatmap: {filepath}")
    
    plt.close()

def create_heatmap(X_grid, Z_grid, field_data, field_name, units='T', output_dir='', polygons=None, loadstep=0, i_case=1):
    """Create and save a heatmap for a magnetic field component."""
    print(f"Creating heatmap for {field_name} (Case {i_case}, Loadstep {loadstep})...")
    
    # Calculate aspect ratio based on data dimensions
    x_range = X_grid.max() - X_grid.min()
    z_range = Z_grid.max() - Z_grid.min()
    aspect_ratio = z_range / x_range
    
    # Adjust figure size to maintain equal spacing
    base_width = 10
    fig_height = base_width * aspect_ratio + 2  # +2 for colorbar space
    
    plt.figure(figsize=(base_width, fig_height))
    
    # Create heatmap with equal aspect ratio
    im = plt.imshow(field_data, extent=[X_grid.min(), X_grid.max(), Z_grid.min(), Z_grid.max()], 
                    origin='lower', aspect='equal', cmap='RdBu_r', interpolation='bilinear')
    
    # Add polygon overlays if provided
    if polygons:
        add_polygon_overlays(plt.gca(), polygons, X_grid, Z_grid)
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label(f'{field_name} ({units})', fontsize=12)
    
    # Labels and title
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Z Position (m)', fontsize=12)
    plt.title(f'Magnetic Field Component: {field_name} (Case {i_case}, Loadstep {loadstep})', fontsize=14, fontweight='bold')
    
    # Grid
    plt.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    filename = f'{field_name.lower()}_heatmap_case_{i_case}_loadstep_{loadstep:02d}.svg'
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap: {filepath}")
    
    # Show statistics
    valid_data = field_data[~np.isnan(field_data)]
    if len(valid_data) > 0:
        print(f"  {field_name} statistics:")
        print(f"    Min: {np.min(valid_data):.6f} {units}")
        print(f"    Max: {np.max(valid_data):.6f} {units}")
        print(f"    Mean: {np.mean(valid_data):.6f} {units}")
        print(f"    Std: {np.std(valid_data):.6f} {units}")
    
    plt.close()

def create_combined_heatmap(X_grid, Z_grid, bx_data, by_data, bz_data, bmag_data, output_dir='', polygons=None, loadstep=0, i_case=1):
    """Create a combined 2x2 subplot heatmap showing all magnetic field components."""
    print(f"Creating combined heatmap (Case {i_case}, Loadstep {loadstep})...")
    
    # Calculate aspect ratio based on data dimensions
    x_range = X_grid.max() - X_grid.min()
    z_range = Z_grid.max() - Z_grid.min()
    aspect_ratio = z_range / x_range
    
    # Adjust figure size to maintain equal spacing in subplots
    base_width = 16
    fig_height = base_width * aspect_ratio * 0.5 + 3  # 0.5 for 2 rows, +3 for titles/labels
    
    fig, axes = plt.subplots(2, 2, figsize=(base_width, fig_height))
    fig.suptitle(f'Magnetic Field Components Heatmaps (Case {i_case}, Loadstep {loadstep})', fontsize=16, fontweight='bold')
    
    # Component data and titles
    components = [
        (bx_data, 'Bx', axes[0, 0]),
        (by_data, 'By', axes[0, 1]),
        (bz_data, 'Bz', axes[1, 0]),
        (bmag_data, 'B_magnitude', axes[1, 1])
    ]
    
    # Calculate global min/max for consistent color scale
    all_data = np.concatenate([
        bx_data[np.isfinite(bx_data)].flatten(),
        by_data[np.isfinite(by_data)].flatten(),
        bz_data[np.isfinite(bz_data)].flatten(),
        bmag_data[np.isfinite(bmag_data)].flatten()
    ])
    vmin, vmax = np.min(all_data), np.max(all_data)
    print(f"Using consistent color scale: [{vmin:.6f}, {vmax:.6f}] T")
    
    for field_data, field_name, ax in components:
        # Create heatmap with equal aspect ratio and consistent color scale
        im = ax.imshow(field_data, extent=[X_grid.min(), X_grid.max(), Z_grid.min(), Z_grid.max()], 
                       origin='lower', aspect='equal', cmap='RdBu_r', interpolation='bilinear',
                       vmin=vmin, vmax=vmax)
        
        # Add polygon overlays if provided
        if polygons:
            add_polygon_overlays(ax, polygons, X_grid, Z_grid)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(f'{field_name} (T)', fontsize=10)
        
        # Labels and title
        ax.set_xlabel('X Position (m)', fontsize=10)
        ax.set_ylabel('Z Position (m)', fontsize=10)
        ax.set_title(f'{field_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    filename = f'magnetic_field_components_combined_case_{i_case}_loadstep_{loadstep:02d}.svg'
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved combined heatmap: {filepath}")
    plt.close()

def find_enhanced_vtu_files(base_dir='.'):
    """Find all enhanced VTU files from multiple loadsteps."""
    # Look for enhanced loadstep files
    pattern = os.path.join(base_dir, '*enhanced_loadstep_*.vtu')
    vtu_files = glob.glob(pattern)
    
    if not vtu_files:
        # Fallback: look for single enhanced file
        pattern = os.path.join(base_dir, '*enhanced.vtu')
        vtu_files = glob.glob(pattern)
    
    # Sort files by loadstep number
    def extract_loadstep(filename):
        match = re.search(r'loadstep_(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0  # For files without loadstep number
    
    vtu_files.sort(key=extract_loadstep)
    
    print(f"Found {len(vtu_files)} enhanced VTU files:")
    for i, filepath in enumerate(vtu_files):
        loadstep = extract_loadstep(os.path.basename(filepath))
        print(f"  {i+1:2d}. Loadstep {loadstep:2d}: {os.path.basename(filepath)}")
    
    return vtu_files

def extract_loadstep_from_filename(filename):
    """Extract loadstep number from filename."""
    match = re.search(r'loadstep_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def save_combined_data_to_csv(number_list, center_x_list, center_z_list, interfilamentary_areas, 
                             bx_list, by_list, bz_list, bmag_list, loadstep=0, i_case=1, output_dir=''):
    """
    Save all combined data (centers, areas, and magnetic field) to ANSYS APDL input file only (no CSV).
    
    Args:
        number_list: Array of group IDs
        center_x_list: Array of polygon center X coordinates
        center_z_list: Array of polygon center Z coordinates
        interfilamentary_areas: Dictionary of area_id -> interfilamentary_area
        bx_list: Array of Bx field values at polygon centers
        by_list: Array of By field values at polygon centers
        bz_list: Array of Bz field values at polygon centers
        bmag_list: Array of B_magnitude field values at polygon centers
        loadstep: Loadstep number for filename
        i_case: Case number for filename
        output_dir: Directory to save the APDL file
    """
    print(f"Saving combined data to ANSYS APDL file for Case {i_case}, Loadstep {loadstep}...")
    
    # Create mapping from area_id to strand number (1-21)
    area_to_strand = {
        10000: 1, 10100: 2, 10200: 3, 10300: 4, 10400: 5, 10500: 6, 10600: 7,
        10700: 8, 10800: 9, 10900: 10, 11000: 11, 11100: 12, 11200: 13, 11300: 14,
        11400: 15, 11500: 16, 11600: 17, 11700: 18, 11800: 19, 11900: 20, 12000: 21
    }
    
    combined_apdl_filename = f'nb3sn_combined_data_case_{i_case}_{loadstep}.inp'

    # Always write to the canonical location used by submodel_simulation.
    apdl_dir = INPUT_FOR_SUBMODEL_DIR
    os.makedirs(apdl_dir, exist_ok=True)
    
    # Save all .inp files in the input_for_submodel directory
    combined_apdl_path = os.path.join(apdl_dir, combined_apdl_filename)
    
    try:
        # Save ANSYS APDL input file
        with open(combined_apdl_path, 'w', encoding='utf-8') as apdlfile:
            # Write APDL header
            apdlfile.write('! ANSYS APDL Input File - Nb3Sn Combined Data\n')
            apdlfile.write(f'! Generated from Case {i_case}, Loadstep {loadstep} magnetic field data\n')
            apdlfile.write(f'! Date: {import_datetime().datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            apdlfile.write('!\n')
            apdlfile.write('! Define arrays with 21 elements for each data column\n')
            apdlfile.write('!\n\n')
            
            # Define array dimensions only for the first file (loadstep 0)
            if loadstep == 0:
                apdlfile.write('! Define array dimensions (21 strands) - Only for first loadstep\n')
                apdlfile.write('*DIM,Strand_Number,ARRAY,21\n')
                apdlfile.write('*DIM,Group_ID,ARRAY,21\n')
                apdlfile.write('*DIM,Center_X_m,ARRAY,21\n')
                apdlfile.write('*DIM,Center_Z_m,ARRAY,21\n')
                apdlfile.write('*DIM,Interfilamentary_Area_m2,ARRAY,21\n')
                apdlfile.write('*DIM,Bx_T,ARRAY,21\n')
                apdlfile.write('*DIM,By_T,ARRAY,21\n')
                apdlfile.write('*DIM,Bz_T,ARRAY,21\n')
                apdlfile.write('*DIM,B_magnitude_T,ARRAY,21\n')
                apdlfile.write('\n')
            else:
                apdlfile.write('! Arrays already defined in loadstep 0 file\n')
                apdlfile.write('! Skipping *DIM commands to avoid redefinition\n')
                apdlfile.write('\n')
            
            # Prepare data arrays - ensure we have exactly 21 elements
            # Sort by strand number to ensure proper ordering
            data_tuples = []
            for i in range(len(number_list)):
                group_id = int(number_list[i])
                strand_number = area_to_strand.get(group_id, 0)
                interf_area = interfilamentary_areas.get(group_id, 0.0)
                
                # Handle NaN values by replacing with zero
                bx_val = bx_list[i] if not np.isnan(bx_list[i]) else 0.0
                by_val = by_list[i] if not np.isnan(by_list[i]) else 0.0
                bz_val = bz_list[i] if not np.isnan(bz_list[i]) else 0.0
                bmag_val = bmag_list[i] if not np.isnan(bmag_list[i]) else 0.0
                
                data_tuples.append((strand_number, group_id, center_x_list[i], center_z_list[i], 
                                  interf_area, bx_val, by_val, bz_val, bmag_val))
            
            # Sort by strand number
            data_tuples.sort(key=lambda x: x[0])
            
            # Pad with zeros if we have less than 21 strands
            while len(data_tuples) < 21:
                data_tuples.append((0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            
            # Take only first 21 if we have more
            data_tuples = data_tuples[:21]
            
            # Write data to arrays
            apdlfile.write('! Populate arrays with data\n')
            for i, (strand_num, group_id, cx, cz, area, bx, by, bz, bmag) in enumerate(data_tuples, 1):
                apdlfile.write(f'Strand_Number({i}) = {strand_num}\n')
                apdlfile.write(f'Group_ID({i}) = {group_id}\n')
                apdlfile.write(f'Center_X_m({i}) = {cx:13.6E}\n')
                apdlfile.write(f'Center_Z_m({i}) = {cz:13.6E}\n')
                apdlfile.write(f'Interfilamentary_Area_m2({i}) = {area:13.6E}\n')
                apdlfile.write(f'Bx_T({i}) = {bx:13.6E}\n')
                apdlfile.write(f'By_T({i}) = {by:13.6E}\n')
                apdlfile.write(f'Bz_T({i}) = {bz:13.6E}\n')
                apdlfile.write(f'B_magnitude_T({i}) = {bmag:13.6E}\n')
                apdlfile.write('\n')
            
            # Add some utility commands
            apdlfile.write('! Utility commands\n')
            apdlfile.write('! List all arrays (uncomment to use):\n')
            apdlfile.write('! *VLIST,Strand_Number(1),21\n')
            apdlfile.write('! *VLIST,Group_ID(1),21\n')
            apdlfile.write('! *VLIST,Center_X_m(1),21\n')
            apdlfile.write('! *VLIST,Center_Z_m(1),21\n')
            apdlfile.write('! *VLIST,Interfilamentary_Area_m2(1),21\n')
            apdlfile.write('! *VLIST,Bx_T(1),21\n')
            apdlfile.write('! *VLIST,By_T(1),21\n')
            apdlfile.write('! *VLIST,Bz_T(1),21\n')
            apdlfile.write('! *VLIST,B_magnitude_T(1),21\n')
            apdlfile.write('\n')
            apdlfile.write('! End of data definition\n')
        
        print(f"Saved ANSYS APDL input file to {combined_apdl_path}")
        
    except Exception as e:
        print(f"Error saving combined data files: {e}")


def import_datetime():
    """Import datetime module (helper function for APDL file generation)"""
    import datetime
    return datetime

        
        
def main():
    """Main function to generate magnetic field heatmaps for all cases and loadsteps."""
    print("="*60)
    print("MULTI-CASE MULTI-LOADSTEP MAGNETIC FIELD HEATMAP GENERATOR")
    print("="*60)
    
    try:
        # Find all enhanced VTU files (same VTU files used for all cases)
        vtu_files = find_enhanced_vtu_files(BASE_DIR)
        
        if not vtu_files:
            print("No enhanced VTU files found!")
            return
        
        print(f"Found {len(vtu_files)} VTU files to process for each case")
        print(f"Processing cases {min(CASE_RANGE)} to {max(CASE_RANGE)}")
        
        # Process each case (1 to 10)
        for i_case in CASE_RANGE:
            print(f"\n" + "="*80)
            print(f"PROCESSING CASE {i_case} ({i_case}/{max(CASE_RANGE)})")
            print("="*80)
            
            # Get keypoints file for this case
            keypoints_file = get_keypoints_file(i_case)
            print(f"Using keypoints file: {keypoints_file}")
            
            # Check if keypoints file exists
            if not os.path.exists(keypoints_file):
                print(f"Warning: Keypoints file not found: {keypoints_file}")
                print(f"Skipping case {i_case}")
                continue
            
            # Load polygon data from keypoints file for this case
            polygons = parse_keypoints_file(keypoints_file)
            
            # Calculate polygon areas for this case
            if polygons:
                polygon_areas = calculate_all_polygon_areas(polygons)
                interfilamentary_areas = calculate_nb3sn_interfilamentary_areas(polygon_areas)
                # Create area visualizations for this case
                case_output_dir = get_case_output_dir(i_case)
                visualize_polygon_areas(polygon_areas, output_dir=case_output_dir)
            else:
                print(f"No polygons found for case {i_case} - skipping area calculations")
                interfilamentary_areas = {}
            
            # Process each VTU file for this case
            for j, vtu_file in enumerate(vtu_files):
                loadstep = extract_loadstep_from_filename(os.path.basename(vtu_file))
                
                print(f"\n" + "-"*60)
                print(f"PROCESSING CASE {i_case}, LOADSTEP {loadstep} ({j+1}/{len(vtu_files)})")
                print(f"File: {os.path.basename(vtu_file)}")
                print("-"*60)
                
                # Create case and loadstep-specific output directory
                case_output_dir = get_case_output_dir(i_case)
                loadstep_output_dir = os.path.join(case_output_dir, f'loadstep_{loadstep:02d}')
                
                # Load and filter data for this loadstep
                slice_mesh = load_and_filter_data(vtu_file)
                
                # Extract magnetic field data
                z_coords, x_coords, bx, by, bz, b_mag = extract_magnetic_data(slice_mesh)

                # Create regular grid for interpolation
                X_grid, Z_grid = create_regular_grid(x_coords, z_coords, nx=150, nz=150)
                
                # Interpolate each magnetic field component
                print(f"Interpolating magnetic field components for case {i_case}, loadstep {loadstep}...")
                bx_interp = interpolate_data(x_coords, z_coords, bx, X_grid, Z_grid)
                by_interp = interpolate_data(x_coords, z_coords, by, X_grid, Z_grid)
                bz_interp = interpolate_data(x_coords, z_coords, bz, X_grid, Z_grid)
                bmag_interp = interpolate_data(x_coords, z_coords, b_mag, X_grid, Z_grid)
                
                # Calculate polygon centers for this case and loadstep
                print(f"Calculating polygon centers for case {i_case}, loadstep {loadstep}...")
                centers = save_polygon_centers_once(polygons, X_grid, Z_grid)
                
                if centers:
                    centers = np.array(centers)
                    number_list = centers[:,0]
                    center_x_list = centers[:,1]
                    center_z_list = centers[:,2]
                    
                    # Interpolate magnetic field values at polygon centers
                    bx_list = interpolate_data(x_coords, z_coords, bx, center_x_list, center_z_list)
                    by_list = interpolate_data(x_coords, z_coords, by, center_x_list, center_z_list)
                    bz_list = interpolate_data(x_coords, z_coords, bz, center_x_list, center_z_list)
                    bmag_list = interpolate_data(x_coords, z_coords, b_mag, center_x_list, center_z_list)
                    
                    print(f'Processing {len(number_list)} polygon centers for case {i_case}, loadstep {loadstep}')
                    
                    # Save combined data to APDL file for this case and loadstep
                    save_combined_data_to_csv(number_list, center_x_list, center_z_list, interfilamentary_areas,
                                            bx_list, by_list, bz_list, bmag_list, loadstep=loadstep, 
                                            i_case=i_case, output_dir=loadstep_output_dir)
                else:
                    print(f"No polygon centers found for case {i_case}, loadstep {loadstep} - skipping field interpolation")

                # Create individual heatmaps with polygon overlays
                print(f"Creating individual heatmaps for case {i_case}, loadstep {loadstep}...")
                create_heatmap(X_grid, Z_grid, bx_interp, 'Bx', output_dir=loadstep_output_dir, 
                              polygons=polygons, loadstep=loadstep, i_case=i_case)
                create_heatmap(X_grid, Z_grid, by_interp, 'By', output_dir=loadstep_output_dir, 
                              polygons=polygons, loadstep=loadstep, i_case=i_case)
                create_heatmap(X_grid, Z_grid, bz_interp, 'Bz', output_dir=loadstep_output_dir, 
                              polygons=polygons, loadstep=loadstep, i_case=i_case)
                create_heatmap(X_grid, Z_grid, bmag_interp, 'B_magnitude', output_dir=loadstep_output_dir, 
                              polygons=polygons, loadstep=loadstep, i_case=i_case)
                
                # Create simplified B_magnitude heatmap
                print(f"Creating simplified B_magnitude heatmap for case {i_case}, loadstep {loadstep}...")
                create_simple_heatmap(X_grid, Z_grid, bmag_interp, output_dir=loadstep_output_dir, 
                                     polygons=polygons, loadstep=loadstep, i_case=i_case)
                
                # Create combined heatmap with polygon overlays
                print(f"Creating combined heatmap for case {i_case}, loadstep {loadstep}...")
                create_combined_heatmap(X_grid, Z_grid, bx_interp, by_interp, bz_interp, bmag_interp, 
                                      output_dir=loadstep_output_dir, polygons=polygons, loadstep=loadstep, i_case=i_case)
                
                print(f"Completed processing case {i_case}, loadstep {loadstep}")
            
            print(f"Completed processing case {i_case}")
        
        print("\n" + "="*80)
        print("MULTI-CASE MULTI-LOADSTEP HEATMAP GENERATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Processed {len(CASE_RANGE)} cases")
        print(f"Processed {len(vtu_files)} loadsteps per case")
        print(f"Total combinations: {len(CASE_RANGE)} cases × {len(vtu_files)} loadsteps = {len(CASE_RANGE) * len(vtu_files)}")
        
        print("\nGenerated case directories:")
        for i_case in CASE_RANGE:
            case_dir = get_case_output_dir(i_case)
            if os.path.exists(case_dir):
                print(f"  {case_dir}/ - Contains loadstep subdirectories for case {i_case}")
                for vtu_file in vtu_files:
                    loadstep = extract_loadstep_from_filename(os.path.basename(vtu_file))
                    loadstep_dir = f'loadstep_{loadstep:02d}'
                    print(f"    {loadstep_dir}/ - Heatmaps for case {i_case}, loadstep {loadstep}")
        print("="*80)
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
