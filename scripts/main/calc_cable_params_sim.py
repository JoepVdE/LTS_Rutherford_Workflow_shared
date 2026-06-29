"""
Python conversion of R2D2_LF_size_and_LSDYNA_inp.m
Cable geometry calculations for copper cable analysis
"""

import numpy as np
import json
import os

# ============================================================================
# LOAD CABLE PARAMETERS FROM JSON FILE
# ============================================================================
def load_cable_parameters():
    """Load cable parameters from cable_parameters_user.json"""
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cable_parameters_user.json')
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cable parameters file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get the active cable name — env var takes priority (set by parallel subprocess dispatch)
    active_cable = os.environ.get('ACTIVE_CABLE') or data.get('active_cable')
    if not active_cable:
        raise ValueError("No 'active_cable' specified in cable_parameters_user.json")
    
    # Get the parameters for the active cable
    cables = data.get('cables', {})
    if active_cable not in cables:
        raise ValueError(f"Cable '{active_cable}' not found in cables dictionary. Available: {list(cables.keys())}")
    
    params = cables[active_cable]
    print(f"Loaded cable configuration: {active_cable}")
    
    return params

# Load parameters from JSON
user_params = load_cable_parameters()

# ============================================================================
# CABLE PARAMETERS (loaded from JSON)
# ============================================================================

# Parameters from JSON file
cable_name = user_params['cable_name']
cable_width = user_params['cable_width']  # [mm]
cable_height = user_params['cable_height']  # [mm]
T_pitch = user_params['T_pitch']  # [mm] transposition pitch
N_Strands = user_params['N_Strands']  # Number of strands
D_Strand_base = user_params['D_Strand']  # Base strand diameter [mm]

# Fixed parameters (not in JSON)
offset_from_cable = 1  # [mm] left/right (x) plate offset
offset_from_cable_y = offset_from_cable / 8  # [mm] top/bottom (y) plate offset — 1/8 of left/right
cable_length = 0.5 * T_pitch  # [mm]

# Calculated parameters
angle_factor = cable_width / (cable_height + cable_width)
angle = np.arctan(T_pitch * angle_factor / cable_width)
geometry_correction = 1 / np.sin(angle)

D_Strand = D_Strand_base * 1.02 * geometry_correction  # Strand diameter [mm], geometry correction for thermal expansion
r_strand = D_Strand / 2
d_strand = 0.1  # Initial distance between strands [mm]
N_offsets = 10  # Number of offsets

# Rutherford perimeter geometry  (matches generate_rutherford.FCMacro)
# W = outer cable width derived from N strands at minimum touching pitch
import math as _math
_pitch_min = _math.pi * D_Strand_base * (N_Strands - 2) / (2 * N_Strands)
_Weff      = N_Strands * _pitch_min / 2.0
W_rutherford = _Weff + D_Strand_base   # outer width of strand arrangement [mm]

# Thickness of plate
thickness = 0.1  # 1 mm thickness
half_thickness = thickness / 2.0

# Calculate initial perimeter and radius
# Perimeter_initial = N_Strands * (D_Strand + d_strand)
# Radius_initial = Perimeter_initial / (2 * np.pi)
# Radius_initial = cable_width/2 - offset_from_cable
Radius_initial = (cable_width - D_Strand) / 2

# Rutherford plate travel distances:
#   Plate_Top/Bottom start at y = D_Strand_base + offset_from_cable_y  (1/8 of left/right offset)
#   Plate_Left/Right start at x = W_rutherford/2 + offset_from_cable
#   They travel to the final compressed cable half-dimensions
distance_y = (D_Strand_base + offset_from_cable_y) - cable_height / 2
distance_x = (W_rutherford / 2 + offset_from_cable) - cable_width  / 2

extra_distance_y = 0
extra_distance_x = 0

distance_y = distance_y + extra_distance_y
distance_x = distance_x + extra_distance_x
axial_strain = 1
distance_z = axial_strain * T_pitch / (N_Strands * 100)

# time and velocities are set by main.py (--time arg) before export_parameters_to_json is called
time = None
velocity_y = None
velocity_x = None
velocity_z = None

def export_parameters_to_json(output_dir=None):
    """Export all calculated parameters to a JSON file for use in FreeCAD macro"""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Validate cable name
    if not cable_name or cable_name.strip() == "":
        raise ValueError("Cable name must be specified! Please set the 'cable_name' in cable_parameters_user.json.")
    
    # Create parameter dictionary with all calculated values
    parameters = {
        # Cable name for output files
        'cable_name': str(cable_name),
        
        # Basic cable parameters (from JSON)
        'cable_width': float(cable_width),
        'cable_height': float(cable_height),
        'T_pitch': float(T_pitch),
        'N_Strands': int(N_Strands),
        'D_Strand_base': float(D_Strand_base),
        'n_stacks': int(user_params.get('n_stacks', 10)),
        'stack_height_mm': float(user_params.get('stack_height_mm', cable_height + 0.3)),
        
        # Fixed parameters
        'offset_from_cable': float(offset_from_cable),
        'offset_from_cable_y': float(offset_from_cable_y),
        'cable_length': float(cable_length),
        
        # Calculated geometry parameters
        'angle_factor': float(angle_factor),
        'angle': float(angle),
        'geometry_correction': float(geometry_correction),
        
        # Calculated strand parameters
        'D_Strand': float(D_Strand),
        'r_strand': float(r_strand),
        'd_strand': float(d_strand),
        'N_offsets': int(N_offsets),
        
        # Plate parameters
        'thickness': float(thickness),
        'half_thickness': float(half_thickness),
        
        # Calculated radii and distances
        'Radius_initial': float(Radius_initial),
        'W_rutherford': float(W_rutherford),
        'distance_y': float(distance_y),
        'distance_x': float(distance_x),
        'distance_z': float(distance_z),
        
        # Velocities
        'time': float(time),
        'velocity_y': float(velocity_y),
        'velocity_x': float(velocity_x),
        'velocity_z': float(velocity_z)
    }
    
    # Export to JSON file
    json_file = os.path.join(output_dir, 'cable_parameters.json')
    with open(json_file, 'w') as f:
        json.dump(parameters, f, indent=4)
    
    print(f"Parameters exported to: {json_file}")
    print(f"Cable name: {cable_name}")
    return json_file

# Print results and export parameters (if run as main script)
if __name__ == "__main__":
    # Apply default time when run standalone (main.py overrides this via --time)
    if time is None:
        time = 0.001
        velocity_y = distance_y / time
        velocity_x = distance_x / time
        velocity_z = distance_z / time
    print("=" * 70)
    print(f"Cable Geometry Parameters for: {cable_name}")
    print("=" * 70)
    print(f"Cable width: {cable_width} mm")
    print(f"Cable height: {cable_height} mm")
    print(f"Angle factor: {angle_factor}")
    print(f"Angle: {angle} radians ({np.degrees(angle):.2f} degrees)")
    print(f"Geometry correction: {geometry_correction}")
    print(f"Strand diameter: {D_Strand:.4f} mm")
    print(f"Initial radius: {Radius_initial:.4f} mm")
    print(f"Distance Y: {distance_y:.4f} mm")
    print(f"Distance X: {distance_x:.4f} mm")
    print(f"Distance Z: {distance_z:.6f} mm")
    print(f"Velocity Y: {velocity_y:.4f} mm/ms")
    print(f"Velocity X: {velocity_x:.4f} mm/ms")
    print(f"Velocity Z: {velocity_z:.6f} mm/ms")
    print()
    
    # Export parameters for FreeCAD macro
    export_parameters_to_json()
