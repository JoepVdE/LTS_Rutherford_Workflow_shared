"""
LS-DYNA mesh.k file reader
Reads parts, nodes, and elements from mesh.k file
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os
from datetime import datetime

def read_segments_from_file(filename):
    """Read existing SET_SEGMENT definitions from input file"""
    segments_by_id = {}
    current_segment_id = None
    reading_segments = False
    
    with open(filename, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                continue
            
            # Check for *SET_SEGMENT keyword
            if line_stripped == '*SET_SEGMENT':
                reading_segments = True
                current_segment_id = None
                continue
            
            # Stop reading if we hit another keyword
            if line_stripped.startswith('*') and line_stripped != '*SET_SEGMENT':
                reading_segments = False
                current_segment_id = None
                continue
            
            if reading_segments:
                # Skip comment lines
                if line_stripped.startswith('$'):
                    continue
                
                try:
                    values = line_stripped.split()
                    
                    # If single value, it's the segment ID
                    if len(values) == 1:
                        current_segment_id = int(values[0])
                        if current_segment_id not in segments_by_id:
                            segments_by_id[current_segment_id] = []
                    
                    # If 4 values, it's a segment (4 nodes)
                    elif len(values) == 4 and current_segment_id is not None:
                        segment_nodes = [int(v) for v in values]
                        segments_by_id[current_segment_id].append(segment_nodes)
                    
                except (ValueError, IndexError):
                    continue
    
    return segments_by_id


def read_mesh_file(filename):
    """Read mesh.k file and extract parts, nodes globally, and elements per part"""
    parts = {}
    nodes = {}
    reading_nodes = False
    reading_elements = False
    last_element_pid = None  # Track which part the last element belongs to
    
    with open(filename, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            
            # Skip empty lines and comments
            if not line_stripped or line_stripped.startswith('$'):
                continue
            
            # Check for *PART keyword to capture part definitions
            if line_stripped == '*PART':
                # Next non-comment line after *PART has part info
                continue
            
            # Check for *NODE keyword
            if line_stripped == '*NODE':
                reading_nodes = True
                reading_elements = False
                continue
            
            # Check for *ELEMENT_SOLID keyword (including variations)
            if line_stripped.startswith('*ELEMENT_SOLID'):
                reading_nodes = False
                reading_elements = True
                continue
            
            # Stop reading if we hit another keyword
            if line_stripped.startswith('*'):
                reading_nodes = False
                reading_elements = False
                continue
            
            # Read nodes (global section)
            if reading_nodes:
                try:
                    # LS-DYNA nodes can be in fixed or free format
                    # Free format: node_id x y z
                    # Fixed format: 8 chars for node_id, then 16 chars each for x,y,z
                    # Handle concatenated values (e.g., "1.23e+00-4.56e+00") by splitting on e+/e-
                    
                    values = line_stripped.split()
                    if len(values) >= 4:
                        # Standard free format with proper spacing
                        node_id = int(values[0])
                        x = float(values[1])
                        y = float(values[2])
                        z = float(values[3])
                        nodes[node_id] = {'x': x, 'y': y, 'z': z}
                    elif len(values) == 3 and 'e' in line_stripped.lower():
                        # Likely concatenated scientific notation (missing spaces)
                        # Try to parse as fixed-width: 8 chars node_id, 16 chars each for coords
                        if len(line_stripped) >= 56:
                            node_id = int(line_stripped[0:8])
                            x = float(line_stripped[8:24])
                            y = float(line_stripped[24:40])
                            z = float(line_stripped[40:56])
                            nodes[node_id] = {'x': x, 'y': y, 'z': z}
                except (ValueError, IndexError):
                    continue
            
            # Read elements
            elif reading_elements:
                try:
                    values = line_stripped.split()
                    
                    # Standard TET4 format: EID PID N1 N2 N3 N4 (6 values on one line)
                    if len(values) == 6:
                        eid = int(values[0])
                        pid = int(values[1])
                        node_ids = [int(v) for v in values[2:6]]
                        
                        # Initialize part if not exists
                        if pid not in parts:
                            parts[pid] = {'part_id': pid, 'elements': []}
                        
                        # Add element with nodes
                        parts[pid]['elements'].append({'eid': eid, 'nodes': node_ids})
                    
                    # Legacy 2-line format: EID PID on first line, nodes on next line
                    elif len(values) == 2:
                        # This is element ID and part ID line
                        eid = int(values[0])
                        pid = int(values[1])
                        
                        # Initialize part if not exists
                        if pid not in parts:
                            parts[pid] = {'part_id': pid, 'elements': []}
                        
                        # Store element info (will add nodes in next line)
                        parts[pid]['elements'].append({'eid': eid, 'nodes': []})
                        last_element_pid = pid  # Remember which part this element belongs to
                    
                    elif len(values) >= 4:
                        # This is the nodes line for legacy 2-line format
                        # For tetrahedral (4), wedge (6), hexahedral (8), or 10-node format
                        # For 10-value lines (8 nodes + 2 extra), take only first 8 values
                        # For tetrahedral elements with duplicates (n1 n2 n3 n4 n4 n4 n4 n4), take first 4 unique
                        if len(values) == 10:
                            # 10-node or 8-node + extras format, take first 8
                            node_ids_raw = [int(v) for v in values[:8]]
                            # Check if it's a tetrahedral (first 4 unique, rest duplicates)
                            unique_nodes = []
                            for nid in node_ids_raw:
                                if nid not in unique_nodes:
                                    unique_nodes.append(nid)
                            # Use only unique nodes if it's a tet4 with padding
                            node_ids = unique_nodes
                        else:
                            # Standard 4, 6, or 8 node format
                            node_ids = [int(v) for v in values]
                        
                        # Add nodes to the last element that was created
                        if last_element_pid is not None and last_element_pid in parts:
                            if parts[last_element_pid]['elements']:
                                parts[last_element_pid]['elements'][-1]['nodes'] = node_ids
                except (ValueError, IndexError):
                    continue
    
    return parts, nodes


def order_face_nodes_ccw(face_nodes, nodes, normal_direction='up'):
    """Order face nodes counter-clockwise when viewed from specified direction
    Works for both triangular (3 nodes) and quadrilateral (4 nodes) faces"""
    # Remove duplicate nodes (for degenerate quads that are actually triangles)
    unique_nodes = []
    seen = set()
    for nid in face_nodes:
        if nid not in seen:
            unique_nodes.append(nid)
            seen.add(nid)
    
    if len(unique_nodes) < 3:
        return unique_nodes  # Degenerate face, return as-is
    
    # Get coordinates of the unique nodes
    coords = np.array([[nodes[nid]['x'], nodes[nid]['y'], nodes[nid]['z']] 
                       for nid in unique_nodes])
    
    # Find center of the face
    center = np.mean(coords, axis=0)
    
    # Create vectors from center to each node
    vectors = coords - center
    
    # For 'up' normal: use XY plane, measure angles counter-clockwise
    # For 'down' normal: use XY plane, measure angles clockwise (or reverse the order)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    
    # Sort by angle
    sorted_indices = np.argsort(angles)
    
    if normal_direction == 'down':
        # Reverse for downward normal
        sorted_indices = sorted_indices[::-1]
    
    ordered_nodes = [unique_nodes[i] for i in sorted_indices]
    return ordered_nodes


def extract_surface_segments(parts, nodes, boundaries, tolerance=1e-4):
    """Extract surface segments for min and max Z surfaces of each part"""
    surface_segments = {}
    
    for pid in sorted(parts.keys()):
        # Skip parts that don't have valid boundaries
        if pid not in boundaries:
            continue
            
        part = parts[pid]
        boundary = boundaries[pid]
        
        min_z_nodes = set(boundary['nodes_at_min'])
        max_z_nodes = set(boundary['nodes_at_max'])
        
        min_z_segments = []
        max_z_segments = []
        
        # Go through each element
        for elem in part['elements']:
            elem_nodes = elem['nodes']
            num_nodes = len(elem_nodes)
            
            # Define faces based on element type
            if num_nodes == 4:  # Tetrahedral element (4 triangular faces)
                faces = [
                    [elem_nodes[0], elem_nodes[1], elem_nodes[2]],  # base
                    [elem_nodes[0], elem_nodes[1], elem_nodes[3]],  # side 1
                    [elem_nodes[1], elem_nodes[2], elem_nodes[3]],  # side 2
                    [elem_nodes[2], elem_nodes[0], elem_nodes[3]]   # side 3
                ]
            elif num_nodes == 6:  # Wedge/Prism element (2 triangular + 3 quadrilateral faces)
                faces = [
                    [elem_nodes[0], elem_nodes[1], elem_nodes[2]],              # triangular base
                    [elem_nodes[3], elem_nodes[4], elem_nodes[5]],              # triangular top
                    [elem_nodes[0], elem_nodes[1], elem_nodes[4], elem_nodes[3]],  # quad side 1
                    [elem_nodes[1], elem_nodes[2], elem_nodes[5], elem_nodes[4]],  # quad side 2
                    [elem_nodes[2], elem_nodes[0], elem_nodes[3], elem_nodes[5]]   # quad side 3
                ]
            elif num_nodes == 8:  # Hexahedral element (6 quadrilateral faces)
                faces = [
                    [elem_nodes[0], elem_nodes[1], elem_nodes[2], elem_nodes[3]],  # bottom
                    [elem_nodes[4], elem_nodes[5], elem_nodes[6], elem_nodes[7]],  # top
                    [elem_nodes[0], elem_nodes[1], elem_nodes[5], elem_nodes[4]],  # front
                    [elem_nodes[2], elem_nodes[3], elem_nodes[7], elem_nodes[6]],  # back
                    [elem_nodes[0], elem_nodes[3], elem_nodes[7], elem_nodes[4]],  # left
                    [elem_nodes[1], elem_nodes[2], elem_nodes[6], elem_nodes[5]]   # right
                ]
            else:
                continue  # Unknown element type
            
            for face in faces:
                # Check if all nodes of this face are at min Z
                if all(nid in min_z_nodes for nid in face):
                    # Order counter-clockwise for upward normal
                    ordered_face = order_face_nodes_ccw(face, nodes, normal_direction='up')
                    min_z_segments.append(ordered_face)
                    break  # Only one face per element should match
                
                # Check if all nodes of this face are at max Z
                elif all(nid in max_z_nodes for nid in face):
                    # Order counter-clockwise for downward normal
                    ordered_face = order_face_nodes_ccw(face, nodes, normal_direction='down')
                    max_z_segments.append(ordered_face)
                    break
        
        surface_segments[pid] = {
            'min_z_segments': min_z_segments,
            'max_z_segments': max_z_segments
        }
        
        # Count triangular (3-node) and quadrilateral (4-node) segments
        tri_count_min = sum(1 for seg in min_z_segments if len(seg) == 3)
        quad_count_min = sum(1 for seg in min_z_segments if len(seg) == 4)
        tri_count_max = sum(1 for seg in max_z_segments if len(seg) == 3)
        quad_count_max = sum(1 for seg in max_z_segments if len(seg) == 4)
        
        print(f"Part {pid}: Min Z surface - {tri_count_min} triangular, {quad_count_min} quadrilateral segments")
        print(f"Part {pid}: Max Z surface - {tri_count_max} triangular, {quad_count_max} quadrilateral segments")
    
    return surface_segments


def get_segment_xy_bbox(segment_nodes, nodes):
    """Calculate XY bounding box for a segment (ignore Z coordinate)"""
    x_coords = [nodes[nid]['x'] for nid in segment_nodes if nid in nodes]
    y_coords = [nodes[nid]['y'] for nid in segment_nodes if nid in nodes]
    
    if not x_coords or not y_coords:
        return None
    
    return {
        'x_min': min(x_coords),
        'x_max': max(x_coords),
        'y_min': min(y_coords),
        'y_max': max(y_coords)
    }


def bboxes_overlap_xy(bbox1, bbox2, tolerance=1e-2):
    """Check if two XY bounding boxes overlap within tolerance"""
    if bbox1 is None or bbox2 is None:
        return False
    
    # Check X overlap
    x_overlap = (bbox1['x_max'] >= bbox2['x_min'] - tolerance and 
                 bbox1['x_min'] <= bbox2['x_max'] + tolerance)
    
    # Check Y overlap
    y_overlap = (bbox1['y_max'] >= bbox2['y_min'] - tolerance and 
                 bbox1['y_min'] <= bbox2['y_max'] + tolerance)
    
    return x_overlap and y_overlap


def find_part_connections(surface_segments, nodes, tolerance=1e-2):
    """Find which parts are connected (max_z of part A overlaps with min_z of part B in XY plane)
    Returns connectivity matrix where matrix[A][B] = 1 means Part A max_z connects to Part B min_z"""
    
    part_ids = sorted(surface_segments.keys())
    n_parts = len(part_ids)
    
    # Create connectivity matrix
    connectivity_matrix = np.zeros((n_parts, n_parts), dtype=int)
    
    print("\nChecking part connections (max_z to min_z overlap in XY plane)...")
    
    for i, pid_a in enumerate(part_ids):
        max_z_segments_a = surface_segments[pid_a]['max_z_segments']
        
        # Pre-calculate bounding boxes for Part A max_z segments
        bboxes_a = [get_segment_xy_bbox(seg, nodes) for seg in max_z_segments_a]
        
        for j, pid_b in enumerate(part_ids):
            if pid_a == pid_b:
                continue  # Don't check part against itself
            
            min_z_segments_b = surface_segments[pid_b]['min_z_segments']
            
            # Check if any segment from A's max_z overlaps with any segment from B's min_z
            connection_found = False
            for bbox_a in bboxes_a:
                if bbox_a is None:
                    continue
                    
                for seg_b in min_z_segments_b:
                    bbox_b = get_segment_xy_bbox(seg_b, nodes)
                    
                    if bboxes_overlap_xy(bbox_a, bbox_b, tolerance):
                        connection_found = True
                        break
                
                if connection_found:
                    break
            
            if connection_found:
                connectivity_matrix[i, j] = 1
    
    return connectivity_matrix, part_ids


def find_boundary_nodes_and_elements(parts, nodes, tolerance=1e-4):
    """Find nodes at max/min Z for each part and their associated elements"""
    part_boundaries = {}
    
    for pid in parts.keys():
        part = parts[pid]
        
        # Collect all unique nodes for this part
        part_nodes = set()
        for elem in part['elements']:
            part_nodes.update(elem['nodes'])
        
        # Find min and max Z for this part
        z_coords = [nodes[nid]['z'] for nid in part_nodes if nid in nodes]
        if not z_coords:
            continue
            
        min_z = min(z_coords)
        max_z = max(z_coords)
        
        # Find nodes at min and max Z (within tolerance)
        nodes_at_min = [nid for nid in part_nodes 
                       if nid in nodes and abs(nodes[nid]['z'] - min_z) < tolerance]
        nodes_at_max = [nid for nid in part_nodes 
                       if nid in nodes and abs(nodes[nid]['z'] - max_z) < tolerance]
        
        # Find elements containing these nodes
        elements_at_min = []
        elements_at_max = []
        
        for elem in part['elements']:
            elem_nodes = set(elem['nodes'])
            if any(nid in elem_nodes for nid in nodes_at_min):
                elements_at_min.append(elem['eid'])
            if any(nid in elem_nodes for nid in nodes_at_max):
                elements_at_max.append(elem['eid'])
        
        part_boundaries[pid] = {
            'min_z': min_z,
            'max_z': max_z,
            'nodes_at_min': nodes_at_min,
            'nodes_at_max': nodes_at_max,
            'elements_at_min': elements_at_min,
            'elements_at_max': elements_at_max
        }
    
    return part_boundaries


def plot_all_parts(parts, nodes, segments):
    """Plot all parts in one 3D window"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for different parts
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    print("Plotting all parts...")
    for i, pid in enumerate(sorted(parts.keys())):
        part = parts[pid]
        color = colors[i % len(colors)]
        
        # Collect unique nodes for this part
        unique_nodes = set()
        for elem in part['elements']:
            unique_nodes.update(elem['nodes'])
        
        # Plot nodes
        node_coords = []
        for node_id in unique_nodes:
            if node_id in nodes:
                node_coords.append([nodes[node_id]['x'], 
                                  nodes[node_id]['y'], 
                                  nodes[node_id]['z']])
        
        if node_coords:
            node_coords = np.array(node_coords)
            ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], 
                      c=color, marker='.', s=1, alpha=0.6, label=f'Part {pid}')
        
        # Print element and segment info
        if pid in segments:
            min_segs = len(segments[pid]['min_z_segments'])
            max_segs = len(segments[pid]['max_z_segments'])
            print(f"  Part {pid}: {len(part['elements'])} elements, {min_segs} segments on min Z, {max_segs} segments on max Z")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('All Parts')
    ax.legend()
    plt.tight_layout()
    plt.savefig('all_parts.svg')
    plt.close()
    

def plot_surface_segments(segments, nodes, parts, show_vectors=True):
    """Plot only the surface segments with proper node ordering"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for different parts
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    title_suffix = " with Normal Vectors" if show_vectors else ""
    print(f"Plotting surface segments{title_suffix.lower()}...")
    
    for i, pid in enumerate(sorted(parts.keys())):
        if pid not in segments:
            continue
        segment = segments[pid]
        color = colors[i % len(colors)]
        
        # Plot min Z surface segments
        for seg in segment['min_z_segments']:
            coords = np.array([[nodes[nid]['x'], nodes[nid]['y'], nodes[nid]['z']] 
                              for nid in seg if nid in nodes])
            if len(coords) >= 3:  # Need at least 3 points for a face
                # Close the polygon by adding the first point at the end
                coords_closed = np.vstack([coords, coords[0]])
                ax.plot3D(coords_closed[:, 0], coords_closed[:, 1], coords_closed[:, 2], 
                         color=color, linewidth=0.5, alpha=0.7)
                
                if show_vectors:
                    # Calculate surface normal using cross product (right-hand rule)
                    # Vector from node 0 to node 1
                    v1 = coords[1] - coords[0]
                    # Vector from node 0 to node 2
                    v2 = coords[2] - coords[0]
                    # Normal = v1 x v2
                    normal = np.cross(v1, v2)
                    norm_magnitude = np.linalg.norm(normal)
                    
                    # Only draw arrow if normal is valid (non-zero)
                    if norm_magnitude > 1e-10:
                        normal = normal / norm_magnitude  # Normalize
                        
                        # Center of the segment
                        center = np.mean(coords, axis=0)
                        
                        # Scale normal for visualization
                        normal_scaled = normal * 0.5
                        
                        ax.quiver(center[0], center[1], center[2],
                                 normal_scaled[0], normal_scaled[1], normal_scaled[2],
                                 color=color, arrow_length_ratio=0.3, linewidth=2, alpha=0.9)
        
        # Plot max Z surface segments
        for seg in segment['max_z_segments']:
            coords = np.array([[nodes[nid]['x'], nodes[nid]['y'], nodes[nid]['z']] 
                              for nid in seg if nid in nodes])
            if len(coords) >= 3:  # Need at least 3 points for a face
                # Close the polygon
                coords_closed = np.vstack([coords, coords[0]])
                ax.plot3D(coords_closed[:, 0], coords_closed[:, 1], coords_closed[:, 2], 
                         color=color, linewidth=0.5, alpha=0.7, linestyle='--')
                
                if show_vectors:
                    # Calculate surface normal using cross product
                    v1 = coords[1] - coords[0]
                    v2 = coords[2] - coords[0]
                    normal = np.cross(v1, v2)
                    norm_magnitude = np.linalg.norm(normal)
                    
                    # Only draw arrow if normal is valid (non-zero)
                    if norm_magnitude > 1e-10:
                        normal = normal / norm_magnitude
                        
                        # Center of the segment
                        center = np.mean(coords, axis=0)
                        
                        # Scale normal for visualization
                        normal_scaled = normal * 0.5
                        
                        ax.quiver(center[0], center[1], center[2],
                                 normal_scaled[0], normal_scaled[1], normal_scaled[2],
                                 color=color, arrow_length_ratio=0.3, linewidth=2, alpha=0.9)
        
        min_segs = len(segment['min_z_segments'])
        max_segs = len(segment['max_z_segments'])
        print(f"  Part {pid}: {min_segs} segments on min Z, {max_segs} segments on max Z")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Surface Segments{title_suffix} (CCW ordering)')
    ax.legend([f'Part {pid}' for pid in sorted(parts.keys())], 
             loc='upper right')
    plt.tight_layout()
    plt.savefig(f'surface_segments{"_with_vectors" if show_vectors else ""}.svg')
    plt.close()


def write_segments_to_files(segments, parts, output_folder):
    """Write segment node data to text files"""
    print(f"\nWriting segments to files in '{output_folder}'...")
    
    for pid in sorted(parts.keys()):
        # Skip parts that don't have segments
        if pid not in segments:
            continue
            
        segment = segments[pid]
        
        # Write min Z segments
        if segment['min_z_segments']:
            filename = os.path.join(output_folder, f'part_{pid}_min_z.txt')
            with open(filename, 'w') as f:
                for seg in segment['min_z_segments']:
                    # For triangular segments (3 nodes), repeat the last node to make it 4-node format
                    if len(seg) == 3:
                        output_seg = [seg[0], seg[1], seg[2], seg[2]]
                    else:
                        output_seg = seg
                    f.write(' '.join(map(str, output_seg)) + '\n')
            print(f"  Part {pid} min Z: {len(segment['min_z_segments'])} segments -> {filename}")
        
        # Write max Z segments
        if segment['max_z_segments']:
            filename = os.path.join(output_folder, f'part_{pid}_max_z.txt')
            with open(filename, 'w') as f:
                for seg in segment['max_z_segments']:
                    # For triangular segments (3 nodes), repeat the last node to make it 4-node format
                    if len(seg) == 3:
                        output_seg = [seg[0], seg[1], seg[2], seg[2]]
                    else:
                        output_seg = seg
                    f.write(' '.join(map(str, output_seg)) + '\n')
            print(f"  Part {pid} max Z: {len(segment['max_z_segments'])} segments -> {filename}")


if __name__ == '__main__':
    import sys
    
    # Get run directory from command line or use latest
    run_dir = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.isdir(arg):
            run_dir = arg
        else:
            # Assume it's a run_id
            workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            run_dir = os.path.join(workspace_root, "data", "runs", arg)
    
    if run_dir is None:
        # Find latest run directory
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        runs_dir = os.path.join(workspace_root, "data", "runs")
        if os.path.exists(runs_dir):
            run_folders = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
            if run_folders:
                latest_run = sorted(run_folders)[-1]
                run_dir = os.path.join(runs_dir, latest_run)
    
    if not run_dir or not os.path.exists(run_dir):
        print("Error: No run directory found")
        print("Usage: python lsdyna_mesh_reader.py [run_directory_or_run_id]")
        sys.exit(1)
    
    # Locate mesh.k file
    mesh_file = os.path.join(run_dir, "LSDYNA", "mesh.k")
    if not os.path.exists(mesh_file):
        print(f"Error: mesh.k not found at {mesh_file}")
        sys.exit(1)
    
    # Create output folder in the same LSDYNA directory
    output_folder = os.path.join(run_dir, "LSDYNA", "segments_analysis")
    os.makedirs(output_folder, exist_ok=True)
    print(f"Run directory: {run_dir}")
    print(f"Mesh file: {mesh_file}")
    print(f"Output folder: {output_folder}\n")
    
    # Read the mesh file
    print("Reading mesh file...")
    parts, nodes = read_mesh_file(mesh_file)
    
    # Print summary
    print(f"\nTotal nodes: {len(nodes)}")
    print(f"Total parts: {len(parts)}\n")
    
    # Find min and max z coordinates globally
    if nodes:
        z_coords = [node['z'] for node in nodes.values()]
        min_z = min(z_coords)
        max_z = max(z_coords)
        print(f"Global Z coordinate range:")
        print(f"  Minimum Z: {min_z:.6f}")
        print(f"  Maximum Z: {max_z:.6f}")
        print(f"  Z Range: {max_z - min_z:.6f}\n")
    
    # Find boundary nodes and elements for each part
    print("Finding boundary nodes and elements for each part...\n")
    boundaries = find_boundary_nodes_and_elements(parts, nodes)
    
    # Extract surface segments
    print("Extracting surface segments with proper node ordering...\n")
    segments = extract_surface_segments(parts, nodes, boundaries)
    
    # Write segments to files
    write_segments_to_files(segments, parts, output_folder)
    
    # Show segment info
    for pid in sorted(parts.keys()):
        if pid not in segments:
            continue
        segment = segments[pid]
        print(f"Part {pid}:")
        print(f"  Min Z surface: {len(segment['min_z_segments'])} segments")
        print(f"  Max Z surface: {len(segment['max_z_segments'])} segments")
    
    # Plot all parts in one window
    print("\nGenerating plot of all parts...")
    try:
        plot_all_parts(parts, nodes, segments)
    except Exception as e:
        print(f"⚠ Plotting failed: {e}")
    
    # Print totals after plotting
    total_min_z = sum(len(segments[pid]['min_z_segments']) for pid in segments.keys())
    total_max_z = sum(len(segments[pid]['max_z_segments']) for pid in segments.keys())
    print(f"\nTotal segments:")
    print(f"  Min Z side: {total_min_z} segments")
    print(f"  Max Z side: {total_max_z} segments")
    print(f"  Grand total: {total_min_z + total_max_z} segments")
    
    # Plot surface segments without vectors
    print("\nGenerating plot of surface segments (without vectors)...")
    try:
        plot_surface_segments(segments, nodes, parts, show_vectors=False)
    except Exception as e:
        print(f"⚠ Plotting failed: {e}")
    
    # Plot surface segments with vectors
    print("\nGenerating plot of surface segments (with normal vectors)...")
    try:
        plot_surface_segments(segments, nodes, parts, show_vectors=True)
    except Exception as e:
        print(f"⚠ Plotting failed: {e}")
    
    # Find part connections
    print("\n" + "="*60)
    connectivity_matrix, part_ids = find_part_connections(segments, nodes, tolerance=1e-2)
    
    # Print connectivity matrix
    print("\nConnectivity Matrix (max_z to min_z connections):")
    print("Rows = Part max_z surface, Columns = Part min_z surface")
    print("1 = connected, 0 = not connected\n")
    
    # Print header
    header = "Part  |" + "".join(f"{pid:4d}" for pid in part_ids)
    print(header)
    print("-" * len(header))
    
    # Print matrix rows
    for i, pid_row in enumerate(part_ids):
        row_str = f"{pid_row:4d}  |"
        for j in range(len(part_ids)):
            row_str += f"{connectivity_matrix[i, j]:4d}"
        print(row_str)
    
    # Print connection summary
    print("\nConnection Summary:")
    connections = []
    for i, pid_a in enumerate(part_ids):
        for j, pid_b in enumerate(part_ids):
            if connectivity_matrix[i, j] == 1:
                connections.append((pid_a, pid_b))
                print(f"  Part {pid_a} (max_z) -> Part {pid_b} (min_z)")
    
    print(f"\nTotal connections found: {len(connections)}")
    print("="*60)
