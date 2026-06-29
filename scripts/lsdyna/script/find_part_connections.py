"""
Find connections between parts in LS-DYNA mesh based on shared faces
Built from ground up for the actual mesh structure
"""
import sys
import os
from collections import defaultdict


def read_mesh_structure(filename):
    print(f"Reading mesh file: {filename}")
    
    nodes = {}
    parts_info = {}
    elements_by_part = defaultdict(list)
    
    reading_nodes = False
    reading_elements = False
    current_part_id = None
    current_element_id = None
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line or line.startswith('$'):
            i += 1
            continue
        
        if line == '*PART':
            i += 1
            while i < len(lines) and lines[i].strip().startswith('$'):
                if 'TITLE' not in lines[i]:
                    i += 1
                    continue
                i += 1
                break
            
            if i < len(lines):
                part_name = lines[i].strip()
                i += 1
                
                while i < len(lines) and lines[i].strip().startswith('$'):
                    i += 1
                
                if i < len(lines):
                    values = lines[i].strip().split()
                    if len(values) >= 3:
                        pid = int(values[0])
                        secid = int(values[1])
                        mid = int(values[2])
                        parts_info[pid] = {
                            'name': part_name,
                            'secid': secid,
                            'mid': mid
                        }
            i += 1
            continue
        
        if line == '*NODE':
            reading_nodes = True
            reading_elements = False
            i += 1
            continue
        
        if line.startswith('*ELEMENT_SOLID'):
            reading_nodes = False
            reading_elements = True
            i += 1
            continue
        
        if line.startswith('*'):
            reading_nodes = False
            reading_elements = False
            i += 1
            continue
        
        if reading_nodes:
            try:
                values = line.split()
                if len(values) >= 4:
                    nid = int(values[0])
                    x = float(values[1])
                    y = float(values[2])
                    z = float(values[3])
                    nodes[nid] = {'x': x, 'y': y, 'z': z}
            except (ValueError, IndexError):
                pass
            i += 1
            continue
        
        if reading_elements:
            try:
                values = line.split()
                
                if len(values) == 2:
                    current_element_id = int(values[0])
                    current_part_id = int(values[1])
                    i += 1
                    continue
                
                if len(values) >= 4 and current_element_id is not None and current_part_id is not None:
                    node_ids = []
                    for v in values[:8]:
                        try:
                            nid = int(v)
                            if nid != 0 and nid not in node_ids:
                                node_ids.append(nid)
                        except ValueError:
                            break
                    
                    if len(node_ids) >= 4:
                        elements_by_part[current_part_id].append({
                            'eid': current_element_id,
                            'nodes': node_ids
                        })
                    
                    current_element_id = None
                    current_part_id = None
            except (ValueError, IndexError):
                pass
            
            i += 1
            continue
        
        i += 1
    
    print(f"  Loaded {len(nodes)} nodes")
    print(f"  Loaded {len(parts_info)} parts")
    print(f"  Loaded {sum(len(v) for v in elements_by_part.values())} elements")
    
    return parts_info, nodes, elements_by_part


def get_element_faces(nodes):
    if len(nodes) < 4:
        return []
    
    faces = [
        frozenset([nodes[0], nodes[1], nodes[2]]),
        frozenset([nodes[0], nodes[1], nodes[3]]),
        frozenset([nodes[1], nodes[2], nodes[3]]),
        frozenset([nodes[0], nodes[2], nodes[3]])
    ]
    return faces


def find_shared_faces_between_parts(elements_by_part):
    print("\nAnalyzing face sharing between parts...")
    
    face_to_parts = defaultdict(set)
    
    for part_id, elements in elements_by_part.items():
        for elem in elements:
            faces = get_element_faces(elem['nodes'])
            for face in faces:
                if len(face) == 3:
                    face_to_parts[face].add(part_id)
    
    shared_faces = {face: parts for face, parts in face_to_parts.items() if len(parts) > 1}
    
    print(f"  Total unique faces: {len(face_to_parts)}")
    print(f"  Shared faces (in multiple parts): {len(shared_faces)}")
    
    return shared_faces


def build_connectivity_matrix(parts_info, shared_faces):
    print("\nBuilding part connectivity matrix...")
    
    part_ids = sorted(parts_info.keys())
    connectivity = defaultdict(lambda: defaultdict(int))
    
    for face, parts_set in shared_faces.items():
        parts_list = sorted(parts_set)
        
        for i in range(len(parts_list)):
            for j in range(i + 1, len(parts_list)):
                pid_a = parts_list[i]
                pid_b = parts_list[j]
                connectivity[pid_a][pid_b] += 1
                connectivity[pid_b][pid_a] += 1
    
    return connectivity, part_ids


def print_connection_summary(parts_info, connectivity, shared_faces, output_file=None):
    output_lines = []
    
    def write_line(line=""):
        output_lines.append(line)
        print(line)
    
    write_line("\n" + "="*80)
    write_line("PART CONNECTION ANALYSIS - FACE-BASED")
    write_line("="*80)
    
    write_line(f"\nTotal shared faces: {len(shared_faces)}")
    write_line(f"Total parts: {len(parts_info)}")
    
    write_line("\n" + "-"*80)
    write_line("FACE-BASED CONNECTIONS (parts sharing faces)")
    write_line("-"*80)
    
    total_connections = 0
    for pid_a in sorted(connectivity.keys()):
        for pid_b in sorted(connectivity[pid_a].keys()):
            if pid_a < pid_b:
                count = connectivity[pid_a][pid_b]
                name_a = parts_info[pid_a]['name'] if pid_a in parts_info else f"Part {pid_a}"
                name_b = parts_info[pid_b]['name'] if pid_b in parts_info else f"Part {pid_b}"
                write_line(f"\n  Part {pid_a:3d} <-> Part {pid_b:3d} : {count:5d} shared faces")
                write_line(f"    [{name_a}]")
                write_line(f"    [{name_b}]")
                total_connections += 1
    
    write_line(f"\nTotal face-based connections: {total_connections}")
    write_line("="*80)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"\nConnection report saved to: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python find_part_connections.py <run_directory_or_mesh_file>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if os.path.isfile(input_path):
        mesh_file = input_path
    elif os.path.isdir(input_path):
        mesh_file = os.path.join(input_path, "LSDYNA", "mesh.k")
    else:
        print(f"Error: Invalid path: {input_path}")
        sys.exit(1)
    
    if not os.path.exists(mesh_file):
        print(f"Error: Mesh file not found: {mesh_file}")
        sys.exit(1)
    
    output_file = os.path.join(os.path.dirname(mesh_file), "part_connections_report.txt")
    
    print("="*80)
    print("PART CONNECTION FINDER (FACE-BASED)")
    print("="*80)
    print(f"Mesh file: {mesh_file}")
    print()
    
    parts_info, nodes, elements_by_part = read_mesh_structure(mesh_file)
    
    if not parts_info:
        print("\nError: No parts found in mesh file!")
        sys.exit(1)
    
    shared_faces = find_shared_faces_between_parts(elements_by_part)
    
    connectivity, part_ids = build_connectivity_matrix(parts_info, shared_faces)
    
    print_connection_summary(parts_info, connectivity, shared_faces, output_file)


if __name__ == "__main__":
    main()
