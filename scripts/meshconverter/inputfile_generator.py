import os

class MeshParser:
    def __init__(self, mesh_file):
        self.mesh_file = mesh_file
        self.nodes = {}
        self.elements = {}
        
    def parse(self):
        """Parse the mesh.k file to extract nodes and elements."""
        import re
        with open(self.mesh_file, 'r') as f:
            current_section = None
            element_header = None
            
            for line in f:
                # Detect section headers
                if line[0] == '*':
                    if line.startswith('*NODE'):
                        current_section = 'nodes'
                    elif line.startswith('*ELEMENT_SOLID'):
                        current_section = 'elements'
                    else:
                        current_section = None
                    element_header = None
                    continue
                
                # Parse nodes
                if current_section == 'nodes':
                    numbers = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', line)
                    if len(numbers) >= 4:
                        self.nodes[int(numbers[0])] = (float(numbers[1]), float(numbers[2]), float(numbers[3]))
                
                # Parse elements (two-line format)
                elif current_section == 'elements':
                    if element_header is None:
                        # First line: element_id and part_id
                        parts = line.split()
                        if len(parts) >= 2:
                            element_header = (int(parts[0]), int(parts[1]))
                    else:
                        # Second line: 8 node IDs
                        parts = line.split()
                        if parts:
                            raw_nodes = [int(parts[i]) for i in range(min(8, len(parts)))]
                            seen = set()
                            unique_nodes = []
                            for n in raw_nodes:
                                if n > 0 and n not in seen:
                                    seen.add(n)
                                    unique_nodes.append(n)
                            self.elements[element_header[0]] = (element_header[1], unique_nodes)
                            element_header = None
        
        return self.nodes, self.elements
    
    def get_node(self, node_id):
        """Get coordinates of a specific node."""
        return self.nodes.get(node_id)
    
    def get_element(self, elem_id):
        """Get element data (part and node IDs)."""
        elem = self.elements.get(elem_id)
        return {'part': elem[0], 'nodes': elem[1]} if elem else None


class MeshProcessor:
    def __init__(self, nodes, elements):
        """Initialize processor with parsed mesh data.
        
        Args:
            nodes: Dictionary of {node_id: (x, y, z)}
            elements: Dictionary of {element_id: (part_id, [node_ids])}
        """
        self.nodes = nodes
        self.elements = elements
        self.parts = None
        self.part_nodes = None
        self.part_info = None
        self.face_nodes = None
        self.connecting_faces = None
        self.all_segments = None
        self.segment_connections = None
        self.filtered_connections = None
        
    def analyze(self):
        """Run complete analysis: find parts, connections, and segments."""
        self._identify_parts()
        self._analyze_part_geometry()
        self._find_connections()
        self._create_segments()
        
    def _identify_parts(self):
        """Identify unique parts in the mesh."""
        self.parts = set(elem[0] for elem in self.elements.values())
        
        # Get nodes for each part
        self.part_nodes = {part_id: set() for part_id in self.parts}
        for elem_id, (part_id, node_list) in self.elements.items():
            for node_id in node_list:
                if node_id > 0:
                    self.part_nodes[part_id].add(node_id)
    
    def _analyze_part_geometry(self):
        """Analyze geometry of each part (min/max Z and bounds)."""
        # Calculate z tolerance
        coords = list(self.nodes.values())
        z_vals = [c[2] for c in coords]
        z_range = max(z_vals) - min(z_vals)
        self.z_tol = z_range / 10000
        
        # Calculate min/max z and bounds for each part
        self.part_info = {}
        for part_id in sorted(self.parts):
            part_coords = [self.nodes[nid] for nid in self.part_nodes[part_id] if nid in self.nodes]
            if part_coords:
                z_vals = [c[2] for c in part_coords]
                min_z = min(z_vals)
                max_z = max(z_vals)
                
                # Get nodes at min and max z
                nodes_at_min_z = [nid for nid in self.part_nodes[part_id] 
                                 if nid in self.nodes and abs(self.nodes[nid][2] - min_z) < self.z_tol]
                nodes_at_max_z = [nid for nid in self.part_nodes[part_id] 
                                 if nid in self.nodes and abs(self.nodes[nid][2] - max_z) < self.z_tol]
                
                # Get xy bounds at min and max z
                if nodes_at_min_z:
                    min_z_coords = [self.nodes[nid] for nid in nodes_at_min_z]
                    min_z_x = [c[0] for c in min_z_coords]
                    min_z_y = [c[1] for c in min_z_coords]
                    min_z_bounds = (min(min_z_x), max(min_z_x), min(min_z_y), max(min_z_y))
                else:
                    min_z_bounds = None
                    
                if nodes_at_max_z:
                    max_z_coords = [self.nodes[nid] for nid in nodes_at_max_z]
                    max_z_x = [c[0] for c in max_z_coords]
                    max_z_y = [c[1] for c in max_z_coords]
                    max_z_bounds = (min(max_z_x), max(max_z_x), min(max_z_y), max(max_z_y))
                else:
                    max_z_bounds = None
                
                self.part_info[part_id] = {
                    'min_z': min_z,
                    'max_z': max_z,
                    'min_z_bounds': min_z_bounds,
                    'max_z_bounds': max_z_bounds
                }
        
        # Store face nodes
        self.face_nodes = {}
        for part_id in sorted(self.parts):
            info = self.part_info[part_id]
            min_z = info['min_z']
            max_z = info['max_z']
            
            nodes_at_bottom = [nid for nid in self.part_nodes[part_id] 
                              if nid in self.nodes and abs(self.nodes[nid][2] - min_z) < self.z_tol]
            nodes_at_top = [nid for nid in self.part_nodes[part_id] 
                           if nid in self.nodes and abs(self.nodes[nid][2] - max_z) < self.z_tol]
            
            self.face_nodes[part_id] = {
                'bottom': sorted(nodes_at_bottom),
                'top': sorted(nodes_at_top)
            }
    
    def _find_connections(self):
        """Find which parts connect to each other."""
        import numpy as np
        
        part_list = sorted(self.parts)
        n_parts = len(part_list)
        xy_tol = 0.01
        
        connections_found = []
        
        for i, part_a in enumerate(part_list):
            for j, part_b in enumerate(part_list):
                if i >= j:
                    continue
                
                info_a = self.part_info[part_a]
                info_b = self.part_info[part_b]
                
                match_found = False
                
                # Check if top of part_a matches bottom of part_b
                bounds_a_top = info_a['max_z_bounds']
                bounds_b_bot = info_b['min_z_bounds']
                
                if bounds_a_top and bounds_b_bot:
                    x_match = (abs(bounds_a_top[0] - bounds_b_bot[0]) < xy_tol and 
                              abs(bounds_a_top[1] - bounds_b_bot[1]) < xy_tol)
                    y_match = (abs(bounds_a_top[2] - bounds_b_bot[2]) < xy_tol and 
                              abs(bounds_a_top[3] - bounds_b_bot[3]) < xy_tol)
                    
                    if x_match and y_match:
                        match_found = True
                
                # Check if bottom of part_a matches top of part_b
                if not match_found:
                    bounds_a_bot = info_a['min_z_bounds']
                    bounds_b_top = info_b['max_z_bounds']
                    
                    if bounds_a_bot and bounds_b_top:
                        x_match = (abs(bounds_a_bot[0] - bounds_b_top[0]) < xy_tol and 
                                  abs(bounds_a_bot[1] - bounds_b_top[1]) < xy_tol)
                        y_match = (abs(bounds_a_bot[2] - bounds_b_top[2]) < xy_tol and 
                                  abs(bounds_a_bot[3] - bounds_b_top[3]) < xy_tol)
                        
                        if x_match and y_match:
                            match_found = True
                
                if match_found:
                    connections_found.append((part_a, part_b))
        
        # Filter connections to keep only one direction
        self.filtered_connections = {}
        for part_a, part_b in connections_found:
            info_a = self.part_info[part_a]
            info_b = self.part_info[part_b]
            
            bounds_a_top = info_a['max_z_bounds']
            bounds_b_bot = info_b['min_z_bounds']
            
            if bounds_a_top and bounds_b_bot:
                x_match = (abs(bounds_a_top[0] - bounds_b_bot[0]) < xy_tol and 
                          abs(bounds_a_top[1] - bounds_b_bot[1]) < xy_tol)
                y_match = (abs(bounds_a_top[2] - bounds_b_bot[2]) < xy_tol and 
                          abs(bounds_a_top[3] - bounds_b_bot[3]) < xy_tol)
                
                if x_match and y_match:
                    self.filtered_connections[part_a] = part_b
                else:
                    self.filtered_connections[part_b] = part_a
            else:
                self.filtered_connections[part_b] = part_a
    
    def _order_face_nodes_ccw(self, node_ids, view_from_positive_z=True):
        """Order nodes counter-clockwise when viewed from specified direction."""
        import math
        
        if len(node_ids) < 3:
            return node_ids
        
        coords = [self.nodes[nid] for nid in node_ids]
        cx = sum(c[0] for c in coords) / len(coords)
        cy = sum(c[1] for c in coords) / len(coords)
        
        angles = []
        for i, (x, y, z) in enumerate(coords):
            angle = math.atan2(y - cy, x - cx)
            angles.append((angle, node_ids[i]))
        
        angles.sort()
        ordered = [nid for angle, nid in angles]
        
        if not view_from_positive_z:
            ordered = ordered[::-1]
        
        return ordered
    
    def _extract_face_elements(self, part_id, face_type, z_value):
        """Extract face elements from solid elements at a specific Z plane.
        
        For hex elements (8 nodes): faces are quads with 4 nodes
        For tet elements (4 nodes): faces are triangles with 3 nodes
        
        LS-DYNA hex node numbering (1-indexed):
          Bottom face: 1-2-3-4
          Top face: 5-6-7-8
          
        LS-DYNA tet node numbering (1-indexed):
          Base face: 1-2-3
          Apex: 4
        """
        face_elements = []
        
        part_elements = [(eid, edata) for eid, edata in self.elements.items() 
                        if edata[0] == part_id]
        
        for elem_id, (pid, node_list) in part_elements:
            n_nodes = len([n for n in node_list if n > 0])
            
            if n_nodes == 8:
                # Hex element - check both faces regardless of which slot they occupy,
                # because mesh generators don't guarantee that the interface face is
                # always in the "bottom" (0-3) or "top" (4-7) position.
                bottom_nodes = [node_list[i] for i in [0, 1, 2, 3] if node_list[i] > 0]
                top_nodes = [node_list[i] for i in [4, 5, 6, 7] if node_list[i] > 0]

                face_at_z = None

                # Check if the "bottom" face (nodes 0-3) is at z_value
                if len(bottom_nodes) >= 3:
                    bottom_z_vals = [self.nodes[nid][2] for nid in bottom_nodes if nid in self.nodes]
                    if bottom_z_vals and all(abs(z - z_value) < self.z_tol for z in bottom_z_vals):
                        face_at_z = bottom_nodes

                # If not, check if the "top" face (nodes 4-7) is at z_value
                if face_at_z is None and len(top_nodes) >= 3:
                    top_z_vals = [self.nodes[nid][2] for nid in top_nodes if nid in self.nodes]
                    if top_z_vals and all(abs(z - z_value) < self.z_tol for z in top_z_vals):
                        face_at_z = top_nodes

                if face_at_z is not None:
                    # Use face_type only to set outward-normal direction for CCW ordering
                    view_from_pos_z = (face_type == 'top')
                    ordered = self._order_face_nodes_ccw(face_at_z, view_from_positive_z=view_from_pos_z)
                    if len(ordered) == 3:
                        ordered.append(ordered[-1])
                    face_elements.append(ordered[:4])
            
            elif n_nodes == 4:
                # Tet element - check all 4 faces
                # Face definitions for tet (0-indexed): 
                # Face 0 (base): 0-2-1, Face 1: 0-1-3, Face 2: 1-2-3, Face 3: 2-0-3
                tet_faces = [
                    [node_list[0], node_list[2], node_list[1]],  # base
                    [node_list[0], node_list[1], node_list[3]],
                    [node_list[1], node_list[2], node_list[3]],
                    [node_list[2], node_list[0], node_list[3]],
                ]
                
                for face_nodes in tet_faces:
                    valid_nodes = [n for n in face_nodes if n > 0 and n in self.nodes]
                    if len(valid_nodes) >= 3:
                        z_vals = [self.nodes[nid][2] for nid in valid_nodes]
                        if all(abs(z - z_value) < self.z_tol for z in z_vals):
                            # This face is on the Z plane
                            view_from_pos_z = (face_type == 'top')
                            ordered = self._order_face_nodes_ccw(valid_nodes, view_from_positive_z=view_from_pos_z)
                            # For triangles, duplicate last node for LS-DYNA format
                            ordered.append(ordered[-1])
                            face_elements.append(ordered[:4])
                            break  # Only one face per tet can be on the plane
            
            else:
                # Wedge or other element type - use original logic as fallback
                face_nodes_list = []
                for nid in node_list:
                    if nid > 0 and nid in self.nodes:
                        if abs(self.nodes[nid][2] - z_value) < self.z_tol:
                            face_nodes_list.append(nid)
                
                if len(face_nodes_list) >= 3:
                    view_from_positive_z = (face_type == 'top')
                    ordered_nodes = self._order_face_nodes_ccw(face_nodes_list, view_from_positive_z)
                    
                    if len(ordered_nodes) == 3:
                        ordered_nodes.append(ordered_nodes[-1])
                    elif len(ordered_nodes) > 4:
                        ordered_nodes = ordered_nodes[:4]
                    
                    face_elements.append(ordered_nodes)
        
        return face_elements
    
    def _create_segments(self):
        """Create segments for connecting faces."""
        self.connecting_faces = {}
        self.all_segments = {}
        segment_counter = 1
        self.segment_connections = {}
        
        for bottom_part, top_part in sorted(self.filtered_connections.items()):
            bottom_part_top_face_elems = self._extract_face_elements(
                bottom_part, 'top', self.part_info[bottom_part]['max_z'])
            top_part_bottom_face_elems = self._extract_face_elements(
                top_part, 'bottom', self.part_info[top_part]['min_z'])
            
            bottom_segment_id = segment_counter
            segment_counter += 1
            
            top_segment_id = segment_counter
            segment_counter += 1
            
            self.all_segments[bottom_segment_id] = {
                'segment_id': bottom_segment_id,
                'part': bottom_part,
                'face': 'top',
                'face_elements': bottom_part_top_face_elems,
                'num_elements': len(bottom_part_top_face_elems)
            }
            
            self.all_segments[top_segment_id] = {
                'segment_id': top_segment_id,
                'part': top_part,
                'face': 'bottom',
                'face_elements': top_part_bottom_face_elems,
                'num_elements': len(top_part_bottom_face_elems)
            }
            
            self.segment_connections[bottom_segment_id] = top_segment_id
            self.segment_connections[top_segment_id] = bottom_segment_id
            
            self.connecting_faces[(bottom_part, 'top')] = {
                'part': bottom_part,
                'face': 'top',
                'segment_id': bottom_segment_id,
                'nodes': self.face_nodes[bottom_part]['top'],
                'face_elements': bottom_part_top_face_elems,
                'connects_to': (top_part, 'bottom'),
                'connects_to_segment': top_segment_id
            }
            self.connecting_faces[(top_part, 'bottom')] = {
                'part': top_part,
                'face': 'bottom',
                'segment_id': top_segment_id,
                'nodes': self.face_nodes[top_part]['bottom'],
                'face_elements': top_part_bottom_face_elems,
                'connects_to': (bottom_part, 'top'),
                'connects_to_segment': bottom_segment_id
            }


class InputFileWriter:
    def __init__(self, template_dir, output_file="processed_input.k", end_time=0.001,
                 coord2_id=163, coord2_xO=-1.709686, coord2_yO=5.5426653, coord2_zO=7.9999995,
                 coord2_xL=9.17182, coord2_yL=5.5426653, coord2_zL=7.9999995,
                 coord2_xP=-1.709686, coord2_yP=16.424171, coord2_zP=7.9999995,
                 plate_velocity_y=5940, plate_velocity_x=200,
                 wire_sigy=20, wire_etan=5000):
        """Initialize input file writer.
        
        Args:
            template_dir: Directory containing template files (1_solversettings.k, etc.)
            output_file: Path to output file (default: processed_input.k)
            end_time: Simulation end time (default: 0.001)
            coord2_id: ID for second coordinate system (default: 163)
            coord2_xO, coord2_yO, coord2_zO: Origin point coordinates (defaults: -1.709686, 5.5426653, 7.9999995)
            coord2_xL, coord2_yL, coord2_zL: X-axis direction point (defaults: 9.17182, 5.5426653, 7.9999995)
            coord2_xP, coord2_yP, coord2_zP: X-Y plane point (defaults: -1.709686, 16.424171, 7.9999995)
            plate_velocity_y: Velocity magnitude for Y-direction plates (default: 5940)
            plate_velocity_x: Velocity magnitude for X-direction plates (default: 200)
            wire_sigy: Wire yield stress in MPa for MAT_PLASTIC_KINEMATIC (default: 20)
            wire_etan: Wire tangent modulus in MPa for MAT_PLASTIC_KINEMATIC (default: 5000)
        """
        self.template_dir = template_dir
        self.output_file = output_file
        self.end_time = end_time
        self.mesh_file = None
        
        # Second coordinate system parameters
        self.coord2_id = coord2_id
        self.coord2_xO = coord2_xO
        self.coord2_yO = coord2_yO
        self.coord2_zO = coord2_zO
        self.coord2_xL = coord2_xL
        self.coord2_yL = coord2_yL
        self.coord2_zL = coord2_zL
        self.coord2_xP = coord2_xP
        self.coord2_yP = coord2_yP
        self.coord2_zP = coord2_zP
        
        # Plate velocity parameters (note: swapped to match expected directions in boundary conditions)
        self.plate_velocity_y = plate_velocity_y
        self.plate_velocity_x = plate_velocity_x

        # Wire material parameters
        self.wire_sigy = wire_sigy
        self.wire_etan = wire_etan
    def write(self, parts, mesh_file=None, segment_connections=None, all_segments=None,
              nodes=None, part_nodes=None, part_info=None, face_nodes=None, z_tol=None):
        """Assemble and write the LS-DYNA input file.
        
        Args:
            parts: Set or list of part IDs from MeshProcessor
            mesh_file: Path to mesh file for nodes/elements
            segment_connections: Dict mapping segment_id -> connected_segment_id
            all_segments: Dict with segment info including 'face_elements'
            nodes: Dict of {node_id: (x, y, z)} for determining plate positions
            part_nodes: Dict of {part_id: set of node_ids} for determining plate positions
            part_info: Dict with part min/max z and bounds (from MeshProcessor)
            face_nodes: Dict with 'top' and 'bottom' node lists per part (from MeshProcessor)
            z_tol: tolerance for z comparisons (from MeshProcessor)
        """
        # Debug: print presence/size of key inputs to diagnose skipping of boundary segments
        try:
            nodes_present = bool(nodes)
            nodes_len = len(nodes) if nodes is not None else 0
        except Exception:
            nodes_present = False
            nodes_len = 0
        try:
            all_segments_present = all_segments is not None
            all_segments_len = len(all_segments) if all_segments is not None else 0
        except Exception:
            all_segments_present = False
            all_segments_len = 0
        part_info_present = part_info is not None
        face_nodes_present = face_nodes is not None
        z_tol_present = z_tol is not None
        print(f"  Debug: nodes_present={nodes_present}, nodes_len={nodes_len}, all_segments_present={all_segments_present}, all_segments_len={all_segments_len}, part_info_present={part_info_present}, face_nodes_present={face_nodes_present}, z_tol_present={z_tol_present}")
        import shutil
        
        # Start by copying 1_solversettings.k
        solver_settings = os.path.join(self.template_dir, '1_solversettings.k')
        
        if not os.path.exists(solver_settings):
            raise FileNotFoundError(f"Template file not found: {solver_settings}")
        
        # Read solver settings and modify end time
        with open(solver_settings, 'r') as f:
            lines = f.readlines()
        
        # Find and modify the CONTROL_TERMINATION section
        modified = False
        for i, line in enumerate(lines):
            if '*CONTROL_TERMINATION' in line:
                # The endtim value is on the line after the comment line (2 lines after the keyword)
                if i + 2 < len(lines):
                    data_line = lines[i + 2]
                    # Parse the line and replace the first field (endtim)
                    # LS-DYNA uses 10-character fields
                    # Format: right-aligned, 6 significant figures
                    new_endtim = f"{self.end_time:>10.6g}"
                    # Replace first 10 characters with new value
                    lines[i + 2] = new_endtim + data_line[10:]
                    modified = True
                    break
        
        # Write modified content to output file
        with open(self.output_file, 'w') as f:
            f.writelines(lines)
        
        print(f"Created {self.output_file}")
        print(f"  Copied: 1_solversettings.k")
        if modified:
            print(f"  Modified end time: {self.end_time}")
        
        # Determine plate movement directions before writing materials so con1 is set correctly per plate.
        # Y-moving plates (top/bottom): con1=6 (lock X+Z translation, free Y)
        # X-moving plates (right/left): con1=5 (lock Y+Z translation, free X)
        self._y_plate_parts: set = set()
        self._x_plate_parts: set = set()
        if nodes and part_nodes:
            _sorted = sorted(parts)
            if len(_sorted) >= 4:
                _plate_ids = _sorted[-4:]
                _centroids = {}
                for _pid in _plate_ids:
                    if _pid in part_nodes:
                        _coords = [nodes[nid] for nid in part_nodes[_pid] if nid in nodes]
                        if _coords:
                            _centroids[_pid] = (
                                sum(c[0] for c in _coords) / len(_coords),
                                sum(c[1] for c in _coords) / len(_coords),
                            )
                if len(_centroids) == 4:
                    self._y_plate_parts = {
                        max(_centroids, key=lambda p: _centroids[p][1]),  # top
                        min(_centroids, key=lambda p: _centroids[p][1]),  # bottom
                    }
                    self._x_plate_parts = {
                        max(_centroids, key=lambda p: _centroids[p][0]),  # right
                        min(_centroids, key=lambda p: _centroids[p][0]),  # left
                    }

        # 2. Add material properties
        self._write_materials(parts)
        
        # 3. Add nodes section from mesh file
        if mesh_file:
            self._write_nodes(mesh_file)
        
        # 4. Add elements section from mesh file
        if mesh_file:
            self._write_elements(mesh_file)
        
        # 5. Add coordinate system definitions
        self._write_coordinate_system()
        
        # 6. Add contact definitions
        num_contacts = 0
        if segment_connections and all_segments:
            num_contacts = self._write_contacts(segment_connections, all_segments, parts)
        
        # 7. Add body interaction contacts
        self._write_body_interactions(parts, num_contacts)
        
        # 8. Add boundary conditions for plates
        if nodes and part_nodes:
            self._write_plate_boundary_conditions(parts, nodes, part_nodes)
        
        # 9. Append boundary node segments (min Z and max Z)
        if nodes and all_segments is not None and part_info is not None and face_nodes is not None and z_tol is not None:
            self._append_boundary_segments(nodes, all_segments, part_info, face_nodes, z_tol)
        else:
            print("  Skipping appending boundary segments: missing part_info/face_nodes/z_tol or nodes")

        # 10. Append final template (10_end.k) at the bottom if present
        end_file = os.path.join(self.template_dir, '10_end.k')
        if os.path.exists(end_file):
            with open(end_file, 'r') as ef, open(self.output_file, 'a') as outf:
                outf.writelines(ef.readlines())
            print(f"  Appended final template: 10_end.k")
        else:
            print(f"  Warning: final template 10_end.k not found in {self.template_dir}")
    
    def _write_materials(self, parts):
        """Write material properties for all parts."""
        # Read the material template file
        materials_template = os.path.join(self.template_dir, '2_part_materials.k')
        
        if not os.path.exists(materials_template):
            raise FileNotFoundError(f"Template file not found: {materials_template}")
        
        with open(materials_template, 'r') as f:
            template_lines = f.readlines()
        
        # Extract elastic and rigid material templates
        elastic_template = self._extract_template(template_lines, 'elastic')
        rigid_template = self._extract_template(template_lines, 'rigid')
        
        # Determine which parts are elastic vs rigid
        number_of_parts = len(parts)
        part_list = sorted(parts)
        
        # Append materials to output file
        with open(self.output_file, 'a') as f:
            for part_id in part_list:
                # Last 4 parts are rigid, all others are elastic
                if part_id <= number_of_parts - 4:
                    # Substitute wire material values before _replace_ids runs,
                    # because _replace_ids replaces 'n' everywhere (including inside 'etan')
                    prepped = elastic_template.replace('      sigy', f'{self.wire_sigy:>10g}')
                    prepped = prepped.replace('      etan', f'{self.wire_etan:>10g}')
                    prepped = prepped.replace('sigy_comment', str(self.wire_sigy))
                    prepped = prepped.replace('etan_comment', str(self.wire_etan))
                    material_section = self._replace_ids(prepped, 'n', part_id)
                else:
                    # Write rigid material (title placeholder plate_m replaced by _replace_ids)
                    material_section = self._replace_ids(rigid_template, 'm', part_id)
                    # Set con1 based on movement direction:
                    # Y-moving plates (top/bottom): con1=6 (lock X+Z, free Y)
                    # X-moving plates (right/left): con1=5 (lock Y+Z, free X)
                    if part_id in getattr(self, '_y_plate_parts', set()):
                        con1 = 6
                    else:
                        con1 = 5
                    # Replace the cmo/con1/con2 data line (appears after the $cmo con1 con2 comment)
                    lines_mat = material_section.split('\n')
                    for _i, _line in enumerate(lines_mat):
                        if '$      cmo      con1      con2' in _line and _i + 1 < len(lines_mat):
                            # Data line is next non-empty line
                            _j = _i + 1
                            while _j < len(lines_mat) and not lines_mat[_j].strip():
                                _j += 1
                            if _j < len(lines_mat):
                                _fields = lines_mat[_j]
                                lines_mat[_j] = f'         1{con1:>10d}         7                                                  '
                            break
                    material_section = '\n'.join(lines_mat)
                
                f.write(material_section)
        
        print(f"  Added materials for {number_of_parts} parts ({number_of_parts - 4} elastic, 4 rigid)")
    
    def _write_nodes(self, mesh_file):
        """Copy nodes section from mesh.k file verbatim."""
        if not os.path.exists(mesh_file):
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

        node_count = 0
        in_nodes = False

        with open(mesh_file, 'r') as mf, open(self.output_file, 'a') as f:
            for line in mf:
                if line.startswith('*NODE'):
                    in_nodes = True
                    f.write(line)
                    continue

                if in_nodes:
                    if line.startswith('*'):
                        break
                    f.write(line)
                    if line.strip() and not line.strip().startswith('$'):
                        node_count += 1

        print(f"  Added {node_count} nodes from mesh file")
    
    def _write_elements(self, mesh_file):
        """Copy elements section from mesh.k file verbatim."""
        if not os.path.exists(mesh_file):
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

        in_elements = False
        elem_count = 0

        with open(mesh_file, 'r') as mf, open(self.output_file, 'a') as f:
            for line in mf:
                if line.startswith('*ELEMENT_SOLID'):
                    in_elements = True
                    f.write('*ELEMENT_SOLID\n')
                    continue

                if in_elements:
                    if line.startswith('*'):
                        break
                    f.write(line)
                    if line.strip() and not line.strip().startswith('$'):
                        elem_count += 1

        # Each element spans 2 lines (header + nodes), so divide by 2
        print(f"  Added {elem_count // 2} elements from mesh file")
    
    def _write_coordinate_system(self):
        """Append coordinate system definitions from template file with customizable second coord system."""
        coord_system_file = os.path.join(self.template_dir, '5_coordinatesystem.k')
        
        if not os.path.exists(coord_system_file):
            print(f"  Warning: Coordinate system file not found: {coord_system_file}")
            return
        
        # Read coordinate system file
        with open(coord_system_file, 'r') as f:
            lines = f.readlines()
        
        # Process lines and replace second coordinate system values
        result_lines = []
        coord_system_count = 0
        in_second_coord = False
        
        for i, line in enumerate(lines):
            if line.strip().startswith('*DEFINE_COORDINATE_SYSTEM'):
                coord_system_count += 1
                if coord_system_count == 2:
                    in_second_coord = True
                    result_lines.append(line)
                    continue
            
            if in_second_coord and coord_system_count == 2:
                # Check if this is the data line with ID, xO, yO, etc.
                if not line.strip().startswith('$') and line.strip():
                    # Check if this is the first data line (has ID and 7 values)
                    parts = line.split()
                    if len(parts) >= 7:
                        # This is the first data line: ID xO yO zO xL yL zL
                        # Format to preserve exact precision as in template
                        new_line = f"{self.coord2_id:>10d}{self.coord2_xO:>10.8g}{self.coord2_yO:>10.8g}{self.coord2_zO:>10.8g}{self.coord2_xL:>10.8g}{self.coord2_yL:>10.8g}{self.coord2_zL:>10.8g}          \n"
                        result_lines.append(new_line)
                        continue
                    elif len(parts) >= 3:
                        # This is the second data line: xP yP zP
                        new_line = f"{self.coord2_xP:>10.8g}{self.coord2_yP:>10.8g}{self.coord2_zP:>10.8g}             \n"
                        result_lines.append(new_line)
                        in_second_coord = False
                        continue
            
            result_lines.append(line)
        
        # Append to output file
        with open(self.output_file, 'a') as f:
            f.writelines(result_lines)
        
        print(f"  Added coordinate system definitions")
    
    def _write_contacts(self, segment_connections, all_segments, parts):
        """Append contact definitions for all segment connection pairs.
        
        Args:
            segment_connections: Dict mapping segment_id -> connected_segment_id (bidirectional)
            all_segments: Dict with segment info including 'face_elements' and 'part'
            parts: Set or list of all part IDs (used to identify plates - last 4 parts)
        """
        contact_template_file = os.path.join(self.template_dir, '6_contacts.k')
        
        if not os.path.exists(contact_template_file):
            print(f"  Warning: Contact template file not found: {contact_template_file}")
            return
        
        # Read contact template
        with open(contact_template_file, 'r') as f:
            template_content = f.read()
        
        # Identify plate parts (last 4 in sorted list)
        sorted_parts = sorted(parts)
        plate_parts = set(sorted_parts[-4:]) if len(sorted_parts) >= 4 else set()
        print(f"  Plate parts (excluded from plate-to-plate contacts): {sorted(plate_parts)}")
        
        # Get unique pairs (avoid duplicates since segment_connections is bidirectional)
        processed_pairs = set()
        contact_count = 0
        skipped_count = 0
        
        with open(self.output_file, 'a') as f:
            for seg_xx in sorted(segment_connections.keys()):
                seg_yy = segment_connections[seg_xx]
                
                # Create canonical pair to avoid duplicates
                pair = tuple(sorted([seg_xx, seg_yy]))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)
                
                # Get segment info
                seg_xx_info = all_segments[seg_xx]
                seg_yy_info = all_segments[seg_yy]
                
                # Skip if BOTH parts are plates (plate-to-plate connection)
                part_xx = seg_xx_info['part']
                part_yy = seg_yy_info['part']
                if part_xx in plate_parts and part_yy in plate_parts:
                    skipped_count += 1
                    continue
                
                # Write SET_SEGMENT for segment xx
                f.write(f"*SET_SEGMENT\n")
                f.write(f"{seg_xx:>10d}\n")
                for face_elem in seg_xx_info['face_elements']:
                    f.write(f"{face_elem[0]:>10d}{face_elem[1]:>10d}{face_elem[2]:>10d}{face_elem[3]:>10d}\n")
                
                # Write SET_SEGMENT for segment yy
                f.write(f"*SET_SEGMENT\n")
                f.write(f"{seg_yy:>10d}\n")
                for face_elem in seg_yy_info['face_elements']:
                    f.write(f"{face_elem[0]:>10d}{face_elem[1]:>10d}{face_elem[2]:>10d}{face_elem[3]:>10d}\n")
                
                # Write CONTACT definition
                contact_id = 180 + contact_count  # Unique contact ID
                f.write(f"*CONTACT_TIED_SURFACE_TO_SURFACE_OFFSET_ID\n")
                f.write(f"$       ID                                                               heading\n")
                f.write(f"{contact_id:>10d}Bonded - Segment {seg_xx} to Segment {seg_yy}\n")
                f.write(f"$     ssid      msid     sstyp     mstyp    sboxid    mboxid       spr       mpr\n")
                f.write(f"{seg_xx:>10d}{seg_yy:>10d}         0         0         0         0         1         1\n")
                f.write(f"$       fs        fd        dc        vc       vdc    penchk        bt        dt\n")
                f.write(f"         0         0         0         0        10         0         0         0\n")
                f.write(f"$      sfs       sfm       sst       mst      sfst      sfmt       fsf       vsf\n")
                f.write(f"         0         0      -250      -250         0         0         0         0\n")
                f.write(f"$     soft   softscl    lcidab    maxpar     sbopt     depth     bsort    frcfrq\n")
                f.write(f"         0         0         0         0         3         5         0         0\n")
                f.write(f"$   penmax    tkhopt    shlthk     snlog      isym     i2d3d    sldthk    sldstf\n")
                f.write(f"         0         0         0         0         0         0         0         0\n")
                
                contact_count += 1
        
        print(f"  Added {contact_count} contact definitions ({contact_count * 2} segments)")
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} plate-to-plate connections")
        
        return contact_count  # Return count for use in body interactions
    
    def _write_body_interactions(self, parts, num_contacts):
        """Append body interaction contacts for self-contact between parts.
        
        Args:
            parts: Set or list of all part IDs
            num_contacts: Number of tied contacts written (to calculate next segment ID)
        """
        body_interaction_template = os.path.join(self.template_dir, '7_contacts.k')
        
        if not os.path.exists(body_interaction_template):
            print(f"  Warning: Body interaction template not found: {body_interaction_template}")
            return
        
        # Calculate nn: 1 higher than last segment (each contact has 2 segments)
        last_segment = num_contacts * 2
        nn1 = last_segment + 1
        nn2 = last_segment + 2
        
        # Identify plate parts (last 4 in sorted list)
        sorted_parts = sorted(parts)
        n_parts = len(sorted_parts)
        
        if n_parts < 4:
            print(f"  Warning: Not enough parts for body interactions (need at least 4)")
            return
        
        # Plates are the last 4 parts
        plate_parts = sorted_parts[-4:]  # e.g., [35, 36, 37, 38]
        
        # Contact 1: Parts 1 to (n-2), i.e., all except last 2 plates
        # Include all parts except plate_parts[-2] and plate_parts[-1]
        parts_contact1 = [p for p in sorted_parts if p not in plate_parts[-2:]]
        
        # Contact 2: Parts 1 to (n-4) + last 2 plates
        # Include all non-plate parts + plate_parts[-2] and plate_parts[-1]
        non_plate_parts = sorted_parts[:-4]
        parts_contact2 = non_plate_parts + plate_parts[-2:]
        
        with open(self.output_file, 'a') as f:
            # Write first body interaction contact
            f.write(f"*SET_PART_LIST\n")
            f.write(f"{nn1:>10d}\n")
            f.write(f"$Body interaction 1: all parts except last 2 plates ({plate_parts[-2]}, {plate_parts[-1]})\n")
            # Write parts in rows of 8
            for i in range(0, len(parts_contact1), 8):
                row = parts_contact1[i:i+8]
                f.write("".join(f"{p:>10d}" for p in row) + "\n")
            
            f.write(f"*CONTACT_AUTOMATIC_SINGLE_SURFACE_ID\n")
            f.write(f"$       ID                                                               heading\n")
            f.write(f"       140                                                      Body Interaction\n")
            f.write(f"$     ssid      msid     sstyp     mstyp    sboxid    mboxid       spr       mpr\n")
            f.write(f"{nn1:>10d}         0         2         2         0         0         0         0\n")
            f.write(f"$       fs        fd        dc        vc       vdc    penchk        bt        dt\n")
            f.write(f"         0         0         0         0        10         0         0         0\n")
            f.write(f"$      sfs       sfm       sst       mst      sfst      sfmt       fsf       vsf\n")
            f.write(f"         0         0         0         0         0         0         0         0\n")
            f.write(f"$     soft   softscl    lcidab    maxpar     sbopt     depth     bsort    frcfrq\n")
            f.write(f"         2         0         0         0         3         5         0         0\n")
            f.write(f"$   penmax    tkhopt    shlthk     snlog      isym     i2d3d    sldthk    sldstf\n")
            f.write(f"         0         0         0         0         0         0         0         0\n")
            
            # Write second body interaction contact
            f.write(f"*SET_PART_LIST\n")
            f.write(f"{nn2:>10d}\n")
            f.write(f"$Body interaction 2: all parts except first 2 plates ({plate_parts[0]}, {plate_parts[1]})\n")
            # Write parts in rows of 8
            for i in range(0, len(parts_contact2), 8):
                row = parts_contact2[i:i+8]
                f.write("".join(f"{p:>10d}" for p in row) + "\n")
                # print("".join(f"{p:>10d}" for p in row) + "\n")
            
            f.write(f"*CONTACT_AUTOMATIC_SINGLE_SURFACE_ID\n")
            f.write(f"$       ID                                                               heading\n")
            f.write(f"       150                                                    Body Interaction 2\n")
            f.write(f"$     ssid      msid     sstyp     mstyp    sboxid    mboxid       spr       mpr\n")
            f.write(f"{nn2:>10d}         0         2         2         0         0         0         0\n")
            f.write(f"$       fs        fd        dc        vc       vdc    penchk        bt        dt\n")
            f.write(f"         0         0         0         0        10         0         0         0\n")
            f.write(f"$      sfs       sfm       sst       mst      sfst      sfmt       fsf       vsf\n")
            f.write(f"         0         0         0         0         0         0         0         0\n")
            f.write(f"$     soft   softscl    lcidab    maxpar     sbopt     depth     bsort    frcfrq\n")
            f.write(f"         2         0         0         0         3         5         0         0\n")
            f.write(f"$   penmax    tkhopt    shlthk     snlog      isym     i2d3d    sldthk    sldstf\n")
            f.write(f"         0         0         0         0         0         0         0         0\n")
        
        print(f"  Added 2 body interaction contacts (SET_PART_LIST IDs: {nn1}, {nn2})")
        print(f"    Contact 1: {len(parts_contact1)} parts (excludes plates {plate_parts[-2]}, {plate_parts[-1]})")
        print(f"    Contact 2: {len(parts_contact2)} parts (excludes plates {plate_parts[0]}, {plate_parts[1]})")
    
    def _write_plate_boundary_conditions(self, parts, nodes, part_nodes):
        """Write boundary conditions for the 4 plates based on their positions.
        
        Determines plate positions from coordinates and assigns:
        - Top plate (max Y): moves -Y toward center
        - Bottom plate (min Y): moves +Y toward center  
        - Right plate (max X): moves -X toward center
        - Left plate (min X): moves +X toward center
        
        Args:
            parts: Set or list of all part IDs
            nodes: Dict of {node_id: (x, y, z)}
            part_nodes: Dict of {part_id: set of node_ids}
        """
        # Identify plate parts (last 4 in sorted list)
        sorted_parts = sorted(parts)
        if len(sorted_parts) < 4:
            print(f"  Warning: Not enough parts for plate boundary conditions")
            return
        
        plate_parts = sorted_parts[-4:]
        
        # Calculate centroid (average X, Y) for each plate
        plate_centroids = {}
        for plate_id in plate_parts:
            if plate_id not in part_nodes:
                continue
            plate_node_ids = part_nodes[plate_id]
            coords = [nodes[nid] for nid in plate_node_ids if nid in nodes]
            if coords:
                avg_x = sum(c[0] for c in coords) / len(coords)
                avg_y = sum(c[1] for c in coords) / len(coords)
                plate_centroids[plate_id] = (avg_x, avg_y)
        
        if len(plate_centroids) < 4:
            print(f"  Warning: Could not determine centroids for all plates")
            return
        
        # Find which plate is at each position
        # Top = max Y, Bottom = min Y, Right = max X, Left = min X
        top_plate = max(plate_centroids.keys(), key=lambda p: plate_centroids[p][1])
        bottom_plate = min(plate_centroids.keys(), key=lambda p: plate_centroids[p][1])
        right_plate = max(plate_centroids.keys(), key=lambda p: plate_centroids[p][0])
        left_plate = min(plate_centroids.keys(), key=lambda p: plate_centroids[p][0])
        
        # Define boundary conditions:
        # (part_id, dof, velocity, curve_id, position_name)
        # DOF: 1=X, 2=Y
        # Velocity sign: negative moves toward center for top/right, positive for bottom/left
        boundary_conditions = [
            (top_plate, 2, -self.plate_velocity_y, 1, "top"),      # Top plate: -Y
            (bottom_plate, 2, self.plate_velocity_y, 2, "bottom"), # Bottom plate: +Y
            (right_plate, 1, -self.plate_velocity_x, 3, "right"), # Right plate: -X
            (left_plate, 1, self.plate_velocity_x, 4, "left"),    # Left plate: +X
        ]
        
        with open(self.output_file, 'a') as f:
            for part_id, dof, velocity, curve_id, position in boundary_conditions:
                # Write DEFINE_CURVE
                f.write(f"*DEFINE_CURVE\n")
                f.write(f"$       ID      sidr       sfa       sfo      offa      offo    dattyp   unused1\n")
                f.write(f"{curve_id:>10d}         0         0         0         0         0         0          \n")
                f.write(f"$                 a1                  o1                                 unused1\n")
                f.write(f"{0:>20}{velocity:>20}                                        \n")
                f.write(f"$                 a1                  o1                                 unused1\n")
                f.write(f"{self.end_time:>20}{velocity:>20}                                        \n")
                
                # Write BOUNDARY_PRESCRIBED_MOTION_RIGID
                f.write(f"*BOUNDARY_PRESCRIBED_MOTION_RIGID\n")
                f.write(f"$      sid       dof       vad      lcid        sf       vid     death     birth\n")
                f.write(f"{part_id:>10d}{dof:>10d}         0{curve_id:>10d}         1         0         0         0\n")
        
        print(f"  Added 4 plate boundary conditions:")
        for part_id, dof, velocity, curve_id, position in boundary_conditions:
            dof_name = "Y" if dof == 2 else "X"
            print(f"    Part {part_id} ({position}): DOF={dof_name}, velocity={velocity}")
    
    def _append_boundary_segments(self, nodes, all_segments, part_info, face_nodes, z_tol):
        """Append two *SET_NODE_LIST blocks to the output file using previously computed face node sets.
        - First block (ID = nn): all node IDs on parts whose bottom face is at global min Z
        - Second block (ID = nn+1): all node IDs on parts whose top face is at global max Z
        Node lists are written as rows of 8 node IDs, right-aligned (10-char fields).
        nn is computed as 1 higher than the current maximum segment id in all_segments.
        """
        # Validate inputs
        if not part_info or not face_nodes or not nodes:
            print("  Warning: missing data to append boundary segments (need nodes, part_info, face_nodes)")
            return
        # Determine global min/max from part_info
        part_z_mins = [info['min_z'] for pid, info in part_info.items() if 'min_z' in info]
        part_z_maxs = [info['max_z'] for pid, info in part_info.items() if 'max_z' in info]
        if not part_z_mins or not part_z_maxs:
            print("  Warning: part_info missing min/max z values")
            return
        global_min_z = min(part_z_mins)
        global_max_z = max(part_z_maxs)
        
        # Collect nodes from face_nodes for parts on global min/max within z_tol
        min_nodes_set = set()
        max_nodes_set = set()
        for pid, info in part_info.items():
            # bottom face
            if 'min_z' in info and abs(info['min_z'] - global_min_z) < z_tol:
                nodes_list = face_nodes.get(pid, {}).get('bottom', [])
                min_nodes_set.update(nodes_list)
            # top face
            if 'max_z' in info and abs(info['max_z'] - global_max_z) < z_tol:
                nodes_list = face_nodes.get(pid, {}).get('top', [])
                max_nodes_set.update(nodes_list)
        
        # Filter nodes to those actually present in nodes dict and sort
        min_nodes = sorted(nid for nid in min_nodes_set if nid in nodes)
        max_nodes = sorted(nid for nid in max_nodes_set if nid in nodes)
        
        # Determine nn from existing segments
        if all_segments and len(all_segments) > 0:
            max_seg = max(all_segments.keys())
        else:
            max_seg = 0
        nn = max_seg + 3
        nn2 = nn + 1
        
        def write_node_rows(f, node_list):
            for i in range(0, len(node_list), 8):
                row = node_list[i:i+8]
                f.write(''.join(f"{p:>10d}" for p in row) + "\n")
        
        with open(self.output_file, 'a') as f:
            # First SET_NODE_LIST (min Z)
            f.write("*SET_NODE_LIST\n")
            f.write(f"{nn:>10d}\n")
            # f.write(f"$ Nodes at min Z = {global_min_z:.6g} (total {len(min_nodes)})\n")
            write_node_rows(f, min_nodes)
            f.write("*BOUNDARY_SLIDING_PLANE\n")
            f.write("$     nsid        vx        vy        vz      copt                       unused1\n")
            f.write(f"{nn:>10d}         0         0         1         0                        \n")
            # Second SET_NODE_LIST (max Z)
            f.write("*SET_NODE_LIST\n")
            f.write(f"{nn2:>10d}\n")
            # f.write(f"$ Nodes at max Z = {global_max_z:.6g} (total {len(max_nodes)})\n")
            write_node_rows(f, max_nodes)
            f.write("*BOUNDARY_SLIDING_PLANE\n")
            f.write("$     nsid        vx        vy        vz      copt                       unused1\n")
            f.write(f"{nn2:>10d}         0         0         1         0                        \n")
        print(f"  Appended boundary segments: SET_NODE_LIST {nn} (min Z, {len(min_nodes)} nodes), {nn2} (max Z, {len(max_nodes)} nodes)")
    
    def _extract_template(self, lines, material_type):
        """Extract template section between $start here and $end here markers."""
        in_section = False
        section_lines = []
        marker_prefix = '$elastic' if material_type == 'elastic' else '$rigid'
        
        for i, line in enumerate(lines):
            if marker_prefix in line.lower():
                # Found the material type marker, look for $start here
                for j in range(i, min(i + 5, len(lines))):
                    if '$start here' in lines[j]:
                        in_section = True
                        continue
            
            if in_section:
                if '$end here' in line:
                    break
                section_lines.append(line)
        
        return ''.join(section_lines)
    
    def _replace_ids(self, template, placeholder, part_id):
        """Replace placeholder (n or m) with actual part_id and format all numeric fields properly."""
        lines = template.split('\n')
        result_lines = []
        prev_line_was_part = False
        
        for line in lines:
            # Check if this is a title line (comes right after *PART)
            is_title_line = prev_line_was_part and line.strip() and not line.strip().startswith('$') and not line.strip().startswith('*')
            
            # Track if this line is *PART for next iteration
            prev_line_was_part = line.strip().startswith('*PART')
            
            # Only replace in data lines (not comments, not titles, not keyword lines)
            if line.strip() and not line.strip().startswith('$') and not line.strip().startswith('*') and not is_title_line:
                # Format all numeric fields in the line properly
                line = self._format_lsdyna_line(line, placeholder, part_id)
            elif is_title_line:
                # For title lines, replace the placeholder in the title text
                line = line.replace(placeholder, str(part_id))
            
            result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _format_lsdyna_line(self, line, placeholder, part_id):
        """Format a data line according to LS-DYNA fixed-format rules (10 chars per field)."""
        # Determine how many fields we need based on original line length
        # LS-DYNA typically uses 8 fields (80 chars) but can extend
        original_length = len(line)
        num_fields = max(8, (original_length + 9) // 10)  # Round up to nearest 10-char field
        
        # Split line into 10-character fields
        fields = []
        for i in range(num_fields):
            start = i * 10
            end = start + 10
            if start < original_length:
                field = line[start:end]
                # Pad to 10 chars if needed
                if len(field) < 10:
                    field = field.ljust(10)
            else:
                field = ' ' * 10
            fields.append(field)
        
        # Process each field
        formatted_fields = []
        for field in fields:
            field_content = field.strip()
            
            # Skip empty fields - preserve as spaces
            if not field_content:
                formatted_fields.append(' ' * 10)
                continue
            
            # Check if field contains the placeholder
            if placeholder in field_content:
                # Replace placeholder with part_id
                try:
                    # Handle case where placeholder is the entire field
                    if field_content == placeholder:
                        formatted_fields.append(f"{part_id:>10d}")
                    else:
                        # Placeholder might be part of an expression - just replace it
                        field_content = field_content.replace(placeholder, str(part_id))
                        # Try to convert to number and format
                        try:
                            value = int(eval(field_content))
                            formatted_fields.append(f"{value:>10d}")
                        except:
                            formatted_fields.append(f"{field_content:>10}")
                except:
                    formatted_fields.append(field)
            else:
                # Field doesn't have placeholder - check if it's numeric
                try:
                    # Try integer
                    value = int(field_content)
                    formatted_fields.append(f"{value:>10d}")
                except ValueError:
                    try:
                        # Try float (scientific notation)
                        value = float(field_content)
                        # Preserve scientific notation if present
                        if 'E' in field_content.upper() or 'e' in field_content:
                            formatted_fields.append(f"{value:>10.2E}")
                        else:
                            formatted_fields.append(f"{value:>10}")
                    except ValueError:
                        # Not a number, keep as-is (right-aligned)
                        formatted_fields.append(f"{field_content:>10}")
        
        return ''.join(formatted_fields)
    
    def _insert_part_name(self, material_section, part_name):
        """Insert part name as title line after *PART keyword."""
        lines = material_section.split('\n')
        result_lines = []

        for line in lines:
            result_lines.append(line)
            # Insert part name as a new line immediately after *PART
            if line.strip().startswith('*PART') and not line.strip().startswith('*PARAMETER'):
                result_lines.append(part_name)

        return '\n'.join(result_lines)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import os
    
    # ========== CONFIGURATION ==========
    END_TIME = 0.001  # Simulation end time
    ENABLE_PLOTTING = False  # Set to False to skip visualization
    # ===================================
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_file = os.path.join(script_dir, 'mesh.k')
    
    # Parse mesh
    parser = MeshParser(mesh_file)
    nodes, elements = parser.parse()
    
    print(f"Parsed {len(nodes)} nodes and {len(elements)} elements")
    
    # Process mesh
    processor = MeshProcessor(nodes, elements)
    processor.analyze()
    
    # Get results
    parts = processor.parts
    part_nodes = processor.part_nodes
    part_info = processor.part_info
    face_nodes = processor.face_nodes
    connecting_faces = processor.connecting_faces
    all_segments = processor.all_segments
    segment_connections = processor.segment_connections
    filtered_connections = processor.filtered_connections
    z_tol = processor.z_tol
    
    # Print results
    print(f"Number of parts: {len(parts)}")
    print(f"Part IDs: {sorted(parts)}")
    
    # Extract coordinates for visualization
    coords = list(nodes.values())
    x = [c[0] for c in coords]
    y = [c[1] for c in coords]
    z = [c[2] for c in coords]
    
    print(f"X range: [{min(x)}, {max(x)}]")
    print(f"Y range: [{min(y)}, {max(y)}]")
    print(f"Z range: [{min(z)}, {max(z)}]")
    print(f"\nZ tolerance: {z_tol}")
    
    print(f"\nFiltered connections (one direction only): {len(filtered_connections)}")
    for bottom_part, top_part in sorted(filtered_connections.items()):
        print(f"  Part {bottom_part} -> {top_part}")
    
    # Print face node information
    print(f"\n{'='*60}")
    print("FACE NODES FOR ALL PARTS")
    print(f"{'='*60}")
    for part_id in sorted(parts):
        print(f"\nPart {part_id}:")
        print(f"  Bottom face ({len(face_nodes[part_id]['bottom'])} nodes): {face_nodes[part_id]['bottom']}")
        print(f"  Top face ({len(face_nodes[part_id]['top'])} nodes): {face_nodes[part_id]['top']}")
    
    print(f"\n{'='*60}")
    print("CONNECTING FACES (SEGMENTS)")
    print(f"{'='*60}")
    for (part_id, face_type), face_info in sorted(connecting_faces.items()):
        connects_to_part, connects_to_face = face_info['connects_to']
        seg_id = face_info['segment_id']
        connects_to_seg = face_info['connects_to_segment']
        print(f"\nSegment {seg_id}: Part {part_id} ({face_type} face)")
        print(f"  Connects to: Segment {connects_to_seg} [Part {connects_to_part} ({connects_to_face} face)]")
        print(f"  Number of nodes: {len(face_info['nodes'])}")
        print(f"  Number of face elements: {len(face_info['face_elements'])}")
        print(f"  First 3 face elements (nodes in CCW order from outside):")
        for i, elem_nodes in enumerate(face_info['face_elements'][:3]):
            print(f"    Element {i+1}: {elem_nodes}")
    
    # Print segment connections
    print(f"\n{'='*60}")
    print("SEGMENT CONNECTIONS")
    print(f"{'='*60}")
    print(f"Total segments: {len(all_segments)}")
    print(f"Total segment connections: {len(segment_connections)}")
    print("\nSegment Connections:")
    for seg_id in sorted(segment_connections.keys()):
        connected_seg = segment_connections[seg_id]
        seg_info = all_segments[seg_id]
        connected_info = all_segments[connected_seg]
        print(f"Segment {seg_id} (Part {seg_info['part']}, {seg_info['face']}) <-> Segment {connected_seg} (Part {connected_info['part']}, {connected_info['face']})")
    
    # Plot with different colors for each part
    if ENABLE_PLOTTING:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Generate random distinct colors for each part
        np.random.seed(42)
        part_list = sorted(parts)
        n_parts = len(part_list)
        colors = np.random.rand(n_parts, 3)
        
        # Plot each part with its own color
        for i, part_id in enumerate(part_list):
            part_node_ids = part_nodes[part_id]
            part_coords = [nodes[nid] for nid in part_node_ids if nid in nodes]
            if part_coords:
                px = [c[0] for c in part_coords]
                py = [c[1] for c in part_coords]
                pz = [c[2] for c in part_coords]
                ax.scatter(px, py, pz, s=0.5, c=[colors[i]], label=f'Part {part_id}')
        
        # Calculate appropriate arrow length (5% of model height)
        model_height = max(z) - min(z)
        arrow_length = model_height * 0.05
        
        # Plot arrows for connecting face elements (every 10th element to avoid clutter)
        n_skip = 10
        for seg_id, seg_info in all_segments.items():
            face_elements = seg_info['face_elements']
            face_type = seg_info['face']
            
            # Determine arrow direction
            if face_type == 'top':
                arrow_dir = np.array([0, 0, arrow_length])  # +Z direction
                arrow_color = 'red'
            else:  # bottom
                arrow_dir = np.array([0, 0, -arrow_length])  # -Z direction
                arrow_color = 'blue'
            
            # Plot arrows for every n_skip-th element
            for i, elem_nodes in enumerate(face_elements):
                if i % n_skip == 0:
                    # Calculate centroid of face element
                    elem_coords = [nodes[nid] for nid in elem_nodes if nid in nodes]
                    if elem_coords:
                        cx = sum(c[0] for c in elem_coords) / len(elem_coords)
                        cy = sum(c[1] for c in elem_coords) / len(elem_coords)
                        cz = sum(c[2] for c in elem_coords) / len(elem_coords)
                        
                        # Draw arrow
                        ax.quiver(cx, cy, cz, 
                                 arrow_dir[0], arrow_dir[1], arrow_dir[2],
                                 color=arrow_color, alpha=0.6, arrow_length_ratio=0.3,
                                 linewidth=1.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Mesh - All Parts with Contact Normals (Red=Top/+Z, Blue=Bottom/-Z)')
        
        # Add legend outside plot area
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=2, fontsize=8, markerscale=4)
        plt.tight_layout()
        plt.show()
    
    # Write input file
    writer = InputFileWriter(script_dir, os.path.join(script_dir, 'processed_input.k'), end_time=END_TIME)
    writer.write(parts, mesh_file=mesh_file, 
                 segment_connections=segment_connections, 
                 all_segments=all_segments,
                 nodes=nodes, part_nodes=part_nodes,
                 part_info=part_info, face_nodes=face_nodes, z_tol=z_tol)


