import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, tan
from hexagon import HexagonMesh  # Add this import


class StrandMesh:
    def __init__(self, diameter, radial_divisions=3, core_divisions=2):
        self.diameter = diameter
        self.radius = diameter / 2
        self.radial_divisions = radial_divisions
        self.core_divisions = core_divisions
        self.core_size = 0.1* diameter
        self.circumferential_divisions = core_divisions * 4
        self.nodes = []  # List to store (x, y) coordinates of nodes
        self.elements = []  # List to store quad element connectivity
        self.generate_mesh()

    def order_core_boundary_nodes(self, core_boundary_nodes):
        """Orders core boundary nodes in azimuthal direction starting from (x=+core_size/2, y=0)."""
        core_half = self.core_size / 2
        
        # Extract coordinates of core boundary nodes
        boundary_coords = [self.nodes[i] for i in core_boundary_nodes]
        
        # Compute azimuthal angles
        angles = [np.arctan2(y, x) for x, y in boundary_coords]
        
        # Adjust angles to start from (x=+core_size/2, y=0)
        angles = [(angle if angle >= 0 else angle + 2 * np.pi) for angle in angles]
        
        # Sort indices based on azimuthal angle
        core_boundary_nodes_ordered = [x for _, x in sorted(zip(angles, core_boundary_nodes))]
        
        return core_boundary_nodes_ordered
    
    def generate_mesh(self):
        """Generates a hybrid circular quad mesh with a central Cartesian block."""
        core_half = self.core_size / 2
        core_element_size = self.core_size / self.core_divisions
        inner_circ_radius = np.sqrt(2) * core_half + 0.5 * core_element_size
        
        # Generate core Cartesian nodes
        x_vals = np.linspace(-core_half, core_half, self.core_divisions + 1)
        y_vals = np.linspace(-core_half, core_half, self.core_divisions + 1)
        core_nodes = [(x, y) for y in y_vals for x in x_vals]
        self.nodes.extend(core_nodes)
        
        # Generate core elements
        core_nx = self.core_divisions + 1
        for i in range(self.core_divisions):
            for j in range(self.core_divisions):
                n1 = i * core_nx + j
                n2 = i * core_nx + j + 1
                n3 = (i + 1) * core_nx + j + 1
                n4 = (i + 1) * core_nx + j
                self.elements.append((n1, n2, n3, n4))
        
        # Generate circular mesh nodes
        offset = len(self.nodes)
        for i in range(0, self.radial_divisions):
            r = inner_circ_radius + (i / self.radial_divisions) * (self.radius - inner_circ_radius)
            for j in range(self.circumferential_divisions):
                theta = (j / self.circumferential_divisions) * 2 * np.pi
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                self.nodes.append((x, y))
        
        # Generate circular mesh elements
        for i in range(self.radial_divisions - 1):
            for j in range(self.circumferential_divisions):
                n1 = offset + i * self.circumferential_divisions + j
                n2 = offset + i * self.circumferential_divisions + (j + 1) % self.circumferential_divisions
                n3 = offset + (i + 1) * self.circumferential_divisions + (j + 1) % self.circumferential_divisions
                n4 = offset + (i + 1) * self.circumferential_divisions + j
                self.elements.append((n1, n2, n3, n4))
        
        # Reorder core boundary nodes to match circumferential nodes
        core_boundary_nodes = list()
        for i in range(0, self.core_divisions + 1):
            core_boundary_nodes.append(i)
        for i in range(1, self.core_divisions):
            core_boundary_nodes.append(i * (self.core_divisions + 1))
            core_boundary_nodes.append(i * (self.core_divisions + 1) + self.core_divisions)
        for i in range(self.core_divisions + 1):
            core_boundary_nodes.append((self.core_divisions + 1) * (self.core_divisions + 1) - self.core_divisions - 1 + i)
        
        core_boundary_nodes_ordered = self.order_core_boundary_nodes(core_boundary_nodes)

        # Generate transition elements between core and circular mesh
        first_circ_layer_nodes = self.nodes[offset:offset + self.circumferential_divisions]
        
        for j in range(len(first_circ_layer_nodes)):
            n1 = core_boundary_nodes_ordered[(j) % len(core_boundary_nodes_ordered)]
            n2 = core_boundary_nodes_ordered[(j + 1) % len(core_boundary_nodes_ordered)]
            n3 = offset + (j + 1) % self.circumferential_divisions
            n4 = offset + j
            self.elements.append((n1, n2, n3, n4))
    
    def plot_mesh(self):
        """Plots the generated mesh."""
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot nodes
        x_vals, y_vals = zip(*self.nodes)
        ax.scatter(x_vals, y_vals, color='blue', s=10)
        
        # Plot elements as lines
        for elem in self.elements:
            try:
                elem_nodes = [self.nodes[i] for i in elem] + [self.nodes[elem[0]]]
                x_e, y_e = zip(*elem_nodes)
                ax.plot(x_e, y_e, color='black', linewidth=0.5)
            except IndexError:
                print(f"Skipping invalid element: {elem}")
        
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Core and Circular Mesh with Transition Elements")
        ax.set_aspect('equal')
        ax.grid(True)
        plt.show()


# De-duplicate StrandMesh_Hexa clamp log lines: a strand-mesh is built once
# per (strand, stack, cable), so the same (diameter, scale) tuple recurs.
_CLAMP_LOGGED: set = set()


class StrandMesh_Hexa:
    def __init__(self, diameter=0.85, radial_divisions=2, core_divisions=6,
                 hex_scale=(159/444), hex_outer_scale=400/444, angle=30,
                 inner_circumradius_mm=None, outer_circumradius_mm=None):
        """
        hex_scale: fraction of diameter for inner hexagon vertex-to-vertex
            (so inner hex circumradius = 0.5 * hex_scale * diameter).
        hex_outer_scale: fraction of diameter for outer hexagon outermost
            boundary vertex-to-vertex (so outer hex circumradius = 0.5 *
            hex_outer_scale * diameter). The internal /1.3 factor sets the
            inner edge of the outer-ring transition layer.
        inner_circumradius_mm / outer_circumradius_mm: optional absolute
            circumradii in mm. If given, override hex_scale / hex_outer_scale
            via 2*R/diameter so the same scale-fraction semantics hold.
        """
        # --- Robustness: clamp the hex template to fit inside the strand ----
        # The transition layers (radial_divisions of them) are generated by
        #   r = R_outer + (i+1)/n_circ * (strand_radius - R_outer)
        # so when R_outer > strand_radius (e.g. a wire block designed for a
        # bigger strand pasted onto a small one) the transition layers go
        # INWARD and end up smaller than the last hex layer. Layer 8 (writer's
        # "middle") then envelops layer 11 (writer's "outer"), producing a
        # degenerate ASBA topology that crashes 2-geo.inp.
        #
        # Clamp R_outer to <= 0.9 * (D/2) so the 3 transition layers have room
        # to grow OUTWARD from the hex to the strand outline. Scale R_inner by
        # the same factor so the hex aspect (Nb3Sn:Cu area split) is preserved.
        # Bypass entirely when the caller passes neither circumradius — the
        # legacy hex_scale/hex_outer_scale defaults are already strand-relative.
        if outer_circumradius_mm is not None:
            safe_outer = 0.45 * diameter  # = 0.9 * strand_radius
            if outer_circumradius_mm > safe_outer:
                _scale = safe_outer / outer_circumradius_mm
                outer_circumradius_mm = safe_outer
                if inner_circumradius_mm is not None:
                    inner_circumradius_mm *= _scale
                # Log once per (D, scale) combo to avoid spamming 200 lines
                # when the strand-mesh constructor is called for every strand
                # of every stack of every cable.
                _key = (round(diameter, 6), round(_scale, 4))
                if _key not in _CLAMP_LOGGED:
                    _CLAMP_LOGGED.add(_key)
                    print(f"[StrandMesh_Hexa] Clamped hex circumradii by "
                          f"{_scale:.3f} to fit strand D={diameter*1e3:.0f} um: "
                          f"R_inner={inner_circumradius_mm*1e3:.1f} um, "
                          f"R_outer={outer_circumradius_mm*1e3:.1f} um.")

        if inner_circumradius_mm is not None:
            hex_scale = 2.0 * inner_circumradius_mm / diameter
        if outer_circumradius_mm is not None:
            hex_outer_scale = 2.0 * outer_circumradius_mm / diameter
        self.diameter = diameter
        self.radius = diameter / 2
        self.radial_divisions = radial_divisions
        self.core_divisions = core_divisions
        self.angle = angle
        self.hex_scale = hex_scale
        self.hex_outer_scale = hex_outer_scale / 1.3
        self.core_size = self.hex_scale * diameter
        self.hex_outer_size = self.hex_outer_scale * diameter
        self.circumferential_divisions = core_divisions * 6  # for hexagon symmetry
        self.edge_divisions = self.circumferential_divisions // 6   # cache for segment size
        self.nodes = []
        self.node_index = {}  # For fast node lookup
        self.elements = []
        self.generate_mesh()

    def hex_corner(self, center, radius, i):
        angle_rad = np.deg2rad(60 * i - self.angle)
        return (center[0] + radius * np.cos(angle_rad),
                center[1] + radius * np.sin(angle_rad))

    def find_or_add_node(self, node, tol=1e-8):
        key = (round(node[0]/tol)*tol, round(node[1]/tol)*tol)
        if key in self.node_index:
            return self.node_index[key]
        idx = len(self.nodes)
        self.nodes.append(node)
        self.node_index[key] = idx
        return idx

    def _unique(self, nodes, tol=1e-8):
        seen = set(); out = []
        for x, y in nodes:
            key = (round(x/tol)*tol, round(y/tol)*tol)
            if key not in seen:
                seen.add(key); out.append((x, y))
        return out

    def generate_hex_layer_nodes(self, n_layers, inner_radius, outer_radius):
        """Generate nodes for n_layers between two hexagons (inner and outer), avoiding duplicates."""
        nodes = []
        for layer in range(n_layers + 1):
            frac = layer / n_layers
            r = inner_radius * (1 - frac) + outer_radius * frac
            for j in range(self.circumferential_divisions):
                edge = j // (self.circumferential_divisions // 6)
                t = (j % (self.circumferential_divisions // 6)) / (self.circumferential_divisions // 6)
                p1 = self.hex_corner((0, 0), r, edge)
                p2 = self.hex_corner((0, 0), r, (edge + 1) % 6)
                x = (1 - t) * p1[0] + t * p2[0]
                y = (1 - t) * p1[1] + t * p2[1]
                nodes.append((x, y))
        # Remove duplicates efficiently
        return self._unique(nodes)

    # def generate_inner_hex_mesh(self, n_inner_div):
    #     """Mesh the inner hexagon with a conformal quad mesh, node numbering starts at 0."""
    #     nodes = [(0.0, 0.0)]
    #     # Generate n_inner_div radial layers from center to inner hexagon
    #     for layer in range(1, n_inner_div + 1):
    #         frac = layer / n_inner_div
    #         r = self.core_size / 2
    #         for j in range(self.circumferential_divisions):
    #             edge = j // (self.circumferential_divisions // 6)
    #             t = (j % (self.circumferential_divisions // 6)) / (self.circumferential_divisions // 6)
    #             p1 = self.hex_corner((0, 0), r, edge)
    #             p2 = self.hex_corner((0, 0), r, (edge + 1) % 6)
    #             x_hex = (1 - t) * p1[0] + t * p2[0]
    #             y_hex = (1 - t) * p1[1] + t * p2[1]
    #             x = frac * x_hex
    #             y = frac * y_hex
    #             nodes.append((x, y))
    #     # Remove duplicates efficiently
    #     nodes = self._unique(nodes)
    #     elements = []
    #     # Elements: quads between layers
    #     # First layer: triangles from center to first ring (as degenerate quads)
    #     for j in range(self.circumferential_divisions):
    #         n1 = 0
    #         n2 = 1 + j
    #         n3 = 1 + (j + 1) % self.circumferential_divisions
    #         n4 = 0
    #         elements.append((n1, n2, n3, n4))
    #     # Next layers: quads between rings
    #     for layer in range(1, n_inner_div):
    #         base1 = 1 + (layer - 1) * self.circumferential_divisions
    #         base2 = 1 + layer * self.circumferential_divisions
    #         for j in range(self.circumferential_divisions):
    #             n1 = base1 + j
    #             n2 = base1 + (j + 1) % self.circumferential_divisions
    #             n3 = base2 + (j + 1) % self.circumferential_divisions
    #             n4 = base2 + j
    #             elements.append((n1, n2, n3, n4))
    #     return nodes, elements

    def generate_inner_hex_mesh(self, n_edge=12, n_height=6):
        """
        Generate a conformal quad mesh inside a regular hexagon.
        n_edge: number of divisions along each hex edge (horizontal direction)
        n_height: number of divisions from center to edge (vertical direction)
        """
        r = self.core_size 
        nodes = []
        grid = {}
        idx = 0
        # Hexagon corners
        corners = [self.hex_corner((0, 0), r, n) for n in range(6)]
        # Loop over grid in (v, u) space
        for v in range(n_height + 1):
            frac_v = v / n_height
            for u in range(n_edge + 1):
                frac_u = u / n_edge
                # Find which two corners this row is between
                # Each row goes from one edge to the opposite edge
                # For each v, interpolate between corners[0] and corners[3], etc.
                # Map (u, v) to barycentric coordinates
                # For each v, the row starts at a point on edge 5-0 and ends at edge 2-3
                start = (
                    (1 - frac_v) * corners[5][0] + frac_v * corners[0][0],
                    (1 - frac_v) * corners[5][1] + frac_v * corners[0][1]
                )
                end = (
                    (1 - frac_v) * corners[2][0] + frac_v * corners[3][0],
                    (1 - frac_v) * corners[2][1] + frac_v * corners[3][1]
                )
                x = (1 - frac_u) * start[0] + frac_u * end[0]
                y = (1 - frac_u) * start[1] + frac_u * end[1]
                nodes.append((x, y))
                grid[(v, u)] = idx
                idx += 1
        elements = []
        for v in range(n_height):
            for u in range(n_edge):
                n1 = grid[(v, u)]
                n2 = grid[(v, u + 1)]
                n3 = grid[(v + 1, u + 1)]
                n4 = grid[(v + 1, u)]
                elements.append((n1, n2, n3, n4))
        return nodes, elements

    def generate_mesh(self):
        # --- Inner hexagonal region (using HexagonMesh) ---
        n_edge = 12
        n_height = 6
        hex_width = self.core_size * sqrt(3) / 2
        hex_height = hex_width / np.cos(np.deg2rad(30)) / 2  # rectangle height

        hex_mesh = HexagonMesh(n_edge=n_edge, n_height=n_height, width=hex_width, height=hex_height)
        hex_mesh.rotate_nodes(np.deg2rad(30 - self.angle))

        hex_node_map = [self.find_or_add_node(n) for n in hex_mesh.nodes]
        for elem in hex_mesh.elements:
            self.elements.append(tuple(hex_node_map[i] for i in elem))

        # --- Inner hexagonal ring region (core) ---
        n_hex_layers = self.core_divisions
        core_inner_radius = self.core_size / 2
        core_outer_radius = self.hex_outer_size / 2
        hex_nodes = self.generate_hex_layer_nodes(n_hex_layers, core_inner_radius, core_outer_radius)
        n_per_layer = self.circumferential_divisions
        hex_node_map = [self.find_or_add_node(n) for n in hex_nodes]
        for layer in range(n_hex_layers):
            base1 = layer * n_per_layer
            base2 = (layer + 1) * n_per_layer
            for j in range(n_per_layer):
                n1 = hex_node_map[base1 + j]
                n2 = hex_node_map[base1 + (j + 1) % n_per_layer]
                n3 = hex_node_map[base2 + (j + 1) % n_per_layer]
                n4 = hex_node_map[base2 + j]
                self.elements.append((n1, n2, n3, n4))
        # --- Outer hexagonal region (between two hexagons) ---
        n_hex_outer_layers = 2  # can be parameterized
        hex_outer_nodes = self.generate_hex_layer_nodes(n_hex_outer_layers, core_outer_radius, core_outer_radius * 1.3)
        hex_outer_node_map = [self.find_or_add_node(n) for n in hex_outer_nodes]
        for layer in range(n_hex_outer_layers):
            base1 = layer * n_per_layer
            base2 = (layer + 1) * n_per_layer
            for j in range(n_per_layer):
                n1 = hex_outer_node_map[base1 + j]
                n2 = hex_outer_node_map[base1 + (j + 1) % n_per_layer]
                n3 = hex_outer_node_map[base2 + (j + 1) % n_per_layer]
                n4 = hex_outer_node_map[base2 + j]
                self.elements.append((n1, n2, n3, n4))
        # --- Transition from hexagon to circle ---
        n_circ_layers = self.radial_divisions
        circ_node_map = []
        
        
        alpha = 30*np.pi/180  # 30 degrees in radians
        beta = np.arctan(np.tan(alpha)/3)
        gamma = np.arctan(2*np.tan(alpha)/3)-beta
        eta = alpha - beta - gamma
        
        # print(f"Angles: beta={beta*180/np.pi}, gamma={gamma*180/np.pi}, eta={eta*180/np.pi}",'sum:', (beta + gamma + eta)*180/np.pi)


        for i in range(n_circ_layers):
            r = core_outer_radius * 1.3 + ((i + 1) / n_circ_layers) * (self.radius - (core_outer_radius * 1.3))
            if np.isclose(r, core_outer_radius * 1.0):
                continue
            for j in range(self.circumferential_divisions):
                # print('j:', j, 'circumferential_divisions:', self.circumferential_divisions)
                # Repeat [one, two, three] pattern until circumferential_divisions is reached
                pattern = [eta, gamma, beta, beta, gamma, eta]
                theta = -np.deg2rad(self.angle) + sum(pattern[k % 6] for k in range(j))
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                idx = self.find_or_add_node((x, y))
                circ_node_map.append(idx)
        n_circ_layers_actual = len(circ_node_map) // self.circumferential_divisions
        for i in range(n_circ_layers_actual - 1):
            for j in range(self.circumferential_divisions):
                n1 = circ_node_map[i * self.circumferential_divisions + j]
                n2 = circ_node_map[i * self.circumferential_divisions + (j + 1) % self.circumferential_divisions]
                n3 = circ_node_map[(i + 1) * self.circumferential_divisions + (j + 1) % self.circumferential_divisions]
                n4 = circ_node_map[(i + 1) * self.circumferential_divisions + j]
                self.elements.append((n1, n2, n3, n4))
        # Connect outermost hex ring to closest circle nodes (no crossing lines)
        hex_outermost_offset = hex_outer_node_map[-n_per_layer:]
        if n_circ_layers_actual > 0 and circ_node_map:
            for j in range(n_per_layer):
                hx_idx = hex_outermost_offset[j]
                hx, hy = self.nodes[hx_idx]
                # Find closest circle node index for this hex node
                dists = [np.hypot(hx - self.nodes[circ_node_map[k]][0], hy - self.nodes[circ_node_map[k]][1]) for k in range(n_per_layer)]
                best_k = int(np.argmin(dists))
                n1 = hx_idx
                n2 = hex_outermost_offset[(j + 1) % n_per_layer]
                n3 = circ_node_map[(best_k + 1) % n_per_layer]
                n4 = circ_node_map[best_k]
                # self.elements.append((n1, n2, n3, n4))
                self.elements.insert(len(self.elements) - (n_circ_layers_actual-1) * self.circumferential_divisions, (n1, n2, n3, n4))

                
                
                
        #mesh inner part
        
        

    def plot_mesh(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        x_vals, y_vals = zip(*self.nodes)
        ax.scatter(x_vals, y_vals, color='blue', s=15, alpha=0.01)
        
        
        for elem_idx, elem in enumerate(self.elements):
            try:
                elem_nodes = [self.nodes[i] for i in elem] + [self.nodes[elem[0]]]
                # Color logic
                if elem_idx < 72:
                    facecolor = '#b87333'  # copper orange
                elif 72 <= elem_idx < 360:
                    facecolor = '#c0c0c0'  # silver
                else:
                    facecolor = '#b87333'  # copper orange
                poly = plt.Polygon(elem_nodes, closed=True, facecolor=facecolor, edgecolor='none', linewidth=0.001, alpha=1)
                ax.add_patch(poly)
                
                
                # # Plot element number at centroid
                # xs, ys = zip(*[self.nodes[i] for i in elem])
                # centroid_x = sum(xs) / len(xs)
                # centroid_y = sum(ys) / len(ys)
                # ax.text(centroid_x, centroid_y, str(elem_idx), fontsize=10, color='green', ha='center', va='center')
            except IndexError:
                print(f"Skipping invalid element: {elem}")
        ax.set_title("Hexagonal Core, Hexagonal Middle, and Outer")
        ax.set_aspect('equal')
        # ax.grid(True)
        plt.show()



# Example usage
if __name__ == "__main__":
    # mesh = StrandMesh(diameter=0.85, radial_divisions=9, core_divisions=12)
    mesh = StrandMesh_Hexa(diameter=0.85, radial_divisions=3, core_divisions=6, hex_scale=(129.52/425), hex_outer_scale=334.5/425, angle=0)

    mesh.plot_mesh()
