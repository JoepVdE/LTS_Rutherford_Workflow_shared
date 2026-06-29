import numpy as np
import matplotlib.pyplot as plt
from strandMeshGenerator import StrandMesh_Hexa
from strandMeshGenerator import StrandMesh
from deformedStrandInterpolator import DeformedStrandInterpolator
from scipy.optimize import minimize_scalar

class MeshMapping:
    def __init__(self, mesh, deformed_strand):
        """Initializes the mesh mapping process."""
        self.mesh = mesh
        self.deformed_strand = deformed_strand
        self.translated_nodes = None
        self.mapped_nodes = None

    @staticmethod
    def _polygon_centroid(pts):
        """Area centroid of a closed polygon (shoelace formula).
        pts: (N, 2) array of boundary points (last point need not repeat first).
        Returns (cx, cy)."""
        x, y = pts[:, 0], pts[:, 1]
        xn, yn = np.roll(x, -1), np.roll(y, -1)
        cross = x * yn - xn * y
        area = 0.5 * np.sum(cross)
        if abs(area) < 1e-30:
            return np.mean(pts, axis=0)
        cx = np.sum((x + xn) * cross) / (6.0 * area)
        cy = np.sum((y + yn) * cross) / (6.0 * area)
        return np.array([cx, cy])

    def translate_mesh_to_barycenter(self):
        """Translates the circular mesh to align with the barycenter of the deformed B-spline region."""
        circular_nodes = np.array(self.mesh.nodes)
        bspline_nodes = self.deformed_strand.evaluate_bspline(num_points=500)
        
        # Compute barycenters
        barycenter_circular = np.mean(circular_nodes, axis=0)
        barycenter_bspline = self._polygon_centroid(bspline_nodes)
        
        # Compute translation vector
        translation_vector = barycenter_bspline - barycenter_circular
        
        # Apply translation
        self.translated_nodes = circular_nodes + translation_vector
    
    def find_bspline_intersection(self, origin, direction):
        """Finds the intersection of a radial direction with the B-spline."""
        def distance_to_bspline(t):
            point = origin + t * direction
            bspline_nodes = self.deformed_strand.evaluate_bspline(num_points=500)
            distances = np.linalg.norm(bspline_nodes - point, axis=1)
            return np.min(distances)
        
        result = minimize_scalar(distance_to_bspline, bounds=(0, 2), method='bounded')
        return origin + result.x * direction
    
    def map_circumferential_layer_to_bspline(self) -> None:
        """Moves all nodes of the circumferential region along their radial direction to maintain uniform spacing."""
        if self.translated_nodes is None:
            raise ValueError("Mesh must be translated before mapping.")
        
        num_nodes = len(self.translated_nodes)
        # print('Num_nodes:', num_nodes)
        # print('Circumferential divisions:', self.mesh.circumferential_divisions)
        outer_layer_indices = range(num_nodes - self.mesh.circumferential_divisions, num_nodes)
        # print('Outer layer indices:', outer_layer_indices)
        circumferential_layer_indices = range(num_nodes - 2 * self.mesh.circumferential_divisions, num_nodes - self.mesh.circumferential_divisions)
        
        # print('Circumferential layer indices:', circumferential_layer_indices)
        
        # Compute barycenter of the deformed region (polygon area centroid, not mean of boundary points)
        bspline_nodes = self.deformed_strand.evaluate_bspline(num_points=500)
        barycenter_bspline = self._polygon_centroid(bspline_nodes)
        
        # Move outer layer nodes to B-spline positions along radial direction and store displacement
        mapped_nodes = self.translated_nodes.copy()
        original_nodes = self.translated_nodes.copy()

        # --- Pre-scale: ensure all outer ring nodes start strictly inside the B-spline
        # so that the radial mapping is always outward (t > 0), which prevents cave-in
        # on heavily deformed strands where the template outer ring would otherwise sit
        # outside the B-spline in some directions.
        outer_layer_list = list(outer_layer_indices)
        outer_r = np.linalg.norm(mapped_nodes[outer_layer_list] - barycenter_bspline, axis=1)
        max_outer_r = outer_r.max()
        bspline_r_min = np.min(np.linalg.norm(bspline_nodes - barycenter_bspline, axis=1))
        if max_outer_r >= bspline_r_min:
            pre_scale = 0.99 * bspline_r_min / max_outer_r
            mapped_nodes = barycenter_bspline + (mapped_nodes - barycenter_bspline) * pre_scale
        # Save pre-scaled outer ring radii as denominator for inner-node proportional scaling.
        # Using the pre-scaled template radius (not the post-mapping B-spline radius) ensures
        # the pre-scale factor cancels in the ratio r_inner / r_outer_template.
        outer_r_template = np.linalg.norm(mapped_nodes[outer_layer_list] - barycenter_bspline, axis=1)
        # -------------------------------------------------------------------------
        
        displacement_vectors = []
        direction_vectors = []
        original_nodes_for_plotting = []	
        
        for idx in outer_layer_indices:
            node = mapped_nodes[idx]
            original_nodes_for_plotting.append(original_nodes[idx])
            direction = node - barycenter_bspline  # Compute radial direction relative to deformed barycenter
            
            direction_vectors.append(direction)
            
            if np.linalg.norm(direction) != 0:
                direction /= np.linalg.norm(direction)
            else:
                print("Cannot normalize a zero vector.")
                
            new_position = self.find_bspline_intersection(node, direction)
            displacement_vectors.append(new_position - node)
            mapped_nodes[idx] = new_position
        
        # print('radial divisions:', self.mesh.radial_divisions)
        
        if hasattr(self.mesh, 'radial_divisions') and hasattr(self.mesh, 'core_divisions'):
            mapping_adjuster = self.mesh.radial_divisions + 10
        else:
            mapping_adjuster = self.mesh.radial_divisions
        
        # Precompute inner-node distances from the pre-scaled mapped_nodes using
        # barycenter_bspline as reference.  These are used as the numerator in
        # factor = r_inner / r_outer_template for proportional displacement.
        original_distances = []
        for i, idx in enumerate(circumferential_layer_indices):
            node_distances = []
            for j in range(mapping_adjuster - 1):
                node_index = idx - j * self.mesh.circumferential_divisions
                scaled_node = mapped_nodes[node_index]
                node_distances.append(np.linalg.norm(scaled_node - barycenter_bspline))
            original_distances.append(node_distances)

        core_nodes = 91

        count = 0
        # Apply displacement proportionally to hex and circ region nodes
        for i, idx in enumerate(circumferential_layer_indices):
            corresponding_outer_idx = outer_layer_list[i]
            displacement = displacement_vectors[i]
            # Denominator: pre-scaled template outer ring radius for this ray.
            # Using the template radius (not the post-mapping B-spline radius) lets the
            # pre-scale cancel: factor = (pre_scale * r_inner) / (pre_scale * r_outer)
            # = r_inner / r_outer, independent of the pre-scale.
            ray_outer_r = outer_r_template[i]
            factor_list = []
            count2 = 0
            for j in range(mapping_adjuster - 1):
                node_index = idx - j * self.mesh.circumferential_divisions
                if node_index < core_nodes:
                    continue
                # Proportional scaling: inner node distance / outer ring distance
                factor = original_distances[i][j] / ray_outer_r if ray_outer_r > 0 else 0.0
                factor_list.append(factor)
                mapped_nodes[node_index] = mapped_nodes[node_index] + factor * displacement
                # print('Factor:', factor)
                # print('Mapped node:', mapped_nodes[node_index])
                # print('Node index:', node_index)
                # print('adjust', (mapping_adjuster - 1 - j) * displacement)
                # count2 += 1
            # print('Count of iterations for this rad-line:', count2)
            # count += 1
            # print('Factor list:', factor_list)
        self.mapped_nodes = mapped_nodes
        # print('Count of rad lines:', count)

    def plot_mesh(self):
        """Plots the original mesh nodes and elements before mapping."""
        fig, ax = plt.subplots(figsize=(4, 4))
        x_vals, y_vals = zip(*self.mesh.nodes)
        # ax.scatter(x_vals, y_vals, color='blue', s=15, alpha=0.5, label='Original Nodes')

        for elem_idx, elem in enumerate(self.mesh.elements):
            try:
                elem_nodes = [self.mesh.nodes[i] for i in elem] + [self.mesh.nodes[elem[0]]]
                # Color logic
                if elem_idx < 72+36:
                    facecolor = '#b87333'  # copper orange
                elif 72+36 <= elem_idx < 360:
                    facecolor = '#c0c0c0'  # silver
                else:
                    facecolor = '#b87333'  # copper orange
                poly = plt.Polygon(elem_nodes, closed=True, facecolor=facecolor, edgecolor='none', linewidth=0.001, alpha=1)
                ax.add_patch(poly)
            except IndexError:
                print(f"Skipping invalid element: {elem}")

        # Set axis limits to fit all nodes with a small margin
        margin = 0.05 * max(max(x_vals) - min(x_vals), max(y_vals) - min(y_vals))
        ax.set_xlim(min(x_vals) - margin, max(x_vals) + margin)
        ax.set_ylim(min(y_vals) - margin, max(y_vals) + margin)

        ax.set_title(
            "RRP strand consisting of three regions",
            # fontsize=11,
            wrap=True
        )
        ax.set_aspect('equal')
        ax.set_xlabel("x / mm")
        ax.set_ylabel("y / mm")
        plt.savefig("original_mesh.svg", format="svg")

        ax.legend()

    def plot_mapped_mesh(self):
        """Plots the mapped mesh with colored regions, similar to plot_mesh."""
        fig, ax = plt.subplots(figsize=(4, 4))

        # Plot B-spline interpolation
        # bspline_nodes = self.deformed_strand.evaluate_bspline(num_points=200)
        # ax.plot(bspline_nodes[:, 0], bspline_nodes[:, 1], color='blue', linestyle='--', label='B-Spline Interpolation')

        if self.mapped_nodes is not None:
            # Plot mapped mesh nodes (optional, low alpha)
            # ax.scatter(self.mapped_nodes[:, 0], self.mapped_nodes[:, 1], color='red', s=15, alpha=1)

            # Calculate barycenters to align original mesh with mapped mesh
            original_barycenter = np.mean(self.mesh.nodes, axis=0)
            mapped_barycenter = np.mean(self.mapped_nodes, axis=0)
            translation_vector = mapped_barycenter - original_barycenter

            # # First plot the original mesh (translated to mapped mesh barycenter)
            # ax.plot([], [], color='black', linestyle='-', linewidth=0.7, label='Original Mesh')
            # for elem_idx, elem in enumerate(self.mesh.elements):
            #     try:
            #         # Get original element nodes and translate them
            #         original_elem_nodes = [np.array(self.mesh.nodes[i]) + translation_vector for i in elem]
            #         original_elem_nodes.append(np.array(self.mesh.nodes[elem[0]]) + translation_vector)  # Close the polygon
                    
            #         # Plot only the outlines of the original mesh for clarity
            #         poly = plt.Polygon(original_elem_nodes, closed=True, facecolor='none', 
            #                           edgecolor='black', linewidth=0.1, alpha=0.1)
            #         ax.add_patch(poly)
            #     except IndexError:
            #         print(f"Skipping invalid element for original mesh: {elem}")
            # Plot a circle at the mapped barycenter with diameter 0.88 mm

            # ax.plot(mapped_barycenter[0], mapped_barycenter[1], 'ko', markersize=5, label='Mapped Barycenter')
            # Then plot the mapped mesh with colored regions
            ax.plot([], [], color='none', marker='s', markerfacecolor='#b87333', 
                    markersize=10, label='Copper')
            ax.plot([], [], color='none', marker='s', markerfacecolor='#c0c0c0', 
                    markersize=10, label='Nb$_3$Sn interfilamentary area')
                    
            # Plot elements with region coloring
            for elem_idx, elem in enumerate(self.mesh.elements):
                try:
                    elem_nodes = [self.mapped_nodes[i] for i in elem] + [self.mapped_nodes[elem[0]]]
                    # Color logic as in plot_mesh
                    if elem_idx < 72+36:
                        facecolor = '#b87333'  # copper orange
                    elif 72+36 <= elem_idx < 360:
                        facecolor = '#c0c0c0'  # silver
                    else:
                        facecolor = '#b87333'  # copper orange
                    poly = plt.Polygon(elem_nodes, closed=True, facecolor=facecolor, edgecolor='none', linewidth=0.001, alpha=1)
                    ax.add_patch(poly)
                except IndexError:
                    print(f"Skipping invalid element: {elem}")
            circle = plt.Circle(mapped_barycenter, 0.882 / 2, color='black', fill=False, linestyle='--', linewidth=1.2, label='Outline strand before mapping')
            ax.add_patch(circle)
        ax.set_xlabel("x / mm")
        ax.set_ylabel("y / mm")
        ax.set_title("Mesh Mapping: Original (Outlines) vs. Mapped Mesh")
        # Place legend outside the plot
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        ax.set_aspect('equal')
        # Save the figure with the legend included
        plt.savefig("mapped_mesh.svg", format="svg", bbox_inches='tight')
        plt.show()

# Example usage
if __name__ == "__main__":
    # mesh = StrandMesh(diameter=.8, radial_divisions=5, core_divisions=6)
    # mesh = StrandMesh_Hexa(diameter=0.75, radial_divisions=2, core_divisions=6, hex_scale=(159/444), hex_outer_scale=402/444)
    mesh = StrandMesh_Hexa(diameter=0.85, radial_divisions=3, core_divisions=6, hex_scale=.2, hex_outer_scale=.85, angle=0)
    
    # figure = mesh.plot_mesh()
    
    
    deformed_strand = DeformedStrandInterpolator("./Stack_1_Part17.csv")
    deformed_strand.fit_bspline()
    
    mapper = MeshMapping(mesh, deformed_strand)


    mapper.translate_mesh_to_barycenter()
    # mapper.plot_mapped_mesh()
    
    mapper.map_circumferential_layer_to_bspline()
    figure3 = mapper.plot_mesh()
    figure2 =  mapper.plot_mapped_mesh()
