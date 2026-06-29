"""Diagnose mesh structure to verify radial connectivity."""
import numpy as np
from strandMeshGenerator import StrandMesh_Hexa

# Create a test mesh matching your cable parameters
mesh = StrandMesh_Hexa(
    diameter=0.85,  # CD1 strand diameter
    radial_divisions=3,
    core_divisions=6,
    inner_circumradius_mm=0.02,  # Example: 20 µm
    outer_circumradius_mm=0.04,  # Example: 40 µm
    angle=0
)

print(f"Total nodes: {len(mesh.nodes)}")
print(f"Total elements: {len(mesh.elements)}")
print(f"Circumferential divisions: {mesh.circumferential_divisions}")
print(f"Core size: {mesh.core_size}")
print(f"Hex outer size: {mesh.hex_outer_size}")

# Count elements by type
elem_lens = [len(e) for e in mesh.elements]
tri_count = sum(1 for e in mesh.elements if len(e) == 3)  # Triangles (degenerate quads)
quad_count = sum(1 for e in mesh.elements if len(e) == 4)  # Quads

print(f"\nElement counts:")
print(f"  Triangles (degenerate): {tri_count}")
print(f"  Quads: {quad_count}")

# Check if any elements touch the center (find the node closest to origin)
center_node_idx = np.argmin([np.sqrt(x**2 + y**2) for x, y in mesh.nodes])
center_elems = sum(1 for e in mesh.elements if center_node_idx in e)
print(f"  Elements touching center node {center_node_idx} (actual center): {center_elems}")
print(f"  Center node coordinates: {mesh.nodes[center_node_idx]}")

# Check connectivity: find all nodes adjacent to center
adjacent_nodes = set()
for elem in mesh.elements:
    if center_node_idx in elem:
        adjacent_nodes.update(elem)
adjacent_nodes.discard(center_node_idx)
print(f"  Nodes adjacent to center: {len(adjacent_nodes)}")

# Find radial lines: pairs of nodes where one is at center and one is at outer ring
outer_ring = set(range(len(mesh.nodes) - mesh.circumferential_divisions, len(mesh.nodes)))
radial_connections = sum(
    1 for e in mesh.elements 
    if center_node_idx in e and any(node_idx in outer_ring for node_idx in e if node_idx != center_node_idx)
)
print(f"  Direct center-to-outer-ring elements: {radial_connections}")

# Print first 10 nodes
print(f"\nFirst 10 nodes:")
for i in range(min(10, len(mesh.nodes))):
    print(f"  {i}: {mesh.nodes[i]}")

# Print first 10 elements
print(f"\nFirst 10 elements:")
for i in range(min(10, len(mesh.elements))):
    print(f"  {i}: {mesh.elements[i]}")

# Check innermost nodes (should be near center)
print(f"\nInnermost 5 nodes (by distance from origin):")
distances = [np.sqrt(x**2 + y**2) for x, y in mesh.nodes]
sorted_nodes = sorted(enumerate(distances), key=lambda x: x[1])
for idx, dist in sorted_nodes[:5]:
    print(f"  Node {idx}: distance={dist:.6f}, coords={mesh.nodes[idx]}")
