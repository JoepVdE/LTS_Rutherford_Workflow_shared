
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from strandMeshGenerator import StrandMesh
from strandMeshGenerator import StrandMesh_Hexa
from deformedStrandInterpolator import DeformedStrandInterpolator
# from strandsContact import StrandsContact
from meshMapping import MeshMapping
from insulationlayer import InsulationLayer, align_interface_keypoints
from contact_nodes import ContactNodes
from itertools import combinations
import random
import os


def compute_hex_circumradii_mm(wire):
    """Inner/outer big-hex circumradii (mm) from the wire JSON block.

    Mirrors the homogenisation in scripts/apdl/submodel/RVE/prep.py:
        d_outer    = D_CORE_EQ_UM + CU_SLEEVE_THICKNESS_UM        [um]
        side_outer = sqrt(pi * d_outer**2 / (6*sqrt(3)))          [um]
        R_inner    = sqrt(N_TOTAL - N_NB3SN) * side_outer         [um]
        R_outer    = sqrt(N_TOTAL)            * side_outer         [um]

    Returns
    -------
    (R_inner_mm, R_outer_mm) or (None, None) if `wire` is None.
    """
    if wire is None:
        return None, None
    d_core = float(wire["D_CORE_EQ_UM"])
    t_cu = float(wire["CU_SLEEVE_THICKNESS_UM"])
    n_total = int(wire["N_TOTAL"])
    n_nb3sn = int(wire["N_NB3SN"])
    n_cu = max(n_total - n_nb3sn, 0)
    d_outer_um = d_core + t_cu
    side_outer_um = np.sqrt(np.pi * d_outer_um**2 / (6.0 * np.sqrt(3.0)))
    R_inner_um = np.sqrt(max(n_cu, 1)) * side_outer_um
    R_outer_um = np.sqrt(n_total) * side_outer_um
    return R_inner_um * 1e-3, R_outer_um * 1e-3


def _shells_from_hex_count(n: int) -> int:
    """Inverse of the hex-number sequence 1, 7, 19, 37, 61, 91, 127, 169, ..."""
    k = 0
    while 3 * k * (k + 1) + 1 < n:
        k += 1
    if 3 * k * (k + 1) + 1 != n:
        raise ValueError(f"{n} is not a valid hex number (1,7,19,37,61,91,127,169,...)")
    return k


def _hex_center_positions(n_rings, side):
    """Axial-coordinate hex grid: returns list of (q, r, x, y, shell_index)."""
    out = []
    for q in range(-n_rings, n_rings + 1):
        r1 = max(-n_rings, -q - n_rings)
        r2 = min(n_rings, -q + n_rings)
        for r in range(r1, r2 + 1):
            s_ax = -q - r
            shell = max(abs(q), abs(r), abs(s_ax))
            x = side * 1.5 * q
            y = side * np.sqrt(3.0) * (r + q / 2.0)
            out.append((q, r, x, y, shell))
    return out


def plot_hex_overlay(mesh, R_inner_mm, R_outer_mm, filename, title=None,
                     wire=None, strand_diameter_mm=None):
    """Overlay the inner / outer homogenisation hexagons on the undeformed
    sub-element FE mesh of a single strand. When ``wire`` and
    ``strand_diameter_mm`` are supplied, the actual RRP sub-element pattern
    (Cu sleeves, Nb3Sn rings, Nb barrier, bronze cores) is drawn underneath
    so the homogenisation can be visually verified against the real
    geometry (cf. ``scripts/apdl/submodel/RVE/prep.py``).
    """
    from matplotlib.patches import Circle, RegularPolygon

    fig, ax = plt.subplots(figsize=(8, 8))

    # --- RRP sub-element pattern (optional, drawn underneath) ----------------
    if wire is not None and strand_diameter_mm is not None:
        try:
            d_core = float(wire["D_CORE_EQ_UM"])
            t_cu = float(wire["CU_SLEEVE_THICKNESS_UM"])
            n_total = int(wire["N_TOTAL"])
            n_nb3sn = int(wire["N_NB3SN"])
            t_nb = float(wire.get("NB_THICKNESS_UM", 2.0))
            bronze_frac = float(wire.get("BRONZE_FRAC", 0.55))
            n_cu_dummies = max(n_total - n_nb3sn, 0)
            n_shells = _shells_from_hex_count(n_total)
            try:
                cu_shells = _shells_from_hex_count(n_cu_dummies) if n_cu_dummies >= 1 else -1
                _is_cu_dummy = lambda q, r, shell, _cs=cu_shells: shell <= _cs
            except ValueError:
                # 78/91 layout: inner 7 (shell<=1) + 6 outer corner hexagons
                if n_cu_dummies != 13:
                    raise
                _is_cu_dummy = lambda q, r, shell, _ns=n_shells: (
                    shell <= 1 or (shell == _ns and (q == 0 or r == 0 or (-q - r) == 0))
                )

            d_outer_um = d_core + t_cu
            side_outer_um = np.sqrt(np.pi * d_outer_um**2 / (6.0 * np.sqrt(3.0)))
            side_nb_um = np.sqrt(np.pi * d_core**2 / (6.0 * np.sqrt(3.0)))
            side_inner_um = side_nb_um - t_nb * 2.0 / np.sqrt(3.0)
            bronze_r_um = np.sqrt(bronze_frac * (3.0 * np.sqrt(3.0) / 2.0
                                                  * side_inner_um**2) / np.pi)
            wire_r_mm = strand_diameter_mm / 2.0

            # All sub-element geometry in mm
            side_outer = side_outer_um * 1e-3
            side_nb = side_nb_um * 1e-3
            side_inner = max(side_inner_um, 0.0) * 1e-3
            bronze_r = bronze_r_um * 1e-3
            positions = _hex_center_positions(n_shells, side_outer)

            ax.add_patch(Circle((0, 0), wire_r_mm,
                                facecolor="#cc7733", edgecolor="none", zorder=1))
            for q, r, x, y, shell in positions:
                ax.add_patch(RegularPolygon((x, y), numVertices=6,
                                             radius=side_outer,
                                             orientation=np.pi / 6,
                                             facecolor="#b87333",
                                             edgecolor="none", zorder=2))
                if not _is_cu_dummy(q, r, shell) and side_inner > 0:
                    ax.add_patch(RegularPolygon((x, y), numVertices=6,
                                                 radius=side_nb,
                                                 orientation=np.pi / 6,
                                                 facecolor="#888888",
                                                 edgecolor="none", zorder=3))
                    ax.add_patch(RegularPolygon((x, y), numVertices=6,
                                                 radius=side_inner,
                                                 orientation=np.pi / 6,
                                                 facecolor="#3070d0",
                                                 edgecolor="none", zorder=4))
                    ax.add_patch(Circle((x, y), bronze_r,
                                        facecolor="#d4a017",
                                        edgecolor="none", zorder=5))
        except (KeyError, ValueError) as exc:
            print(f"plot_hex_overlay: skipping RRP overlay ({exc})")

    # --- FE sub-element mesh -------------------------------------------------
    # Rotate FE nodes by +30 deg in the plot only so the (flat-top) FE hex
    # regions visually align with the pointy-top analytical hexes / RRP
    # filaments. This does NOT modify the production mesh.
    _ca, _sa = np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0))
    _raw = np.array(mesh.nodes)
    nodes = np.column_stack((_ca * _raw[:, 0] - _sa * _raw[:, 1],
                             _sa * _raw[:, 0] + _ca * _raw[:, 1]))
    for elem in mesh.elements:
        pts = [nodes[i] for i in elem] + [nodes[elem[0]]]
        x, y = zip(*pts)
        ax.plot(x, y, color="white", linewidth=0.4, zorder=6, alpha=0.7)
    ax.scatter(nodes[:, 0], nodes[:, 1], s=1.5, color="white",
               zorder=7, alpha=0.8)

    # --- Hexagon outlines + strand circle -----------------------------------
    def hex_outline(R, angle_deg):
        # Pointy-top to match strandMeshGenerator FE hex (rotated 60-angle)
        # and the RRP sub-element pattern (orientation = pi/6).
        ang = np.deg2rad(60.0 * np.arange(7) - angle_deg + 30.0)
        return R * np.cos(ang), R * np.sin(ang)
    if R_inner_mm is not None:
        xi, yi = hex_outline(R_inner_mm, mesh.angle)
        ax.plot(xi, yi, color="red", linewidth=2.5, zorder=8,
                label=f"Inner hex  R={R_inner_mm*1e3:.1f} um")
    if R_outer_mm is not None:
        xo, yo = hex_outline(R_outer_mm, mesh.angle)
        ax.plot(xo, yo, color="cyan", linewidth=2.5, zorder=8,
                label=f"Outer hex  R={R_outer_mm*1e3:.1f} um")

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(mesh.radius * np.cos(theta), mesh.radius * np.sin(theta),
            color="black", linewidth=1.5, linestyle="--", zorder=8,
            label=f"Strand  R={mesh.radius*1e3:.0f} um")

    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def _bspline_samples(strand, num_points):
    """Memoised wrapper around strand.evaluate_bspline(num_points=...).

    The B-spline is fitted once per strand (fit_bspline) and then sampled
    repeatedly with the same num_points across every pairwise operation, so
    the samples are cached on the strand object, keyed by num_points. The
    cache is tied to the current spline_function tuple: re-running
    fit_bspline() replaces spline_function and invalidates the cache.
    Returned arrays are shared -- callers must not mutate them in place.
    """
    cache = getattr(strand, "_bspline_sample_cache", None)
    if cache is None or cache[0] is not strand.spline_function:
        cache = (strand.spline_function, {})
        strand._bspline_sample_cache = cache
    samples = cache[1]
    if num_points not in samples:
        samples[num_points] = strand.evaluate_bspline(num_points=num_points)
    return samples[num_points]


class ConformalRutherfordMesh:
    def __init__(self, strand1_file, strand2_file, hex_angle1=30,hex_angle2=30) -> None:
        """Initializes the conformal mesh process for two strands."""
        
        # Load and interpolate strands
        self.strand1 = DeformedStrandInterpolator(strand1_file)
        self.strand2 = DeformedStrandInterpolator(strand2_file)
        

        self.strand1.fit_bspline()
        self.strand2.fit_bspline()
        
        # Generate meshes (legacy entry point: keeps hardcoded ratios; the
        # JSON-driven path goes through run() and from_existing()).
        self.mesh1 = StrandMesh_Hexa(diameter=.6,radial_divisions=3,angle=hex_angle1,hex_scale=(149.12/425), hex_outer_scale=385.54/425)
        self.mesh2 = StrandMesh_Hexa(diameter=.6,radial_divisions=3,angle=hex_angle2,hex_scale=(149.12/425), hex_outer_scale=385.54/425)
        
        # Map meshes to deformed B-splines
        self.mapper1 = MeshMapping(self.mesh1, self.strand1)
        self.mapper2 = MeshMapping(self.mesh2, self.strand2)
        
        self.mapper1.translate_mesh_to_barycenter()
        self.mapper1.map_circumferential_layer_to_bspline()
        
        self.mapper2.translate_mesh_to_barycenter()
        self.mapper2.map_circumferential_layer_to_bspline()

    @classmethod
    def from_existing(cls, strand1, mapper1, strand2, mapper2):
        """Creates a ConformalRutherfordMesh from pre-built strands and mappers (no CSV re-reading)."""
        obj = cls.__new__(cls)
        obj.strand1 = strand1
        obj.strand2 = strand2
        obj.mapper1 = mapper1
        obj.mapper2 = mapper2
        obj.mesh1 = mapper1.mesh
        obj.mesh2 = mapper2.mesh
        return obj

    def identify_contact_region(self):
        """Identifies the contact region between the two B-splines and returns the start and end contact points.

        Detection rule (union of two tests, evaluated on the bspline1 sample
        ring of 200 points):
          (a) Distance test  -- min distance to bspline2 <= 15 um.
          (b) Penetration test -- bspline1 sample lies INSIDE the bspline2
              polygon (i.e. the strands actually overlap geometrically).

        The penetration test catches cases where deformed strands intersect by
        more than the B-spline sample stride (~10-15 um around the perimeter):
        in such cases (a) alone reports too few contact samples because the
        nearest-neighbour distance grows quickly as the sample moves into the
        other strand, even though the sample is clearly inside it.
        """
        bspline1 = _bspline_samples(self.strand1, num_points=200)
        bspline2 = _bspline_samples(self.strand2, num_points=200)

        # (a) Distance test
        tree2 = cKDTree(bspline2)
        min_distances, _ = tree2.query(bspline1)
        dist_mask = min_distances <= 0.015

        # (b) Penetration test: which bspline1 samples lie inside bspline2?
        # Ray-casting via matplotlib.path.Path is robust for closed polygons.
        from matplotlib.path import Path
        poly2 = Path(bspline2)
        inside_mask = poly2.contains_points(bspline1)

        contact_indices = np.where(dist_mask | inside_mask)[0]


        
        # print('contact_indices =', contact_indices)
        if contact_indices.size == 0:
            return None, None
        


        def sort_circular(nums, max_val=199):
            nums = sorted(set(nums))  # remove duplicates and sort
            # Find the largest gap (with wraparound)
            gaps = [((nums[(i + 1) % len(nums)] - nums[i]) % (max_val + 1)) for i in range(len(nums))]
            split_index = (gaps.index(max(gaps)) + 1) % len(nums)
            return nums[split_index:] + nums[:split_index]

        sorted_indices = sort_circular(contact_indices)

        # print('contact_indices',contact_indices)
        # print('sorted_indices',sorted_indices)
        
        
        start_point = bspline1[sorted_indices[0]]
        end_point = bspline1[sorted_indices[-1]]
        
        return start_point, end_point
        
    

    def find_next_node_index(self, current_idx, outer_layer, start_point, end_point):
        """Finds the correct next node index along the azimuthal direction, ensuring adjacency."""
        direction_vector = end_point - start_point
        # node_vectors = outer_layer - start_point
        node_vectors = outer_layer - outer_layer[current_idx]
        
        angles = np.arctan2(node_vectors[:, 1], node_vectors[:, 0])
        target_angle = np.arctan2(direction_vector[1], direction_vector[0])
        
        # Ensure the indices wrap correctly and follow the azimuthal order
        num_nodes = len(outer_layer)
        forward_idx = (current_idx + 1) % num_nodes
        backward_idx = (current_idx - 1) % num_nodes
        
        # Compute the difference in angles to determine direction
        forward_diff = abs(angles[forward_idx] - target_angle)
        backward_diff = abs(angles[backward_idx] - target_angle)
        
        # Choose the index that moves towards end_point along the azimuthal direction
        if forward_diff < backward_diff:
            return forward_idx
        else:
            return backward_idx
    
    def rotate_outer_layer_nodes(self):
        """Aligns outermost nodes in both strands to ensure the same number of nodes in the contact region."""
        start_point, end_point = self.identify_contact_region()
        if start_point is None or end_point is None:
            # print("No valid contact region found.")
            contact = False
            return contact

        # Diagnostic label set by caller (run(...) loop). Falls back to '?' if not set.
        pair_label = getattr(self, "_pair_label", "?")
        # List of human-readable reasons this pair tripped a debug-plot trigger.
        # The caller inspects this after rotate_outer_layer_nodes() returns and
        # writes a focused plot to plots/align_debug/ when non-empty.
        self._align_debug_reasons = []

        def _outer_dup_neighbours(mapper, idx):
            """Return list of neighbour offsets (in {-1,+1}) where the outer-ring
            entry at `idx` coincides with its neighbour. Empty list = no dup."""
            CD = mapper.mesh.circumferential_divisions
            total = mapper.mapped_nodes.shape[0]
            outer_start = total - CD
            # Normalise negative Python indices before computing local position
            if idx < 0:
                idx = total + idx
            local = idx - outer_start
            if local < 0 or local >= CD:
                return []
            this = mapper.mapped_nodes[idx]
            dups = []
            for d in (-1, +1):
                neighbour_local = (local + d) % CD
                neighbour_idx = outer_start + neighbour_local
                if np.allclose(this[:2], mapper.mapped_nodes[neighbour_idx][:2], atol=1e-10):
                    dups.append(d)
            return dups

        def assign(mapper, idx, value, branch_label, mapper_label):
            """Assign value to mapper.mapped_nodes[idx] and warn if it creates
            an intra-strand adjacent duplicate in the outer ring."""
            mapper.mapped_nodes[idx] = value
            dups = _outer_dup_neighbours(mapper, idx)
            if dups:
                CD = mapper.mesh.circumferential_divisions
                total = mapper.mapped_nodes.shape[0]
                outer_start = total - CD
                norm_idx = total + idx if idx < 0 else idx
                local = norm_idx - outer_start
                msg = (
                    f"INTRA-STRAND DUP pair={pair_label} mapper={mapper_label} "
                    f"branch='{branch_label}' outer_pos={local} "
                    f"neighbour_offsets={dups} "
                    f"coord=({value[0]:.10e},{value[1]:.10e})"
                )
                print(f"[align_nodes] {msg}")
                self._align_debug_reasons.append(msg)

        def align_nodes(mapper1, mapper2, start_point, end_point):
            # ----------------------------------------------------------
            # Robust outer-ring contact-node matching between two strands.
            #
            # Key invariants:
            #  * `outer_orig{1,2}` are SNAPSHOTS taken at entry; every
            #    geometric search reads from the snapshot so iterations
            #    are independent of intermediate writes.
            #  * Endpoints of the contact arc are PAIRED REAL NODES, not
            #    B-spline midpoints.  start_idx/end_idx are real outer-
            #    ring indices on each strand; their values are set to the
            #    average of the two existing node positions.
            #  * Promotion: when shorter side has 1 node OR the count
            #    delta is >=2, the shorter arc is extended by exactly one
            #    real outer-ring neighbour so both endpoints can be
            #    real-node pairs.  Examples: 1-vs-2 -> 2-vs-2,
            #    2-vs-4 -> 3-vs-4, 3-vs-6 -> 4-vs-6.
            #  * A swap-fix re-pairs strand-2 endpoints if argmin produced
            #    a crossing pairing (the 2-vs-2 bug).
            #  * Wrap-around uses modular arithmetic everywhere; no
            #    hard-coded 0/35 special cases.
            # ----------------------------------------------------------
            CD = mapper1.mesh.circumferential_divisions
            assert mapper2.mesh.circumferential_divisions == CD, (
                f"CD mismatch: m1={CD} m2={mapper2.mesh.circumferential_divisions}")

            # `outer_layer{1,2}` are views (kept for any later observers);
            # `outer_orig{1,2}` are immutable snapshots used for all geometry.
            outer_layer1 = mapper1.mapped_nodes[-CD:]
            outer_layer2 = mapper2.mapped_nodes[-CD:]
            outer_orig1 = outer_layer1.copy()
            outer_orig2 = outer_layer2.copy()

            bspline1 = _bspline_samples(mapper1.deformed_strand, num_points=200)
            bspline2 = _bspline_samples(mapper2.deformed_strand, num_points=200)

            num_nodes1 = num_nodes2 = CD

            # All endpoint indices computed from the ORIGINAL outer rings,
            # using the contact-region samples returned by identify_contact_region.
            start_idx1 = int(np.argmin(np.linalg.norm(outer_orig1 - start_point, axis=1)))
            end_idx1 = int(np.argmin(np.linalg.norm(outer_orig1 - end_point, axis=1)))
            start_idx2 = int(np.argmin(np.linalg.norm(outer_orig2 - start_point, axis=1)))
            end_idx2 = int(np.argmin(np.linalg.norm(outer_orig2 - end_point, axis=1)))

            # ---- Swap-fix (handles the 2-vs-2 crossing-lines bug) ----
            # argmin alone may pair start_point with the strand-2 node that
            # geometrically corresponds to end_point and vice versa.  Pick
            # the strand-2 endpoint pairing that minimises the sum of
            # node-pair distances.
            direct = (np.linalg.norm(outer_orig1[start_idx1] - outer_orig2[start_idx2]) +
                      np.linalg.norm(outer_orig1[end_idx1] - outer_orig2[end_idx2]))
            swapped = (np.linalg.norm(outer_orig1[start_idx1] - outer_orig2[end_idx2]) +
                       np.linalg.norm(outer_orig1[end_idx1] - outer_orig2[start_idx2]))
            if swapped + 1e-12 < direct:
                start_idx2, end_idx2 = end_idx2, start_idx2
                msg = (f"swap-fix pair={pair_label} "
                       f"direct={direct:.6e} swapped={swapped:.6e}")
                print(f"[align_nodes] {msg}")
                self._align_debug_reasons.append(msg)

            # Path-length analysis -> traversal direction & contact-node count.
            forward_distance1 = (end_idx1 - start_idx1) % num_nodes1
            backward_distance1 = (start_idx1 - end_idx1) % num_nodes1
            forward_distance2 = (end_idx2 - start_idx2) % num_nodes2
            backward_distance2 = (start_idx2 - end_idx2) % num_nodes2

            direction1 = 1 if forward_distance1 < backward_distance1 else -1
            direction2 = 1 if forward_distance2 < backward_distance2 else -1

            # Inclusive count of nodes traversed (start..end along chosen direction).
            num_points1 = min(forward_distance1, backward_distance1) + 1
            num_points2 = min(forward_distance2, backward_distance2) + 1

            # ---- Promotion: extend the shorter contact arc by one real
            # outer-ring node so that BOTH endpoints can be paired with
            # existing nodes (not B-spline midpoints).
            #
            # Trigger:
            #   * shorter == 1 (singleton has no distinct start/end), or
            #   * |delta| >= 2  (e.g. 2-vs-4 -> 3-vs-4, 3-vs-6 -> 4-vs-6).
            # NOT triggered when shorter >= 2 and |delta| == 1, because the
            # change1/change2 mechanism already handles that cleanly with
            # both endpoints already on real nodes.
            #
            # Direction: extend on whichever side (start or end) brings the
            # newly-included shorter-side node geometrically closest to the
            # corresponding longer-side endpoint.
            short_n = min(num_points1, num_points2)
            long_n = max(num_points1, num_points2)
            if short_n != long_n and (short_n == 1 or long_n - short_n >= 2):
                if num_points1 < num_points2:
                    s_idx, e_idx, s_dir = start_idx1, end_idx1, direction1
                    cand_start = (s_idx - s_dir) % CD
                    cand_end = (e_idx + s_dir) % CD
                    d_start = np.linalg.norm(outer_orig1[cand_start] - outer_orig2[start_idx2])
                    d_end = np.linalg.norm(outer_orig1[cand_end] - outer_orig2[end_idx2])
                    if d_start <= d_end:
                        start_idx1 = cand_start
                        which = "start"
                    else:
                        end_idx1 = cand_end
                        which = "end"
                    num_points1 += 1
                    msg = (f"promote pair={pair_label} m1 {num_points1-1}->{num_points1} "
                           f"side={which} (n2={num_points2})")
                else:
                    s_idx, e_idx, s_dir = start_idx2, end_idx2, direction2
                    cand_start = (s_idx - s_dir) % CD
                    cand_end = (e_idx + s_dir) % CD
                    d_start = np.linalg.norm(outer_orig2[cand_start] - outer_orig1[start_idx1])
                    d_end = np.linalg.norm(outer_orig2[cand_end] - outer_orig1[end_idx1])
                    if d_start <= d_end:
                        start_idx2 = cand_start
                        which = "start"
                    else:
                        end_idx2 = cand_end
                        which = "end"
                    num_points2 += 1
                    msg = (f"promote pair={pair_label} m2 {num_points2-1}->{num_points2} "
                           f"side={which} (n1={num_points1})")
                print(f"[align_nodes] {msg}")
                self._align_debug_reasons.append(msg)

            # ---- Real-node endpoint snap ----
            # Both endpoints are placed at the AVERAGE of the two existing
            # outer-ring node positions on the (possibly promoted) contact
            # arc -- never at a synthesised B-spline projection.
            start_point = 0.5 * (outer_orig1[start_idx1] + outer_orig2[start_idx2])
            end_point = 0.5 * (outer_orig1[end_idx1] + outer_orig2[end_idx2])

            # ---- Narrow-arc collapse to single point ----
            # If the contact arc is much smaller than a single outer-ring
            # node spacing, the two endpoint-pair averages collapse to
            # nearly the same coordinates and the result is two adjacent
            # near-coincident nodes on each strand (1-2 um apart). In that
            # case the contact is geometrically a point: drop end_idx and
            # write only the start pair (1-vs-1).
            #
            # Spacing reference: nominal outer-ring step on either strand.
            spacing1 = float(np.linalg.norm(
                outer_orig1[(start_idx1 + 1) % CD] - outer_orig1[start_idx1]))
            spacing2 = float(np.linalg.norm(
                outer_orig2[(start_idx2 + 1) % CD] - outer_orig2[start_idx2]))
            ref_spacing = 0.5 * (spacing1 + spacing2)
            arc_dist = float(np.linalg.norm(start_point - end_point))
            if (start_idx1 != end_idx1 or start_idx2 != end_idx2) and \
               arc_dist < 0.5 * ref_spacing:
                # Pick the geometrically-closest pair across the two candidate
                # endpoint pairings, snap both strands to it, drop the other.
                d_ss = np.linalg.norm(outer_orig1[start_idx1] - outer_orig2[start_idx2])
                d_ee = np.linalg.norm(outer_orig1[end_idx1] - outer_orig2[end_idx2])
                if d_ee < d_ss:
                    start_idx1, start_idx2 = end_idx1, end_idx2
                    start_point = end_point
                end_idx1 = start_idx1
                end_idx2 = start_idx2
                num_points1 = 1
                num_points2 = 1
                msg = (f"narrow-arc-collapse pair={pair_label} arc={arc_dist*1e3:.3f} um "
                       f"ref_spacing={ref_spacing*1e3:.1f} um")
                print(f"[align_nodes] {msg}")
                self._align_debug_reasons.append(msg)

            # First writes: coincident start endpoints on both strands.
            assign(mapper1, -CD + start_idx1, start_point, "start_point", "m1")
            assign(mapper2, -CD + start_idx2, start_point, "start_point", "m2")

            current_idx1 = start_idx1
            current_idx2 = start_idx2

            change1 = change2 = change11 = change22 = False

            # ---- Imbalance flags (computed AFTER promotion) ----
            if num_points1 != num_points2:
                if num_points1 == num_points2 + 1:
                    change1 = True
                elif num_points1 > num_points2 + 1:
                    change11 = True
                elif num_points2 == num_points1 + 1:
                    change2 = True
                else:
                    change22 = True

            # ---- Main midpoint loop (equal counts and Delta=1) ----
            saved_current_idx1 = start_idx1
            saved_current_idx2 = start_idx2
            saved_last_idx1 = end_idx1
            saved_last_idx2 = end_idx2

            march_iter = 0
            while not (change11 or change22):
                march_iter += 1
                if march_iter > 10000:
                    msg = (f"iteration-cap pair={pair_label} "
                           f"midpoint loop exceeded 10000 iterations")
                    print(f"[align_nodes] {msg}")
                    self._align_debug_reasons.append(msg)
                    break
                next_idx1 = (current_idx1 + direction1) % num_nodes1
                next_idx2 = (current_idx2 + direction2) % num_nodes2

                saved_current_idx1 = current_idx1
                saved_current_idx2 = current_idx2

                # 1-vs-1: nothing in between to write.
                if num_points1 == 1 and num_points2 == 1:
                    break
                # Stop before assigning a non-endpoint write at end_idx.
                if next_idx1 == end_idx1 or next_idx2 == end_idx2:
                    break

                # Project from the snapshot, not the live array.
                p1 = bspline1[np.argmin(np.linalg.norm(bspline1 - outer_orig1[next_idx1], axis=1))]
                p2 = bspline2[np.argmin(np.linalg.norm(bspline2 - outer_orig2[next_idx2], axis=1))]
                midpoint = 0.5 * (p1 + p2)

                # Bidirectional coincidence check on BOTH mappers.
                skip_assign = False
                for _mapper, _next, _label in ((mapper1, next_idx1, "m1"),
                                               (mapper2, next_idx2, "m2")):
                    for _d in (-1, +1):
                        _nb_pos = (_next + _d) % CD
                        _nb_val = _mapper.mapped_nodes[-CD + _nb_pos]
                        if np.allclose(midpoint[:2], _nb_val[:2], atol=1e-9):
                            msg = (f"midpoint_loop_skip_coincident pair={pair_label} "
                                   f"mapper={_label} pos={_next} coincides_with={_nb_pos}")
                            print(f"[align_nodes] {msg}")
                            self._align_debug_reasons.append(msg)
                            skip_assign = True
                            break
                    if skip_assign:
                        break

                if not skip_assign:
                    assign(mapper1, -CD + next_idx1, midpoint, "midpoint_loop", "m1")
                    assign(mapper2, -CD + next_idx2, midpoint, "midpoint_loop", "m2")

                current_idx1 = next_idx1
                current_idx2 = next_idx2

            # End-point writes (coincident).
            assign(mapper1, -CD + end_idx1, end_point, "end_point", "m1")
            assign(mapper2, -CD + end_idx2, end_point, "end_point", "m2")

            # ---- Delta=1 imbalance correction ----
            # Fill the orphan slot on the longer side with the average of
            # the two nearest slots on the shorter side. Modular wrap, no
            # 0/35 magic numbers.
            if change1:
                avg = 0.5 * (mapper2.mapped_nodes[-CD + saved_current_idx2] +
                             mapper2.mapped_nodes[-CD + saved_last_idx2])
                target_local = (end_idx1 - direction1) % CD
                if not np.allclose(avg[:2], end_point[:2], atol=1e-9):
                    assign(mapper1, -CD + target_local, avg, "change1", "m1")
                else:
                    self._align_debug_reasons.append(
                        f"change1_skip_coincident pair={pair_label} pos={target_local}")
            if change2:
                avg = 0.5 * (mapper1.mapped_nodes[-CD + saved_current_idx1] +
                             mapper1.mapped_nodes[-CD + saved_last_idx1])
                target_local = (end_idx2 - direction2) % CD
                if not np.allclose(avg[:2], end_point[:2], atol=1e-9):
                    assign(mapper2, -CD + target_local, avg, "change2", "m2")
                else:
                    self._align_debug_reasons.append(
                        f"change2_skip_coincident pair={pair_label} pos={target_local}")

            # ---- |Delta|>=2 sub-loop ----
            if change11 or change22:
                temp_idx1 = current_idx1
                temp_idx2 = current_idx2
                march_iter = 0
                while True:
                    march_iter += 1
                    if march_iter > 10000:
                        msg = (f"iteration-cap pair={pair_label} "
                               f"delta-ge-2 loop exceeded 10000 iterations")
                        print(f"[align_nodes] {msg}")
                        self._align_debug_reasons.append(msg)
                        break
                    next_idx1 = (current_idx1 + direction1) % num_nodes1
                    if change11:
                        temp_idx1 = next_idx1
                        next_idx1 = (next_idx1 + direction1) % num_nodes1

                    next_idx2 = (current_idx2 + direction2) % num_nodes2
                    if change22:
                        temp_idx2 = next_idx2
                        next_idx2 = (next_idx2 + direction2) % num_nodes2

                    saved_current_idx1 = current_idx1
                    saved_current_idx2 = current_idx2

                    if next_idx1 == end_idx1 or next_idx2 == end_idx2:
                        break

                    p1 = bspline1[np.argmin(np.linalg.norm(bspline1 - outer_orig1[next_idx1], axis=1))]
                    p2 = bspline2[np.argmin(np.linalg.norm(bspline2 - outer_orig2[next_idx2], axis=1))]
                    midpoint = 0.5 * (p1 + p2)

                    assign(mapper1, -CD + next_idx1, midpoint, "midpoint_loop_11_22", "m1")
                    assign(mapper2, -CD + next_idx2, midpoint, "midpoint_loop_11_22", "m2")

                    if change11:
                        first = mapper2.mapped_nodes[-CD + saved_current_idx2]
                        second = mapper2.mapped_nodes[-CD + next_idx2]
                        assign(mapper1, -CD + temp_idx1, 0.5 * (first + second),
                               "change11_loop", "m1")
                    if change22:
                        first = mapper1.mapped_nodes[-CD + saved_current_idx1]
                        second = mapper1.mapped_nodes[-CD + next_idx1]
                        assign(mapper2, -CD + temp_idx2, 0.5 * (first + second),
                               "change22_loop", "m2")

                    current_idx1 = next_idx1
                    current_idx2 = next_idx2

                # End-point writes (coincident).
                assign(mapper1, -CD + end_idx1, end_point, "end_point_11_22", "m1")
                assign(mapper2, -CD + end_idx2, end_point, "end_point_11_22", "m2")

                # Final orphan write for the temp slot on the longer side.
                if change11:
                    first = mapper2.mapped_nodes[-CD + saved_current_idx2]
                    second = mapper2.mapped_nodes[-CD + next_idx2]
                    assign(mapper1, -CD + temp_idx1, 0.5 * (first + second),
                           "change11_post", "m1")
                if change22:
                    first = mapper1.mapped_nodes[-CD + saved_current_idx1]
                    second = mapper1.mapped_nodes[-CD + next_idx1]
                    assign(mapper2, -CD + temp_idx2, 0.5 * (first + second),
                           "change22_post", "m2")

            # ---- Post-condition: verify endpoint coincidence (warn-only) ----
            sp_dist = np.linalg.norm(mapper1.mapped_nodes[-CD + start_idx1][:2] -
                                     mapper2.mapped_nodes[-CD + start_idx2][:2])
            ep_dist = np.linalg.norm(mapper1.mapped_nodes[-CD + end_idx1][:2] -
                                     mapper2.mapped_nodes[-CD + end_idx2][:2])
            if sp_dist > 1e-10 or ep_dist > 1e-10:
                msg = (f"POSTCOND pair={pair_label} endpoint-pair distances: "
                       f"start={sp_dist:.3e} end={ep_dist:.3e}")
                print(f"[align_nodes] {msg}")
                self._align_debug_reasons.append(msg)
                



        
        # align_nodes(self.mapper1, self.mapper2, self.mapper3, start_point, end_point)
        align_nodes(self.mapper1, self.mapper2, start_point, end_point)
        
        
        
        contact = True    #used later for plotting the contact region
        return contact

    def plot_conformal_mesh(self, filename):
        """Plots the B-splines, mapped meshes, and elements for both strands, with node indices for the external layer."""
        

        plt.figure(figsize=(15, 8))
        
        # Plot B-spline interpolation for both strands
        bspline1 = _bspline_samples(self.strand1, num_points=200)
        bspline2 = _bspline_samples(self.strand2, num_points=200)
        
        
        plt.plot(bspline1[:, 0], bspline1[:, 1], color='blue', linestyle='--', label='B-Spline Strand 1')
        plt.plot(bspline2[:, 0], bspline2[:, 1], color='green', linestyle='--', label='B-Spline Strand 2')
        
        # Plot mapped meshes with elements
        mapped_nodes1 = self.mapper1.mapped_nodes
        mapped_nodes2 = self.mapper2.mapped_nodes
        
        if mapped_nodes1 is not None:
            plt.scatter(mapped_nodes1[:, 0], mapped_nodes1[:, 1], color='red', label='Mapped Mesh 1', s=5)
            for i, elem in enumerate(self.mesh1.elements):
                elem_nodes = [mapped_nodes1[j] for j in elem] + [mapped_nodes1[elem[0]]]
                x_e, y_e = zip(*elem_nodes)
                plt.plot(x_e, y_e, color='black', linewidth=0.5)
            # Plot external layer node indices
            outer_layer1 = mapped_nodes1[-self.mesh1.circumferential_divisions:]
            for i, node in enumerate(outer_layer1):
                plt.text(node[0], node[1], str(i), color='red', fontsize=8)
        
        if mapped_nodes2 is not None:
            plt.scatter(mapped_nodes2[:, 0], mapped_nodes2[:, 1], color='purple', label='Mapped Mesh 2', s=5)
            for i, elem in enumerate(self.mesh2.elements):
                elem_nodes = [mapped_nodes2[j] for j in elem] + [mapped_nodes2[elem[0]]]
                x_e, y_e = zip(*elem_nodes)
                plt.plot(x_e, y_e, color='black', linewidth=0.5)
            # Plot external layer node indices
            outer_layer2 = mapped_nodes2[-self.mesh2.circumferential_divisions:]
            for i, node in enumerate(outer_layer2):
                plt.text(node[0], node[1], str(i), color='purple', fontsize=8)
        
        
        plt.xlabel("x [mm]]")
        plt.ylabel("y [mm]")
        plt.title("Conformal Rutherford Mesh: B-Splines, Meshes, and Elements with External Layer Indices")
        plt.legend()
        plt.axis('equal')


        # Save the plot as a svg file
        plt.savefig(filename)
        # plt.show()
        plt.close()

    def plot_outer_nodes_and_bsplines(self, filename):
        """Plots only the B-splines and outer layer nodes for both strands, with node indices and outlines."""
        
        plt.figure(figsize=(6, 4))
        
        # Plot B-spline interpolation for both strands
        bspline1 = _bspline_samples(self.strand1, num_points=100)
        bspline2 = _bspline_samples(self.strand2, num_points=100)
        
        plt.plot(bspline1[:, 0], bspline1[:, 1], color='blue', linestyle='--', label='B-Spline Strand 1')
        plt.plot(bspline2[:, 0], bspline2[:, 1], color='green', linestyle='--', label='B-Spline Strand 2')
        
        # Plot only outer layer nodes
        mapped_nodes1 = self.mapper1.mapped_nodes
        mapped_nodes2 = self.mapper2.mapped_nodes
        
        if mapped_nodes1 is not None:
            outer_layer1 = mapped_nodes1[-self.mesh1.circumferential_divisions:]
            plt.scatter(outer_layer1[:, 0], outer_layer1[:, 1], color='blue', label='Outer Nodes 1', s=20)
            # Draw outline connecting the outer nodes
            outline_nodes1 = np.vstack([outer_layer1, outer_layer1[0]])  # Close the loop
            plt.plot(outline_nodes1[:, 0], outline_nodes1[:, 1], color='black', linewidth=2, alpha=1, label='New Outline 1')
            # Plot external layer node indices
            for i, node in enumerate(outer_layer1):
                plt.text(node[0], node[1], str(i), color='black', fontsize=8)
        
        if mapped_nodes2 is not None:
            outer_layer2 = mapped_nodes2[-self.mesh2.circumferential_divisions:]
            plt.scatter(outer_layer2[:, 0], outer_layer2[:, 1], color='green', label='Outer Nodes 2', s=20)
            # Draw outline connecting the outer nodes
            outline_nodes2 = np.vstack([outer_layer2, outer_layer2[0]])  # Close the loop
            plt.plot(outline_nodes2[:, 0], outline_nodes2[:, 1], color='black', linewidth=2, alpha=1, label='New Outline 2')
            # Plot external layer node indices
            for i, node in enumerate(outer_layer2):
                plt.text(node[0], node[1], str(i), color='black', fontsize=8)
        
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.title("B-Splines and Outer Layer Nodes")
        plt.legend()
        plt.axis('equal')
        
        # Save the plot as a svg file
        plt.savefig(filename)
        # plt.show()
        plt.close()
        
    # def write_mesh_one_go(self, filename_nodes, filename_elements, saved_strands, saved_mappers, stack_nr, stack_height,square_size=72,e_per_layer=36,radial_layers_nb=8,radial_layers_cu=3):
    #     """Writes the mesh nodes and elements to a file with the specified format, using saved strands and mappers."""
    #     node_offset_stacknr = (stack_nr-1)*1e5
    #     node_offset = 0
        
        
        
    #     elem_strand_offset_increase = 10000
    #     elem_strand_offset_strack_nr = (stack_nr-1)*elem_strand_offset_increase*100
    #     elem_strand_offset = elem_strand_offset_strack_nr + 1
        
    #     with open(filename_nodes, 'w') as f_nodes, open(filename_elements, 'w') as f_elements:
    #         for i, (strand, mapper) in enumerate(zip(saved_strands, saved_mappers)):
    #             if strand is None or mapper is None:
    #                 continue
                    
    #             # Write nodes for the current strand
    #             for j, (x, y) in enumerate(mapper.mapped_nodes):
    #                 f_nodes.write(f"N, {j + 1 + node_offset + node_offset_stacknr}, {(x)/1000:.6e}, {(y+-(1-stack_nr)*stack_height)/1000:.6e}, 0.0\n")
                    
    #             # Write elements for the current strand
                
    #             counter_write = 0

                
    #             for elem in mapper.mesh.elements:
    #                 if counter_write == 0:
    #                     f_elements.write(f"NUMSTR,ELEM,{elem_strand_offset} \n")
    #                     f_elements.write(f'mat,2 \n')
    #                 elif counter_write == square_size:
    #                     f_elements.write(f'mat,3 \n')
    #                 elif counter_write == square_size + e_per_layer*radial_layers_nb:
    #                     f_elements.write(f'mat,2 \n')
                        
    #                 n1 = elem[0] + 1 + node_offset + node_offset_stacknr
    #                 n2 = elem[1] + 1 + node_offset + node_offset_stacknr
    #                 n3 = elem[2] + 1 + node_offset + node_offset_stacknr
    #                 n4 = elem[3] + 1 + node_offset + node_offset_stacknr
    #                 if elem[3] == elem[0]:
    #                     f_elements.write(f"E, {n1}, {n2}, {n3}\n")
    #                 else:
    #                     f_elements.write(f"E, {n1}, {n2}, {n3}, {n4}\n")
                        
                    
    #                 counter_write += 1
    #                 if counter_write == square_size + (radial_layers_cu + radial_layers_nb) * e_per_layer:
    #                     f_elements.write(f"EREFINE, {elem_strand_offset}, {elem_strand_offset+elem_strand_offset_increase-1}, 1, 1,,, \n")
    #                     elem_strand_offset = elem_strand_offset + elem_strand_offset_increase
    #                     counter_write = 0
                    
    #             node_offset += len(mapper.mapped_nodes)
    #             f_nodes.write(f"\n!New part\n")
    #             f_elements.write(f"\n!New part\n")

    def write_mesh_one_go(self, filename_nodes, filename_elements, saved_strands, saved_mappers, stack_nr, stack_height, stacking=True, square_size=72,e_per_layer=36,radial_layers_nb=8,radial_layers_cu=3):
        """Writes the mesh nodes and elements to a file with the specified format, using saved strands and mappers."""
        node_offset_stacknr = (stack_nr-1)*2e4  # Reduced from 1e5 (nodes are separate entity space)
        node_offset = 0
        
        # Use stack_height for vertical offset only if stacking is enabled
        vertical_offset = stack_height if stacking else 0
        
        elem_strand_offset_increase = 10000
        elem_strand_offset_strack_nr = (stack_nr-1)*elem_strand_offset_increase*55  # Reduced from *100
        elem_strand_offset = elem_strand_offset_strack_nr + 1
        
        with open(filename_nodes, 'w') as f_nodes, open(filename_elements, 'w') as f_elements:
            for i, (strand, mapper) in enumerate(zip(saved_strands, saved_mappers)):
                if strand is None or mapper is None:
                    continue
                    
                # Write nodes for the current strand
                for j, (x, y) in enumerate(mapper.mapped_nodes):
                    f_nodes.write(f"N, {j + 1 + node_offset + node_offset_stacknr}, {(x)/1000:.6e}, {(y+-(1-stack_nr)*vertical_offset)/1000:.6e}, 0.0\n")
                    
                # Write elements for the current strand
                
                counter_write = 0

                
                for elem in mapper.mesh.elements:
                    if counter_write == 0:
                        f_elements.write(f"NUMSTR,ELEM,{elem_strand_offset} \n")
                        f_elements.write(f'mat,2 \n')
                    elif counter_write == square_size:
                        f_elements.write(f'mat,3 \n')
                    elif counter_write == square_size + e_per_layer*radial_layers_nb:
                        f_elements.write(f'mat,2 \n')
                        
                    n1 = elem[0] + 1 + node_offset + node_offset_stacknr
                    n2 = elem[1] + 1 + node_offset + node_offset_stacknr
                    n3 = elem[2] + 1 + node_offset + node_offset_stacknr
                    n4 = elem[3] + 1 + node_offset + node_offset_stacknr
                    if elem[3] == elem[0]:
                        f_elements.write(f"E, {n1}, {n2}, {n3}\n")
                    else:
                        f_elements.write(f"E, {n1}, {n2}, {n3}, {n4}\n")
                        
                    
                    counter_write += 1
                    if counter_write == square_size + (radial_layers_cu + radial_layers_nb) * e_per_layer:
                        f_elements.write(f"EREFINE, {elem_strand_offset}, {elem_strand_offset+elem_strand_offset_increase-1}, 1, 1,,, \n")
                        elem_strand_offset = elem_strand_offset + elem_strand_offset_increase
                        counter_write = 0
                    
                node_offset += len(mapper.mapped_nodes)
                f_nodes.write(f"\n!New part\n")
                f_elements.write(f"\n!New part\n")


    def write_impregnation_keypoints(self, connections_file, saved_mappers, stack_nr, stack_height, stacking=True, output_dir=None):
        """Writes the area in between the strands as keypoints for the epoxy region."""
        node_offset_stacknr = (stack_nr-1)*3e4  # Reduced from 1e5 to match APDL stack_offset
        
        # Use stack_height for vertical offset only if stacking is enabled
        vertical_offset = stack_height if stacking else 0
        
        # Read the connections file
        connections_df = pd.read_csv(connections_file)
        
        # Function to find all cycles of a given length
        def find_cycles_of_length(graph, length):
            cycles = []
            for cycle in nx.simple_cycles(graph):
                if len(cycle) == length:
                    cycles.append(cycle)
            return cycles

        def filter_quads_by_triangles(loops_3, loops_4):
            """
            Filters out 4-loops (quads) that contain any of the 3-loops (triangles) as subsets.

            Args:
                loops_3 (list of tuples): 3-node cycles.
                loops_4 (list of tuples): 4-node cycles.

            Returns:
                list of tuples: Filtered 4-node cycles (quads) not containing any triangle.
            """
            triangle_sets = {frozenset(t) for t in loops_3}
            filtered_quads = []

            for quad in loops_4:
                quad_set = set(quad)
                # Check all 3-element subsets of this quad
                if not any(frozenset(sub) in triangle_sets for sub in combinations(quad, 3)):
                    filtered_quads.append(quad)

            return filtered_quads
            
        def find_overlapping_coords(arr1, arr2, threshold):
            overlapping_coords = set()
            tree2 = cKDTree(arr2)
            # query_ball_point is inclusive (<= threshold); the per-pair norm
            # check below keeps the original strict < threshold semantics.
            for i, neighbours in enumerate(tree2.query_ball_point(arr1, threshold)):
                coord1 = arr1[i]
                for j in neighbours:
                    coord2 = arr2[j]
                    if np.linalg.norm(coord1 - coord2) < threshold:
                        overlapping_coords.add(tuple(coord1))
                        overlapping_coords.add(tuple(coord2))
            if not overlapping_coords:
                return False
            return overlapping_coords
        
        # Find the point in overlapping_coords_x1_x2 that is closest to any point in overlapping_coords_x2_x3 and overlapping_coords_x1_x3
        def find_closest_point(source_set, target_sets):
            min_distance = float('inf')
            closest_point = None
            target_trees = []
            for target_set in target_sets:
                target_points = np.array(list(target_set))
                if len(target_points) > 0:
                    target_trees.append(cKDTree(target_points))
            for point in source_set:
                point_arr = np.array(point)
                for tree in target_trees:
                    distance, _ = tree.query(point_arr)
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = point
            return closest_point
        
        def find_closest_point_index(array, point, threshold): #Find the index of the closest point in the array
            return np.where(np.all(np.abs(array - point) < threshold, axis=1))[0][0]


        def calculate_indices_between_good(index_a, index_b, array_length):
            if index_a > index_b:
                index_a, index_b = index_b, index_a
            
            #so index_b is always greater than index_a
            indexlist = []
            
            if index_b - index_a > array_length/2:
                indexlist.append(index_b)
                for i in range(1, array_length):
                    next_index = (index_b + i) % array_length
                    indexlist.append(next_index)
                    if next_index == index_a:
                        break    

            else:
                indexlist.append(index_a)
                for i in range(1, array_length):
                    next_index = (index_a + i) % array_length
                    indexlist.append(next_index)
                    if next_index == index_b:
                        break    
                    
            return indexlist
            

        def calculate_indices_between(index_a, index_b, array_length):
            if index_a > index_b:
                if (index_a - index_b) < array_length / 2:
                    indices_between = np.linspace(index_b, index_a + 1, num=abs(index_a - index_b) + 1, endpoint=False, dtype=int)
                else:
                    indices_between_first = np.linspace(index_a, array_length, num=abs(array_length - index_a), endpoint=False, dtype=int)
                    indices_between_second = np.linspace(0, index_b, num=abs(index_b+1), endpoint=False, dtype=int)
                    indices_between = np.concatenate((indices_between_first, indices_between_second))
            elif index_a < index_b:
                if (index_b - index_a) < array_length / 2:
                    indices_between = np.linspace(index_a, index_b + 1, num=abs(index_b - index_a), endpoint=False, dtype=int)
                else:
                    indices_between = np.linspace(index_b, index_a + 1, num=abs(index_b - index_a), endpoint=False, dtype=int)
            else:
                indices_between = np.array([index_a])
            return indices_between
        
        
        # Find loops of size 3-4
        
        # Create a graph from the dataframe
        G = nx.from_pandas_edgelist(connections_df, 'Connection_a', 'Connection_b')

        
        cycles_of_size_3 = find_cycles_of_length(G, 3)
        cycles_of_size_4 = find_cycles_of_length(G, 4)
        

        filtered_cycles_4 = filter_quads_by_triangles(cycles_of_size_3, cycles_of_size_3)

    
        # # Display the loops
        # print("Length of cycles of size 3:", len(cycles_of_size_3))
        # print("Length of cycles of size 4:", len(cycles_of_size_4))
        # print("Length of filtered cycles of size 4:", len(filtered_cycles_4))
        

        
        cycles_of_size_4 = filtered_cycles_4 #prevent writing cycles of 4 that are also cycles of 3.
        
        outer_layer1 = []
        outer_layer2 = []
        outer_layer3 = []
        
        sorted_unique_coords = []
        # Loops of three
        threshold = 1e-2 #For determining node overlap
        j_index = 1 #used for naming the keypoints apdl
        # Initialize the keypoints file
        _kp_file = (str(output_dir / f"keypoints_{stack_nr}.txt") if output_dir is not None else f"keypoints_{stack_nr}.txt")
        with open(_kp_file, "w") as f:
            f.write("! Keypoints for impregnation area\n")
        j_index_old = j_index
        for i in range(len(cycles_of_size_3)):
            
            # print('first strand', cycles_of_size_3[i][0], 'second strand', cycles_of_size_3[i][1], 'third strand', cycles_of_size_3[i][2])
            mapper1 = saved_mappers[cycles_of_size_3[i][0]-1] # correct the index
            mapper2 = saved_mappers[cycles_of_size_3[i][1]-1]
            mapper3 = saved_mappers[cycles_of_size_3[i][2]-1]
            
            outer_layer1.append(mapper1.mapped_nodes[-mapper1.mesh.circumferential_divisions:])
            outer_layer2.append(mapper2.mapped_nodes[-mapper2.mesh.circumferential_divisions:])
            outer_layer3.append(mapper3.mapped_nodes[-mapper3.mesh.circumferential_divisions:])
            
            outer_1np = np.array(outer_layer1[i])
            outer_2np = np.array(outer_layer2[i])
            outer_3np = np.array(outer_layer3[i])
            

            overlapping_coords_1_2 = find_overlapping_coords(outer_1np, outer_2np, threshold) #Find overlapping coordinates between part 1 and 2
            overlapping_coords_2_3 = find_overlapping_coords(outer_2np, outer_3np, threshold) #Find overlapping coordinates between part 2 and 3
            overlapping_coords_1_3 = find_overlapping_coords(outer_1np, outer_3np, threshold) #Find overlapping coordinates between part 1 and 3
            

            
            closest_point_1_2 = find_closest_point(overlapping_coords_1_2, [overlapping_coords_2_3, overlapping_coords_1_3]) #Closest point on intersection of part 1 and 2
            closest_point_2_3 = find_closest_point(overlapping_coords_2_3, [overlapping_coords_1_2, overlapping_coords_1_3]) #Closest point on intersection of part 2 and 3
            closest_point_1_3 = find_closest_point(overlapping_coords_1_3, [overlapping_coords_1_2, overlapping_coords_2_3]) #Closest point on intersection of part 1 and 3


            # Find the indices of the edge in the x1 array
            closest_point_1_2_index = find_closest_point_index(outer_1np, closest_point_1_2, threshold)
            closest_point_1_3_index = find_closest_point_index(outer_1np, closest_point_1_3, threshold)

            index_a = closest_point_1_2_index
            index_b = closest_point_1_3_index

            # print("Index of a in x1_array:", index_a)
            # print("Index of b in x1_array:", index_b)

            indices_between_1 = calculate_indices_between_good(index_a, index_b, len(outer_1np))
            # print('indices between a and b: for 1', indices_between_1)


            # Find the indices of the edge in the x2 array
            closest_point_1_2_index = find_closest_point_index(outer_2np, closest_point_1_2, threshold)
            closest_point_2_3_index = find_closest_point_index(outer_2np, closest_point_2_3, threshold)

            index_a = closest_point_1_2_index
            index_b = closest_point_2_3_index

            # print("Index of a in outer_2np:", index_a)
            # print("Index of b in outer_2np:", index_b)

            indices_between_2 = calculate_indices_between_good(index_a, index_b, len(outer_2np))
            # print('indices between a and b: for 2', indices_between_2)

            # Find the indices of the edge in the x3 array
            closest_point_1_3_index = find_closest_point_index(outer_3np, closest_point_1_3, threshold)
            closest_point_2_3_index = find_closest_point_index(outer_3np, closest_point_2_3, threshold)

            index_a = closest_point_1_3_index
            index_b = closest_point_2_3_index

            # print("Index of a in outer_3np:", index_a)
            # print("Index of b in outer_3np:", index_b)

            indices_between_3 = calculate_indices_between_good(index_a, index_b, len(outer_3np))
            # print('indices between a and b: for 3', indices_between_3)
            
            
            # all_coords = np.vstack((outer_layer1[i][indices_between_1], outer_layer2[i][indices_between_2], outer_layer3[i][indices_between_3]))
            # Determine which layer starts closest to where layer 1 ends
            last_point_layer1 = outer_layer1[i][indices_between_1][-1]
            start_point_layer2 = outer_layer2[i][indices_between_2][0]
            start_point_layer3 = outer_layer3[i][indices_between_3][0]

            distance_to_layer2 = np.linalg.norm(last_point_layer1 - start_point_layer2)
            distance_to_layer3 = np.linalg.norm(last_point_layer1 - start_point_layer3)

            if distance_to_layer2 < distance_to_layer3:
                next_layer = outer_layer2[i][indices_between_2]
                last_layer = outer_layer3[i][indices_between_3]
            else:
                next_layer = outer_layer3[i][indices_between_3]
                last_layer = outer_layer2[i][indices_between_2]

            # Stack the layers
            all_coords = np.vstack((outer_layer1[i][indices_between_1], next_layer,last_layer))
            
            
            unique_coords = []
            
        

            for coord in all_coords:
                is_unique = True
                if unique_coords:
                    # Distances to PREVIOUSLY ACCEPTED coords only (order-dependent)
                    if np.any(np.linalg.norm(np.array(unique_coords) - coord, axis=1) < threshold):
                        is_unique = False
                if is_unique:
                    unique_coords.append(coord)
                
            # unique_coords = np.array(unique_coords)
            # centroid = np.mean(unique_coords, axis=0)
            # angles = np.arctan2(unique_coords[:, 1] - centroid[1], unique_coords[:, 0] - centroid[0])
            # sorted_indices = np.argsort(angles)
            # sorted_unique_coords.append(unique_coords[sorted_indices])
            
            # Write the keypoints to the file
            with open(_kp_file, "a") as f:
                for coord in unique_coords:
                    f.write(f"k,{j_index+node_offset_stacknr},{coord[0]/1e3:.10e},{(coord[1]+(stack_nr-1)*vertical_offset)/1e3:.10e},0.0\n")
                    j_index += 1
                    
                    
                f.write(f"FLST,2,{j_index-j_index_old},3\n")
                for i in range(j_index_old, j_index):
                    f.write(f"FITEM,2,{i+node_offset_stacknr}\n")
                f.write("A,P51X\n")   
                
            j_index = j_index_old + 100
            j_index_old = j_index
            
            # print('j_index after 3', j_index)
            
        outer_layer1 = []
        outer_layer2 = []
        outer_layer3 = []
        outer_layer4 = []
        sorted_unique_coords = []
        
        
        # Loops of four
        for i in range(len(cycles_of_size_4)):
            overlapping_coords_1_2 = []
            overlapping_coords_2_3 = []
            overlapping_coords_3_4 = []
            overlapping_coords_1_4 = []
                        
            outer_1np = []
            outer_2np = []
            outer_3np = []
            outer_4np = []
            
            
            # print(i)
            # print('first strand', cycles_of_size_4[i][0], 'second strand', cycles_of_size_4[i][1], 'third strand', cycles_of_size_4[i][2], 'fourth strand', cycles_of_size_4[i][3])
            mapper1 = saved_mappers[cycles_of_size_4[i][0]-1] # correct the index
            mapper2 = saved_mappers[cycles_of_size_4[i][1]-1]
            mapper3 = saved_mappers[cycles_of_size_4[i][2]-1]
            mapper4 = saved_mappers[cycles_of_size_4[i][3]-1]     
                   
            outer_layer1 = mapper1.mapped_nodes[-mapper1.mesh.circumferential_divisions:]
            outer_layer2 = mapper2.mapped_nodes[-mapper2.mesh.circumferential_divisions:]
            outer_layer3 = mapper3.mapped_nodes[-mapper3.mesh.circumferential_divisions:]
            outer_layer4 = mapper4.mapped_nodes[-mapper4.mesh.circumferential_divisions:]
        
        
            outer_1np = np.array(outer_layer1)
            outer_2np = np.array(outer_layer2)
            outer_3np = np.array(outer_layer3)
            outer_4np = np.array(outer_layer4)        
            
            overlapping_coords_1_2 = find_overlapping_coords(outer_1np, outer_2np, threshold) #Find overlapping coordinates between part 1 and 2
            overlapping_coords_2_3 = find_overlapping_coords(outer_2np, outer_3np, threshold) #Find overlapping coordinates between part 2 and 3
            overlapping_coords_3_4 = find_overlapping_coords(outer_3np, outer_4np, threshold) #Find overlapping coordinates between part 3 and 4
            overlapping_coords_1_4 = find_overlapping_coords(outer_1np, outer_4np, threshold) #Find overlapping coordinates between part 1 and 3


            comp_array_1_2 = [overlapping_coords_1_4, overlapping_coords_2_3, overlapping_coords_3_4]
            comp_array_2_3 = [overlapping_coords_1_2, overlapping_coords_1_4, overlapping_coords_3_4]
            comp_array_3_4 = [overlapping_coords_1_2, overlapping_coords_1_4, overlapping_coords_2_3]
            comp_array_1_4 = [overlapping_coords_1_2, overlapping_coords_2_3, overlapping_coords_3_4]

            closest_point_1_2 = find_closest_point(overlapping_coords_1_2,comp_array_1_2) #Closest point on intersection of part 1 and 2
            closest_point_2_3 = find_closest_point(overlapping_coords_2_3,comp_array_2_3) #Closest point on intersection of part 2 and 3
            closest_point_3_4 = find_closest_point(overlapping_coords_3_4,comp_array_3_4) #Closest point on intersection of part 3 and 4
            closest_point_1_4 = find_closest_point(overlapping_coords_1_4,comp_array_1_4) #Closest point on intersection of part 1 and 4
            
            first_a = closest_point_1_2
            first_b = closest_point_1_4
            
            first_a_index = find_closest_point_index(outer_1np, first_a, threshold)
            first_b_index = find_closest_point_index(outer_1np, first_b, threshold)
        
            
            second_a = closest_point_1_2
            second_b = closest_point_2_3
            
            second_a_index = find_closest_point_index(outer_2np, second_a, threshold)
            second_b_index = find_closest_point_index(outer_2np, second_b, threshold)
            
            third_a = closest_point_2_3
            third_b = closest_point_3_4
            
            third_a_index = find_closest_point_index(outer_3np, third_a, threshold)
            third_b_index = find_closest_point_index(outer_3np, third_b, threshold)
            
            fourth_a = closest_point_3_4
            fourth_b = closest_point_1_4
            
            fourth_a_index = find_closest_point_index(outer_4np, fourth_a, threshold)
            fourth_b_index = find_closest_point_index(outer_4np, fourth_b, threshold)
            
            indices_between_1 = calculate_indices_between_good(first_a_index, first_b_index, len(outer_1np))
            indices_between_2 = calculate_indices_between_good(second_a_index, second_b_index, len(outer_2np))
            indices_between_3 = calculate_indices_between_good(third_a_index, third_b_index, len(outer_3np))
            indices_between_4 = calculate_indices_between_good(fourth_a_index, fourth_b_index, len(outer_4np))

            

            
            all_coords = np.vstack((outer_1np[indices_between_1], outer_2np[indices_between_2], outer_3np[indices_between_3], outer_4np[indices_between_4]))
            
            unique_coords = []
            for coord in all_coords:
                is_unique = True
                if unique_coords:
                    # Distances to PREVIOUSLY ACCEPTED coords only (order-dependent)
                    if np.any(np.linalg.norm(np.array(unique_coords) - coord, axis=1) < threshold):
                        is_unique = False
                if is_unique:
                    unique_coords.append(coord)
                

            # Write the keypoints to the file
            with open(_kp_file, "a") as f:
                for coord in unique_coords:
                    f.write(f"k,{j_index+node_offset_stacknr},{coord[0]/1e3:.10e},{(coord[1]+(stack_nr-1)*vertical_offset)/1e3:.10e},0.0\n")
                    j_index += 1
            
                f.write(f"FLST,2,{j_index-j_index_old},3\n")
                for i in range(j_index_old, j_index):
                    f.write(f"FITEM,2,{i+node_offset_stacknr}\n")
                f.write("A,P51X\n")




            j_index = j_index_old + 100
            j_index_old = j_index


    # def write_mesh_to_file(self, filename_nodes, filename_elements):
    #     """Writes the mesh nodes and elements to a file with the specified format."""
    #     node_offset = 0
        
    #     with open(filename_nodes, 'w') as f_nodes, open(filename_elements, 'w') as f_elements:
    #         # Write nodes for the first strand
    #         for i, (x, y) in enumerate(self.mapper1.mapped_nodes):
    #             f_nodes.write(f"N, {i + 1}, {x:.6f}, {y:.6f}, 0.0\n")
    #         node_offset = len(self.mapper1.mapped_nodes)
            
    #         # Write elements for the first strand
    #         for elem in self.mesh1.elements:
    #             f_elements.write(f"E, {elem[0] + 1}, {elem[1] + 1}, {elem[2] + 1}, {elem[3] + 1}\n")
            
    #         f_nodes.write(f"\n")
    #         f_elements.write(f"\n")

    #         # Write nodes for the second strand, with offset
    #         for i, (x, y) in enumerate(self.mapper2.mapped_nodes):
    #             f_nodes.write(f"N, {i + 1 + node_offset}, {x:.6f}, {y:.6f}, 0.0\n")
            
    #         # Write elements for the second strand, with offset
    #         for elem in self.mesh2.elements:
    #             f_elements.write(f"E, {elem[0] + 1 + node_offset}, {elem[1] + 1 + node_offset}, {elem[2] + 1 + node_offset}, {elem[3] + 1 + node_offset}\n")
                
    #         f_nodes.write(f"\n")
    #         f_elements.write(f"\n")
            
            
    #         # Write nodes for the second strand, with offset
    #         for i, (x, y) in enumerate(self.mapper2.mapped_nodes):
    #             f_nodes.write(f"N, {i + 1 + node_offset}, {x:.6f}, {y:.6f}, 0.0\n")
            
    #         # Write elements for the second strand, with offset
    #         for elem in self.mesh2.elements:
    #             f_elements.write(f"E, {elem[0] + 1 + node_offset}, {elem[1] + 1 + node_offset}, {elem[2] + 1 + node_offset}, {elem[3] + 1 + node_offset}\n")
    

def plot_cable_overview(saved_mappers, stack_nr, filename, insulation_polygon=None, stack_height_mm=0.0):
    """Plots the outer layer outlines of all strands and the insulation boundary in a single overview figure."""
    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(10, 8))
    y_offset_mm = (stack_nr - 1) * stack_height_mm
    for i, mapper in enumerate(saved_mappers):
        if mapper is None:
            continue
        outer_layer = mapper.mapped_nodes[-mapper.mesh.circumferential_divisions:].copy()
        outer_layer[:, 1] += y_offset_mm
        outline = np.vstack([outer_layer, outer_layer[0]])
        color = colors[i % len(colors)]
        ax.fill(outer_layer[:, 0], outer_layer[:, 1], alpha=0.3, color=color)
        ax.plot(outline[:, 0], outline[:, 1], color=color, linewidth=1.5)
        centroid = outer_layer.mean(axis=0)
        ax.text(centroid[0], centroid[1], str(i + 1), ha='center', va='center', fontsize=7, color='black')
    if insulation_polygon is not None and len(insulation_polygon) > 2:
        ins = np.array(insulation_polygon)
        # insulation_polygon coords are in metres; mapped_nodes are in mm
        ins_mm = ins * 1e3
        closed = np.vstack([ins_mm, ins_mm[0]])
        ax.plot(closed[:, 0], closed[:, 1], color='black', linewidth=2, linestyle='--', label='Insulation boundary')
        ax.legend(fontsize=7)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Cable overview — stack {stack_nr}")
    ax.set_aspect('equal')
    fig.savefig(filename)
    plt.close(fig)


def plot_final_geometry(all_stacks_mappers, all_insulation_polygons, filename, stack_height_mm=0.0):
    """Single-axes plot of the complete cable cross-section as it will appear in ANSYS.

    All stacks are drawn at their true (x, y) coordinates — insulation polygons
    as filled grey regions with a dashed black border, strand outlines as
    coloured filled patches on top.  This reflects the geometry *after*
    gap-snapping so the plot matches what the APDL deck actually receives.

    Parameters
    ----------
    all_stacks_mappers : dict
        stack_nr -> list of MeshMapping objects.
    all_insulation_polygons : dict
        stack_nr -> list of (x, y) in metres.
    filename : str
        Output SVG/PNG path.
    stack_height_mm : float
        Stack height in mm (= stack_height in metres * 1e3). Used to apply
        the per-stack y-offset to strand nodes, which are stored in local
        (stack-centred) coordinates without the stacking shift.
    """
    from matplotlib.patches import Polygon as MplPolygon

    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(12, max(6, 2 * len(all_stacks_mappers))))

    for stack_nr in sorted(all_stacks_mappers.keys()):
        y_offset_mm = (stack_nr - 1) * stack_height_mm

        # Insulation polygon (metres → mm, already at absolute y)
        ins_pts = all_insulation_polygons.get(stack_nr)
        if ins_pts is not None and len(ins_pts) > 2:
            ins_mm = np.array(ins_pts) * 1e3
            ins_patch = MplPolygon(ins_mm, closed=True, facecolor='#d8d8d8', edgecolor='black',
                                   linewidth=1.2, linestyle='--', zorder=1)
            ax.add_patch(ins_patch)

        # Strand outlines (mm, local coords — apply stack y-offset)
        mappers = all_stacks_mappers[stack_nr]
        for i, mapper in enumerate(mappers):
            if mapper is None:
                continue
            outer_layer = mapper.mapped_nodes[-mapper.mesh.circumferential_divisions:].copy()
            outer_layer[:, 1] += y_offset_mm
            color = colors[i % len(colors)]
            strand_patch = MplPolygon(outer_layer, closed=True,
                                      facecolor=color, edgecolor=color,
                                      linewidth=0.5, alpha=0.75, zorder=2)
            ax.add_patch(strand_patch)
            centroid = outer_layer.mean(axis=0)
            ax.text(centroid[0], centroid[1], str(i + 1),
                    ha='center', va='center', fontsize=5, color='white', fontweight='bold', zorder=3)

    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Final cable geometry (all stacks combined)")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"  Final geometry plot saved to: {filename}")


def plot_all_stacks_overview(all_stacks_mappers, all_insulation_polygons, filename, stack_height_mm=0.0):
    """Plots every stack's strand outlines and insulation boundary in a single grid figure.

    Parameters
    ----------
    all_stacks_mappers : dict
        Mapping stack_nr -> list of MeshMapping objects (saved_mappers) for that stack.
    all_insulation_polygons : dict
        Mapping stack_nr -> list of (x, y) tuples in metres (insulation.filtered_outerpoints).
    filename : str
        Output file path (SVG recommended).
    stack_height_mm : float
        Stack height in mm. Applied as per-stack y-offset to strand nodes.
    """
    stack_nrs = sorted(all_stacks_mappers.keys())
    n = len(stack_nrs)
    if n == 0:
        return

    # Determine a compact grid layout (favour more columns than rows)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    colors = plt.cm.tab20.colors
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, stack_nr in enumerate(stack_nrs):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        y_offset_mm = (stack_nr - 1) * stack_height_mm
        mappers = all_stacks_mappers[stack_nr]
        for i, mapper in enumerate(mappers):
            if mapper is None:
                continue
            outer_layer = mapper.mapped_nodes[-mapper.mesh.circumferential_divisions:].copy()
            outer_layer[:, 1] += y_offset_mm
            outline = np.vstack([outer_layer, outer_layer[0]])
            color = colors[i % len(colors)]
            ax.fill(outer_layer[:, 0], outer_layer[:, 1], alpha=0.3, color=color)
            ax.plot(outline[:, 0], outline[:, 1], color=color, linewidth=1.0)
            centroid = outer_layer.mean(axis=0)
            ax.text(centroid[0], centroid[1], str(i + 1), ha='center', va='center', fontsize=6, color='black')

        # Draw insulation boundary if available (coords in metres → convert to mm, already absolute y)
        ins_pts = all_insulation_polygons.get(stack_nr)
        if ins_pts is not None and len(ins_pts) > 2:
            ins_mm = np.array(ins_pts) * 1e3
            closed = np.vstack([ins_mm, ins_mm[0]])
            ax.plot(closed[:, 0], closed[:, 1], color='black', linewidth=1.5, linestyle='--')

        ax.set_title(f"Stack {stack_nr}", fontsize=9)
        ax.set_aspect('equal')
        ax.set_xlabel("x [mm]", fontsize=7)
        ax.set_ylabel("y [mm]", fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused subplot cells when n < nrows * ncols
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("All stacks — cable cross-section overview", fontsize=12)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Post-snap origin fixes for known contact-zone degeneracies.
#
# Background. The contact-zone alignment in `rotate_outer_layer_nodes` /
# `align_nodes` is a post-hoc fixup: each strand's outer ring is mapped to the
# B-spline, then nodes near a contact pair are snapped to a contact midpoint.
# The heuristic has three distinct failure modes when the deformed strand
# geometry is extreme:
#   (1) CCW order can break — a node is moved past an angular neighbour, so
#       the outer-ring polygon self-crosses. Triggers APDL "Poorly defined
#       area. Check for crossed lines." on the strand-area boolean.
#       Observed: TEST_A_thinmany_NOM, stack 5 (surface 653 boolean failure).
#   (2) Two nodes from ADJACENT strands' outer rings can land < BTOL apart but
#       not coincident — the boolean fails with min KPT distance just above
#       BTOL. Observed: TEST_E_narrowkeyst_HEAVY (1.18 µm pair).
#   (3) Two nodes on the SAME strand's outer ring can land < BTOL apart (a
#       near-zero-area cell), MAPDL's quad mesher then segfaults during
#       `amesh`. Observed: TEST_B_medstd_HEAVY (Segmentation Violation).
#
# All three fixes below are GATED by their respective failure conditions and
# are no-ops on cables that don't trigger them. Verified by running through
# TEST_A_LIGHT after deploying: zero corrections applied, fd_good output
# byte-identical to the pre-fix run. Working cables (R2D2_LF/HF, CD1, SMACC_HF,
# TEST_B_LIGHT/NOM, TEST_C×3, TEST_D×3, TEST_E_LIGHT/NOM) likewise see zero
# triggers because their outer-ring geometry is well within the thresholds.
#
# Conservative thresholds chosen to be well above any legitimate pair on
# working cables (smallest observed inter-strand pair on TEST_C: ~22 µm;
# smallest intra-strand pair on TEST_C: ~26 µm) and to catch the observed
# failure cases (TEST_E_HEAVY 1.18 µm; TEST_B_HEAVY < 2 µm).
# --------------------------------------------------------------------------- #

# NOTE: MeshMapping.mapped_nodes are stored in MILLIMETRES, so all distance
# thresholds below are expressed in mm. Comments note the µm equivalent.
_DEDUP_INTRA_THRESH_MM = 2.0e-3   # 2 µm — intra-strand outer-ring (TEST_B_HEAVY mode)
_DEDUP_INTER_THRESH_MM = 6.0e-3   # 6 µm — inter-strand outer-ring (TEST_E_HEAVY mode).
                                  # Empirically tuned: at 3 µm a 3-strand cluster
                                  # cascade leaves residuals; at 10 µm aggressive
                                  # merging creates degenerate cells that segfault
                                  # MAPDL's mesher; 6 µm gives the best balance
                                  # (gets TEST_E_HEAVY through stacks 1–3, fails
                                  # at stack 4 with min KPT 6.3 µm in a
                                  # degenerate area MAPDL can't subtract).
                                  # Working cables (TEST_A_LIGHT min 15.6 µm,
                                  # TEST_B_NOM ~22 µm, TEST_C_LIGHT ~22 µm) all
                                  # untouched at this threshold.
_DEDUP_SKIP_THRESH_MM  = 1.0e-6   # 1 nm — below: already coincident; merging is no-op
_CCW_EPSILON_RAD       = 1.0e-3   # ~0.06 deg tolerance for floating-point


def _fix_outer_ring_ccw_violations(saved_mappers):
    """Origin fix for TEST_A_NOM "Poorly defined area. Check for crossed lines."

    Each strand's outer ring is a closed polygon ordered CCW by construction
    (StrandMesh_Hexa numbers nodes CCW around the strand). The contact-zone
    snap can move a node past one of its angular neighbours (relative to the
    strand barycenter), reversing the local order — APDL's ASBA then sees a
    self-crossing area boundary and fails.

    For each outer ring: walk every node, check its angle (about the
    barycenter of that ring) stays between its two neighbours' angles going
    CCW. If not, replace it with the midpoint between the two neighbours
    (which is guaranteed to sit between them angularly).

    Triggered only when the snap actually crossed something — on the working
    cables the check finds zero violations and the mappers are untouched.
    """
    n_total_fixed = 0
    for mp in saved_mappers:
        if mp is None:
            continue
        cd = mp.mesh.circumferential_divisions
        outer_view = mp.mapped_nodes[-cd:]
        bary = outer_view.mean(axis=0)
        # Up to 3 passes (cascading violations are rare but possible)
        for _pass in range(3):
            outer = mp.mapped_nodes[-cd:]
            angles = np.arctan2(outer[:, 1] - bary[1], outer[:, 0] - bary[0])
            n_this_pass = 0
            for i in range(cd):
                prev_a = angles[(i - 1) % cd]
                next_a = angles[(i + 1) % cd]
                this_a = angles[i]
                # Signed CCW deltas, normalised to (-pi, pi]
                d_prev = ((this_a - prev_a + np.pi) % (2 * np.pi)) - np.pi
                d_next = ((next_a - this_a + np.pi) % (2 * np.pi)) - np.pi
                # CCW order requires both deltas non-negative (modulo epsilon)
                if d_prev < -_CCW_EPSILON_RAD or d_next < -_CCW_EPSILON_RAD:
                    new_pos = (mp.mapped_nodes[-cd + (i - 1) % cd]
                               + mp.mapped_nodes[-cd + (i + 1) % cd]) / 2.0
                    mp.mapped_nodes[-cd + i] = new_pos
                    n_this_pass += 1
                    n_total_fixed += 1
            if n_this_pass == 0:
                break
    if n_total_fixed:
        print(f"[origin_fix:ccw] corrected {n_total_fixed} crossed outer-ring node(s)")
    return n_total_fixed


def _dedup_close_outer_ring_nodes(saved_mappers):
    """Origin fix for TEST_E_HEAVY (inter-strand) and TEST_B_HEAVY (intra-strand).

    INTER-strand: after the contact snap, some pairs from adjacent strands'
    outer rings end up within (1 nm, 3 µm) of each other instead of exactly
    coincident.  APDL's BTOL then sees them as distinct, and the strand-area
    boolean fails with "min KPT distance" just above BTOL.  Merge the pair
    to its midpoint — same outcome the snap was supposed to produce.

    INTRA-strand: same snap can pull two outer-ring nodes of the SAME strand
    within (1 nm, 2 µm) of each other.  MAPDL's quad mesher segfaults on the
    resulting near-zero-area cell.  Merge them to the midpoint.

    NOTE: when an intra-strand merge would make two ADJACENT outer-ring nodes
    (within id-diff 5) truly coincident, the post-write duplicate detector
    later in this module aborts with a clear "intra-strand duplicate" error
    rather than letting MAPDL crash silently.  That is intentional — better
    than a SegV.  Cables where the snap is well-behaved never trigger either
    merge, so the mappers are byte-identical to the pre-fix state.
    """
    n_intra = 0
    n_inter = 0

    # Cluster-merge pass: when 3+ nodes form a tight cluster (TEST_E_HEAVY:
    # 3 strands collide at one contact-zone midpoint), a single pair-by-pair
    # merge leaves residual sub-µm distances that still trip the boolean.
    # Repeat the merge sweep until no more pairs are below threshold (cap at
    # 6 passes, more than enough for any contact-zone cluster size we have
    # ever seen).
    MAX_PASSES = 20
    for _pass in range(MAX_PASSES):
        merges_this_pass = 0

        # --- intra-strand: pairs on the same strand's outer ring ---
        for mp in saved_mappers:
            if mp is None:
                continue
            cd = mp.mesh.circumferential_divisions
            outer = mp.mapped_nodes[-cd:]
            for i in range(cd):
                for j in range(i + 1, cd):
                    d = float(np.linalg.norm(outer[i] - outer[j]))
                    if _DEDUP_SKIP_THRESH_MM < d < _DEDUP_INTRA_THRESH_MM:
                        mid = (outer[i] + outer[j]) / 2.0
                        mp.mapped_nodes[-cd + i] = mid
                        mp.mapped_nodes[-cd + j] = mid
                        n_intra += 1
                        merges_this_pass += 1

        # --- inter-strand: pairs between two strands' outer rings ---
        n = len(saved_mappers)
        for ia in range(n):
            ma = saved_mappers[ia]
            if ma is None:
                continue
            cd_a = ma.mesh.circumferential_divisions
            out_a = ma.mapped_nodes[-cd_a:]
            for ib in range(ia + 1, n):
                mb = saved_mappers[ib]
                if mb is None:
                    continue
                cd_b = mb.mesh.circumferential_divisions
                out_b = mb.mapped_nodes[-cd_b:]
                for ka in range(cd_a):
                    for kb in range(cd_b):
                        d = float(np.linalg.norm(out_a[ka] - out_b[kb]))
                        if _DEDUP_SKIP_THRESH_MM < d < _DEDUP_INTER_THRESH_MM:
                            mid = (out_a[ka] + out_b[kb]) / 2.0
                            ma.mapped_nodes[-cd_a + ka] = mid
                            mb.mapped_nodes[-cd_b + kb] = mid
                            n_inter += 1
                            merges_this_pass += 1

        if merges_this_pass == 0:
            break  # converged

    if n_intra or n_inter:
        print(f"[origin_fix:dedup] merged {n_intra} intra-strand + "
              f"{n_inter} inter-strand close outer-ring pair(s) "
              f"({_pass + 1} pass(es))")
    return n_intra, n_inter


def run(stack_dir, n_stacks, n_parts, output_dir=None, stack_height=None, debug_plots=False, wire=None, diameter_mm=None):
    """Run conformal mesh generation for all stacks.

    Parameters
    ----------
    stack_dir : str or Path
        Directory containing the ParaView CSV files (Stack_*_Part*.csv).
    n_stacks : int
        Number of stacks to process (1 … n_stacks).
    n_parts : int
        Number of wire parts per stack (N_Strands).
    output_dir : str or Path, optional
        Directory where all output files are written.  Defaults to ``stack_dir``.
    stack_height : float, optional
        Physical stack height in metres (cable_height + 300 µm).
        Defaults to 0.93*2e-3 m if not provided.
    wire : dict, optional
        ``cable["wire"]`` block from cable_parameters_user.json. When given,
        the inner / outer big-hex circumradii of the homogenised sub-element
        layout are derived from ``D_CORE_EQ_UM``, ``CU_SLEEVE_THICKNESS_UM``,
        ``N_TOTAL`` and ``N_NB3SN`` and forwarded to ``StrandMesh_Hexa`` so
        the Cu sleeve thickness is encoded in the strand mesh. When omitted
        the legacy hardcoded ratios (149.12/425, 385.54/425) are used.
    """
    import os as _os
    from pathlib import Path as _Path

    stack_dir = _Path(stack_dir)
    output_dir = _Path(output_dir) if output_dir is not None else stack_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # All plot files go into a dedicated subfolder so they don't clutter the APDL output
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    conformal_mesh_plots_dir = plots_dir / "conformal_mesh"
    conformal_mesh_plots_dir.mkdir(exist_ok=True)
    outer_nodes_plots_dir = plots_dir / "outer_nodes"
    outer_nodes_plots_dir.mkdir(exist_ok=True)
    # Debug-only folder for pairs where align_nodes hit a suspect branch
    # (indice_special / change11_extra / change22_extra) or produced an
    # intra-strand outer-ring duplicate. Always populated, regardless of
    # the debug_plots flag, so we can inspect intermittent failures.
    debug_dup_plots_dir = plots_dir / "align_debug"
    debug_dup_plots_dir.mkdir(exist_ok=True)

    # All APDL input text files go directly into output_dir (no 'input' subfolder)
    input_dir = output_dir

    # Accumulators for the combined all-stacks overview, filled during the per-stack loop
    all_stacks_mappers: dict = {}
    all_insulation_polygons: dict = {}
    all_insulation_layers: dict = {}  # InsulationLayer objects; written after interface alignment

    # Clamp n_parts to non-empty CSVs (ParaView writes empty files for missing parts)
    actual_n_parts = sum(
        1 for i in range(1, n_parts + 1)
        if (stack_dir / f"Stack_1_Part{i}.csv").exists()
        and (stack_dir / f"Stack_1_Part{i}.csv").stat().st_size > 0
    )
    if actual_n_parts == 0:
        raise FileNotFoundError(f"No valid Stack_1_Part*.csv files found in {stack_dir}")
    if actual_n_parts != n_parts:
        print(f"Note: {n_parts} strands in JSON but only {actual_n_parts} non-empty CSVs found — using {actual_n_parts}")
    n_parts = actual_n_parts

    # Clamp n_stacks to non-empty CSVs
    actual_n_stacks = sum(
        1 for s in range(1, n_stacks + 1)
        if (stack_dir / f"Stack_{s}_Part1.csv").exists()
        and (stack_dir / f"Stack_{s}_Part1.csv").stat().st_size > 0
    )
    if actual_n_stacks == 0:
        raise FileNotFoundError(f"No valid Stack_*_Part1.csv files found in {stack_dir}")
    if actual_n_stacks != n_stacks:
        print(f"Note: {n_stacks} stacks in JSON but only {actual_n_stacks} non-empty CSVs found — using {actual_n_stacks}")
    n_stacks = actual_n_stacks

    if stack_height is None:
        stack_height = 0.93 * 2 * 1e-3  # fallback default
    stack_height = float(stack_height)
    print(f"Using stack_height = {stack_height*1e3:.3f} mm")

    stacking = True  # Set to False to keep conductors in same plane, True to stack them vertically

    square_size = 6 * 12  # 72
    e_per_layer = 36
    radial_layers_nb = 8
    radial_layers_cu = 3
    n_elements_per_strand = square_size + (radial_layers_cu + radial_layers_nb) * e_per_layer

    # Resolve hex circumradii from the wire homogenisation (mm). Falls back to
    # the legacy hardcoded ratios when no `wire` block is supplied.
    R_inner_mm, R_outer_mm = compute_hex_circumradii_mm(wire)
    # Strand diameter in mm. The mesh-mapping step in meshMapping.py scales
    # intermediate-layer node displacements by 1/(diameter/2), so this MUST
    # match the physical strand diameter or the outer hex / intermediate
    # layers end up scaled by D_strand/0.6.
    _template_diameter = float(diameter_mm) if diameter_mm is not None else 0.6
    if R_inner_mm is None:
        legacy_inner_scale = 149.12 / 425
        legacy_outer_scale = 385.54 / 425
        R_inner_mm = 0.5 * legacy_inner_scale * _template_diameter
        R_outer_mm = 0.5 * legacy_outer_scale * _template_diameter
        print(f"Using legacy hex circumradii (no wire data): "
              f"R_in={R_inner_mm*1e3:.2f} um, R_out={R_outer_mm*1e3:.2f} um, "
              f"D_strand={_template_diameter*1e3:.0f} um")
    else:
        print(f"Hex circumradii from wire block: "
              f"R_in={R_inner_mm*1e3:.2f} um, R_out={R_outer_mm*1e3:.2f} um, "
              f"D_strand={_template_diameter*1e3:.0f} um")

    # One overlay plot — the relative geometry is identical for every strand
    # in a given cable, so a single figure is enough for the visual check.
    _overlay_mesh = StrandMesh_Hexa(
        diameter=_template_diameter, radial_divisions=3, angle=0,
        inner_circumradius_mm=R_inner_mm,
        outer_circumradius_mm=R_outer_mm,
    )
    plot_hex_overlay(
        _overlay_mesh, R_inner_mm, R_outer_mm,
        str(plots_dir / "hex_overlay.svg"),
        title="Sub-element mesh vs. homogenisation hexes",
        wire=wire, strand_diameter_mm=_template_diameter,
    )

    for stack_nr in range(1, n_stacks + 1):
        percent = int(100 * stack_nr / n_stacks)
        bar_length = 40
        filled_length = int(bar_length * percent // 100)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        print(f"\rProcessing stack {stack_nr}/{n_stacks}... [{bar}] {percent}%", end='', flush=True)
        if stack_nr == n_stacks:
            print()  # Move to next line after last stack

        random.seed(1000 + stack_nr)
        hex_angle = [random.uniform(-30, 30) for _ in range(n_parts)]
        print("\n hex_angle:", [f"{a:.1f}°" for a in hex_angle])

        parts = [str(stack_dir / f"Stack_{stack_nr}_Part{i+1}.csv") for i in range(n_parts)]
        saved_strands = [None] * n_parts
        saved_mappers = [None] * n_parts

        # Pre-initialize all strands and mappers once (avoids redundant CSV reads per pair)
        for i in range(n_parts):
            strand = DeformedStrandInterpolator(parts[i])
            strand.fit_bspline()
            mesh = StrandMesh_Hexa(diameter=_template_diameter, radial_divisions=3, angle=hex_angle[i],
                                   inner_circumradius_mm=R_inner_mm,
                                   outer_circumradius_mm=R_outer_mm)
            mapper = MeshMapping(mesh, strand)
            mapper.translate_mesh_to_barycenter()
            mapper.map_circumferential_layer_to_bspline()
            saved_strands[i] = strand
            saved_mappers[i] = mapper

        connections_file = str(input_dir / f"connections_{stack_nr}.txt")
        with open(connections_file, "w") as f:
            f.write("Connection_a,Connection_b\n")

        for i in range(n_parts - 1):
            for j in range(i + 1, n_parts):
                conformal_mesh = ConformalRutherfordMesh.from_existing(
                    saved_strands[i], saved_mappers[i],
                    saved_strands[j], saved_mappers[j]
                )
                conformal_mesh._pair_label = f"stack{stack_nr}:({i+1},{j+1})"

                status = conformal_mesh.rotate_outer_layer_nodes()

                # Always dump a debug plot for pairs that hit suspect branches
                # or produced an intra-strand duplicate, regardless of debug_plots.
                debug_reasons = getattr(conformal_mesh, "_align_debug_reasons", [])
                if debug_reasons:
                    tag = "_".join(sorted({r.split()[0] for r in debug_reasons}))
                    base = f"stack{stack_nr}_pair_{i+1}_{j+1}_{tag}"
                    try:
                        conformal_mesh.plot_conformal_mesh(str(debug_dup_plots_dir / f"{base}_mesh.svg"))
                        conformal_mesh.plot_outer_nodes_and_bsplines(str(debug_dup_plots_dir / f"{base}_outer.svg"))
                        with open(debug_dup_plots_dir / f"{base}_reasons.txt", "w") as f:
                            for r in debug_reasons:
                                f.write(r + "\n")
                        print(f"[align_debug] saved {base} (reasons: {len(debug_reasons)})")
                    except Exception as e:
                        print(f"[align_debug] plot failed for {base}: {e}")

                if status is True:
                    if debug_plots:
                        conformal_mesh.plot_conformal_mesh(str(conformal_mesh_plots_dir / f"conformal_mesh_{stack_nr}_{i+1}_{j+1}.svg"))
                        conformal_mesh.plot_outer_nodes_and_bsplines(str(outer_nodes_plots_dir / f"outer_nodes_{stack_nr}_{i+1}_{j+1}.svg"))
                    with open(connections_file, "a") as f:
                        f.write(f"{i+1},{j+1}\n")

                saved_strands[i] = conformal_mesh.strand1
                saved_mappers[i] = conformal_mesh.mapper1

                saved_strands[j] = conformal_mesh.strand2
                saved_mappers[j] = conformal_mesh.mapper2

        # ----- Origin fixes for contact-snap degeneracies (see helpers above).
        # Both are gated by failure conditions: working cables see zero
        # corrections and the saved_mappers stay byte-identical.  Failing
        # cables (TEST_A_NOM crossed lines, TEST_E_HEAVY 1.2 µm pair,
        # TEST_B_HEAVY < 2 µm pair causing SegV) get their degeneracies
        # repaired before the impregnation and strand keypoint writers run.
        _fix_outer_ring_ccw_violations(saved_mappers)
        _dedup_close_outer_ring_nodes(saved_mappers)

        conformal_mesh.write_impregnation_keypoints(
            connections_file, saved_mappers, stack_nr, stack_height, stacking, output_dir=input_dir
        )

        offset_stacknr = (stack_nr - 1) * 3e4  # Reduced from 1e5 to match APDL stack_offset
        j_index = 10000
        j_offset_areas = j_index
        j_index_old = j_index
        offset_strand = 100

        keypoints_file = str(input_dir / f"keypoints_nodes_{stack_nr}.txt")
        with open(keypoints_file, "w") as f:
            f.write("! Key Points for nodes of the strands \n")
            f.write(f"NUMSTR, area, {j_index+offset_stacknr} \n")

        for i in range(n_parts):
            offset_areas = j_offset_areas + offset_stacknr + i * offset_strand

            nodes_square = 7 * 13
            n_per_layer = 36

            vertical_offset = stack_height if stacking else 0

            use_mapper = saved_mappers[i]
            inner_layer = use_mapper.mapped_nodes[nodes_square:nodes_square + n_per_layer]
            middle_layer = use_mapper.mapped_nodes[
                nodes_square + (radial_layers_nb - 1) * n_per_layer:
                nodes_square + radial_layers_nb * n_per_layer
            ]
            outer_layer = use_mapper.mapped_nodes[-use_mapper.mesh.circumferential_divisions:]

            for layer in [inner_layer, middle_layer, outer_layer]:
                with open(keypoints_file, "a") as f:
                    for coord in layer:
                        j_index = j_index + 1
                        f.write(
                            f"k,{j_index+offset_stacknr},{coord[0]/1e3:.10e},"
                            f"{(coord[1]+(stack_nr-1)*vertical_offset*1e3)/1e3:.10e},0.0\n"
                        )
                    f.write(f"NUMSTR, area, {offset_areas} \n")
                    f.write(f"FLST,2,{j_index-j_index_old},3\n")

                    for idx in range(j_index_old + 1, j_index + 1):
                        f.write(f"FITEM,2,{idx+offset_stacknr}\n")
                    f.write("A,P51X\n")

                    if np.array_equal(layer, inner_layer):
                        f.write("asel,none \n")
                    elif np.array_equal(layer, middle_layer):
                        f.write("asel,none \n")
                    elif np.array_equal(layer, outer_layer):
                        area_nr_strand = offset_areas
                        f.write("allsel\n")
                        f.write(f"numstr,area,{area_nr_strand}\n")
                        f.write(f"asba,{area_nr_strand+2},{area_nr_strand+1},,keep,keep\n")
                        f.write("allsel\n")
                        f.write(f"asba,{area_nr_strand+1},{area_nr_strand},,delete,keep\n")
                        f.write(f"ADELE,{area_nr_strand+2}\n")
                        f.write(f"asel,s,,,{area_nr_strand}\n")
                        f.write(f"mat,2 \n")
                        f.write(f"asel,s,,,{area_nr_strand+4}\n")
                        f.write(f"mat,3 \n")
                        f.write(f"cm,m_{stack_nr}_{i+1},area\n")
                        f.write(f"asel,s,,,{area_nr_strand+3}\n")
                        f.write(f"cm,o_{stack_nr}_{i+1},area\n")
                        f.write(f"mat,2 \n")
                        f.write(f"asel,s,,,{area_nr_strand}\n")
                        f.write(f"cm,i_{stack_nr}_{i+1},area\n")
                        f.write(f"mat,3 \n")

                    j_index_old = j_index

        # --- Post-write scan: detect intra-strand adjacent coincident KPs ---
        # APDL refuses to create a line between two coincident keypoints, so any
        # adjacent duplicate inside one strand's outer ring will crash the deck
        # later. Scan the freshly written keypoints file, plot the offending
        # strand's outer ring, and abort with an actionable message.
        try:
            kps = []
            with open(keypoints_file, "r") as fh:
                for line in fh:
                    if not line.startswith("k,"):
                        continue
                    parts = line.strip().split(",")
                    kps.append((int(float(parts[1])), float(parts[2]), float(parts[3])))
            seen = {}
            stack_dups = []
            for kp_id, x, y in kps:
                key = (round(x, 13), round(y, 13))
                if key in seen and abs(kp_id - seen[key]) <= 5:
                    stack_dups.append((seen[key], kp_id, x, y))
                else:
                    seen[key] = kp_id
            if stack_dups:
                print(f"[align_debug] stack {stack_nr}: {len(stack_dups)} intra-strand duplicate(s) detected in {keypoints_file}")
                offset_stacknr_local = (stack_nr - 1) * 3e4
                for prev_id, dup_id, x, y in stack_dups:
                    j_local = dup_id - 10000 - offset_stacknr_local
                    strand_idx = int((j_local - 1) // 108)  # 0-based
                    pos_in_strand = int((j_local - 1) % 108)
                    layer_name = ["inner", "middle", "outer"][pos_in_strand // 36]
                    print(
                        f"  KP {prev_id} == KP {dup_id} at ({x:.10e},{y:.10e}) "
                        f"-> stack {stack_nr}, strand {strand_idx + 1}, "
                        f"layer {layer_name}, pos_in_layer {pos_in_strand % 36}"
                    )
                    if 0 <= strand_idx < len(saved_mappers):
                        try:
                            import matplotlib.pyplot as plt
                            mapper = saved_mappers[strand_idx]
                            CD = mapper.mesh.circumferential_divisions
                            outer = mapper.mapped_nodes[-CD:]
                            # outer ring & bspline are in mm, with NO stack offset.
                            # Duplicate point read from the .txt file is in metres
                            # and includes the (stack_nr-1)*stack_height shift baked
                            # in by the keypoint writer. Strip that offset here so
                            # the red X lands on the actual strand outline.
                            stack_offset_mm = (stack_nr - 1) * stack_height * 1e3
                            x_mm = x * 1e3
                            y_mm = y * 1e3 - stack_offset_mm
                            fig, ax = plt.subplots(figsize=(8, 8))
                            bspl = _bspline_samples(saved_strands[strand_idx], num_points=200)
                            ax.plot(bspl[:, 0], bspl[:, 1], "b--", lw=0.6, label="B-spline")
                            ax.plot(outer[:, 0], outer[:, 1], "k.-", ms=3, lw=0.5, label="outer ring")
                            for k_idx, (xx, yy) in enumerate(outer):
                                ax.text(xx, yy, str(k_idx), fontsize=6, color="gray")
                            ax.plot([x_mm], [y_mm], "rx", ms=14, mew=2,
                                    label=f"DUP @ pos {pos_in_strand % 36}")
                            ax.set_aspect("equal")
                            ax.set_xlabel("x [mm]")
                            ax.set_ylabel("y [mm]  (strand-local, stacking offset removed)")
                            ax.set_title(
                                f"stack {stack_nr} strand {strand_idx + 1} {layer_name} "
                                f"DUP KP {prev_id}=={dup_id}"
                            )
                            ax.legend(loc="best", fontsize=8)
                            fname = debug_dup_plots_dir / f"stack{stack_nr}_strand{strand_idx + 1}_DUP_{prev_id}_{dup_id}.svg"
                            fig.savefig(str(fname), bbox_inches="tight")
                            plt.close(fig)
                            print(f"  -> debug plot: {fname}")
                        except Exception as e:
                            print(f"  -> debug plot failed: {e}")
                # After duplicate detection, re-run pairwise alignment for all pairs involving this strand in plot-only mode
                try:
                    strand_num = strand_idx + 1
                    pairs = []
                    connections_file_path = str(input_dir / f"connections_{stack_nr}.txt")
                    if os.path.exists(connections_file_path):
                        with open(connections_file_path, "r") as cf:
                            for line in cf:
                                if line.startswith("Connection_a"): continue
                                a, b = [int(x) for x in line.strip().split(",")]
                                if a == strand_num or b == strand_num:
                                    pairs.append((a, b))
                    # For each pair, re-run alignment and plot
                    for a, b in pairs:
                        try:
                            # Reconstruct strand objects for this stack
                            s1 = saved_strands[a-1] if a-1 < len(saved_strands) and saved_strands[a-1] is not None else None
                            s2 = saved_strands[b-1] if b-1 < len(saved_strands) and saved_strands[b-1] is not None else None
                            if s1 is None or s2 is None:
                                print(f"  -> Skipping pair ({a},{b}): missing strand mesh")
                                continue
                            # Re-run alignment (plot-only, catch errors)
                            try:
                                m1 = saved_mappers[a-1] if a-1 < len(saved_mappers) and saved_mappers[a-1] is not None else None
                                m2 = saved_mappers[b-1] if b-1 < len(saved_mappers) and saved_mappers[b-1] is not None else None
                                if m1 is None or m2 is None:
                                    print(f"  -> Skipping pair ({a},{b}): missing mapper")
                                    continue
                                mesh = ConformalRutherfordMesh.from_existing(s1, m1, s2, m2)
                                base = f"stack{stack_nr}_pair_{a}_{b}_ALLCONTACTS"
                                mesh.plot_conformal_mesh(str(debug_dup_plots_dir / f"{base}_mesh.svg"))
                                mesh.plot_outer_nodes_and_bsplines(str(debug_dup_plots_dir / f"{base}_outer.svg"))
                                print(f"  -> plotted ALLCONTACTS for pair ({a},{b})")
                            except Exception as e:
                                print(f"  -> contact plot failed for pair ({a},{b}): {e}")
                        except Exception as e:
                            print(f"  -> error in ALLCONTACTS plotting for pair ({a},{b}): {e}")
                except Exception as e:
                    print(f"  -> failed to plot all contacts for strand {strand_idx+1}: {e}")
                raise RuntimeError(
                    f"Intra-strand duplicate keypoints detected in stack {stack_nr}; "
                    f"see {debug_dup_plots_dir} for debug plots. "
                    f"APDL would fail with 'Cannot create line between coincident keypoints'."
                )
        except FileNotFoundError:
            pass

        insulation = InsulationLayer(keypoints_file, stack_nr, plots_dir=plots_dir)
        insulation.read_keypoints()
        insulation.generate_alpha_shape(alpha=500)
        insulation.select_points_close_to_polygon(tolerance_distance=7.5e-6)
        insulation.scale_polygon(
            offset_distance=100e-6, stack_height=stack_height, stack_nr=stack_nr, stacking=stacking,
            debug_plots=debug_plots,
        )

        # Defer write_keypoints_to_file and plot until after interface alignment (below)
        all_insulation_layers[stack_nr] = insulation

        # Store this stack's data so we can draw the combined all-stacks figure at the end
        all_stacks_mappers[stack_nr] = list(saved_mappers)

        contact_nodes = ContactNodes(
            connections_file,
            str(input_dir / f"contacts_strands_{stack_nr}.txt"),
            stack_nr,
            elem_per_strand=n_elements_per_strand,
            num_strands=n_parts,
            n_stacks=n_stacks,
        )
        contact_nodes.read_connections()
        contact_nodes.write_contacts()

    # ------------------------------------------------------------------
    # Align interface KPs across adjacent stacks, then write KP files
    # ------------------------------------------------------------------
    ordered_layers = [all_insulation_layers[s] for s in sorted(all_insulation_layers)]
    interface_y_list = align_interface_keypoints(ordered_layers) if len(ordered_layers) > 1 else []

    for stack_nr in sorted(all_insulation_layers):
        insulation = all_insulation_layers[stack_nr]
        insulation.write_keypoints_to_file(str(input_dir / f"keypoints_insulation_nodes_{stack_nr}.txt"))
        plot_cable_overview(
            all_stacks_mappers[stack_nr], stack_nr,
            str(plots_dir / f"overview_stack_{stack_nr}.svg"),
            insulation_polygon=insulation.filtered_outerpoints,
            stack_height_mm=stack_height * 1e3,
        )
        all_insulation_polygons[stack_nr] = insulation.filtered_outerpoints

    # Write APDL NUMMRG snippet for each stack interface
    nummrg_file = input_dir / "stack_interface_nummrg.inp"
    with open(nummrg_file, "w") as _nf:
        _nf.write("! Auto-generated: merge coincident KPs at stack-to-stack interfaces\n")
        _nf.write("! Called from 3-mesh.inp before any AMESH\n")
        eps = 1e-10
        for y_if in interface_y_list:
            _nf.write(f"ksel,s,loc,y,{y_if - eps:.15e},{y_if + eps:.15e}\n")
            _nf.write("NUMMRG,ALL,1e-10\n")
            _nf.write("allsel\n")
    print(f"  Stack interface NUMMRG written to: {nummrg_file}")

    # Write cable dimensions including insulation to a text file (use last stack's values)
    insulation = all_insulation_layers[max(all_insulation_layers)]
    if hasattr(insulation, 'cable_height_incl_insulation_m') and hasattr(insulation, 'cable_width_incl_insulation_m'):
        dims_file = output_dir / "cable_dimensions_incl_insulation.txt"
        with open(dims_file, "w") as _f:
            _f.write("Cable dimensions including insulation\n")
            _f.write("=====================================\n")
            _f.write(f"Height (incl. insulation): {insulation.cable_height_incl_insulation_m * 1e3:.6f} mm\n")
            _f.write(f"Width  (incl. insulation): {insulation.cable_width_incl_insulation_m * 1e3:.6f} mm\n")
        print(f"Cable dimensions written to: {dims_file}")

    # Combined figure showing every stack in one grid — saved into the plots subfolder
    plot_all_stacks_overview(
        all_stacks_mappers, all_insulation_polygons,
        str(plots_dir / "overview_all_stacks.svg"),
        stack_height_mm=stack_height * 1e3,
    )

    plot_final_geometry(
        all_stacks_mappers, all_insulation_polygons,
        str(plots_dir / "final_geometry.svg"),
        stack_height_mm=stack_height * 1e3,
    )

    return all_stacks_mappers, all_insulation_polygons


def plot_multi_cable_overview(cables_data, filename):
    """Grid figure showing every cable (rows) × every stack (columns).

    Parameters
    ----------
    cables_data : dict
        ``{cable_label: {'stacks': {stack_nr: {'mappers': [...], 'insulation': [...]}}}``
        Each mapper entry is either ``None`` or a dict with keys
        ``'nodes'`` (list, shape N×2, in mm) and ``'circ_div'`` (int).
        Insulation points are in metres.
    filename : str
        Output file path (SVG recommended).
    """
    cable_labels = sorted(cables_data.keys())
    all_stack_nrs = sorted({
        stack_nr
        for cd in cables_data.values()
        for stack_nr in cd["stacks"].keys()
    })
    n_cables = len(cable_labels)
    n_stacks = len(all_stack_nrs)
    if n_cables == 0 or n_stacks == 0:
        return

    colors = plt.cm.tab20.colors
    fig, axes = plt.subplots(
        n_cables, n_stacks,
        figsize=(3 * n_stacks, 3 * n_cables),
        squeeze=False,
    )

    for row, cable_label in enumerate(cable_labels):
        cd = cables_data[cable_label]
        stack_height_mm = cd.get("stack_height_mm", 0.0)
        for col, stack_nr in enumerate(all_stack_nrs):
            ax = axes[row][col]
            stack_data = cd["stacks"].get(stack_nr)
            if stack_data is None:
                ax.set_visible(False)
                continue

            y_offset_mm = (stack_nr - 1) * stack_height_mm
            for i, mapper_data in enumerate(stack_data["mappers"]):
                if mapper_data is None:
                    continue
                nodes = np.array(mapper_data["nodes"]).copy()
                circ_div = mapper_data["circ_div"]
                outer_layer = nodes[-circ_div:]
                outer_layer[:, 1] += y_offset_mm
                outline = np.vstack([outer_layer, outer_layer[0]])
                color = colors[i % len(colors)]
                ax.fill(outer_layer[:, 0], outer_layer[:, 1], alpha=0.3, color=color)
                ax.plot(outline[:, 0], outline[:, 1], color=color, linewidth=1.0)
                centroid = outer_layer.mean(axis=0)
                ax.text(centroid[0], centroid[1], str(i + 1),
                        ha="center", va="center", fontsize=5, color="black")

            ins = stack_data.get("insulation")
            if ins and len(ins) > 2:
                ins_mm = np.array(ins) * 1e3
                closed = np.vstack([ins_mm, ins_mm[0]])
                ax.plot(closed[:, 0], closed[:, 1], color="black",
                        linewidth=1.5, linestyle="--")

            if row == 0:
                ax.set_title(f"Stack {stack_nr}", fontsize=8)
            if col == 0:
                ax.set_ylabel(cable_label, fontsize=8, labelpad=4)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=5)

    fig.suptitle("All cables — all stacks overview", fontsize=12)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


# Example usage
if __name__ == "__main__":
    import os as _os
    from pathlib import Path as _Path

    _stack_dir = _Path(r"c:\stack11_areas_good_ratios_CunonCu1.2_boxsubmodel")
    _stack_dir.mkdir(parents=True, exist_ok=True)
    run(stack_dir=_stack_dir, n_stacks=10, n_parts=21, output_dir=_stack_dir)
