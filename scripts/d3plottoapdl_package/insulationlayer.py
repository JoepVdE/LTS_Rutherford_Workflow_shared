import numpy as np
import alphashape
from shapely.geometry import Polygon, Point
from shapely.affinity import scale
import matplotlib.pyplot as plt

class InsulationLayer:
    def __init__(self, keypoints_file, stack_nr, plots_dir=None):
        self.keypoints_file = keypoints_file
        self.coordinates = []
        self.selected_points = []
        self.filtered_outerpoints = []
        self.outer_polygon = None
        self.stack_nr = stack_nr
        self.offset_stacknr = (stack_nr-1)*3e4  # Reduced from 1e6 to match APDL stack_offset
        self.plots_dir = plots_dir

    def read_keypoints(self):
        """Reads keypoints from the file and extracts coordinates."""
        with open(self.keypoints_file, 'r') as file:
            keypoints_data = file.readlines()

        for line in keypoints_data:
            if line.startswith('k,'):
                parts = line.split(',')
                x, y = float(parts[2]), float(parts[3])
                self.coordinates.append((x, y))

    def generate_alpha_shape(self, alpha=500):
        """Generates an alpha shape (concave hull) from the coordinates."""
        concave_hull = alphashape.alphashape(self.coordinates, alpha)
        if concave_hull.geom_type == 'Polygon':
            self.outer_polygon = Polygon(concave_hull.exterior.coords)
        return concave_hull

    def select_points_close_to_polygon(self, tolerance_distance=7.5e-6):
        """Selects points close to the polygon line."""
        if not self.outer_polygon:
            raise ValueError("Outer polygon not generated yet.")
        
        for coord in self.coordinates:
            point = Point(coord)
            if self.outer_polygon.exterior.distance(point) <= tolerance_distance:
                self.selected_points.append(coord)

        # Sort selected points in a clockwise manner
        center = np.mean(self.selected_points, axis=0)
        self.selected_points = sorted(
            self.selected_points,
            key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0])
        )

        print(f"  Stack {self.stack_nr}: inner insulation polygon has {len(self.selected_points)} points")

        # Validate inner polygon: fix self-intersections
        if len(self.selected_points) > 2:
            temp_poly = Polygon(self.selected_points)
            if not temp_poly.is_valid:
                print(f"  Stack {self.stack_nr}: fixing self-intersecting inner insulation polygon")
                temp_poly = temp_poly.buffer(0)
                if temp_poly.geom_type == 'MultiPolygon':
                    temp_poly = max(temp_poly.geoms, key=lambda p: p.area)
                self.selected_points = list(temp_poly.exterior.coords[:-1])


    # def scale_polygon(self, offset_distance=150e-6, stack_height=.93e-3*2, omit_distance=5e-5,stack_nr=1):
    #     """Scales the polygon to create an offset loop and omits points that are too close."""
    #     if not self.outer_polygon:
    #         return

    #     min_y = min([pt[1] for pt in self.outer_polygon.exterior.coords])
    #     max_y = max([pt[1] for pt in self.outer_polygon.exterior.coords])
    #     min_x = min([pt[0] for pt in self.outer_polygon.exterior.coords])
    #     max_x = max([pt[0] for pt in self.outer_polygon.exterior.coords])
    #     horizontal_distance = max_x - min_x
    #     vertical_distance = max_y - min_y
    #     print(f"Vertical distance of outer_polygon: {vertical_distance:.2e}")
    #     print(f"Horizontal distance of outer_polygon: {horizontal_distance:.2e}")

    #     thickness_insulation = stack_height / 2 - vertical_distance / 2
    #     print(f"Thickness of insulation layer: {thickness_insulation:.2e}")
    #     scaled_polygon = self.outer_polygon.buffer(thickness_insulation)

    #     # Plot outer polygon coordinates (convert to mm)
    #     plt.figure(figsize=(8, 6))
    #     x, y = zip(*self.outer_polygon.exterior.coords)
    #     x_mm = np.array(x) * 1e3
    #     y_mm = np.array(y) * 1e3
    #     plt.plot(x_mm, y_mm, 'b-', label='Outer Polygon')

    #     # Plot bounding box (convert to mm)
    #     box_x = [min_x, max_x, max_x, min_x, min_x]
    #     box_y = [min_y, min_y, max_y, max_y, min_y]
    #     box_x_mm = np.array(box_x) * 1e3
    #     box_y_mm = np.array(box_y) * 1e3
    #     plt.plot(box_x_mm, box_y_mm, 'r--', label='Bounding Box')
        
    #     # Plot bounding box with extra thickness_insulation added (convert to mm)
    #     box_x_extra = [min_x - thickness_insulation, max_x + thickness_insulation,
    #                    max_x + thickness_insulation, min_x - thickness_insulation, min_x - thickness_insulation]
    #     box_y_extra = [min_y - thickness_insulation, min_y - thickness_insulation,
    #                    max_y + thickness_insulation, max_y + thickness_insulation, min_y - thickness_insulation]
    #     box_x_extra_mm = np.array(box_x_extra) * 1e3
    #     box_y_extra_mm = np.array(box_y_extra) * 1e3
    #     plt.plot(box_x_extra_mm, box_y_extra_mm, 'm--', label='Box + Insulation Thickness')

    #     # Plot scaled polygon (convert to mm)
    #     if scaled_polygon.geom_type == 'Polygon':
    #         sx, sy = scaled_polygon.exterior.xy
    #         sx_mm = np.array(sx) * 1e3
    #         sy_mm = np.array(sy) * 1e3
    #         plt.plot(sx_mm, sy_mm, 'g-', label='Scaled Polygon')

    #     plt.xlabel('X [mm]')
    #     plt.ylabel('Y [mm]')
    #     plt.title('Outer Polygon, Bounding Box, and Scaled Polygon')
    #     plt.legend()
    #     plt.grid(True)
    #     # plt.show()
    #     plt.savefig(f"outer_polygon_and_box_{self.stack_nr}.png")
    #     # plt.close()

    

    #     # Check if any point of the scaled polygon is closer than margin to the bounding box with extra thickness
    #     margin =20e-6
        

    #     for pt in scaled_polygon.exterior.coords:
    #         x, y = pt
    #         # Move points close to top/bottom edge to max_y + thickness_insulation or min_y - thickness_insulation
    #         if abs(y - (max_y + thickness_insulation)) <= margin:
    #             y = max_y + thickness_insulation
    #         elif abs(y - (min_y - thickness_insulation)) <= margin:
    #             y = min_y - thickness_insulation
    #         # Move points close to left/right edge to min_x - thickness_insulation or max_x + thickness_insulation
    #         if abs(x - (max_x + thickness_insulation)) <= margin:
    #             x = max_x + thickness_insulation
    #         elif abs(x - (min_x - thickness_insulation)) <= margin:
    #             x = min_x - thickness_insulation
    #         self.filtered_outerpoints.append((x, y))
    def scale_polygon(self, offset_distance=150e-6, stack_height=.93e-3*2, omit_distance=5e-5, stack_nr=1, stacking=True, debug_plots=False):
        """Scales the polygon to create an offset loop and omits points that are too close."""
        if not self.outer_polygon:
            return

        min_y = min([pt[1] for pt in self.outer_polygon.exterior.coords])
        max_y = max([pt[1] for pt in self.outer_polygon.exterior.coords])
        min_x = min([pt[0] for pt in self.outer_polygon.exterior.coords])
        max_x = max([pt[0] for pt in self.outer_polygon.exterior.coords])
        horizontal_distance = max_x - min_x
        vertical_distance = max_y - min_y
        print(f"Vertical distance of outer_polygon: {vertical_distance:.2e}")
        print(f"Horizontal distance of outer_polygon: {horizontal_distance:.2e}")

        thickness_insulation = stack_height / 2 - vertical_distance / 2
        print(f"Thickness of insulation layer: {thickness_insulation:.2e}")
        # Use convex hull before buffering so top/bottom edges are straight
        # (alpha shape has concavities between strands that prevent the buffer
        #  from reaching stack_height/2 everywhere)
        convex_outer = self.outer_polygon.convex_hull
        scaled_polygon = convex_outer.buffer(thickness_insulation)

        # Clip the buffered polygon to exact cable height boundaries so stacks touch perfectly
        from shapely.geometry import box as shapely_box
        # Keypoints already include vertical stacking offset: (stack_nr-1)*stack_height
        center_y = (stack_nr - 1) * stack_height if stacking else 0
        clip_box = shapely_box(
            min_x - 2 * thickness_insulation,       # wide enough to keep rounded sides
            center_y - stack_height / 2,             # exact bottom boundary for this stack
            max_x + 2 * thickness_insulation,
            center_y + stack_height / 2              # exact top boundary for this stack
        )
        scaled_polygon = scaled_polygon.intersection(clip_box)
        if scaled_polygon.is_empty:
            print(f"  Stack {self.stack_nr}: WARNING clip_box produced empty polygon!")
        elif scaled_polygon.geom_type != 'Polygon':
            print(f"  Stack {self.stack_nr}: WARNING clip_box produced {scaled_polygon.geom_type}")
        else:
            print(f"  Stack {self.stack_nr}: clipped polygon has {len(scaled_polygon.exterior.coords)-1} points")

        # Store cable dimensions including insulation for external use
        self.cable_height_incl_insulation_m = stack_height
        self.cable_width_incl_insulation_m = horizontal_distance + 2 * thickness_insulation
        self.insulation_thickness_m = thickness_insulation

        # Calculate centered box parameters
        # Use stack_height for vertical offset only if stacking is enabled
        center_y_stack = stack_height * (stack_nr - 1) if stacking else 0
        half_height = stack_height / 2
        half_width = horizontal_distance / 2  # Keep original width
        
        # Create exact centered boxes
        # Basic bounding box centered at origin, moved up by stack position
        box_x = [-half_width, half_width, half_width, -half_width, -half_width]
        box_y = [center_y_stack - half_height, center_y_stack - half_height, 
                center_y_stack + half_height, center_y_stack + half_height, center_y_stack - half_height]
        
        # Box with insulation thickness added
        box_x_extra = [-half_width - thickness_insulation, half_width+thickness_insulation,
                    half_width+thickness_insulation, -half_width - thickness_insulation, -half_width - thickness_insulation]
        box_y_extra = [center_y_stack - half_height, center_y_stack - half_height,
                    center_y_stack + half_height, center_y_stack + half_height, 
                    center_y_stack - half_height]

        if debug_plots:
            # Plot outer polygon coordinates (convert to mm)
            plt.figure(figsize=(8, 6))
            x, y = zip(*self.outer_polygon.exterior.coords)
            x_mm = np.array(x) * 1e3
            y_mm = np.array(y) * 1e3
            plt.plot(x_mm, y_mm, 'b-', label='Outer Polygon')

            # Plot centered bounding box (convert to mm)
            box_x_mm = np.array(box_x) * 1e3
            box_y_mm = np.array(box_y) * 1e3
            plt.plot(box_x_mm, box_y_mm, 'r--', label=f'Centered Box (Height: {stack_height*1e3:.2f}mm)')

            # Plot centered box with insulation thickness (convert to mm)
            box_x_extra_mm = np.array(box_x_extra) * 1e3
            box_y_extra_mm = np.array(box_y_extra) * 1e3
            plt.plot(box_x_extra_mm, box_y_extra_mm, 'm--', label='Centered Box + Insulation')

            # Plot scaled polygon (convert to mm)
            if scaled_polygon.geom_type == 'Polygon':
                sx, sy = scaled_polygon.exterior.xy
                sx_mm = np.array(sx) * 1e3
                sy_mm = np.array(sy) * 1e3
                plt.plot(sx_mm, sy_mm, 'g-', label='Scaled Polygon')

            plt.xlabel('X [mm]')
            plt.ylabel('Y [mm]')
            plt.title(f'Centered Boxes - Stack {stack_nr} (Center Y: {center_y_stack*1e3:.2f}mm)')
            plt.legend()
            plt.grid(True)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)  # Origin reference
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)  # Origin reference
            plt.axhline(y=center_y_stack*1e3, color='orange', linestyle=':', alpha=0.5, label=f'Stack {stack_nr} Center')
            _plot_path = (str(self.plots_dir / f"outer_polygon_and_box_{self.stack_nr}.png") if self.plots_dir else f"outer_polygon_and_box_{self.stack_nr}.png")
            plt.savefig(_plot_path)

        # Update the margin checking to use box_x and box_y coordinates
        margin = 30e-6
        box_edges_x = set(box_x_extra)
        box_edges_y = set(box_y_extra)
        for pt in scaled_polygon.exterior.coords:
            x, y = pt
            # Snap x to nearest box_x_extra edge if within margin
            for bx in box_edges_x:
                if abs(x - bx) <= margin:
                    x = bx
                    break
            # Snap y to nearest box_y_extra edge if within margin
            for by in box_edges_y:
                if abs(y - by) <= margin:
                    y = by
                    break
            self.filtered_outerpoints.append((x, y))
        




        def omit_close_points(points, min_distance):
            omitted = []
            for pt in points:
                if not omitted:
                    omitted.append(pt)
                else:
                    if np.linalg.norm(np.array(pt) - np.array(omitted[-1])) >= min_distance:
                        omitted.append(pt)
            return omitted

        self.filtered_outerpoints = omit_close_points(self.filtered_outerpoints, omit_distance)
        print(f"  Stack {self.stack_nr}: outer insulation has {len(self.filtered_outerpoints)} points after snapping/omitting")

        # Validate polygon: fix self-intersections caused by snapping to box edges
        if len(self.filtered_outerpoints) > 2:
            temp_poly = Polygon(self.filtered_outerpoints)
            if not temp_poly.is_valid:
                print(f"  Stack {self.stack_nr}: fixing self-intersecting insulation polygon")
                temp_poly = temp_poly.buffer(0)
                if temp_poly.geom_type == 'MultiPolygon':
                    temp_poly = max(temp_poly.geoms, key=lambda p: p.area)
                self.filtered_outerpoints = list(temp_poly.exterior.coords[:-1])

        # Simplify outer polygon. The alpha-shape over densely-packed small-D
        # strands (D < 0.7 mm, 30+ strands per stack) produces wiggly outlines
        # with many short edges. Beyond ~25 vertices APDL's ASBA fails with
        #   ERROR: Poorly defined area. Check for crossed lines.
        #   ERROR: Cannot project lines to surface N. Surface could be twisted
        #          or lines do not lie on the surface.
        # because adjacent edges of the impregnation boundary cannot be
        # projected reliably onto the strand-area boundary surfaces.  The only
        # workable fix is a hard cap on the vertex count: simplify until at
        # most MAX_INSULATION_PTS remain. This may "round off" tight corners,
        # but the impregnation region is bulk material and a millimetre-scale
        # rounding is acceptable; a failed boolean is not.
        MAX_INSULATION_PTS = 25
        if len(self.filtered_outerpoints) > 2:
            n_before = len(self.filtered_outerpoints)
            temp_poly = Polygon(self.filtered_outerpoints)
            # Pick the more aggressive of (3/4 of input) and (hard cap).
            # Keep at least 15 points so the outline still resembles a cable.
            target = max(min(n_before * 3 // 4, MAX_INSULATION_PTS), 15)
            tol = 1e-6
            # Tol cap raised from 5e-5 to 1e-3 so we ALWAYS meet `target`
            # (the cap is the point of this whole block).
            while True:
                simplified = temp_poly.simplify(tol, preserve_topology=True)
                if len(simplified.exterior.coords) - 1 <= target or tol > 1e-3:
                    break
                tol *= 1.5
            self.filtered_outerpoints = list(simplified.exterior.coords[:-1])
            print(f"  Stack {self.stack_nr}: outer insulation reduced from {n_before} to {len(self.filtered_outerpoints)} points (tol={tol:.1e}, target={target})")

        # Plot filtered outer points as a polygon (convert to mm)
        if len(self.filtered_outerpoints) > 2:
            fx, fy = zip(*self.filtered_outerpoints)
            fx_mm = np.array(fx)*1e3
            fy_mm = np.array(fy)*1e3
            plt.plot(fx_mm, fy_mm, 'c-', label='Filtered Outer Points Polygon')
            plt.legend()
            _plot_path = (str(self.plots_dir / f"outer_polygon_and_box_{self.stack_nr}.png") if self.plots_dir else f"outer_polygon_and_box_{self.stack_nr}.png")
            plt.savefig(_plot_path)
        plt.close()




    def write_keypoints_to_file(self, output_file):
        """Writes the selected and scaled keypoints to a file."""
        import pathlib
        stack_nr = self.stack_nr
        _out_dir = pathlib.Path(output_file).parent
        with open(output_file, "w") as f:
            with open(_out_dir / f"inner_insulation_lines_{stack_nr}.txt","w") as f2:
                with open(_out_dir / f"outer_insulation_lines_{stack_nr}.txt","w") as f3:
            
            
                    f2.write(f"! Insulation layer inner lines for stack {stack_nr}\n")
                    f2.write(f"asel,none\n")
                    f2.write(f"ksel,none\n")
                    f2.write(f"cm,inner_insulation_{stack_nr},kp \n")
                    
                    
                    
                    f3.write(f"! Insulation layer outer lines for stack {stack_nr}\n")
                    f3.write(f"asel,none\n")
                    f3.write(f"ksel,none\n")
                    f3.write(f"cm,outer_insulation_{stack_nr},kp \n")


                    iii = int(25e3)
                    offset_iii = iii
                    j_index_old = iii
                    stack_nr = self.stack_nr
                    

                        
                    
                    f.write(f"NUMSTR,area,{offset_iii+self.offset_stacknr}\n")
                    f.write(f"NUMSTR,line,{offset_iii+self.offset_stacknr}\n")
                    for coord in self.selected_points:
                        iii += 1
                        f.write(f"k,{iii+self.offset_stacknr},{coord[0]:.10e},{coord[1]:.10e},0.0\n")
                        
                        
                        f2.write(f"allsel \n") 
                        f2.write(f"asel,none \n")
                        f2.write(f"cmsel,none \n")
                        
                        
                        f2.write(f"cmsel,s,STR_IMP_{stack_nr} \n")
                        f2.write(f"allsel,belo,area \n")
                        f2.write(f"ksel,r,loc,x,{coord[0]:.10e}\n")
                        f2.write(f"ksel,r,loc,y,{coord[1]:.10e}\n")
                        
                        f2.write(f"cmsel,a,inner_insulation_{stack_nr} \n")
                        f2.write(f"cm,inner_insulation_{stack_nr},kp \n")


                        
                        
                        
                    f.write(f"FLST,2,{iii-j_index_old},3\n")
                    for i in range(j_index_old+1, iii+1):
                        f.write(f"FITEM,2,{i+self.offset_stacknr}\n")
                    f.write("A,P51X\n")
                    

                    # f2.write(f"lsel,s,,,{j_index_old+self.offset_stacknr},{iii+self.offset_stacknr}\n")
                        
                    j_index_old = iii
                    
                    
                    f.write(f"NUMSTR,line,{offset_iii+self.offset_stacknr+1e3}\n")

                    for coord in self.filtered_outerpoints:
                        iii += 1
                        f.write(f"k,{iii+self.offset_stacknr},{coord[0]:.10e},{coord[1]:.10e},0.0\n")
                        f3.write(f"ksel,a,loc,x,{coord[0]:.10e}\n")
                        f3.write(f"ksel,a,loc,y,{coord[1]:.10e}\n")
                        
                        
                        
                        f3.write(f"allsel \n") 
                        f3.write(f"asel,none \n")
                        f3.write(f"cmsel,none \n")
                        
                        
                        f3.write(f"cmsel,s,STR_IMP_{stack_nr} \n")
                        f3.write(f"allsel,belo,area \n")
                        f3.write(f"ksel,r,loc,x,{coord[0]:.10e}\n")
                        f3.write(f"ksel,r,loc,y,{coord[1]:.10e}\n")
                        
                        f3.write(f"cmsel,a,outer_insulation_{stack_nr} \n")
                        f3.write(f"cm,outer_insulation_{stack_nr},kp \n")

                    f.write(f"FLST,2,{iii-j_index_old},3\n")
                    for i in range(j_index_old+1, iii+1):
                        f.write(f"FITEM,2,{i+self.offset_stacknr}\n")
                    f.write("A,P51X\n")
                    

                        # f3.write(f"lsel,s,,,{j_index_old+self.offset_stacknr},{iii+self.offset_stacknr}\n")

    def plot_alpha_shape(self, concave_hull):
        """Plots the alpha shape and selected points."""
        plt.figure(figsize=(12, 6))
        plt.plot(*zip(*self.coordinates), 'o', label='Points')
        if concave_hull.geom_type == 'Polygon':
            x, y = concave_hull.exterior.xy
            plt.plot(x, y, 'r-', label='Alpha Concave Hull')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Alpha Concave Hull of Coordinates')
        plt.legend()
        plt.grid(True)
        # plt.show()
        _plot_path = (str(self.plots_dir / "alpha_shape_plot.png") if self.plots_dir else "alpha_shape_plot.png")
        plt.savefig(_plot_path)

    def plot_selected_points(self):
        """Plots the selected points close to the polygon."""
        plt.figure(figsize=(10, 6))
        plt.plot(*zip(*self.coordinates), 'o', label='All Points')
        plt.plot(*zip(*self.selected_points), 'ro', label='Selected Points (Close to Polygon)')
        x, y = self.outer_polygon.exterior.xy
        plt.plot(x, y, 'b-', label='Polygon Line')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Points Close to Polygon Line')
        plt.legend()
        plt.grid(True)
        # plt.show()
        _plot_path = (str(self.plots_dir / f"selected_points_plot_{self.stack_nr}.png") if self.plots_dir else f"selected_points_plot_{self.stack_nr}.png")
        plt.savefig(_plot_path)
        # plt.close()


# ---------------------------------------------------------------------------
# Stack-interface alignment helpers (module-level, not part of InsulationLayer)
# ---------------------------------------------------------------------------

def _insert_interface_points(pts_list, x_targets, y_if, snap_tol=1e-9):
    """Insert (x, y_if) into pts_list for every x in x_targets not already present.

    Finds the pair of consecutive interface-edge vertices that bracket x and
    inserts the new vertex between them, preserving polygon winding order.
    """
    for x_new in x_targets:
        existing_x = [p[0] for p in pts_list if abs(p[1] - y_if) <= snap_tol]
        if any(abs(x_new - ex) < snap_tol for ex in existing_x):
            continue
        iface_indices = [i for i, p in enumerate(pts_list) if abs(p[1] - y_if) <= snap_tol]
        n = len(pts_list)
        inserted = False
        for k in range(len(iface_indices)):
            ia = iface_indices[k]
            ib = iface_indices[(k + 1) % len(iface_indices)]
            xa, xb = pts_list[ia][0], pts_list[ib][0]
            lo, hi = min(xa, xb), max(xa, xb)
            if lo < x_new < hi:
                if ib == (ia + 1) % n:
                    pts_list.insert(ia + 1, (x_new, y_if))
                else:
                    # Wrapping edge: ib wraps around to index 0; append at end
                    pts_list.append((x_new, y_if))
                inserted = True
                break
        if not inserted:
            print(f"    WARNING: could not insert x={x_new:.6e} at y_if={y_if:.6e} into polygon")


def _fix_shallow_interface_angles(lo, hi, y_if, min_angle_deg=30.0, snap_tol=1e-9):
    """At each end of the shared interface segment between lo and hi, check the
    angle between the narrower polygon's side edge and the wider polygon's
    continued horizontal interface edge.  If that angle is < min_angle_deg,
    slide the narrower polygon's corner outward along y_if until the angle
    reaches min_angle_deg, capped at the wider polygon's outer interface end.

    'Narrower/wider from center' means whose interface endpoint is closer to x=0
    (i.e. the polygon with the shorter interface extent on that side).

    Called before _insert_interface_points so that the corrected x position is
    automatically included in the shared x-set.
    """
    import math
    tan_min = math.tan(math.radians(min_angle_deg))

    lo_x_if = sorted(p[0] for p in lo.filtered_outerpoints if abs(p[1] - y_if) <= snap_tol)
    hi_x_if = sorted(p[0] for p in hi.filtered_outerpoints if abs(p[1] - y_if) <= snap_tol)
    if not lo_x_if or not hi_x_if:
        return

    for side in ('right', 'left'):
        if side == 'right':
            x_lo_end = max(lo_x_if)
            x_hi_end = max(hi_x_if)
            sign = 1.0
            if x_lo_end < x_hi_end - snap_tol:
                narrower_pts = lo.filtered_outerpoints
                narrower_nr = lo.stack_nr
                x_narrow = x_lo_end
                x_wider_limit = x_hi_end
            elif x_hi_end < x_lo_end - snap_tol:
                narrower_pts = hi.filtered_outerpoints
                narrower_nr = hi.stack_nr
                x_narrow = x_hi_end
                x_wider_limit = x_lo_end
            else:
                continue  # same extent on this side
        else:  # left
            x_lo_end = min(lo_x_if)
            x_hi_end = min(hi_x_if)
            sign = -1.0
            if x_lo_end > x_hi_end + snap_tol:
                narrower_pts = lo.filtered_outerpoints
                narrower_nr = lo.stack_nr
                x_narrow = x_lo_end
                x_wider_limit = x_hi_end
            elif x_hi_end > x_lo_end + snap_tol:
                narrower_pts = hi.filtered_outerpoints
                narrower_nr = hi.stack_nr
                x_narrow = x_hi_end
                x_wider_limit = x_lo_end
            else:
                continue

        # Locate the corner vertex in the narrower polygon
        corner_idx = None
        for i, p in enumerate(narrower_pts):
            if abs(p[1] - y_if) <= snap_tol and abs(p[0] - x_narrow) < snap_tol:
                corner_idx = i
                break
        if corner_idx is None:
            continue

        # Find the first adjacent vertex NOT on the interface line
        n = len(narrower_pts)
        Q = None
        for delta in (1, -1):
            nb = narrower_pts[(corner_idx + delta) % n]
            if abs(nb[1] - y_if) > snap_tol:
                Q = nb
                break
        if Q is None:
            continue

        x_q, y_q = Q
        dy = abs(y_q - y_if)
        if dy < 1e-12:
            continue

        # Angle between side edge (P→Q) and outward horizontal direction
        dx = x_q - x_narrow
        vec_len = math.sqrt(dx * dx + dy * dy)
        cos_a = max(-1.0, min(1.0, sign * dx / vec_len))
        angle_deg = math.degrees(math.acos(cos_a))

        if angle_deg >= min_angle_deg:
            continue

        # Target position: sign*(x_q - x_new) = dy/tan_min  →  x_new = x_q - sign*dy/tan_min
        x_new = x_q - sign * dy / tan_min

        # Cap at the wider polygon's outer interface end
        if sign > 0:
            x_new = min(x_new, x_wider_limit)
            if x_new <= x_narrow + snap_tol:
                continue
        else:
            x_new = max(x_new, x_wider_limit)
            if x_new >= x_narrow - snap_tol:
                continue

        narrower_pts[corner_idx] = (x_new, y_if)
        print(
            f"  fix_shallow_angle: stack {narrower_nr} {side} corner "
            f"x={x_narrow:.6e} -> {x_new:.6e} m "
            f"(angle {angle_deg:.1f} deg -> >= {min_angle_deg:.0f} deg, "
            f"limit {x_wider_limit:.6e})"
        )


def align_interface_keypoints(layers):
    """Align the x-discretisation of adjacent-stack outer insulation polygons
    at each stack-to-stack interface so that coincident KPs are created.

    After this call, both stacks' filtered_outerpoints share all x-coordinates
    at the interface y.  NUMMRG,ALL in the APDL deck then merges those KPs
    (and derived lines/nodes) without any bonded contact pair.

    Parameters
    ----------
    layers : list[InsulationLayer]
        Ordered by ascending stack number (stack 1 first).

    Returns
    -------
    list[float]
        Interface y-coordinates in metres (one per adjacent pair).
    """
    interface_y_list = []
    snap_tol = 1e-9
    for i in range(len(layers) - 1):
        lo, hi = layers[i], layers[i + 1]
        lo_arr = np.array(lo.filtered_outerpoints)
        hi_arr = np.array(hi.filtered_outerpoints)
        y_top = float(lo_arr[:, 1].max())
        y_bot = float(hi_arr[:, 1].min())
        y_if = (y_top + y_bot) / 2.0
        if abs(y_top - y_bot) > 1e-7:
            print(
                f"  WARNING align_interface_keypoints: stack {lo.stack_nr} top y={y_top:.6e}"
                f" != stack {hi.stack_nr} bottom y={y_bot:.6e} (diff={abs(y_top-y_bot):.2e})"
            )
        interface_y_list.append(y_if)
        _fix_shallow_interface_angles(lo, hi, y_if, snap_tol=snap_tol)
        x_lo = sorted({p[0] for p in lo.filtered_outerpoints if abs(p[1] - y_if) <= snap_tol})
        x_hi = sorted({p[0] for p in hi.filtered_outerpoints if abs(p[1] - y_if) <= snap_tol})
        # Only share x-coords within the overlapping x-range of both interface edges.
        # Points outside the narrower polygon's edge have no matching segment to insert into.
        if x_lo and x_hi:
            x_min = max(min(x_lo), min(x_hi))
            x_max = min(max(x_lo), max(x_hi))
            x_all = sorted(x for x in set(x_lo) | set(x_hi) if x_min <= x <= x_max)
        else:
            x_all = sorted(set(x_lo) | set(x_hi))
        _insert_interface_points(lo.filtered_outerpoints, x_all, y_if, snap_tol)
        _insert_interface_points(hi.filtered_outerpoints, x_all, y_if, snap_tol)
        print(
            f"  Interface stack {lo.stack_nr}-{hi.stack_nr} at y={y_if:.6e} m: "
            f"{len(x_lo)}+{len(x_hi)} -> {len(x_all)} shared x-coords"
        )
    return interface_y_list


if __name__ == "__main__":
    # Define the input and output files
    keypoints_file = "keypoints_nodes_3.txt"  # Input file containing keypoints
    output_file = "keypoints_insulation_nodes_3.txt"  # Output file for processed keypoints
    stack_nr = 3  # Set the stack number here or get from user/argument

    # Create an instance of the InsulationLayer class
    insulation = InsulationLayer(keypoints_file, stack_nr)

    # Step 1: Read keypoints from the input file
    insulation.read_keypoints()

    # Step 2: Generate the alpha shape (concave hull) with a specified alpha value
    alpha = 500  # Adjust alpha as needed
    concave_hull = insulation.generate_alpha_shape(alpha=alpha)

    # Step 3: Select points close to the polygon with a specified tolerance distance
    tolerance_distance = 7.5e-6  # Adjust tolerance as needed
    insulation.select_points_close_to_polygon(tolerance_distance=tolerance_distance)

    # Step 4: Scale the polygon to create an offset loop
    offset_distance = 100e-6  # Adjust offset as needed
    insulation.scale_polygon(offset_distance=offset_distance)

    # Step 5: Write the selected and scaled keypoints to the output file
    insulation.write_keypoints_to_file(output_file)

    # Step 6: Plot the alpha shape and save the plot
    insulation.plot_alpha_shape(concave_hull)

    # Step 7: Plot the selected points and save the plot
    insulation.plot_selected_points()

    print(f"Processing complete. Results saved to {output_file}.")