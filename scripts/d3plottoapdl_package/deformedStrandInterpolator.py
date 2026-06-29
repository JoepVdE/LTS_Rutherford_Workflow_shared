import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import alphashape
from shapely.geometry import Point

class DeformedStrandInterpolator:

    def __init__(self, filepath):
        """Initializes the interpolator by reading external perimeter nodes from a file."""
        self.filepath = filepath
        self.nodes = self.read_nodes()
        self.spline_function = None

    def read_nodes(self):
        """Reads the CSV and returns the alpha-shape boundary as (N, 2) in CCW order.

        The upstream ParaView extraction (extract_coordinates_stack_sort.py)
        sometimes leaks interior pole nodes from the strand's O-grid into the
        CSV. We filter to the true outer perimeter here so every consumer of
        `self.nodes` (B-spline fit, debug plots, downstream geometry checks)
        sees only the boundary.
        """
        data = pd.read_csv(self.filepath, header=None, skiprows=1)
        raw = data.to_numpy()
        coords = raw[:, :2]
        try:
            shape = alphashape.alphashape(coords, 1.0)
            boundary = [
                c for c in coords
                if shape.exterior.contains(Point(c))
                or shape.exterior.touches(Point(c))
            ]
            boundary = np.array(boundary)
            if boundary.size == 0:
                return coords  # alpha returned an empty shell; fall back to raw
            centre = boundary.mean(axis=0)
            angles = np.arctan2(boundary[:, 1] - centre[1],
                                boundary[:, 0] - centre[0])
            return boundary[np.argsort(angles)]
        except Exception:
            # Defensive: never make the pipeline crash because of upstream noise
            return coords
    
    def fit_bspline(self):
        """Fits a B-spline curve by interpolating through the external perimeter nodes with periodic boundary conditions."""
        x, y = self.nodes[:, 0], self.nodes[:, 1]
        
        
        
        
        # Step 3: Keep only boundary points
        coords = np.column_stack((x, y))
        
        alpha = 1
        alpha_shape = alphashape.alphashape(coords, alpha)

        boundary_coords = []
        for coord in coords:
            point = Point(coord)
            if alpha_shape.exterior.contains(point) or alpha_shape.exterior.touches(point):
                boundary_coords.append(coord)

        
        
        
        # # Plot the alpha shape exterior and interior coordinates
        # plt.figure(figsize=(8, 8))
        # plt.plot(*alpha_shape.exterior.xy, label='Alpha Shape Exterior', color='blue')
        # plt.scatter(coords[:, 0], coords[:, 1], color='gray', label='All Points')
        # plt.scatter(np.array(boundary_coords)[:, 0], np.array(boundary_coords)[:, 1], color='red', label='Boundary Points')
        # plt.xlabel("X Coordinate")
        # plt.ylabel("Y Coordinate")
        # plt.title("Alpha Shape Exterior and Boundary Points")
        # plt.legend()
        # plt.show()
        
        boundary_coords = np.array(boundary_coords)
        # Sort boundary coordinates in clockwise order
        center = np.mean(boundary_coords, axis=0)
        angles = np.arctan2(boundary_coords[:, 1] - center[1], boundary_coords[:, 0] - center[0])
        boundary_coords = boundary_coords[np.argsort(angles)]
        x, y = boundary_coords[:, 0], boundary_coords[:, 1]
    
    
        t = np.linspace(0, 1, len(x), endpoint=False)  # Parametric space
        
        # Using CubicSpline with periodic boundary conditions to ensure smooth closure
        t = np.append(t, 1)  # Ensure the parametric space forms a loop/
        x = np.append(x, x[0])  # Append the first x-coordinate to the end
        y = np.append(y, y[0])  # Append the first y-coordinate to the end
        

        spline_x = CubicSpline(t, x, bc_type='periodic')
        spline_y = CubicSpline(t, y, bc_type='periodic')
        
        # # Plot the parametric space and the coordinates
        # plt.figure(figsize=(12, 6))

        # plt.plot(x, y, 'o-', label='Original Nodes')
        # plt.xlabel('X Coordinate')
        # plt.ylabel('Y Coordinate')
        # plt.title('Original Nodes in XY Space')

        # for i, (xi, yi) in enumerate(zip(x, y)):
        #     plt.text(xi, yi, str(i), fontsize=12, ha='right')
        
        # #also plot the b-spline
        # plt.plot(spline_x(t), spline_y(t), label='B-Spline Interpolation')
        # plt.legend()
        # plt.show()
        

        
        self.spline_function = (spline_x, spline_y)
    
    # def fit_bspline(self):
    #     """Fits a B-spline curve by interpolating through the external perimeter nodes with periodic boundary conditions."""
    #     x, y = self.nodes[:, 0], self.nodes[:, 1]
    #     t = np.linspace(0, 1, len(x), endpoint=False)  # Parametric space
        
    #     # Using CubicSpline with periodic boundary conditions to ensure smooth closure
    #     spline_x = CubicSpline(t, x, bc_type='periodic')
    #     spline_y = CubicSpline(t, y, bc_type='periodic')
        
    #     self.spline_function = (spline_x, spline_y)
    
    def evaluate_bspline(self, num_points=200):
        """Evaluates the B-spline curve at specified number of points."""
        if self.spline_function is None:
            raise ValueError("Spline has not been fitted. Call fit_bspline() first.")
        
        t_new = np.linspace(0, 1, num_points, endpoint=True)
        x_new = self.spline_function[0](t_new)
        y_new = self.spline_function[1](t_new)
        
        # np.append(x_new, x_new[0])
        # np.append(y_new, y_new[0])
        
        return np.array([x_new, y_new]).T
    
    def plot_interpolation(self, num_points=100):
        """Plots the original nodes and the interpolated B-spline curve with equal axis scale."""
        plt.figure(figsize=(6, 6))
        
        # Plot original nodes
        x, y = self.nodes[:, 0], self.nodes[:, 1]
        plt.scatter(x, y, color='red', label='Original Nodes')
        
        # Plot B-spline interpolation
        interpolated_points = self.evaluate_bspline(num_points)
        plt.plot(interpolated_points[:, 0], interpolated_points[:, 1], color='blue', label='B-Spline Interpolation')
        
        # Scatter plot of interpolated points with numbering
        for i, (x, y) in enumerate(interpolated_points):
            plt.scatter(x, y, color='green')
            plt.text(x, y, str(i), fontsize=12, ha='right')
        
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Deformed Strand B-Spline Interpolation (Closed & Smooth Curve)")
        plt.legend()
        plt.axis('equal')  # Ensure equal scaling of x and y axes
        plt.show()

if __name__ == "__main__":
    interpolator = DeformedStrandInterpolator("./Stack_4_Part11.csv")
    interpolator.fit_bspline()
    interpolator.plot_interpolation()