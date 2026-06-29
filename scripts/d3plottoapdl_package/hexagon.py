import numpy as np
import matplotlib.pyplot as plt

class HexagonMesh:
    def __init__(self, n_edge=12, n_height=6, width=1.0, rotate = 0, height=None):
        self.n_edge = n_edge
        self.n_height = n_height
        self.width = width
        if height is None:
            self.height = width / np.cos(np.deg2rad(30)) / 2
        else:
            self.height = height
        self.nodes = []
        self.elements = []
        self.grid = {}
        self._generate_mesh()

    def shift_nodes(self, indices, delta_y_func):
        for i in indices:
            x, y = self.nodes[i]
            self.nodes[i] = (x, y + delta_y_func(i))

    def rotate_nodes(self, angle_rad):
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        self.nodes = [
            (cos_a * x - sin_a * y, sin_a * x + cos_a * y)
            for x, y in self.nodes
        ]

    def _generate_mesh(self):
        n_edge = self.n_edge
        n_height = self.n_height
        width = self.width
        height = self.height
        nodes = []
        grid = {}
        idx = 0
        x0 = -width / 2
        y0 = -height / 2
        for v in range(n_height + 1):
            y = y0 + v * height / n_height
            for u in range(n_edge + 1):
                x = x0 + u * width / n_edge
                nodes.append((x, y))
                grid[(v, u)] = idx
                idx += 1

        horizontal_step = width / n_edge

        angle_deg_30 = 30
        angle_deg_20 = 20
        angle_deg_10 = 10

        angle_rad_30 = np.deg2rad(angle_deg_30)
        angle_rad_20 = np.deg2rad(angle_deg_20)
        angle_rad_10 = np.deg2rad(angle_deg_10)

        step_30 = horizontal_step * np.tan(angle_rad_30)
        step_20 = horizontal_step * np.tan(angle_rad_20)
        step_10 = horizontal_step * np.tan(angle_rad_10)

        # 0-6 down, 6-12 up (main angle)
        self.nodes = nodes
        self.shift_nodes(range(0, 7), lambda i: -i * step_30)
        y6 = self.nodes[6][1]
        self.shift_nodes(range(7, 13), lambda i: (i - 6) * step_30 + (y6 - self.nodes[i][1]))

        # 13-19 down, 19-25 up (20 deg region)
        y13 = self.nodes[13][1]
        self.shift_nodes(range(13, 20), lambda i: -(i - 13) * step_20 + (y13 - self.nodes[i][1]))
        y19 = self.nodes[19][1]
        self.shift_nodes(range(20, 26), lambda i: (i - 19) * step_20 + (y19 - self.nodes[i][1]))

        # 26-32 down, 32-38 up (10 deg region)
        y26 = self.nodes[26][1]
        self.shift_nodes(range(26, 33), lambda i: -(i - 26) * step_10 + (y26 - self.nodes[i][1]))
        y32 = self.nodes[32][1]
        self.shift_nodes(range(33, 39), lambda i: (i - 32) * step_10 + (y32 - self.nodes[i][1]))

        # 52-58 up, 58-64 down (10 deg region)
        y52 = self.nodes[52][1]
        self.shift_nodes(range(52, 59), lambda i: (i - 52) * step_10 + (y52 - self.nodes[i][1]))
        y58 = self.nodes[58][1]
        self.shift_nodes(range(59, 65), lambda i: -(i - 58) * step_10 + (y58 - self.nodes[i][1]))

        # 65-71 up, 71-77 down (20 deg region)
        y65 = self.nodes[65][1]
        self.shift_nodes(range(65, 72), lambda i: (i - 65) * step_20 + (y65 - self.nodes[i][1]))
        y71 = self.nodes[71][1]
        self.shift_nodes(range(72, 78), lambda i: -(i - 71) * step_20 + (y71 - self.nodes[i][1]))

        # 78-84 up, 84-90 down (main angle)
        self.shift_nodes(range(78, 85), lambda i: (i - 78) * step_30)
        y84 = self.nodes[84][1]
        self.shift_nodes(range(85, 91), lambda i: -(i - 84) * step_30 + (y84 - self.nodes[i][1]))

        # Elements
        elements = []
        for v in range(n_height):
            for u in range(n_edge):
                n1 = grid[(v, u)]
                n2 = grid[(v, u + 1)]
                n3 = grid[(v + 1, u + 1)]
                n4 = grid[(v + 1, u)]
                elements.append((n1, n2, n3, n4))
        self.elements = elements
        self.grid = grid

    def plot(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        x_vals, y_vals = zip(*self.nodes)
        ax.scatter(x_vals, y_vals, color='blue', s=15, label='nodes')
        for elem in self.elements:
            elem_nodes = [self.nodes[i] for i in elem] + [self.nodes[elem[0]]]
            x_e, y_e = zip(*elem_nodes)
            ax.plot(x_e, y_e, color='black', linewidth=0.7)
        for idx, (x, y) in enumerate(self.nodes):
            ax.text(x, y, str(idx), fontsize=8, color='red', ha='center', va='center')
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Structured Mesh Inside Rectangle (Two Meshes)")
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        plt.show()

if __name__ == "__main__":
    n_edge = 12  # 12 elements along width
    n_height = 6 # 6 elements along height
    width = 1  # rectangle width
    height = width/np.cos(np.deg2rad(30))/2 # rectangle height

    print(f"Rectangle dimensions: width={width}, height={height}")

    mesh = HexagonMesh(n_edge=n_edge, n_height=n_height, width=width, height=height)
    mesh.rotate_nodes(np.deg2rad(90))
    mesh.plot()