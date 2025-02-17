import numpy as np
import itertools
from scipy.spatial import ConvexHull

class Face_Finder:
    def __init__(self, vertices, faces):
        """
        Initialize the STL object with vertices and faces.
        :param vertices: np.array of shape (N,3) containing vertex coordinates.
        :param faces: np.array of shape (M,3) containing indices of vertices forming faces.
        """
        self.vertices = np.array(vertices)
        self.faces = np.array(faces)

    def generate_combinations(self):
        """
        Generate all unique vertex triplet combinations.
        :return: np.array of valid 3-vertex combinations.
        """
        all_combinations = list(itertools.combinations(range(len(self.vertices)), 3))
        return np.array(all_combinations)

    def check_inside_mesh(self, test_points):
        """
        Checks if given points are inside the convex hull of the object.
        :param test_points: np.array of shape (N,3) containing points to check.
        :return: Boolean array indicating if points are inside the mesh.
        """
        hull = ConvexHull(self.vertices)
        return np.all(hull.equations[:, :-1] @ test_points.T + hull.equations[:, -1] <= 0, axis=0)

    def find_external_faces(self):
        """
        Determine valid and invalid face combinations for the object.
        :return: Tuple of (valid_positions, invalid_positions, combinations)
        """
        combinations = self.generate_combinations()
        valid_positions = []
        invalid_positions = []

        for combo in combinations:
            V1, V2, V3 = self.vertices[combo]
            sample_points = self.generate_sample_points(V1, V2, V3)
            inside = self.check_inside_mesh(sample_points)

            if np.all(inside):
                invalid_positions.append(combo)
            else:
                valid_positions.append(combo)

        return np.array(valid_positions), np.array(invalid_positions), combinations

    def generate_sample_points(self, V1, V2, V3, num_samples=50):
        """
        Generate sample points along the triangle plane.
        :param V1, V2, V3: Triangle vertices.
        :param num_samples: Resolution for point sampling.
        :return: np.array of sampled points.
        """
        u = np.linspace(np.min(self.vertices), np.max(self.vertices), num_samples)
        v = u
        U, V = np.meshgrid(u, v)
        return V1 + U.flatten()[:, None] * (V2 - V1) + V.flatten()[:, None] * (V3 - V1)


# Example Usage:
# STL vertices and faces should be loaded here
vertices = np.array([[0, 0, 0], [0, 0, 200], [0, 20, 20], [0, 20, 200], [0, 200, 0], [0, 200, 20],
                     [100, 0, 0], [100, 0, 200], [100, 20, 20], [100, 20, 200], [100, 200, 0], [100, 200, 20]])

faces = np.array([[11, 5, 8], [8, 5, 2], [5, 4, 2], [2, 4, 0], [2, 0, 3], [3, 0, 1],
                  [7, 1, 6], [6, 1, 0], [9, 7, 8], [8, 7, 6], [8, 6, 11], [11, 6, 10],
                  [5, 11, 4], [4, 11, 10], [10, 6, 4], [4, 6, 0], [3, 9, 2], [2, 9, 8],
                  [7, 9, 1], [1, 9, 3]])

face_finder = Face_Finder(vertices, faces)
valid, invalid, combos = face_finder.find_external_faces()
print("Valid Faces:\n", valid)
print("Invalid Faces:\n", invalid)