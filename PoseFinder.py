import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

class PoseFinder:
    def __init__(self, convex_hull_obj_file: str, self_obj_file: str, tolerance: float = 1e-5):
        """
        Initialize the PoseFinder with the convex hull OBJ file.
        :param convex_hull_obj_file: Path to the convex hull OBJ file.
        """
        self.mesh = trimesh.load_mesh(self_obj_file)
        self.convex_hull_mesh = trimesh.load_mesh(convex_hull_obj_file)

        # Ensure the mesh is centered around the centroid of the convex hull
        centroid = self.convex_hull_mesh.centroid

        self.convex_hull_mesh.apply_translation(-centroid)
        self.mesh.apply_translation(-centroid)

        #Initailze a numerical tolerance for grouping
        self.tolerance = tolerance
    
    def find_candidate_rotations_by_face_normals(self):
        """
        Computes candidate rotations by aligning every face normal with every other face normal.
        :return: A list of candidate rotations as quaternions.
        """
        face_normals = self.convex_hull_mesh.face_normals
        candidate_rotations = []

        # Add the identity quaternion as the first candidate rotation
        candidate_rotations.append(np.array([1.0, 0.0, 0.0, 0.0]))

        # Generate rotations by aligning every face normal with every other face normal
        for i, normal_1 in enumerate(face_normals):
            for j, normal_2 in enumerate(face_normals):
                if i != j and not np.allclose(normal_1, normal_2):  # Avoid redundant checks
                    # Step 1: Calculate the axis of rotation (cross product)
                    axis_of_rotation = np.cross(normal_1, normal_2)

                    # Step 2: Calculate the angle between the normals (dot product)
                    cos_angle = np.dot(normal_1, normal_2) / (np.linalg.norm(normal_1) * np.linalg.norm(normal_2))
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Clip for numerical stability

                    # Step 3: Construct the rotation vector (axis * angle)
                    if np.linalg.norm(axis_of_rotation) > 1e-6:  # Ensure valid axis
                        axis_of_rotation = axis_of_rotation / np.linalg.norm(axis_of_rotation)
                        rotation = R.from_rotvec(axis_of_rotation * angle)
                        candidate_rotations.append(rotation.as_quat())

        return candidate_rotations
    
    def find_candidate_rotations_by_fibonacci_sphere(self, axis_samples=20, angle_samples=4):
        """
        Generate a list of candidate rotation quaternions.
        First entry is the identity quaternion.
        :param axis_samples: Number of candidate axes (from Fibonacci sphere).
        :param angle_samples: Number of rotation orders to generate (2 to 2+angle_samples).
        :return: List of quaternions (w, x, y, z).
        """
        from trimesh.transformations import quaternion_about_axis
        candidates = [(1.0, 0.0, 0.0, 0.0)]  # Identity rotation

        # Fibonacci sphere for axis sampling
        i = np.arange(0, axis_samples)
        phi = np.arccos(1 - 2*(i + 0.5)/axis_samples)
        theta = np.pi * (1 + 5**0.5) * (i + 0.5)
        axes = np.stack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ], axis=1)

        for axis in axes:
            axis = axis / np.linalg.norm(axis)
            for order in range(2, angle_samples + 2):  # e.g. 2, 3, 4, ...
                angle = 2 * np.pi / order
                for i in range(1, order):  # Skip 0 rotation (identity)
                    quat = quaternion_about_axis(i * angle, axis)
                    candidates.append(tuple(quat))  # (w, x, y, z)

        return candidates


    def find_valid_rotations(self, candidate_rotations):
        """
        Filters candidate rotations to find valid rotations where at least three external vertices
        of the convex hull would intersect two 90-degree intersecting planes.
        :param candidate_rotations: A list of candidate rotations as quaternions.
        :return: A list of valid rotations as tuples (index, quaternion).
        """
        vertices = np.array(self.convex_hull_mesh.vertices)
        valid_rotations = []

        # Add the identity quaternion as the first valid rotation
        valid_rotations.append((0, np.array([1.0, 0.0, 0.0, 0.0])))

        count = 1  # Start counting from 1 as 0 is reserved for the identity quaternion

        for quat in candidate_rotations:
            rotation = R.from_quat(quat)
            rotated_vertices = rotation.apply(vertices)

            # --- x–y plane check (same z) ---
            binned_z = np.round(rotated_vertices[:, 2] / self.tolerance) * self.tolerance
            min_z = np.min(binned_z)
            x_plane_count = np.count_nonzero(binned_z == min_z) >= 3

            # --- y–z plane check (same x) ---
            binned_x = np.round(rotated_vertices[:, 0] / self.tolerance) * self.tolerance
            min_x = np.min(binned_x)
            y_plane_count = np.count_nonzero(binned_x == min_x) >= 2

            if True: #x_plane_count and y_plane_count:
                valid_rotations.append((count, quat))
                count += 1

        return valid_rotations
    
    def symmetry_handler(self, rotations):
        """
        Handles symmetry constraints by assigning duplicate rotations the same index as the first detected instance.
        Also removes candidate rotations where all vertices align with a previous candidate rotation within a desired tolerance.
        :param rotations: List of tuples (index, valid rotation vector).
        :return: List of tuples (assigned index, valid rotation vector).
        """
        unique_rotations = {}
        assigned_rotations = []

        for index, rot in rotations:
            rounded_rot = tuple(np.round(rot, decimals=5))  # Round to prevent numerical noise
            if rounded_rot not in unique_rotations:  # Remove duplicates
                unique_rotations[rounded_rot] = index

                # Check if the rotation aligns all vertices with a previous candidate rotation
                #rotation = R.from_rotvec(rot)
                rotation = R.from_quat(rot)
                rotated_vertices = rotation.apply(self.mesh.vertices)
                previous_rotation = R.from_quat(list(unique_rotations.keys())[list(unique_rotations.values()).index(index)])
                previous_rotated_vertices = previous_rotation.apply(self.mesh.vertices)

                if np.allclose(rotated_vertices, previous_rotated_vertices, atol=self.tolerance):
                    #continue  # Skip this rotation as it aligns with a previous one
                    assigned_rotations.append((index, rot))
                else:
                    assigned_rotations.append((index, rot))

        return assigned_rotations
