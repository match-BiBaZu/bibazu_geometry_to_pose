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

        # Check to ensure that the mesh and convex hull meshes are not empty
        if self.mesh.vertices is None or len(self.mesh.vertices) == 0:
                    raise ValueError("Mesh vertices are not initialized or empty.")
        
        if self.convex_hull_mesh.vertices is None or len(self.convex_hull_mesh.vertices) == 0:
                    raise ValueError("Convex hull mesh vertices are not initialized or empty.")

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

            # --- Bin by z ---
            binned_z = np.round(rotated_vertices[:, 2] / self.tolerance) * self.tolerance
            min_z = np.min(binned_z)
            min_z_indices = np.where(binned_z == min_z)[0]

            # --- x–y plane check (same z) ---
            x_plane_count = np.count_nonzero(binned_z == min_z) >= 3

            # --- x-plane for y–z: check for min_x in each x-bin ---
            binned_x = np.round(rotated_vertices[min_z_indices, 0] / self.tolerance) * self.tolerance
            min_x = np.min(binned_x)

            # --- y–z plane check for the values that passed the x-y plane check (same x) ---
            y_plane_count = np.count_nonzero(binned_x == min_x) >= 2

            if x_plane_count and y_plane_count:
                valid_rotations.append((count, quat))
                count += 1

        return valid_rotations
    
    def duplicate_remover(self, rotations):
        """
        Handles duplicate rotations by removing rotations that are too close to each other.
        :param rotations: List of tuples (pose, valid rotation vector).
        :return: List of tuples (assigned pose, valid rotation vector).
        """
        unique_rotations = {}
        assigned_rotations = []
        pose_count = 0

        for index, quat in rotations:
            rounded_quat = tuple(np.round(quat, decimals=int(np.log10(1/self.tolerance))))  # Round to prevent numerical noise
            one_zero_rounded_quat = tuple(0.0 if abs(x) < self.tolerance else x for x in rounded_quat)  # Ensure all zeros are positive
            if one_zero_rounded_quat not in unique_rotations:
                unique_rotations[one_zero_rounded_quat] = index
                assigned_rotations.append((pose_count, one_zero_rounded_quat))
                pose_count += 1
        
        return assigned_rotations
    
    def symmetry_handler(self, rotations):
        """
        Handles symmetry constraints by sorting by pose and assigning the same pose to symmetrically equivalent rotations.
        This is done by checking if the rotation has multiple representations equivalent by rotational symmetry.
        :param rotations: List of tuples (pose, valid rotation vector).
        :return: List of tuples (assigned pose, valid rotation vector).
        """
        assigned_rotations = []
        assymetric_pose_count = 0

        for index, quat in rotations:
            # Check if the rotation is already assigned to a pose
            if not any(np.array_equal(quat, assigned_quat) for _, assigned_quat in assigned_rotations):
                assigned_rotations.append((assymetric_pose_count, quat))

                # Check if the rotation has multiple representations equivalent by rotational symmetry
                rotation = R.from_quat(quat)
                rotated_vertices = rotation.apply(self.mesh.vertices)

                for check_index, check_quat in rotations:
                    if check_index != index:
                        check_rotation = R.from_quat(check_quat)
                        check_rotated_vertices = check_rotation.apply(self.mesh.vertices)

                        sorted_rotated_vertices = np.sort(rotated_vertices, axis=0)
                        sorted_check_rotated_vertices = np.sort(check_rotated_vertices, axis=0)

                        is_close = np.allclose(sorted_rotated_vertices, sorted_check_rotated_vertices, atol=self.tolerance)
                        
                        if is_close:
                            # Append the rotation with the same pose count if it is rotationally symmetric
                            assigned_rotations.append((assymetric_pose_count, check_quat))
                
                assymetric_pose_count += 1

        return assigned_rotations
