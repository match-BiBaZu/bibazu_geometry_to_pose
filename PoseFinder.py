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
    
    def find_valid_rotations(self):
        """
        Computes valid rotations where at least three external vertices of the convex hull
        would intersect two 90-degree intersecting planes, ensuring the object is not cut by the planes.
        Uses pairwise face normal alignments instead of just aligning with coordinate axes.
        :return: A list of valid rotations as numpy arrays.
        """
        vertices = np.array(self.convex_hull_mesh.vertices)
        face_normals = self.convex_hull_mesh.face_normals
        
        valid_rotations = []
        candidate_rotations = []  # Initialize with the identity quaternion

        # the first row of valid rotations is the identity quaternion paired with pose count 0
        valid_rotations.append((0, np.array([1.0, 0.0, 0.0, 0.0])))
        #valid_rotations.append((0, np.array([0.0, 0.0, 0.0])))

        # Generate rotations by aligning every face normal with every other face normal
        for i, normal_1 in enumerate(face_normals):
            for j, normal_2 in enumerate(face_normals):
                if i != j and not np.allclose(normal_1, normal_2):  # Avoid redundant checks

                    # Step 1: Calculate the axis of rotation (cross product)
                    axis_of_rotation = np.cross(normal_1, normal_2)

                    # Step 2: Calculate the angle between vec1 and vec2 (dot product)
                    cos_angle = np.dot(normal_1, normal_2) / (np.linalg.norm(normal_1) * np.linalg.norm(normal_2))
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Clip for numerical stability

                    # Step 3: Construct the rotation vector (axis * angle)
                    rotation_vector = axis_of_rotation * angle

                    if np.linalg.norm(rotation_vector) > 1e-6:  # Ensure valid rotation
                        rotation = R.from_rotvec(rotation_vector)
                        candidate_rotations.append(rotation)

        count = 1 # Start counting from 1 as 0 is reserved for the identity quaternion

        for rotation in candidate_rotations:
            rotated_vertices = rotation.apply(vertices)

            # --- x–y plane check (same z) ---
            binned_z = np.round(rotated_vertices[:, 2] / self.tolerance) * self.tolerance
            unique_z, z_counts = np.unique(binned_z, return_counts=True)
            min_z = np.min(unique_z)
            x_plane_count = np.count_nonzero(binned_z == min_z) >= 3

            # --- y–z plane check (same x) ---
            binned_x = np.round(rotated_vertices[:, 0] / self.tolerance) * self.tolerance
            unique_x, x_counts = np.unique(binned_x, return_counts=True)
            min_x = np.min(unique_x)
            y_plane_count = np.count_nonzero(binned_x == min_x) >= 2

            if (
                    True
                    #np.count_nonzero(binned_z == min_z) >= 3 
                    #and np.count_nonzero(binned_x == min_x) >= 2
                ):
                valid_rotations.append((count, rotation.as_quat()))
                #valid_rotations.append((count, rotation.as_rotvec()))
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
                #previous_rotation = R.from_rotvec(list(unique_rotations.keys())[list(unique_rotations.values()).index(index)])
                previous_rotation = R.from_quat(list(unique_rotations.keys())[list(unique_rotations.values()).index(index)])
                previous_rotated_vertices = previous_rotation.apply(self.mesh.vertices)

                if np.allclose(rotated_vertices, previous_rotated_vertices, atol=self.tolerance):
                    #continue  # Skip this rotation as it aligns with a previous one
                    assigned_rotations.append((index, rot))
                else:
                    assigned_rotations.append((index, rot))

        return assigned_rotations
