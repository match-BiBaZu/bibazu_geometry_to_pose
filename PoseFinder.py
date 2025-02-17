import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

class PoseFinder:
    def __init__(self, convex_hull_obj_file: str):
        """
        Initialize the PoseFinder with the convex hull OBJ file.
        :param convex_hull_obj_file: Path to the convex hull OBJ file.
        """
        self.convex_hull_obj_file = convex_hull_obj_file
        self.mesh = trimesh.load_mesh(self.convex_hull_obj_file)
    
    def find_centroid(self):
        """
        Computes the centroid of the convex hull.
        :return: Centroid coordinates as a numpy array.
        """
        return self.mesh.centroid
    
    def find_normal_vectors(self):
        """
        Computes the normal unit vectors to each face of the convex hull originating from the centroid.
        :return: A list of normal vectors as numpy arrays.
        """
        normals = self.mesh.face_normals
        return normals
    
    def find_valid_rotations(self):
        """
        Computes valid rotations where at least three external vertices of the convex hull
        would intersect two 90-degree intersecting planes, ensuring the object is not cut by the planes.
        Uses pairwise face normal alignments instead of just aligning with coordinate axes.
        :return: A list of valid rotations as numpy arrays.
        """
        vertices = np.array(self.mesh.vertices)
        face_normals = self.find_normal_vectors()
        
        valid_rotations = []
        candidate_rotations = []  # Initialize with the identity quaternion

        # the first row of valid rotations is the identity quaternion paired with pose count 0
        valid_rotations.append((0, np.array([1, 0, 0, 0])))

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


                    #rotation_vector = np.cross(normal_1, normal_2)
                    if np.linalg.norm(rotation_vector) > 1e-6:  # Ensure valid rotation
                        rotation = R.from_rotvec(rotation_vector)
                        candidate_rotations.append(rotation)

        count = 1 # Start counting from 1 as 0 is reserved for the identity quaternion

        for rotation in candidate_rotations:
            rotated_vertices = rotation.apply(vertices)

            # Check if at least three vertices lie on both intersecting planes (x=0 and y=0)
            on_x_plane = np.isclose(rotated_vertices[:, 0], 0)
            on_y_plane = np.isclose(rotated_vertices[:, 1], 0)

            # Ensure the object is not cut by checking if all vertices remain on one side of each plane
            x_side = np.sign(rotated_vertices[:, 0])
            y_side = np.sign(rotated_vertices[:, 1])

            if np.sum(on_x_plane & on_y_plane) >= 3 & np.all(x_side == x_side[0]) & np.all(y_side == y_side[0]):
                quat = rotation.as_quat()
                quat_wxyz = np.array([quat[1], quat[2], quat[3], quat[0]])  # Convert to WXYZ format
                valid_rotations.append((count, quat_wxyz))
                count += 1

        return valid_rotations
    
    def symmetry_handler(self, rotations):
        """
        EDIT WITH THE BOP SYMMETRIES STUFF!!!! LOOKS WEIRD
        Handles symmetry constraints by assigning duplicate rotations the same index as the first detected instance.
        :param rotations: List of tuples (index, valid rotation quaternion).
        :return: List of tuples (assigned index, valid rotation quaternion).
        """
        unique_rotations = {}
        assigned_rotations = []

        for index, quat in rotations:
            rounded_quat = tuple(np.round(quat, decimals=5))  # Round to prevent numerical noise
            if rounded_quat not in unique_rotations:
                unique_rotations[rounded_quat] = index
            assigned_rotations.append((unique_rotations[rounded_quat], quat))
        
        return assigned_rotations
