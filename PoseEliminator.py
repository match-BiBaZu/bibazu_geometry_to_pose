import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from PoseFinder import PoseFinder
from matplotlib.path import Path

class PoseEliminator(PoseFinder):
    def __init__(self, convex_hull_obj_file: str, self_obj_file: str, tolerance: float = 1e-6):
        """
        Initialize the PoseEliminator with the convex hull OBJ file.
        :param convex_hull_obj_file: Path to the convex hull OBJ file.
        """
        super().__init__(convex_hull_obj_file, self_obj_file, tolerance)
        
        # Additional initialization (if any) can go here

    def _is_stable_by_centroid(self, rotation_vector: np.ndarray) -> bool:
        """
        Checks if the centroid's (x, y) coordinates, after applying the given rotation,
        fall within the bounds of the resting plane (the face in contact with the ground).
        Assumes the resting plane is the XY plane after rotation.
        :param rotation_vector: Rotation to apply to the mesh.
        :return: True if centroid projects inside the resting plane polygon, False otherwise.
        """
        # Rotate the mesh
        rotated_mesh = self.mesh.copy()
        rotation = R.from_quat(rotation_vector)
        rot_matrix = np.eye(4)
        rot_matrix[:3, :3] = rotation.as_matrix()
        rotated_mesh.apply_transform(rot_matrix)

        # Find the lowest face (resting plane) after rotation
        z_coords = rotated_mesh.vertices[:, 2]
        min_z = np.min(z_coords)
        # Faces whose all vertices are at min_z (within tolerance)
        resting_faces = []
        for face in rotated_mesh.faces:
            if np.all(np.abs(z_coords[face] - min_z) < self.tolerance):
                resting_faces.append(face)
        if not resting_faces:
            return False

        # Get all vertices of the resting plane
        resting_vertices = np.unique(np.concatenate(resting_faces))
        resting_xy = rotated_mesh.vertices[resting_vertices][:, :2]

        # Project centroid to XY
        centroid = rotated_mesh.centroid[:2]

        # Use ConvexHull to get the 2D polygon of the resting plane
        if len(resting_xy) < 3:
            return False  # Not a valid plane
        hull = ConvexHull(resting_xy)
        hull_points = resting_xy[hull.vertices]

        # Point-in-polygon test using matplotlib.path
        path = Path(hull_points)
        return path.contains_point(centroid)


    
    def remove_duplicates(self, rotations: list[tuple[int, int, int, np.ndarray]], xy_shadows) -> list[tuple[int, int, int, np.ndarray]]:
        """
        Handles duplicate rotations by removing rotations that are too close to each other.
        :param rotations: List of tuples (pose, face_id, shadow_id, valid rotation vector) and list of xy_shadow arrays.
        :return: List of tuples (assigned pose, face_id, shadow_id, valid rotation vector) and list of xy_shadow arrays.
        """
        unique_rotations = {}
        assigned_rotations = []
        assigned_shadows = []
        pose_count = 0

        for index, face_id, edge_id, quat in rotations:
            if quat not in unique_rotations:
                unique_rotations[quat] = index
                assigned_rotations.append((pose_count, face_id, edge_id, quat))
                assigned_shadows.append(xy_shadows[index])
                pose_count += 1
        
        return assigned_rotations, assigned_shadows
    
    def remove_unstable_poses(self, rotations: list[tuple[int, int, int, np.ndarray]], xy_shadows) -> list[tuple[int, int, int, np.ndarray], list[np.ndarray]]:
        """
        Removes unstable poses based on the convex hull.
        :param rotations: List of tuples (pose, face_id, shadow_id, valid rotation vector).
        :return: List of tuples (assigned pose, face_id, shadow_id, valid rotation vector).
        """
        stable_rotations = []
        stable_shadows = []

        for index, face_id, edge_id, quat in rotations:
            if self._is_stable_by_centroid(quat):
                stable_rotations.append((index, face_id, edge_id, quat))
                stable_shadows.append(xy_shadows[index])
        
        return stable_rotations, stable_shadows
    
