import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from PoseFinder import PoseFinder
from matplotlib.path import Path
import matplotlib.pyplot as plt
import trimesh

class PoseEliminator(PoseFinder):
    def __init__(self, convex_hull_obj_file: str, self_obj_file: str, tolerance: float = 1e-5,
                     stable_rotations=None, stable_shadows=None, stable_axis_parameters=None,
                     pose_types=None, pose_cylinder_radius=None, pose_cylinder_axis_direction=None,
                     pose_cylinder_axis_origin=None, pose_cylinder_group=None):
        """
        Initialize the PoseEliminator with the convex hull OBJ file.
        :param convex_hull_obj_file: Path to the convex hull OBJ file.
        """
        super().__init__(convex_hull_obj_file, self_obj_file, tolerance)

        #put this in parent class sometime
        self.stable_rotations = stable_rotations
        self.stable_shadows = stable_shadows
        self.stable_axis_parameters = stable_axis_parameters
        self.pose_types = pose_types
        self.pose_cylinder_radius = pose_cylinder_radius
        self.pose_cylinder_axis_direction = pose_cylinder_axis_direction
        self.pose_cylinder_axis_origin = pose_cylinder_axis_origin
        self.pose_cylinder_group = pose_cylinder_group

    def _is_stable_by_center_mass(self, rotation_vector: np.ndarray) -> bool:
        """
        Checks if the center_mass's (x, y) coordinates, after applying the given rotation,
        fall within the bounds of the resting plane (the face in contact with the ground).
        Assumes the resting plane is the XY plane after rotation.
        :param rotation_vector: Rotation to apply to the mesh.
        :return: True if center_mass projects inside the resting plane polygon, False otherwise.
        """
        # Rotate the mesh
        rotated_mesh = self.mesh.copy()
        rotation = R.from_quat(rotation_vector)
        rot_matrix = np.eye(4)
        rot_matrix[:3, :3] = rotation.as_matrix()
        rotated_mesh.apply_transform(rot_matrix)

       # Find vertices near min Z
        z_coords = rotated_mesh.vertices[:, 2]
        min_z = np.min(z_coords)
        
        # Select only vertices near the lowest Z value (resting surface)
        resting_indices = np.where(np.abs(z_coords - min_z) < self.tolerance)[0]
        resting_vertices = rotated_mesh.vertices[resting_indices]

        # Project to XY and remove duplicates
        projected_points = np.unique(resting_vertices[:, :2], axis=0)

        # Need at least 3 unique points
        if projected_points.shape[0] < 3:
            return False

        # Project center_mass to XY
        center_mass = rotated_mesh.center_mass[:2]

        # Use ConvexHull to get the 2D polygon of the merged resting faces
        hull = ConvexHull(projected_points)
        hull_points = projected_points[hull.vertices]

        # Point-in-polygon test using matplotlib.path
        path = Path(hull_points)

        is_inside = path.contains_point(center_mass, radius=self.tolerance)
        #print(f"Center of mass {center_mass}.")

        # Diagnostic plot (always shown)
        #plt.figure()
        #plt.plot(*hull_points.T, 'k--', lw=1, label='Support Polygon')
        #plt.plot(center_mass[0], center_mass[1],
        #        'go' if is_inside else 'ro', label='Center of Mass')
        #plt.gca().set_aspect('equal')
        #plt.legend()
        #plt.title(f"Stability Check: {'Stable' if is_inside else 'Unstable'}")
        #plt.show()

        return is_inside
    
    # use a tolerance-aware modulo check
    def _is_aligned(self, value: float, step: float, tol: float) -> bool:
        # distance to nearest multiple of step
        return min(abs((value % step)), abs(step - (value % step))) <= tol
    
    def get_stable_rotations(self):
        return self.stable_rotations

    def get_stable_shadows(self):
        return self.stable_shadows
    
    def get_stable_axis_parameters(self):
        return self.stable_axis_parameters
    
    def get_pose_types(self):
        return self.pose_types
    
    def get_pose_cylinder_radius(self):
        return self.pose_cylinder_radius
    
    def get_pose_cylinder_axis_direction(self):
        return self.pose_cylinder_axis_direction
    
    def get_pose_cylinder_axis_origin(self):
        return self.pose_cylinder_axis_origin
    
    def get_pose_cylinder_group(self):
        return self.pose_cylinder_group

    def remove_duplicates(self):
        """
        Handles duplicate rotations by removing rotations that are too close to each other.
        :param rotations: List of tuples (pose, face_id, shadow_id, valid rotation vector).
        :return: List of tuples (assigned pose, face_id, shadow_id, valid rotation vector).
        """
        unique_rotations = {}
        assigned_rotations = []
        assigned_shadows = []
        assigned_cylinder_axis_parameters = []
        filtered_pose_types = []
        filtered_cylinder_radius = []
        filtered_cylinder_axis_direction = []
        filtered_cylinder_axis_origin = []
        filtered_cylinder_group = []
        pose_count = 0

        for index, face_id, edge_id, quat in self.stable_rotations:
            if quat not in unique_rotations:
                unique_rotations[quat] = index
                assigned_rotations.append((pose_count, face_id, edge_id, quat))
                assigned_shadows.append(self.stable_shadows[index])
                if self.stable_axis_parameters:
                    assigned_cylinder_axis_parameters.append(self.stable_axis_parameters[index])
                else:
                    assigned_cylinder_axis_parameters.append(None)
                filtered_pose_types.append(self.pose_types[index])
                filtered_cylinder_radius.append(self.pose_cylinder_radius[index])
                filtered_cylinder_axis_direction.append(self.pose_cylinder_axis_direction[index])
                filtered_cylinder_axis_origin.append(self.pose_cylinder_axis_origin[index])
                filtered_cylinder_group.append(self.pose_cylinder_group[index])


                pose_count += 1
        
        self.stable_rotations = assigned_rotations
        self.stable_shadows = assigned_shadows
        self.stable_axis_parameters = assigned_cylinder_axis_parameters
        self.pose_types = filtered_pose_types
        self.pose_cylinder_radius = filtered_cylinder_radius
        self.pose_cylinder_axis_direction = filtered_cylinder_axis_direction
        self.pose_cylinder_axis_origin = filtered_cylinder_axis_origin
        self.pose_cylinder_group = filtered_cylinder_group
    
    
    def remove_unstable_poses(self):
        """
        Removes unstable poses based on the convex hull.
        :param rotations: List of tuples (pose, face_id, shadow_id, valid rotation vector).
        :return: List of tuples (assigned pose, face_id, shadow_id, valid rotation vector).
        """
        stable_rotations = []
        stable_shadows = []
        stable_cylinder_axis_parameters = []
        filtered_pose_types = []
        filtered_cylinder_radius = []
        filtered_cylinder_axis_direction = []
        filtered_cylinder_axis_origin = []
        filtered_cylinder_group = []
        pose_count = 0

        for index, face_id, edge_id, quat in self.stable_rotations:
            #print(f"Checking pose {index} for stability...")
            is_stable = self._is_stable_by_center_mass(quat)
            if is_stable or self.pose_types[index] > 0: # keep all cylinder poses for now
                #print(f"Pose {index} is stable.")
                stable_rotations.append((pose_count, face_id, edge_id, quat))
                stable_shadows.append(self.stable_shadows[index])
                stable_cylinder_axis_parameters.append(self.stable_axis_parameters[index])
                filtered_pose_types.append(self.pose_types[index])
                filtered_cylinder_radius.append(self.pose_cylinder_radius[index])
                filtered_cylinder_axis_direction.append(self.pose_cylinder_axis_direction[index])
                filtered_cylinder_axis_origin.append(self.pose_cylinder_axis_origin[index])
                filtered_cylinder_group.append(self.pose_cylinder_group[index])
                pose_count += 1
        
        self.stable_rotations = stable_rotations
        self.stable_shadows = stable_shadows
        self.stable_axis_parameters = stable_cylinder_axis_parameters
        self.pose_types = filtered_pose_types
        self.pose_cylinder_radius = filtered_cylinder_radius
        self.pose_cylinder_axis_direction = filtered_cylinder_axis_direction
        self.pose_cylinder_axis_origin = filtered_cylinder_axis_origin
        self.pose_cylinder_group = filtered_cylinder_group
    
    def remove_cylinder_poses(self):
        """
        Removes cylinder poses based on alignment criteria.
        :param rotations: List of tuples (pose, face_id, shadow_id, valid rotation vector).
        :return: List of tuples (assigned pose, face_id, shadow_id, valid rotation vector).
        """
        filtered_rotations = []
        filtered_shadows = []
        filtered_cylinder_axis_parameters = []
        filtered_pose_types = []
        filtered_cylinder_radius = []
        filtered_cylinder_axis_direction = []
        filtered_cylinder_axis_origin = []
        filtered_cylinder_group = []
        pose_count = 0

        for index, face_id, edge_id, quat in self.stable_rotations:
            if ((self.pose_types[index] == 0) or (self.pose_types[index] == 2) or (self.pose_types[index] == 3)): # detect only acceptable cylinder poses and remaining non cylinder poses
                filtered_rotations.append((pose_count, face_id, edge_id, quat))
                filtered_shadows.append(self.stable_shadows[index])
                filtered_cylinder_axis_parameters.append(self.stable_axis_parameters[index])
                filtered_pose_types.append(self.pose_types[index])
                filtered_cylinder_radius.append(self.pose_cylinder_radius[index])
                filtered_cylinder_axis_direction.append(self.pose_cylinder_axis_direction[index])
                filtered_cylinder_axis_origin.append(self.pose_cylinder_axis_origin[index])
                filtered_cylinder_group.append(self.pose_cylinder_group[index])
                pose_count += 1
        
        self.stable_rotations = filtered_rotations
        self.stable_shadows = filtered_shadows
        self.stable_axis_parameters = filtered_cylinder_axis_parameters
        self.pose_types = filtered_pose_types
        self.pose_cylinder_radius = filtered_cylinder_radius
        self.pose_cylinder_axis_direction = filtered_cylinder_axis_direction
        self.pose_cylinder_axis_origin = filtered_cylinder_axis_origin
        self.pose_cylinder_group = filtered_cylinder_group
