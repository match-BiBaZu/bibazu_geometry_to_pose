import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from PoseFinder import PoseFinder
from matplotlib.path import Path
import matplotlib.pyplot as plt
import trimesh

class PoseEliminator(PoseFinder):
    def __init__(self, convex_hull_obj_file: str, self_obj_file: str, tolerance: float = 1e-5,
                     stability_tolerance: float = 1e-3,
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
        self.stability_tolerance = stability_tolerance

    def _is_stable_by_center_mass_old(self, rotation_vector: np.ndarray, index: int) -> bool:
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

        # Project center_mass to XY
        center_mass = rotated_mesh.center_mass[:2]

        # Need at least 3 unique points
        if projected_points.shape[0] < 3:
            # Fallback: plot center mass + raw resting points anyway
            ''' Plots for debugging 
            plt.figure()
            if projected_points.shape[0] > 0:
            plt.scatter(*projected_points.T, c='k', label='Projected Vertices')
            plt.plot(center_mass[0], center_mass[1], 'ro', label='Center of Mass')
            plt.gca().set_aspect('equal')
            plt.legend()
            plt.title(f"Unstable (too few contact points) — pose {index}")
            plt.savefig(f"unstable_pose_{index:03d}.png", dpi=150)
            plt.close()
            '''
            return False

        # Use ConvexHull to get the 2D polygon of the merged resting faces
        hull = ConvexHull(projected_points)
        hull_points = projected_points[hull.vertices]

        # Point-in-polygon test using matplotlib.path
        path = Path(hull_points)

        if self.pose_types[index] == 3:
            is_inside = path.contains_point(center_mass, radius=0)  # looser tolerance for those pesky rolling cylinder poses
        else:
            is_inside = path.contains_point(center_mass, radius=self.stability_tolerance) # small negative radius to be stricter on stability and make sure the center of mass is well within the support polygon (fixes Rl4i behaviour)
        #print(f"radius: {self.stability_tolerance}")
        #print(f"Center of mass {center_mass}.")

        # Diagnostic plot (always shown)
        ''' Plots for debugging
        if is_inside:
            plt.figure()
            plt.plot(*hull_points.T, 'k--', lw=1, label='Support Polygon')
            plt.plot(center_mass[0], center_mass[1],
                    'go' if is_inside else 'ro', label='Center of Mass')
            plt.gca().set_aspect('equal')
            plt.legend()
            plt.title(f"Stability Check: {'Stable' if is_inside else 'Unstable'} pose_number {index}")
            plt.savefig(f"stability_check_{index:03d}.png", dpi=150)
            plt.close()
        '''
        return is_inside
    

    def _is_stable_by_center_mass(self, rotation_vector: np.ndarray, index: int,
                              alpha_tilt: float, beta_tilt: float) -> bool:
        """
        Checks if the pose is stable on a tilted XY plane defined by alpha and beta tilt angles.
        A pose is considered stable if:
        - The center of mass projects inside the support polygon (contact points at min-Z)
        - The CoM's Y-coordinate lies within the Y-span (min–max) of the contact vertices

        Tilts:
        alpha_tilt – CCW tilt (deg) around X-axis (front-back)
        """

        # Step 1: Apply pose rotation
        rotated_mesh = self.mesh.copy()
        rotation = R.from_quat(rotation_vector)
        T_pose = np.eye(4)
        T_pose[:3, :3] = rotation.as_matrix()
        rotated_mesh.apply_transform(T_pose)

        # Step 2: Identify contact vertices at bottom plane (min-Z)
        verts = rotated_mesh.vertices
        min_z = np.min(verts[:, 2])  # base plane
        min_x = np.min(verts[:, 0])  # back plane

        # Step 3: Tilt the system so that the center of mass becomes offset by the tilt angle
        alpha = np.radians(alpha_tilt)
        beta = np.radians(beta_tilt)

        if self.pose_types == 3:
            R_tilt = R.from_euler('xy', [alpha, 0]).as_matrix()  # inverse tilt
        else:
            R_tilt = R.from_euler('xy', [alpha, -beta]).as_matrix()  # inverse tilt

        verts_tilted = verts @ R_tilt.T
        com_tilted   = rotated_mesh.center_mass @ R_tilt.T

        #Step 4: find the location of the verticies which are touching the back and base planes before the tilt is applied
        base_idx = np.where(np.abs(verts[:, 2] - min_z) < self.tolerance)[0]
        back_idx = np.where(np.abs(verts[:, 0] - min_x) < self.tolerance)[0]

        # use the location of these verticies to extract the tilted polygons
        base_contact_tilt = verts_tilted[base_idx]
        back_contact_tilt = verts_tilted[back_idx]

        #Step 5: Do inside 2D convex hull projection check for both the back and base planes
        if len(np.unique(base_contact_tilt[:, :2],axis= 0)) >= 3:
            hull_base = self.find_contact_polygon(base_contact_tilt[:, :2], min_z)
            path_base = Path(hull_base[:, :2])
            inside_base_polygon = path_base.contains_point(com_tilted, radius=self.stability_tolerance)
        else:
            inside_base_polygon = False

        if (beta_tilt > 0.0) and (len(np.unique(back_contact_tilt[:, :2],axis=0)) >= 3):
            hull_back = self.find_contact_polygon(back_contact_tilt[:, :2], min_z)
            path_back = Path(hull_back[:, :2])
            inside_back_polygon = path_back.contains_point(com_tilted, radius=self.stability_tolerance)
        else:
            inside_back_polygon = False
        
        # Step 6: Y-span check at the back (min-X) face
        x_coords = verts[:, 0]
        min_x = np.min(x_coords)
        back_idx = np.where(np.abs(x_coords - min_x) < self.tolerance)[0]
        back_vertices = verts[back_idx]

        if len(back_vertices) < 2:
            inside_y_span = False
        else:
            back_tilted = back_vertices @ R_tilt.T
            y_coords_back = back_tilted[:, 1]
            y_min, y_max = np.min(y_coords_back), np.max(y_coords_back)
            com_y = com_tilted[1]
            if self.pose_types[index] == 2:
                inside_y_span = (y_min - 2.5) <= com_y <= (y_max + 2.5)  # looser tolerance for those pesky base cylinder poses
            else:
                inside_y_span = (y_min - self.tolerance) <= com_y <= (y_max + self.tolerance)
        
        #print('y min:', y_min, 'y max:', y_max, 'com y:', com_y)

        # Optional debug
        '''
        if (inside_polygon and inside_y_span):
            plt.figure()
            plt.plot(*hull_points.T, 'k--', lw=1, label='Support Polygon')
            plt.plot(com_xy[0], com_xy[1], 'go', label='Center of Mass')
            plt.axhline(y_min, color='gray', linestyle='--', lw=0.5)
            plt.axhline(y_max, color='gray', linestyle='--', lw=0.5)
            plt.gca().set_aspect('equal')
            plt.legend()
            plt.title(f"Stable Pose {index} – α={alpha_tilt}°")
            plt.savefig(f"stability_check_{index:03d}.png", dpi=150)
            plt.close()
        '''

        return (inside_back_polygon or inside_base_polygon) and inside_y_span

    # use a tolerance-aware modulo check
    def _is_aligned(self, value: float, step: float, tol: float) -> bool:
        # distance to nearest multiple of step
        return min(abs((value % step)), abs(step - (value % step))) <= tol
    
    def find_contact_polygon(self, contact_tilted_in_plane, min_z):

        # contact_tilted: (M,3)
        contact_plane = np.unique(contact_tilted_in_plane, axis=0)

        if contact_plane.shape[0] < 3:
            hull_3d = None

        # convex hull in 2D
        hull = ConvexHull(contact_plane)
        hull_xy = contact_plane[hull.vertices]        # (N,2)

        # lift polygon back to 3D (resting plane)
        hull_3d = np.column_stack([
            hull_xy[:, 0],
            hull_xy[:, 1],
            np.full(len(hull_xy), min_z)
        ])

        # close polygon
        hull_3d = np.vstack([hull_3d, hull_3d[0]])  # (N+1,3)

        return hull_3d
    
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
    
    
    def remove_unstable_poses(self, alpha_tilt: float, beta_tilt: float):
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
            is_stable_flat = self._is_stable_by_center_mass(quat, index, 0, 0)

            if self.pose_types[index] == 3:
                is_stable_tilted = self._is_stable_by_center_mass(quat, index, alpha_tilt, 0) # ignore beta tilt as it was applied during cylinder pose assignment
            else:
                is_stable_tilted = self._is_stable_by_center_mass(quat, index, alpha_tilt, beta_tilt)

            #print(f"Pose {index} stability: {is_stable_flat} and {is_stable_tilted}")
            if is_stable_flat and is_stable_tilted:
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
        filtered_critical_solid_angle_scores = []
        filtered_centroid_solid_angle_scores = []
        filtered_stability_scores = []
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