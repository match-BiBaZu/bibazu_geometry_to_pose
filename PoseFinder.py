import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from trimesh.transformations import quaternion_about_axis, rotation_matrix, reflection_matrix, quaternion_from_matrix, identity_matrix

class PoseFinder:
    def __init__(self, convex_hull_obj_file: str, self_obj_file: str, tolerance: float = 1e-5):
        """
        Initialize the PoseFinder with the convex hull OBJ file.
        :param convex_hull_obj_file: Path to the convex hull OBJ file.
        """
        # Initialize a numerical tolerance for grouping
        self.tolerance = tolerance

        self.mesh = trimesh.load_mesh(self_obj_file)
        self.convex_hull_mesh = trimesh.load_mesh(convex_hull_obj_file)

        # Ensure the mesh is centered around the centroid of the convex hull
        centroid = self.convex_hull_mesh.centroid

        self.convex_hull_mesh.apply_translation(-centroid)
        self.mesh.apply_translation(-centroid)

        # Compute full polygonal face hulls
        self.full_hull = self._compute_full_hull()

        # Check to ensure that the mesh and convex hull meshes are not empty
        if self.mesh.vertices is None or len(self.mesh.vertices) == 0:
            raise ValueError("Mesh vertices are not initialized or empty.")
        
        if self.convex_hull_mesh.vertices is None or len(self.convex_hull_mesh.vertices) == 0:
            raise ValueError("Convex hull mesh vertices are not initialized or empty.")
        
        # Check if at least 3 vertices align with the lowest z-axis (resting face)
        z_min = np.min(self.mesh.vertices[:, 2])
        aligned_vertices = self.mesh.vertices[np.isclose(self.mesh.vertices[:, 2], z_min, atol=self.tolerance)]
        if len(aligned_vertices) < 3:
            raise ValueError("The object is not resting on the xy plane on one of it's faces. Please check original mesh.")
        
        # Check if the convex hull mesh is actually the convex hull of the original mesh	
        if not np.all(np.isin(self.convex_hull_mesh.vertices, self.mesh.vertices)):
            raise ValueError("The convex hull mesh is not a convex hull of the original mesh. Please check the convex hull mesh and the original mesh.")
    
    def _compute_full_hull(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build full face polygons by grouping coplanar triangles from the convex hull mesh.
        Returns:
            - face_normals: (N, 3) array of outward face normals
            - face_centers: (N, 3) array of face center points
            - face_vertices: list of (M_i, 3) arrays of polygon boundary vertices
            - Attempts to return the full hull in the same form as a trimesh object.
        """
        mesh = self.convex_hull_mesh
        normals = []
        centers = []
        verticies = []

        for facet, normal in zip(mesh.facets, mesh.facets_normal):
            submesh = mesh.submesh([facet], append=True, repair=False)
            boundary = submesh.outline()
            if boundary is None or len(boundary.vertices) < 3:
                continue

            normed_normal = normal / np.linalg.norm(normal)
            normed_normal = np.where(np.abs(normed_normal) < self.tolerance, 0.0, normed_normal)  # Ensure all zeros are positive
            center = boundary.vertices.mean(axis=0)

            normals.append(normed_normal)
            centers.append(center)
            verticies.append(boundary.vertices)
        
        if verticies:
            verticies = np.vstack(verticies)
        else:
            verticies = np.empty((0, 3))

        return np.array(normals), np.array(centers), verticies
    
    def _compute_xy_shadow(self, vertices: np.ndarray = None) -> np.ndarray:
        """
        Compute the 2D convex hull shadow (z=0) from a set of 3D vertices.
        :param vertices: (N, 3) array of vertex coordinates.
        :return: (M, 3) array of 3D shadow vertices on the x-y plane (z=0).
        """
        projected_points = vertices[:, :2]
        hull_2d = ConvexHull(projected_points)
        hull_coords = projected_points[hull_2d.vertices]
        z_min = np.min(vertices[:, 2])
        shadow_vertices = np.hstack([hull_coords, np.full((len(hull_coords), 1), z_min)])  # z = z_min

        return shadow_vertices

    def find_candidate_rotations_by_resting_face_normal_alignment(self) -> tuple[list[tuple[int, int, int, np.ndarray]], list[np.ndarray]]:
        """
        Aligns each face normal to the resting face normal (closest to -Z).
        Returns candidate rotations as quaternions.
        """

        # Compute the full hull so that we can find the boundaries of the convex hull and the outward face normals
        face_normals, _, vertices = self._compute_full_hull() 
        resting_face_idx = np.argmin(np.dot(face_normals, np.array([0, 0, -1])))
        resting_normal = face_normals[resting_face_idx]

        # initialise the outputs of the first candidate which is the identity rotation
        candidate_rotations = [(0, 0, 0, np.array([0.0, 0.0, 0.0, 1.0]))] ## Identity quaternion [x, y, z, w]
        xy_shadows = [self._compute_xy_shadow(vertices)]
        candidate_count = 1

        for i, normal in enumerate(face_normals):
            if i == resting_face_idx: #or np.allclose(normal, resting_normal):
                continue

            dot = np.clip(np.dot(normal, resting_normal), -1.0, 1.0)
            angle = np.arccos(dot)

            #if resting face is at the top, we need to extend the angle by 180 degrees to make sure its facing downwards
            #if dot > 0:
            if normal[2] > 0:
                angle = np.pi - angle

           # Identity rotation
            if np.abs(angle) < 1e-6: 
                quat = np.array([0.0, 0.0, 0.0, 0.1])  
            # 180° rotation: axis undefined → use any axis orthogonal to normal
            elif np.abs(angle - np.pi) < 1e-6:
                ortho = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
                axis = np.cross(normal, ortho)
                axis /= np.linalg.norm(axis)
                quat = R.from_rotvec(axis * angle).as_quat()
            else:
                axis = np.cross(normal, resting_normal)
                axis /= np.linalg.norm(axis)
                quat = R.from_rotvec(axis * angle).as_quat()

            candidate_rotations.append((candidate_count,candidate_count, 0, quat))
            # Apply rotation and compute shadow
            rotated_vertices = R.from_quat(quat).apply(vertices)
            shadow = self._compute_xy_shadow(rotated_vertices)
            xy_shadows.append(shadow)

            candidate_count += 1

        return candidate_rotations, xy_shadows
    
    def find_candidate_rotations_by_shadow_edge_alignment(self,xy_shadow: np.ndarray = None) -> tuple[list[tuple[int, int, int, np.ndarray]], list[np.ndarray]]:
        """
        Generates quaternions rotating around the z-axis by cumulatively adding internal angles.
        The starting edge is the one between the leftmost vertex and its next vertex in the CCW-ordered list.

        :param xy_shadow: (N, 3) array of 3D shadow points on the x-y plane (z=0).
        :return: List of (index, 0, index, quaternion) and corresponding rotated shadows.
        """
        if xy_shadow is None:
            _, _, vertices = self._compute_full_hull()
            xy_shadow = self._compute_xy_shadow(vertices)

        candidate_rotations = []
        xy_shadows = []

        # Step 1: Find leftmost vertex (min x)
        leftmost_idx = np.argmin(xy_shadow[:, 0])

        # Step 2: Define the initial edge from leftmost_idx to (leftmost_idx + 1) % N
        p1 = xy_shadow[leftmost_idx][:2]
        p2 = xy_shadow[(leftmost_idx + 1) % len(xy_shadow)][:2]
        edge_vec = p2 - p1
        edge_dir = edge_vec / np.linalg.norm(edge_vec)

        # Step 3: Compute rotation to align this edge with +Y axis
        y_axis = np.array([0, 1])
        angle = np.arccos(np.clip(np.dot(edge_dir, y_axis), -1.0, 1.0))
        if np.cross(edge_dir, y_axis) < 0:
            angle = -angle

        cumulative_angle = angle

        # Step 4: Compute internal angles between edges
        edge_dirs = []
        for i in range(len(xy_shadow)):
            p1 = xy_shadow[i][:2]
            p2 = xy_shadow[(i + 1) % len(xy_shadow)][:2]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm > 1e-8:
                edge_dirs.append(edge / norm)
            else:
                edge_dirs.append(np.array([0.0, 0.0]))
        edge_dirs = np.array(edge_dirs)

        internal_angles = []
        for i in range(len(edge_dirs)):
            prev = edge_dirs[i - 1]
            curr = edge_dirs[i]
            angle = np.arctan2(np.cross(prev, curr), np.dot(prev, curr))
            internal_angles.append(angle)

        # Step 5: Apply rotations incrementally
        for i in range(len(xy_shadow)):
            quat = R.from_euler('z', -cumulative_angle).as_quat()
            rotated_shadow = R.from_quat(quat).apply(xy_shadow)
            candidate_rotations.append((i, 0, i, quat))
            xy_shadows.append(rotated_shadow)

            # Increment angle CCW
            idx = (leftmost_idx + i + 1) % len(xy_shadow)
            cumulative_angle += internal_angles[idx]

        return candidate_rotations, xy_shadows

    def find_candidate_rotations_by_face_and_shadow_alignment(self) -> tuple[list[tuple[int, int, int, np.ndarray]], list[np.ndarray]]:
        """
        Generates combined re-orientations by first aligning each face normal to the resting face (facing -Z),
        and then rotating around Z to align a shadow edge with +Y.

        Returns:
            candidate_rotations: List of (index, quaternion [x, y, z, w])
            xy_shadows: List of (N, 3) arrays of rotated 2D shadow vertices in XY plane (z=0)
        """
        face_rotations, base_xy_shadows = self.find_candidate_rotations_by_resting_face_normal_alignment()
        combined_rotations = []
        combined_shadows = []
        candidate_count = 0

        for i, (face_id, _, _, face_quat) in enumerate(face_rotations):
            # Compute rotated shadow under face alignment
            shadow = base_xy_shadows[i]

            # Compute shadow-alignment rotations from this configuration
            shadow_rotations, shadow_variants = self.find_candidate_rotations_by_shadow_edge_alignment(shadow)

            for j, (edge_id, _, _, shadow_quat) in enumerate(shadow_rotations):
                # Compose total rotation: shadow_quat * face_quat
                q_total = R.from_quat(shadow_quat) * R.from_quat(face_quat)
                q_total = q_total.as_quat()  # Final rotation quaternion

                combined_rotations.append((candidate_count, face_id, edge_id, q_total))
                combined_shadows.append(shadow_variants[j])

                candidate_count += 1

        return combined_rotations, combined_shadows

    def duplicate_remover(self, rotations: list[tuple[int, int, int, np.ndarray]], xy_shadows) -> list[tuple[int, int, int, np.ndarray]]:
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
            rounded_quat = tuple(np.round(quat, decimals=int(np.log10(1/self.tolerance))))  # Round to prevent numerical noise
            one_zero_rounded_quat = tuple(0.0 if abs(x) < self.tolerance else x for x in rounded_quat)  # Ensure all zeros are positive
            if one_zero_rounded_quat not in unique_rotations:
                unique_rotations[one_zero_rounded_quat] = index
                assigned_rotations.append((pose_count, face_id, edge_id, one_zero_rounded_quat))
                assigned_shadows.append(xy_shadows[index])
                pose_count += 1
        
        return assigned_rotations, assigned_shadows
    
    def symmetry_handler(self, rotations: list[tuple[int, int, int, np.ndarray]]) -> list[tuple[int, int, int, np.ndarray]]:
        """
        Handles symmetry constraints by sorting by pose and assigning the same pose to symmetrically equivalent rotations.
        This is done by checking if the full mesh has multiple rotationally symmetrically equivalent representations.
        :param rotations: List of tuples (pose, valid rotation vector).
        :return: List of tuples (assigned pose, valid rotation vector).
        """
        assigned_rotations = [-1] * len(rotations)  # Initialize with -1 to indicate unassigned
        assymetric_pose_count = 0

        for index, (pose, face_id, edge_id, quat) in enumerate(rotations):
            if assigned_rotations[index] == -1:  # If not yet assigned
                assigned_rotations[index] = assymetric_pose_count

                # Check for symmetrically equivalent rotations
                rotation = R.from_quat(quat)
                rotated_vertices = rotation.apply(self.mesh.vertices)

                for check_index, (_, face_id, edge_id, check_quat) in enumerate(rotations):
                    if assigned_rotations[check_index] == -1:  # Only check unassigned rotations
                        check_rotation = R.from_quat(check_quat)
                        check_rotated_vertices = check_rotation.apply(self.mesh.vertices)

                        sorted_rotated_vertices = np.sort(rotated_vertices, axis=0)
                        sorted_check_rotated_vertices = np.sort(check_rotated_vertices, axis=0)

                        # Check if the vertices are close enough to consider them equivalent
                        is_close = np.allclose(sorted_rotated_vertices, sorted_check_rotated_vertices, atol=self.tolerance, rtol=0.0)

                        if is_close:
                            assigned_rotations[check_index] = assymetric_pose_count

                assymetric_pose_count += 1

        # Return the assigned rotations in the same order as the input
        return [(assigned_rotations[i], rotations[i][1], rotations[i][2], quat) for i, (_, _, _, quat) in enumerate(rotations)]
