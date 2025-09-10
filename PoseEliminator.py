import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from PoseFinder import PoseFinder
from matplotlib.path import Path
import matplotlib.pyplot as plt
import trimesh

class PoseEliminator(PoseFinder):
    def __init__(self, convex_hull_obj_file: str, self_obj_file: str, tolerance: float = 1e-5, rotation_steps: int = 12):
        """
        Initialize the PoseEliminator with the convex hull OBJ file.
        :param convex_hull_obj_file: Path to the convex hull OBJ file.
        """
        super().__init__(convex_hull_obj_file, self_obj_file, tolerance)
        
        # Additional initialization (if any) can go here
        if rotation_steps <= 0:
            raise ValueError("rotation_steps must be >= 1")
        self.rotation_steps = int(rotation_steps)

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
        
    def remove_duplicates(self, rotations: list[tuple[int, int, int, np.ndarray]], xy_shadows, cylinder_axis_parameters) -> list[tuple[int, int, int, np.ndarray], list[np.ndarray], list[tuple[tuple[float, float, float], tuple[float, float, float]] | None]]:
        """
        Handles duplicate rotations by removing rotations that are too close to each other.
        :param rotations: List of tuples (pose, face_id, shadow_id, valid rotation vector) and list of xy_shadow arrays.
        :return: List of tuples (assigned pose, face_id, shadow_id, valid rotation vector) and list of xy_shadow arrays.
        """
        unique_rotations = {}
        assigned_rotations = []
        assigned_shadows = []
        assigned_cylinder_axis_parameters = []
        pose_count = 0

        for index, face_id, edge_id, quat in rotations:
            if quat not in unique_rotations:
                unique_rotations[quat] = index
                assigned_rotations.append((pose_count, face_id, edge_id, quat))
                assigned_shadows.append(xy_shadows[index])
                assigned_cylinder_axis_parameters.append(cylinder_axis_parameters[index])
                pose_count += 1
        
        return assigned_rotations, assigned_shadows, assigned_cylinder_axis_parameters
    
    def remove_unstable_poses(self, rotations: list[tuple[int, int, int, np.ndarray]], xy_shadows, cylinder_axis_parameters) -> list[tuple[int, int, int, np.ndarray], list[np.ndarray], list[tuple[tuple[float, float, float], tuple[float, float, float]] | None]]:
        """
        Removes unstable poses based on the convex hull.
        :param rotations: List of tuples (pose, face_id, shadow_id, valid rotation vector).
        :return: List of tuples (assigned pose, face_id, shadow_id, valid rotation vector).
        """
        stable_rotations = []
        stable_shadows = []
        stable_cylinder_axis_parameters = []
        pose_count = 0

        for index, face_id, edge_id, quat in rotations:
            #print(f"Checking pose {index} for stability...")
            is_stable = self._is_stable_by_center_mass(quat)
            if is_stable:
                #print(f"Pose {index} is stable.")
                stable_rotations.append((pose_count, face_id, edge_id, quat))
                stable_shadows.append(xy_shadows[index])
                stable_cylinder_axis_parameters.append(cylinder_axis_parameters[index])
                pose_count += 1
        
        return stable_rotations, stable_shadows, stable_cylinder_axis_parameters
    
    def discretise_rotations(
        self,
        rotations,                 # list[(old_id, face_id, edge_id, quat[x,y,z,w])]
        xy_shadows,                # list[np.ndarray]
        cylinder_axis_parameters,  # list[ list[ (origin, direction [, radius]) | dict ] ]  (already rotated per pose)
    ):
        if not rotations:
            return [], [], []

        # --- helpers ---
        def _line_dist_perp(p, o, d_hat):
            # || (p - o) x d̂ ||, with d̂ unit
            return np.linalg.norm(np.cross(p - o, d_hat))

        def _extract_axes_with_radius(per_pose_axes):
            """Return list[(o, d_hat, r>0)]. Accept dict or tuple forms. Normalize d."""
            out = []
            fallback_r = getattr(self, "cylinder_radius", None)  # optional parallel radii
            for k, item in enumerate(per_pose_axes or []):
                if isinstance(item, dict):
                    o = np.asarray(item["origin"], float)
                    d = np.asarray(item["direction"], float)
                    r = float(item.get("radius",
                                    (fallback_r[k] if isinstance(fallback_r, (list, tuple)) and k < len(fallback_r) else 0.0)))
                else:
                    # ((o),(d)) or ((o),(d),r)
                    o = np.asarray(item[0], float)
                    d = np.asarray(item[1], float)
                    r = float(item[2]) if len(item) >= 3 else (
                        float(fallback_r[k]) if isinstance(fallback_r, (list, tuple)) and k < len(fallback_r) else 0.0)
                n = np.linalg.norm(d)
                if n == 0.0 or r <= 0.0:
                    continue
                out.append((o, d / n, r))
            return out

        def _resting_sets(rot_mesh):
            V = rot_mesh.vertices
            z = V[:, 2]; x = V[:, 0]
            min_z, min_x = np.min(z), np.min(x)
            idx_z = np.where(np.abs(z - min_z) < self.tolerance)[0]
            idx_x = np.where(np.abs(x - min_x) < self.tolerance)[0]
            return V[idx_z], V[idx_x]

        def _pick_cyl_match(verts, axes_with_r):
            """Return (o, d_hat, r) of first axis with ≥2 verts at radius, else None."""
            if verts.shape[0] < 2 or not axes_with_r:
                return None
            abs_tol = float(self.tolerance)         # length units
            rel_tol = 1e-3                          # 0.1% radius band
            for (o, d_hat, r) in axes_with_r:
                band = max(abs_tol, rel_tol * abs(r))
                close = 0
                for p in verts:
                    if abs(_line_dist_perp(p, o, d_hat) - r) <= band:
                        close += 1
                        if close >= 2:
                            return (o, d_hat, r)
            return None

        def _twist_delta_deg_about_axis(q_prev, q_curr, d_hat):
            """Signed twist angle (deg) of relative rotation about axis d̂ in current orientation."""
            R_rel = (R.from_quat(q_curr) * R.from_quat(q_prev).inv())
            Rm = R_rel.as_matrix()
            # pick u ⟂ d̂ (deterministic)
            a = np.array([1.0, 0.0, 0.0]);  b = np.array([0.0, 1.0, 0.0])
            u = a if abs(np.dot(a, d_hat)) < 0.9 else b
            u = u - np.dot(u, d_hat) * d_hat
            u /= (np.linalg.norm(u) or 1.0)
            u_p = Rm @ u
            # ensure u_p stays on plane ⟂ d̂
            u_p = u_p - np.dot(u_p, d_hat) * d_hat
            nrm = np.linalg.norm(u_p)
            if nrm == 0.0:
                return 0.0
            u_p /= nrm
            # signed angle from u to u_p around d̂
            s = np.dot(d_hat, np.cross(u, u_p))
            c = np.clip(np.dot(u, u_p), -1.0, 1.0)
            return float(np.degrees(np.arctan2(s, c)))

        def _wrap_deg(x):
            return ((x + 180.0) % 360.0) - 180.0

        def _aligned(value_deg, step_deg, tol_deg):
            rem = value_deg % step_deg
            return min(abs(rem), abs(step_deg - rem)) <= tol_deg

        step_deg = 360.0 / float(self.rotation_steps)
        tol_deg  = np.degrees(self.tolerance)  # if your tolerance is radians; else use self.tolerance directly

        # sequence state
        mode = None                # "base" | "back" | None
        sum_deg = 0.0
        prev_quat = None
        tracked_axis = None        # (o, d_hat, r) used for twist projection

        kept_rot, kept_shad, kept_axes = [], [], []
        pose_id = 0

        for idx, (old_id, face_id, edge_id, quat) in enumerate(rotations):
            #print(f"Discretizing pose {old_id} ({idx+1}/{len(rotations)})...")
            #print(f'sum_deg={sum_deg}, mode={mode}, tracked_axis={tracked_axis}')
            # rotate mesh
            rot = R.from_quat(quat); T = np.eye(4); T[:3, :3] = rot.as_matrix()
            rot_mesh = self.mesh.copy(); rot_mesh.apply_transform(T)

            rest_base, rest_back = _resting_sets(rot_mesh)
            axes_with_r = _extract_axes_with_radius(cylinder_axis_parameters[old_id])  # already pose-rotated

            # --- 1) radius/membership check FIRST ---
            match_base = _pick_cyl_match(rest_base, axes_with_r)
            match_back = _pick_cyl_match(rest_back, axes_with_r)

            if match_base is None and match_back is None:
                # reset if no cylindrical resting detected
                mode = None; sum_deg = 0.0; prev_quat = None; tracked_axis = None
                continue

            # decide active mode and matched axis
            new_mode = "base" if match_base is not None else "back"
            new_axis = match_base if match_base is not None else match_back  # (o, d_hat, r)

            # reset sequence if switching side or cylinder axis
            if (mode != new_mode) or (tracked_axis is None or np.linalg.norm(tracked_axis[1] - new_axis[1]) > 1e-6):
                mode = new_mode
                sum_deg = 0.0
                prev_quat = quat
                tracked_axis = new_axis
                # allow the first pose of a new sequence to pass only if aligned at 0°
                if _aligned(0.0, step_deg, tol_deg):
                    print(f"Keeping pose {old_id}, pose id {pose_id} at 0.0° about {mode} axis")
                    kept_rot.append((pose_id, face_id, edge_id, quat))
                    kept_shad.append(xy_shadows[old_id])
                    kept_axes.append(cylinder_axis_parameters[old_id])
                    pose_id += 1
                continue

            # --- 2) incremental twist about the current cylinder axis ---
            delta = _twist_delta_deg_about_axis(prev_quat, quat, tracked_axis[1])
            delta = _wrap_deg(delta)
            sum_deg += delta
            prev_quat = quat

            if _aligned(sum_deg, step_deg, tol_deg):
                print(f"Keeping pose {old_id}, pose id {pose_id} at {sum_deg:.1f}° about {mode} axis")
                kept_rot.append((pose_id, face_id, edge_id, quat))
                kept_shad.append(xy_shadows[old_id])
                kept_axes.append(cylinder_axis_parameters[old_id])
                pose_id += 1

        return kept_rot, kept_shad, kept_axes
    
