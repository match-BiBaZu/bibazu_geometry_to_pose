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
        cylinder_axis_parameters,  # list[ list[(origin, direction[, radius]) | dict] ]  (already rotated per pose)
    ):
        if not rotations:
            return [], [], []

        # ---------- helpers ----------
        def _group_indices_by_axis_direction(cyl_params, tol_dir=1e-3):
            """
            Return groups (list[list[int]]) of pose indices sharing the FIRST cylinder direction.
            Sign-sensitive: +d and -d are DIFFERENT. Poses with no axes become singleton groups.
            """
            groups = []  # list of (ref_dir_unit or None, [indices])
            norm = np.linalg.norm
            for idx, axes in enumerate(cyl_params):
                if not axes:
                    groups.append((None, [idx])); continue
                first = axes[0]
                d = (np.asarray(first["direction"], float)
                    if isinstance(first, dict) else np.asarray(first[1], float))
                nd = norm(d)
                if nd == 0:
                    groups.append((None, [idx])); continue
                d = d / nd
                placed = False
                for gdir, idxs in groups:
                    if gdir is None:
                        continue
                    if norm(d - gdir) < tol_dir:  # sign-sensitive
                        idxs.append(idx); placed = True; break
                if not placed:
                    groups.append((d, [idx]))
            return [idxs for (gdir, idxs) in groups], [gdir for (gdir, idxs) in groups]

        def _line_dist_perp(p, o, d_hat):
            # ⟂ distance from point to axis line (o,d̂): ||(p-o)×d̂||
            return np.linalg.norm(np.cross(p - o, d_hat))

        def _extract_axes_with_radius(per_pose_axes):
            """Return list[(o, d̂, r>0)] from dicts/tuples; normalize d."""
            out = []
            fallback_r = getattr(self, "cylinder_radius", None)
            for k, item in enumerate(per_pose_axes or []):
                if isinstance(item, dict):
                    o = np.asarray(item["origin"], float)
                    d = np.asarray(item["direction"], float)
                    r = float(item.get("radius",
                                    (fallback_r[k] if isinstance(fallback_r, (list, tuple)) and k < len(fallback_r) else 0.0)))
                else:
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

        def _best_axis_match_for_group(verts, axes_with_r, n_ref_hat):
            """
            Among axes_with_r, pick the one whose direction is closest to n_ref_hat
            that also has ≥2 verts near radius. Return (o, d̂, r) or None.
            """
            if verts.shape[0] < 2 or not axes_with_r:
                return None
            abs_tol = float(self.tolerance)  # length units
            rel_tol = 1e-3                   # 0.1% of radius
            best = None
            best_dir_err = np.inf
            for (o, d_hat, r) in axes_with_r:
                band = max(abs_tol, rel_tol * abs(r))
                # check membership
                close = 0
                for p in verts:
                    if abs(_line_dist_perp(p, o, d_hat) - r) <= band:
                        close += 1
                        if close >= 2:
                            break
                if close < 2:
                    continue
                # prefer axis whose direction matches n_ref_hat most
                dir_err = np.linalg.norm(d_hat - n_ref_hat)
                if dir_err < best_dir_err:
                    best_dir_err = dir_err
                    best = (o, d_hat, r)
            return best

        def _fix_quat_sign(q, q_ref):
            # Ensure quaternion continuity (±q represent same rotation)
            q = np.asarray(q, float); q_ref = np.asarray(q_ref, float)
            return q if float(np.dot(q, q_ref)) >= 0.0 else -q

        def _twist_delta_deg_about_fixed_axis(q_prev, q_curr, n_ref_hat):
            """
            Signed twist (deg) of relative rotation about FIXED unit axis n_ref_hat.
            Uses swing–twist via quaternion projection, with sign continuity.
            """
            q_curr = _fix_quat_sign(q_curr, q_prev)
            q_rel = (R.from_quat(q_curr) * R.from_quat(q_prev).inv()).as_quat()  # [x,y,z,w]
            vx, vy, vz, w = q_rel
            v_dot_n = vx * n_ref_hat[0] + vy * n_ref_hat[1] + vz * n_ref_hat[2]
            delta_rad = 2.0 * np.arctan2(v_dot_n, w)
            delta_deg = np.degrees(delta_rad)
            return float(((delta_deg + 180.0) % 360.0) - 180.0)  # wrap to (-180,180]

        def _aligned(value_deg, step_deg, tol_deg):
            rem = value_deg % step_deg
            return min(abs(rem), abs(step_deg - rem)) <= tol_deg

        # ---------- constants ----------
        step_deg = 360.0 / float(self.rotation_steps)
        tol_deg  = np.degrees(self.tolerance)  # if self.tolerance is radians; else use self.tolerance

        kept_rot, kept_shad, kept_axes = [], [], []
        pose_id = 0

        # 1) Build direction groups (+d and −d DIFFERENT)
        groups, group_dirs = _group_indices_by_axis_direction(cylinder_axis_parameters, tol_dir=1e-3)

        # 2) Iterate groups; entering a new group resets counters and locks n_ref_hat
        for g, idx_list in enumerate(groups):
            n_ref_hat = group_dirs[g]
            # if group has None (no axes), process poses as isolated (no accumulation)
            if n_ref_hat is not None:
                n_ref_hat = n_ref_hat / (np.linalg.norm(n_ref_hat) or 1.0)

            sum_deg = 0.0
            q_prev = None  # last quaternion actually used for delta (after sign-fix)
            have_sequence = False  # True once on-cylinder condition satisfied

            for idx in idx_list:
                old_id, face_id, edge_id, quat = rotations[idx]

                # build current pose mesh
                rot = R.from_quat(quat)
                T = np.eye(4); T[:3, :3] = rot.as_matrix()
                rot_mesh = self.convex_hull_mesh.copy(); rot_mesh.apply_transform(T)

                rest_base, rest_back = _resting_sets(rot_mesh)
                axes_with_r = _extract_axes_with_radius(cylinder_axis_parameters[idx])

                if n_ref_hat is None:
                    # No axis in this group -> keep pose as is; no accumulation
                    kept_rot.append((pose_id, face_id, edge_id, quat))
                    kept_shad.append(xy_shadows[idx])
                    kept_axes.append(cylinder_axis_parameters[idx])
                    pose_id += 1
                    # reset inner sequence just in case
                    sum_deg = 0.0; q_prev = None; have_sequence = False
                    continue

                # 3) Radius/membership check FIRST on both resting sets; pick best axis closest to n_ref_hat
                match1 = _best_axis_match_for_group(rest_base, axes_with_r, n_ref_hat)
                match2 = _best_axis_match_for_group(rest_back, axes_with_r, n_ref_hat)
                match = match1 if (match1 and (not match2)) else (match2 if (match2 and (not match1)) else (match1 if (match1 and match2 and
                        np.linalg.norm(match1[1] - n_ref_hat) <= np.linalg.norm(match2[1] - n_ref_hat)) else None))

                if match is None:
                    # Not on-cylinder -> keep pose (if you want), but reset accumulation
                    kept_rot.append((pose_id, face_id, edge_id, quat))
                    kept_shad.append(xy_shadows[idx])
                    kept_axes.append(cylinder_axis_parameters[idx])
                    pose_id += 1
                    sum_deg = 0.0; q_prev = None; have_sequence = False
                    continue

                # 4) Accumulate twist about FIXED group axis
                if not have_sequence:
                    # first cylindrical pose in this group
                    have_sequence = True
                    q_prev = np.asarray(quat, float)
                    # optional: keep if aligned at 0°
                    if _aligned(0.0, step_deg, tol_deg):
                        kept_rot.append((pose_id, face_id, edge_id, quat))
                        kept_shad.append(xy_shadows[idx])
                        kept_axes.append(cylinder_axis_parameters[idx])
                        pose_id += 1
                    continue

                # sign-continuous delta about n_ref_hat
                delta = _twist_delta_deg_about_fixed_axis(q_prev, quat, n_ref_hat)
                sum_deg += delta
                print(f"Pose {old_id} face id {face_id} delta {delta:.2f}° -> sum {sum_deg:.2f}° about axis {n_ref_hat}")
                q_prev = _fix_quat_sign(np.asarray(quat, float), q_prev)  # store the sign-consistent version

                if _aligned(sum_deg, step_deg, tol_deg):
                    kept_rot.append((pose_id, face_id, edge_id, quat))
                    kept_shad.append(xy_shadows[idx])
                    kept_axes.append(cylinder_axis_parameters[idx])
                    pose_id += 1

        return kept_rot, kept_shad, kept_axes


    
