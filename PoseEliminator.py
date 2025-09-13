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
            """Return (groups, group_dirs) where groups is list[list[int]]; +d and −d DIFFERENT."""
            groups = []   # list of (ref_dir_unit or None, [indices])
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
                    if gdir is None: continue
                    if norm(d - gdir) < tol_dir:  # sign-sensitive
                        idxs.append(idx); placed = True; break
                if not placed:
                    groups.append((d, [idx]))
            return [idxs for (gdir, idxs) in groups], [gdir for (gdir, idxs) in groups]

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
                if n == 0.0 or r <= 0.0: continue
                out.append((o, d / n, r))
            return out

        def _resting_sets(rot_mesh):
            V = rot_mesh.vertices
            z = V[:, 2]; x = V[:, 0]
            min_z, min_x = np.min(z), np.min(x)
            idx_z = np.where(np.abs(z - min_z) < self.tolerance)[0]
            idx_x = np.where(np.abs(x - min_x) < self.tolerance)[0]
            return V[idx_z], V[idx_x]

        def _line_dist_perp(p, o, d_hat):
            return np.linalg.norm(np.cross(p - o, d_hat))  # ⟂ distance to axis line

        def _axes_aligned_with_dir(axes_with_r, n_ref_hat, tol_dir=1e-3):
            """Select all axes whose direction matches n_ref_hat (sign-sensitive)."""
            aligned = []
            for (o, d_hat, r) in axes_with_r:
                if np.linalg.norm(d_hat - n_ref_hat) < tol_dir:
                    aligned.append((o, d_hat, r))
            return aligned

        def _two_vertices_at_radius_for_any(verts, axes_subset):
            """
            True iff there exists at least one axis in axes_subset for which
            there are >=2 vertices with |dist_perp - radius| within band.
            Vertices may coincide across different axes (allowed).
            """
            if not axes_subset or verts.shape[0] < 2:
                return False
            for (o, d_hat, r) in axes_subset:
                band = 10 * self.tolerance * abs(r) # abs + relative band
                print(f"  Axis at {o} dir {d_hat} r={r} band={band}")
                count = 0
                for p in verts:
                    print(f' abs(_line_dist_perp(p, o, d_hat) - r) <= band: {abs(_line_dist_perp(p, o, d_hat) - r)} <= {band}')
                    if abs(_line_dist_perp(p, o, d_hat) - r) <= band:
                        count += 1
                        print(f"    Vertex {p} at radius (count={count})")
                        if count >= 2:
                            return True
            return False

        def _is_on_cylinder_for_pose(idx, n_ref_hat):
            """Pose is on-cylinder if base OR back has ≥2 verts at radius for ANY aligned axis."""
            old_id, face_id, edge_id, q = rotations[idx]
            rot = R.from_quat(q); T = np.eye(4); T[:3, :3] = rot.as_matrix()
            m = self.convex_hull_mesh.copy(); m.apply_transform(T)
            rest_base, rest_back = _resting_sets(m)
            axes = _extract_axes_with_radius(cylinder_axis_parameters[idx])
            aligned_axes = _axes_aligned_with_dir(axes, n_ref_hat, tol_dir=1e-3)
            print(f"Pose {idx}: {len(axes)} axes, {len(aligned_axes)} aligned with n_ref.")
            if not aligned_axes:
                return False
            return _two_vertices_at_radius_for_any(rest_base, aligned_axes) or \
                _two_vertices_at_radius_for_any(rest_back, aligned_axes)

        def _fix_quat_sign(q, q_ref):
            q = np.asarray(q, float); q_ref = np.asarray(q_ref, float)
            return q if float(np.dot(q, q_ref)) >= 0.0 else -q

        def _abs_twist_deg_wrt_ref(q_ref, q_i, n_ref_hat):
            """Absolute twist angle (deg) of pose i about fixed axis n_ref, w.r.t. reference pose q_ref."""
            q_i = _fix_quat_sign(q_i, q_ref)
            q_rel = (R.from_quat(q_i) * R.from_quat(q_ref).inv()).as_quat()  # [x,y,z,w]
            vx, vy, vz, w = q_rel
            vdot = vx * n_ref_hat[0] + vy * n_ref_hat[1] + vz * n_ref_hat[2]
            ang = 2.0 * np.arctan2(vdot, w)         # radians
            return float(np.degrees(ang) % 360.0)   # [0,360)

        # ---------- main ----------
        kept_rot, kept_shad, kept_axes = [], [], []
        pose_id = 0

        groups, group_dirs = _group_indices_by_axis_direction(cylinder_axis_parameters, tol_dir=1e-3)
        print(f"Found {len(groups)} groups of poses by cylinder axis direction.")

        for g, idxs in enumerate(groups):
            n_ref = group_dirs[g]
            if n_ref is None:
                # No axis in this group → pass through
                for i in idxs:
                    print(f"Warning: Pose {i} has no cylinder axis; keeping without discretization.")
                    old_id, face_id, edge_id, quat = rotations[i]
                    kept_rot.append((pose_id, face_id, edge_id, quat))
                    kept_shad.append(xy_shadows[i])
                    kept_axes.append(cylinder_axis_parameters[i])
                    pose_id += 1
                continue

            n_ref = n_ref / (np.linalg.norm(n_ref) or 1.0)

            # Pick reference quaternion: first pose in group that satisfies ANY-axis on-cylinder test; else first pose
            q_ref = None
            on_cyl_flag = {}
            for i in idxs:
                on_cyl = _is_on_cylinder_for_pose(i, n_ref)
                print(f"Pose {i} in group {g} is {'on-cylinder' if on_cyl else 'not on-cylinder'}.")
                on_cyl_flag[i] = on_cyl
                if q_ref is None and on_cyl:
                    q_ref = np.asarray(rotations[i][3], float)
                    print(f"Selected pose {i} as reference for group {g}.")
            if q_ref is None:
                q_ref = np.asarray(rotations[idxs[0]][3], float)
                print(f"Warning: No on-cylinder pose found in group {g}; using first pose {idxs[0]} as reference.")
            
            print(f"Group {g}: n_ref={n_ref}, reference pose id={i} (on-cylinder={on_cyl_flag[i]})")

            # Compute absolute twist for cylindrical poses; non-cylindrical kept after
            cyl_items, noncyl_items = [], []
            for i in idxs:
                if on_cyl_flag[i]:
                    theta = _abs_twist_deg_wrt_ref(q_ref, rotations[i][3], n_ref)
                    cyl_items.append((theta, i))
                else:
                    noncyl_items.append(i)

            # Sort cylindrical by increasing twist
            cyl_items.sort(key=lambda t_i: t_i[0])

            print(f"Group {g}: {len(cyl_items)} cylindrical poses, {len(noncyl_items)} non-cylindrical poses.")

            # Segment-based filter on cylindrical list (keep ~4 evenly spaced)
            L = len(cyl_items)
            if L > 0:
                threshold = max(1, int(round(L / self.rotation_steps)))
                for k, (_, i) in enumerate(cyl_items):
                    if (k % threshold) == 0:
                        print(f"Keeping pose {i} in cylindrical group {g} (k={k}, threshold={threshold})")
                        old_id, face_id, edge_id, quat = rotations[i]
                        kept_rot.append((pose_id, face_id, edge_id, quat))
                        kept_shad.append(xy_shadows[i])
                        kept_axes.append(cylinder_axis_parameters[i])
                        pose_id += 1

            # Append non-cylindrical (preserve relative order)
            for i in noncyl_items:
                print(f"Keeping non-cylindrical pose {i} in group {g}")
                old_id, face_id, edge_id, quat = rotations[i]
                kept_rot.append((pose_id, face_id, edge_id, quat))
                kept_shad.append(xy_shadows[i])
                kept_axes.append(cylinder_axis_parameters[i])
                pose_id += 1

        return kept_rot, kept_shad, kept_axes





    
