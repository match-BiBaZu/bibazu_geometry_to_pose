import numpy as np
from scipy.spatial.transform import Rotation as R
from PoseFinder import PoseFinder


class CylinderHandler(PoseFinder):
    """
    Classify resting poses with respect to cylinder/cone features.

    pose_types[i]:
        0 -> non-cylinder pose
        1 -> cylinder pose
        2 -> non-wobbly cylinder pose (axis ~ ±Y or ±Z)
    """

    def __init__(self, convex_hull_obj_file: str, self_obj_file: str, tolerance: float = 1e-5, rotation_steps: int = 12, wobble_angle: float = 2.0, axis_based_cylinder_check: int = 1):

        super().__init__(convex_hull_obj_file, self_obj_file, tolerance)

        #how many discrete poses do you want from a rotation around a cylinder axis
        self.rotation_steps = int(rotation_steps)
        #as a mesh is not perfectly aligned to a cylinder axis, we allow a certain wobble angle
        self.wobble_tolerance = 2 * np.sin((wobble_angle * np.pi) / 360)
        # some extra values to twiddle to get the poses resting on 'cone features' working        
        self.plane_alignment_band = getattr(self, 'plane_alignment_band', 10.0) * self.wobble_tolerance
        # cylinder classification method switch
        self.axis_based_cylinder_check = axis_based_cylinder_check

    # ---------- helper methods ----------

    def _group_by_first_dir(self, params):
        """Return groups of indices; groups share (approx.) same first axis direction."""
        groups = []   # list[(ref_dir or None, [indices])]
        for idx, axes in enumerate(params):
            if not axes:
                groups.append((None, [idx]))
                continue
            first = axes[0]
            d = np.asarray(first["direction"] if isinstance(first, dict) else first[1], float)
            n = np.linalg.norm(d)
            if n == 0:
                groups.append((None, [idx]))
                continue
            d /= n
            placed = False
            for gdir, idxs in groups:
                if gdir is None:
                    continue
                if np.linalg.norm(d - gdir) < self.wobble_tolerance:
                    idxs.append(idx)
                    placed = True
                    break
            if not placed:
                groups.append((d, [idx]))
        return [idxs for (gdir, idxs) in groups], [gdir for (gdir, idxs) in groups]

    def _extract_axes_with_radius(self, per_pose_axes):
        """Return list[(origin, direction_normalized, radius>0)] from dicts/tuples; normalize direction."""
        extracted_axes = []
        fallback_radius = self.cylinder_radius
        for axis_idx, axis_item in enumerate(per_pose_axes or []):
            if isinstance(axis_item, dict):
                origin = np.asarray(axis_item["origin"], float)
                direction = np.asarray(axis_item["direction"], float)
                radius = float(axis_item.get(
                    "radius",
                    (fallback_radius[axis_idx] if isinstance(fallback_radius, (list, tuple)) and axis_idx < len(fallback_radius) else 0.0)
                ))
            else:
                origin = np.asarray(axis_item[0], float)
                direction = np.asarray(axis_item[1], float)
                radius = float(axis_item[2]) if len(axis_item) >= 3 else (
                    float(fallback_radius[axis_idx]) if isinstance(fallback_radius, (list, tuple)) and axis_idx < len(fallback_radius) else 0.0
                )
            direction_norm = np.linalg.norm(direction)
            if direction_norm == 0.0 or radius <= 0.0:
                continue
            extracted_axes.append((origin, direction / direction_norm, radius))
        return extracted_axes

    def _resting_sets(self, mesh_in_pose):
        """Vertices near base/back planes, excluding planar contacts."""
        V = mesh_in_pose.vertices
        bbox = mesh_in_pose.bounds
        diag = float(np.linalg.norm(bbox[1] - bbox[0]))
        band_abs = self.tolerance * diag * self.plane_alignment_band

        z = V[:, 2]
        x = V[:, 0]
        min_z, min_x = float(np.min(z)), float(np.min(x))
        idx_z = np.where((z - min_z) <= band_abs)[0]
        idx_x = np.where((x - min_x) <= band_abs)[0]

        # normals
        try:
            N = mesh_in_pose.vertex_normals
            if N is None or len(N) != len(V):
                raise Exception()
        except Exception:
            Nf = mesh_in_pose.face_normals
            N = np.zeros_like(V)
            for fi, tri in enumerate(mesh_in_pose.faces):
                N[tri] += Nf[fi]
            L = np.linalg.norm(N, axis=1)
            ok = L > 0
            N[ok] /= L[ok][:, None]

        g_base = np.array([0.0, 0.0, 1.0])
        g_back = np.array([1.0, 0.0, 0.0])

        idx_base = idx_z[np.where(np.abs(N[idx_z] @ g_base) < 0.94)[0]]
        idx_back = idx_x[np.where(np.abs(N[idx_x] @ g_back) < 0.94)[0]]
        return V[idx_base], V[idx_back]

    @staticmethod
    def _line_dist_perp(p, o, d_hat):
        """Perpendicular distance point→axis."""
        return np.linalg.norm(np.cross(p - o, d_hat))

    def _axes_aligned(self, axes_with_r, n_ref, canonicalize=True):
        """Pick axes whose direction matches n_ref (optionally flip sign to match)."""
        out = []
        for (o, d, r) in axes_with_r:
            d2 = (-d if (canonicalize and np.dot(d, n_ref) < 0.0) else d)
            if np.linalg.norm(d2 - n_ref) < self.wobble_tolerance:
                out.append((o, d2, r))
        return out
    
    def _group_coaxial_axes(self, aligned_axes):
        """
        Groups axes that are approximately coaxial (same or opposite direction, collinear origins).
        Returns a list of group IDs per axis (e.g. [0, 0, 1]).
        """
        if not aligned_axes:
            return []

        tol = self.wobble_tolerance
        n = len(aligned_axes)
        group_ids = [-1] * n
        next_group = 0

        for i in range(n):
            if group_ids[i] != -1:
                continue  # already assigned

            # Start a new group for axis i
            oi, di, _ = aligned_axes[i]
            di = di / np.linalg.norm(di)
            group_ids[i] = next_group

            for j in range(i + 1, n):
                if group_ids[j] != -1:
                    continue

                oj, dj, _ = aligned_axes[j]
                dj = dj / np.linalg.norm(dj)

                # directions aligned or opposite
                if min(np.linalg.norm(di - dj), np.linalg.norm(di + dj)) > tol:
                    continue

                # check if oj lies on line through oi in direction di
                v = oj - oi
                projection = np.dot(v, di) * di
                residual = v - projection
                if np.linalg.norm(residual) <= tol:
                    group_ids[j] = next_group

            next_group += 1

        return group_ids


    def _is_at_cylinder_radius(self, verts, axis, coaxial_radii):
        """all verts at the radius of a single axis (band = 10 * tolerance * |r|)."""
        (o, d, r) = axis
        rad_cnt = 0

        band = 10.0 * float(self.tolerance) * abs(r)
        
        if len(verts) == 0:
            return False
        
        for p in verts:
            dist = self._line_dist_perp(p, o, d)
            # Check if vertex matches the primary radius
            if abs(dist - r) <= band:
                rad_cnt += 1
            # Also check against any coaxial radii
            else:
                for coax_r in coaxial_radii:
                    band_coax = 10.0 * float(self.tolerance) * abs(coax_r)
                    if abs(dist - coax_r) <= band_coax:
                        rad_cnt += 1
                        break
        print(f"Axis check  rad_cnt={rad_cnt}, total verts={len(verts)}")

        return rad_cnt == len(verts) 
    
    def _is_at_cylinder_radius_perpendicular(self, plane_axis, axis_origin, radius):
        """
        check if the axis origin is perpedicularly radius away from the plane defined by the contact vertices)
        """

        #always return true if axis based cylinder check is enabled
        if self.axis_based_cylinder_check == 1:
            return True

        band = 500.0 * float(self.tolerance) * abs(radius)

        dist_to_plane = abs(plane_axis - axis_origin)
        
        print(f"Axis origin check distance to plane={dist_to_plane}, radius={radius}, band={band}")

        return abs(dist_to_plane - radius) <= band       

    def _get_aligned_cylinder(self, idx, n_ref, rotations, cylinder_axis_parameters,cylinder_alignment_type=0):
        """Return radius, direction, and origin of ANY aligned axis with base or back plate at its radius."""
        old_id, face_id, edge_id, q = rotations[idx]
        rot = R.from_quat(q)
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        m = self.convex_hull_mesh.copy()
        m.apply_transform(T)
        baseV, backV = self._resting_sets(m)
        axes = self._extract_axes_with_radius(cylinder_axis_parameters[idx])
        aligned = self._axes_aligned(axes, n_ref)
        group_ids = self._group_coaxial_axes(aligned)

        if not aligned:
            return 0.0, None, None
        for i, ax in enumerate(aligned):
            print(f"Checking axis with radius {ax[2]} for pose {idx}...")
            #find list of radii for coaxial axes, specifically for workpieces such as kk1a where there are multiple coaxial cylinders
            coaxial_radii = [aligned[j][2] for j in range(len(aligned)) if group_ids[j] == group_ids[i]]
            if self._is_at_cylinder_radius(baseV, ax, coaxial_radii) and self._is_at_cylinder_radius_perpendicular(baseV[0][2], ax[0][2], ax[2]):
                if self._is_at_cylinder_radius(backV, ax, coaxial_radii) and self._is_at_cylinder_radius_perpendicular(backV[0][0], ax[0][0], ax[2]):
                    cylinder_alignment_type = 3 # full cylinder pose
                    print(f"Found full cylinder pose (3) for pose {idx} with radius {ax[2]}.")
                    origin, direction, radius = ax
                    return radius, direction, origin,cylinder_alignment_type
                else:
                    print(f"Found back cylinder pose (2) for pose {idx} with radius {ax[2]}.")
                    cylinder_alignment_type = 2 # cylinder resting on back only
                    origin, direction, radius = ax
                    return radius, direction, origin,cylinder_alignment_type
            else:

                if self._is_at_cylinder_radius(backV, ax, coaxial_radii) and self._is_at_cylinder_radius_perpendicular(backV[0][0], ax[0][0], ax[2]):
                    print(f"Found base cylinder pose (1) for pose {idx} with radius {ax[2]}.")
                    cylinder_alignment_type = 1 # cylinder resting on base only
                    origin, direction, radius = ax
                    return radius, direction, origin,cylinder_alignment_type
                else:
                    cylinder_alignment_type = 0
                    print(f"No cylinder pose found for pose {idx} with radius {ax[2]}.")
                    return 0.0, None, None,cylinder_alignment_type

    def _pose_has_pure_y_or_z(self, idx, n_ref, cylinder_axis_parameters):
        """Check if any aligned axis is ~ ±Y or ±Z."""
        axes = self._extract_axes_with_radius(cylinder_axis_parameters[idx])
        aligned = self._axes_aligned(axes, n_ref)
        if not aligned:
            return False

        ey = np.array([0.0, 1.0, 0.0])
        ez = np.array([0.0, 0.0, 1.0])
        tol = float(self.wobble_tolerance)
        for (_, d, _r) in aligned:
            if min(np.linalg.norm(d - ey), np.linalg.norm(d + ey)) < tol:
                return True
            if min(np.linalg.norm(d - ez), np.linalg.norm(d + ez)) < tol:
                return True
        return False

    def _pose_has_pure_y(self, idx, n_ref, cylinder_axis_parameters):
        """Check if any aligned axis is ~ ±Y."""
        axes = self._extract_axes_with_radius(cylinder_axis_parameters[idx])
        aligned = self._axes_aligned(axes, n_ref)
        if not aligned:
            return False

        ey = np.array([0.0, 1.0, 0.0])
        for (_, d, _r) in aligned:
            if min(np.linalg.norm(d - ey), np.linalg.norm(d + ey)) < self.wobble_tolerance*10: #special value for Kk1a
                return True
        return False

    def _pose_has_pure_z(self, idx, n_ref, cylinder_axis_parameters):
        """Check if any aligned axis is ~ ±Z."""
        axes = self._extract_axes_with_radius(cylinder_axis_parameters[idx])
        aligned = self._axes_aligned(axes, n_ref)
        if not aligned:
            return False

        ez = np.array([0.0, 0.0, 1.0])
        for (_, d, _r) in aligned:
            if min(np.linalg.norm(d - ez), np.linalg.norm(d + ez)) < self.wobble_tolerance*0.1:
                return True
        return False
     
    # ---------- main method ----------
    def find_cylinder_poses(self,
                            rotations,
                            xy_shadows,
                            cylinder_axis_parameters):
        """
        Classify each pose as cylinder-related or not.

        Returns:
            rotations, xy_shadows, cylinder_axis_parameters,
            pose_types (list[int]), cylinder_radius (list[float]),
            cylinder_axis_direction (list[np.ndarray]),
            cylinder_axis_origin (list[np.ndarray]),
            cylinder_group (list[int])
        """
        if not rotations:
            return [], [], [], [], [], [], [], []

        n_poses = len(rotations)
        pose_types = [0] * n_poses
        cylinder_radius = [0] * n_poses
        cylinder_axis_direction = [[0, 0, 0]] * n_poses
        cylinder_axis_origin = [[0, 0, 1]] * n_poses
        cylinder_group = [0] * n_poses

        groups, group_dirs = self._group_by_first_dir(cylinder_axis_parameters)
        print(f"Classifying {n_poses} poses in {len(groups)} direction groups.")

        for g, idxs in enumerate(groups):
            n_group = group_dirs[g]
            if n_group is None:
                continue

            n_group = n_group / (np.linalg.norm(n_group) or 1.0)
            candidates_base, candidates_side, candidate_all = [], [], []

            for i in idxs:
                radius, direction, origin, cylinder_alignment_type = self._get_aligned_cylinder(
                    i, n_group, rotations, cylinder_axis_parameters, 0)

                if cylinder_alignment_type == 0:
                    pose_types[i] = 0  # non-cylinder pose
                    cylinder_radius[i] = 0.0
                    cylinder_axis_direction[i] = [0, 0, 0]
                    continue
                else:
                    if self._pose_has_pure_z(i, n_group, cylinder_axis_parameters):
                        if cylinder_alignment_type in (3, 1):
                            candidates_base.append((i, rotations[i], direction, origin, radius))
                            candidate_all.append(i)
                            pose_types[i] = 1  # mark as repeat cylinder pose to seperate from non cylinder pose
                        continue

                    if self._pose_has_pure_y(i, n_group, cylinder_axis_parameters):
                        if cylinder_alignment_type in (3, 2):
                            candidates_side.append((i, rotations[i], direction, origin, radius))
                            candidate_all.append(i)
                            pose_types[i] = 1  # mark as repeat cylinder pose to seperate from non cylinder pose
                        continue

                    pose_types[i] = 4  # wobbly cylinder pose
                    cylinder_radius[i] = radius
                    cylinder_axis_direction[i] = direction
                    cylinder_axis_origin[i] = origin
                    cylinder_group[i] = g
                    candidate_all.append(i)


            # Assign lowest-z and highest-z pose from candidates_side as type 3
            self._assign_aligned_z_pose(candidates_side, pose_types, cylinder_axis_direction,
                                    cylinder_axis_origin, cylinder_radius, cylinder_group, group_id=g)

            # Assign lowest-y and highest-y pose from candidates_base as type 2
            self._assign_aligned_y_pose(candidates_base, pose_types, cylinder_axis_direction,
                                    cylinder_axis_origin, cylinder_radius, cylinder_group, group_id=g)

        return (rotations, xy_shadows, cylinder_axis_parameters,
                pose_types, cylinder_radius,
                cylinder_axis_direction, cylinder_axis_origin, cylinder_group)

    def _assign_aligned_z_pose(self, candidates, pose_types, dirs, origins, radii, groups, group_id):
        if not candidates:
            return

        def get_z(candidate):
            idx, rot, *_ = candidate
            q = rot[3]
            Rmat = R.from_quat(q).as_matrix()
            T = np.eye(4)
            T[:3, :3] = Rmat
            m = self.mesh.copy()
            m.apply_transform(T)
            return m.center_mass[2]

        idx_min = min(candidates, key=get_z)[0]
        idx_max = max(candidates, key=get_z)[0]

        for idx, _, d, o, r in candidates:
            if idx == idx_min or idx == idx_max:
                pose_types[idx] = 3
                dirs[idx] = d
                origins[idx] = o
                radii[idx] = r
                groups[idx] = group_id


    def _assign_aligned_y_pose(self, candidates, pose_types, dirs, origins, radii, groups, group_id):
        if not candidates:
            return

        def get_y(candidate):
            idx, rot, *_ = candidate
            q = rot[3]
            Rmat = R.from_quat(q).as_matrix()
            T = np.eye(4)
            T[:3, :3] = Rmat
            m = self.mesh.copy()
            m.apply_transform(T)
            return m.center_mass[1]

        idx_min = min(candidates, key=get_y)[0]
        idx_max = max(candidates, key=get_y)[0]

        for idx, _, d, o, r in candidates:
            if idx == idx_min or idx == idx_max:
                pose_types[idx] = 2
                dirs[idx] = d
                origins[idx] = o
                radii[idx] = r
                groups[idx] = group_id
