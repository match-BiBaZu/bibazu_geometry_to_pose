import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from PoseFinder import PoseFinder
from matplotlib.path import Path
import matplotlib.pyplot as plt
import trimesh

class PoseEliminator(PoseFinder):
    def __init__(self, convex_hull_obj_file: str, self_obj_file: str, tolerance: float = 1e-5, rotation_steps: int = 12, wobble_angle: float = 2.0):
        """
        Initialize the PoseEliminator with the convex hull OBJ file.
        :param convex_hull_obj_file: Path to the convex hull OBJ file.
        """
        super().__init__(convex_hull_obj_file, self_obj_file, tolerance)
        
        #how many discrete poses do you want from a rotation around a cylinder axis
        self.rotation_steps = int(rotation_steps)
        #as a mesh is not perfectly aligned to a cylinder axis, we allow a certain wobble angle
        self.wobble_tolerance = 2 * np.sin((wobble_angle * np.pi) / 360)
        # some extra values to twiddle to get the poses resting on 'cone features' working        
        self.cone_plane_band_scale = getattr(self, 'cone_plane_band_scale', 10.0) * self.wobble_tolerance
        self.cone_radius_band_mult = getattr(self, 'cone_radius_band_mult', 25.0) * self.wobble_tolerance


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
        cylinder_axis_parameters,  # list[list[(origin, direction[, radius]) | dict]] (already pose-rotated)
    ):
        """
        Groups by (+/-) axis direction, detects cone-vs-cylinder contact on base/back,
        orders by absolute twist within each subgroup, and downsamples by self.rotation_steps.

        Extras:
        - Cylinder side: only keep non-rolling (axis ~ ±Y/±Z).
        - Cone side: REQUIRE two *different* radii (same-direction axes) such that
            **each radius touches BOTH the base (bottom) band and the back band**.

        Notes:
        - Radius band:  band = 10 * self.tolerance * |r|
        - Downsample:   threshold = max(1, round(L / self.rotation_steps))
        - Spherical mean: explicit None handling
        """
        if not rotations:
            return [], [], []

        # ---------- helpers ----------
        def _group_by_first_dir(params):
            """Return (groups, group_dirs) with +d and −d treated as different."""
            groups = []   # list[(ref_dir or None, [indices])]
            for idx, axes in enumerate(params):
                if not axes:
                    groups.append((None, [idx])); continue
                first = axes[0]
                d = np.asarray(first["direction"] if isinstance(first, dict) else first[1], float)
                n = np.linalg.norm(d)
                if n == 0:
                    groups.append((None, [idx])); continue
                d /= n
                placed = False
                for gdir, idxs in groups:
                    if gdir is None: continue
                    if np.linalg.norm(d - gdir) < self.wobble_tolerance:
                        print(f"Pose {idx} grouped with direction {gdir}.  {np.linalg.norm(d - gdir)} < {self.wobble_tolerance}")
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
                                    (fallback_r[k] if isinstance(fallback_r,(list,tuple)) and k < len(fallback_r) else 0.0)))
                else:
                    o = np.asarray(item[0], float)
                    d = np.asarray(item[1], float)
                    r = float(item[2]) if len(item) >= 3 else (
                        float(fallback_r[k]) if isinstance(fallback_r,(list,tuple)) and k < len(fallback_r) else 0.0)
                n = np.linalg.norm(d)
                if n == 0.0 or r <= 0.0: continue
                out.append((o, d/n, r))
            return out

        def _resting_sets(mesh_in_pose):
            """Vertices near base/back planes, excluding planar contacts."""
            V = mesh_in_pose.vertices
            bbox = mesh_in_pose.bounds
            diag = float(np.linalg.norm(bbox[1] - bbox[0]))
            band_abs = self.tolerance * diag * self.cone_plane_band_scale

            z = V[:,2]; x = V[:,0]
            min_z, min_x = float(np.min(z)), float(np.min(x))
            idx_z = np.where((z - min_z) <= band_abs)[0]
            idx_x = np.where((x - min_x) <= band_abs)[0]
            print(f"Resting sets: {len(idx_z)} base, {len(idx_x)} back vertices within ±{band_abs:.3g} band.")

            # normals
            try:
                N = mesh_in_pose.vertex_normals
                if N is None or len(N) != len(V): raise Exception()
            except Exception:
                Nf = mesh_in_pose.face_normals
                N = np.zeros_like(V)
                for fi, tri in enumerate(mesh_in_pose.faces):
                    N[tri] += Nf[fi]
                L = np.linalg.norm(N, axis=1); ok = L > 0
                N[ok] /= L[ok][:,None]

            g_base = np.array([0.0,0.0,1.0])
            g_back = np.array([1.0,0.0,0.0])

            idx_base = idx_z[np.where(np.abs(N[idx_z] @ g_base) < 0.94)[0]]
            idx_back = idx_x[np.where(np.abs(N[idx_x] @ g_back) < 0.94)[0]]
            return V[idx_base], V[idx_back]

        def _line_dist_perp(p, o, d_hat):  # ⟂ distance point→axis
            return np.linalg.norm(np.cross(p - o, d_hat))

        def _axes_aligned(axes_with_r, n_ref, canonicalize=True):
            """Pick axes whose direction matches n_ref (optionally flip sign to match)."""
            out = []
            for (o, d, r) in axes_with_r:
                d2 = (-d if (canonicalize and np.dot(d, n_ref) < 0.0) else d)
                if np.linalg.norm(d2 - n_ref) < self.wobble_tolerance:
                    out.append((o, d2, r))
            return out

        def _two_on_same_axis(verts, axis):
            """≥2 verts at the radius of a single axis (band = 10 * tolerance * |r|)."""
            (o,d,r) = axis
            band =  10.0 * float(self.tolerance) * abs(r)
            cnt = 0
            for p in verts:
                if abs(_line_dist_perp(p,o,d) - r) <= band:
                    cnt += 1
                    if cnt >= 4: return True
            return False

        def _is_cyl_contact(idx, n_ref):
            """ANY aligned axis with ≥2 verts at its radius (base OR back)."""
            old_id, face_id, edge_id, q = rotations[idx]
            rot = R.from_quat(q); T = np.eye(4); T[:3,:3] = rot.as_matrix()
            m = self.convex_hull_mesh.copy(); m.apply_transform(T)
            baseV, backV = _resting_sets(m)
            axes = _extract_axes_with_radius(cylinder_axis_parameters[idx])
            aligned = _axes_aligned(axes, n_ref)
            if not aligned: 
                return False
            return any(_two_on_same_axis(baseV, ax) or _two_on_same_axis(backV, ax) for ax in aligned)

        # --- NEW helpers for cone contact requiring each radius to hit BOTH planes ---
        def _axis_hit_on_plane(verts, axis):
            """True if ≥1 vertex lies at axis radius within band on this plane's vertex set."""
            (o,d,r) = axis
            band = self.cone_radius_band_mult * float(self.tolerance) * abs(r)
            for p in verts:
                if abs(_line_dist_perp(p,o,d) - r) <= band:
                    return True
            return False

        def _is_cone_contact(idx, n_ref):
            """
            Cone contact if there exist TWO same-direction axes with DIFFERENT radii
            such that *each* of those radii is evidenced on BOTH planes:
            - radius r1 has ≥1 hit on BASE and ≥1 hit on BACK;
            - radius r2 has ≥1 hit on BASE and ≥1 hit on BACK;
            - |r1 - r2| > radius_diff_thresh.
            (Hit points can be the same/different across planes.)
            """
            old_id, face_id, edge_id, q = rotations[idx]
            rot = R.from_quat(q); T = np.eye(4); T[:3,:3] = rot.as_matrix()
            m = self.convex_hull_mesh.copy(); m.apply_transform(T)
            baseV, backV = _resting_sets(m)
            axes = _extract_axes_with_radius(cylinder_axis_parameters[idx])
            aligned = _axes_aligned(axes, n_ref)
            if len(aligned) < 2:
                return False

            # radii that hit BOTH planes
            radii_both = []
            for ax in aligned:
                hit_base = _axis_hit_on_plane(baseV, ax)
                hit_back = _axis_hit_on_plane(backV, ax)
                if hit_base and hit_back:
                    radii_both.append(ax[2])  # r
                    #print(f"Pose {idx} axis with radius {ax[2]} hits both planes.")

            if len(radii_both) < 2:
                return False

            # need two distinct radii (separated more than threshold)
            thresh = 100.0 * float(self.tolerance)
            for i in range(len(radii_both)):
                for j in range(i+1, len(radii_both)):
                    #print(f"Pose {idx} comparing radii {radii_both[i]} and {radii_both[j]}. is thresh {thresh} < {abs(radii_both[i] - radii_both[j])}?")
                    if abs(radii_both[i] - radii_both[j]) > thresh:
                        return True
            return False

        def _fix_quat_sign(q, q_ref):
            q = np.asarray(q,float); q_ref = np.asarray(q_ref,float)
            return q if float(np.dot(q,q_ref)) >= 0.0 else -q

        def _abs_twist_deg_wrt_ref(q_ref, q_i, n_ref):
            """Absolute twist (deg) of pose i about fixed axis n_ref, vs. reference pose q_ref."""
            q_i = _fix_quat_sign(q_i, q_ref)
            q_rel = (R.from_quat(q_i) * R.from_quat(q_ref).inv()).as_quat()  # [x,y,z,w]
            vx,vy,vz,w = q_rel
            vdot = vx*n_ref[0] + vy*n_ref[1] + vz*n_ref[2]
            ang = 2.0*np.arctan2(vdot, w)
            return float(np.degrees(ang) % 360.0)

        def _spherical_mean(directions):
            """Mean of unit vectors with sign canonicalized to first; returns None if empty."""
            if not directions:
                return None
            dirs = np.asarray(directions, float)
            m = dirs[0] / (np.linalg.norm(dirs[0]) or 1.0)
            for _ in range(5):
                aligned = np.where((dirs @ m)[:,None] >= 0.0, dirs, -dirs)
                m_new = aligned.mean(axis=0)
                n = np.linalg.norm(m_new)
                if n == 0.0: break
                m_new /= n
                if np.allclose(m_new, m, atol=1e-6): break
                m = m_new
            return m

        # Non-wobbly cylinder poses: axis ~ pure ±Y or ±Z
        def _pose_has_pure_y_or_z(idx, n_ref):
            axes = _extract_axes_with_radius(cylinder_axis_parameters[idx])
            aligned = _axes_aligned(axes, n_ref)
            if not aligned:
                return False
            
            ex = np.array([1.0, 0.0, 0.0])
            ey = np.array([0.0, 1.0, 0.0])
            ez = np.array([0.0, 0.0, 1.0])
            tol = float(self.wobble_tolerance) * 0.3
            for (_, d, _r) in aligned:
                if min(np.linalg.norm(d - ex), np.linalg.norm(d + ex)) < tol:
                    return True
                if min(np.linalg.norm(d - ey), np.linalg.norm(d + ey)) < tol:
                    return True
                if min(np.linalg.norm(d - ez), np.linalg.norm(d + ez)) < tol:
                    return True
            return False

        # finding best starting point for twist angle 0
        def _twist_about_axis_deg(q, n_ref):
            """Twist (deg) of quaternion q about axis n_ref (no ref pose; identity as ref)."""
            import numpy as np
            from scipy.spatial.transform import Rotation as R
            q = np.asarray(q, float)
            # ensure same hemisphere as identity so w >= 0 (shortest)
            if q[3] < 0: q = -q
            vx, vy, vz, w = q
            vdot = vx*n_ref[0] + vy*n_ref[1] + vz*n_ref[2]
            ang = 2.0*np.arctan2(vdot, w)
            return float(np.degrees(ang) % 360.0)

        def _order_indices_by_identity_start(indices, n_ref, rotations):
            """
            Pick start = pose with minimal |twist about n_ref| wrt identity.
            Return indices ordered from start, then wrap around to one before start (looping).
            """
            import numpy as np
            # twist about axis for each pose
            angles = [ _twist_about_axis_deg(rotations[i][3], n_ref) for i in indices ]
            # measure closeness to 0 or 360
            def wrapdist(a): 
                a = a % 360.0
                return min(a, 360.0 - a)
            start = int(np.argmin([wrapdist(a) for a in angles]))
            ordered = indices[start:] + indices[:start]  # wrap
            return ordered, [angles[indices.index(i)] for i in ordered]
        # ---------- main ----------
        kept_rot, kept_shad, kept_axes = [], [], []
        pose_id = 0

        groups, group_dirs = _group_by_first_dir(cylinder_axis_parameters)
        print(f"Discretizing {len(rotations)} poses into {len(groups)} direction groups.")

        for g, idxs in enumerate(groups):
            n_group = group_dirs[g]
            if n_group is None:
                # no axis: pass through
                for i in idxs:
                    old_id, face_id, edge_id, quat = rotations[i]
                    kept_rot.append((pose_id, face_id, edge_id, quat))
                    kept_shad.append(xy_shadows[i])
                    kept_axes.append(cylinder_axis_parameters[i])
                    pose_id += 1
                continue

            n_group = n_group / (np.linalg.norm(n_group) or 1.0)

            # classify poses in this direction group
            cone_idx, cyl_idx, other_idx = [], [], []
            for i in idxs:
                if _is_cyl_contact(i, n_group):
                    cyl_idx.append(i)
                elif _is_cone_contact(i, n_group):
                    cone_idx.append(i)
                else:
                    other_idx.append(i)
            
            cyl_set  = set(cyl_idx)
            cone_idx = [i for i in cone_idx if i not in cyl_set]

            # --- CONE SIDE: local axis from spherical mean of aligned axes used here ---
            if cone_idx:
                cone_dirs = []
                for i in cone_idx:
                    axes = _extract_axes_with_radius(cylinder_axis_parameters[i])
                    cone_dirs.extend([(-d if np.dot(d, n_group) < 0 else d) for (_,d,_) in _axes_aligned(axes, n_group)])
                n_cone = _spherical_mean(cone_dirs)
                n_cone = n_group if n_cone is None else n_cone
                n_cone = n_cone / (np.linalg.norm(n_cone) or 1.0)

                # pick reference quat (first cone pose)
                q_ref = np.asarray(rotations[cone_idx[0]][3], float)

                # order by absolute twist about n_cone
                # choose start closest to identity-about-axis, then order and wrap
                ordered_cone_idx, cone_angles = _order_indices_by_identity_start(cone_idx, n_cone, rotations)
                # keep ~rotation_steps evenly spaced
                L = len(ordered_cone_idx)
                thr = max(1, int(round(L / float(self.rotation_steps))))
                for k, i in enumerate(ordered_cone_idx):
                    if (k % thr) == 0:
                        print(f"Cone pose kept: {i} at new id {pose_id} (angle {cone_angles[k]:.1f}°)")
                        old_id, face_id, edge_id, quat = rotations[i]
                        kept_rot.append((pose_id, face_id, edge_id, quat))
                        kept_shad.append(xy_shadows[i])
                        kept_axes.append(cylinder_axis_parameters[i])
                        pose_id += 1


            # --- CYLINDER SIDE: only keep non-rolling (axis ~ pure ±Y or ±Z), then segment filter ---
            if cyl_idx:
                cyl_idx_pure = [i for i in cyl_idx if _pose_has_pure_y_or_z(i, n_group)]
                if cyl_idx_pure:
                    # first filter to ±Y/±Z as you already do → cyl_idx_pure
                    ordered_cyl_idx, cyl_angles = _order_indices_by_identity_start(cyl_idx_pure, n_group, rotations)
                    Lc = len(ordered_cyl_idx)
                    thr_c = max(1, int(round(Lc / float(self.rotation_steps))))
                    for k, i in enumerate(ordered_cyl_idx):
                        if (k % thr_c) == 0:
                            print(f"Cylinder pose kept: {i} at new id {pose_id}")
                            old_id, face_id, edge_id, quat = rotations[i]
                            kept_rot.append((pose_id, face_id, edge_id, quat))
                            kept_shad.append(xy_shadows[i])
                            kept_axes.append(cylinder_axis_parameters[i])
                            pose_id += 1

                #else:
                    #print(f"No cylinder poses with axis ~ ±Y/±Z in group {g}; skipping cylinder saves for this group.")

            # --- NON-CYL poses: pass through (preserve order) ---
            for i in other_idx:
                print("Non-cylinder pose kept: {i} at new id {pose_id}")
                old_id, face_id, edge_id, quat = rotations[i]
                kept_rot.append((pose_id, face_id, edge_id, quat))
                kept_shad.append(xy_shadows[i])
                kept_axes.append(cylinder_axis_parameters[i])
                pose_id += 1

        return kept_rot, kept_shad, kept_axes

