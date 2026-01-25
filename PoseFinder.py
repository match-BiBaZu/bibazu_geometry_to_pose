import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from pathlib import Path
import csv
from scipy.spatial import cKDTree

class PoseFinder:
    def __init__(self, convex_hull_obj_file: str, obj_file: str, tolerance: float = 1e-5, is_workpiece_centered: int = 0):
        """
        Initialize the PoseFinder with the convex hull OBJ file.
        :param convex_hull_obj_file: Path to the convex hull OBJ file.
        :param obj_file: Path to the OBJ file.
        :param tolerance: Numerical tolerance for grouping rotations.
        :raises ValueError: If the mesh or convex hull meshes are empty or not initialized.
        """
        self.obj_stem = Path(obj_file).stem

        # Initialize a numerical tolerance for grouping
        self.tolerance = tolerance

        self.mesh = trimesh.load_mesh(obj_file)

        self.convex_hull_mesh = trimesh.load_mesh(convex_hull_obj_file)

        # Cylinder data (from CSV); None if not present
        self.cylinder_radius: list[float] | None = None
        self.cyl_axis_origin: list[np.ndarray] | None = None  # world before rotation (model frame)
        self.cyl_axis_dir: list[np.ndarray] | None = None     # unit vector

        # Check to ensure that the mesh and convex hull meshes are not empty
        if self.mesh.vertices is None or len(self.mesh.vertices) == 0:
            raise ValueError("Mesh vertices are not initialized or empty.")
        
        if self.convex_hull_mesh.vertices is None or len(self.convex_hull_mesh.vertices) == 0:
            raise ValueError("Convex hull mesh vertices are not initialized or empty.")

        # fix normals of normal mesh to ensure they point outward
        self.mesh.fix_normals()

        # fix normals of convex hull to ensure they point outward
        self.convex_hull_mesh.fix_normals()

        centroid = self.convex_hull_mesh.centroid

        self.convex_hull_mesh.apply_translation(-centroid)
        self.mesh.apply_translation(-centroid)

        # Ensure the mesh is centered around the centroid of the convex hull if not centered in input file
        if is_workpiece_centered == 0:
            # Only subtract centroid from x and y, keep z unchanged
            self._load_cylinder_from_csv(centroid)
        else:  
            self._load_cylinder_from_csv()

        # Check if at least 3 vertices align with the lowest z-axis (resting face)
        z_min = np.min(self.mesh.vertices[:, 2])
        aligned_vertices = self.mesh.vertices[np.isclose(self.mesh.vertices[:, 2], z_min, atol=self.tolerance)]
        if len(aligned_vertices) < 3:
            raise ValueError("The object is not resting on the xy plane on one of it's faces. Please check original mesh.")
        
        # Check if the convex hull mesh is actually the convex hull of the original mesh	
        if not np.all(np.isin(self.convex_hull_mesh.vertices, self.mesh.vertices)):
            raise ValueError("The convex hull mesh is not a convex hull of the original mesh. Please check the convex hull mesh and the original mesh.")
    
    # ---------- helpers (kept INSIDE PoseFinder) ----------
    def _compute_xy_shadow(self, vertices: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the 2D convex hull shadow (z=0) from a set of 3D vertices.
        Also computes the internal angles at each shadow vertex, measured towards the centroid.

        :param vertices: (N, 3) array of vertex coordinates.
        :return: 
            - (M, 3) array of 3D shadow vertices on the x-y plane (z=0).
            - (M,) array of internal angles (radians), each pointing towards the centroid.
        """
        projected_points = vertices[:, :2]
        hull_2d = ConvexHull(projected_points)
        hull_coords = projected_points[hull_2d.vertices]
        z_min = np.min(vertices[:, 2])
        shadow_vertices = np.hstack([hull_coords, np.full((len(hull_coords), 1), z_min)])  # z = z_min

        # Compute centroid of the shadow polygon
        centroid = np.mean(hull_coords, axis=0)

        # Compute internal angles at each vertex, measured towards the centroid
        angles = []
        n = len(hull_coords)
        for i in range(n):
            prev = hull_coords[i - 1] - hull_coords[i]
            next = hull_coords[(i + 1) % n] - hull_coords[i]
            to_centroid = centroid - hull_coords[i]

            # Normalize vectors
            prev_norm = prev / (np.linalg.norm(prev) + 1e-12)
            next_norm = next / (np.linalg.norm(next) + 1e-12)
            centroid_norm = to_centroid / (np.linalg.norm(to_centroid) + 1e-12)

            # Angle between prev and next (internal angle)
            angle = np.arccos(np.clip(np.dot(prev_norm, next_norm), -1.0, 1.0))

            # Check if centroid is inside the angle (using cross products)
            cross1 = np.cross(prev_norm, centroid_norm)
            cross2 = np.cross(centroid_norm, next_norm)
            # If both cross products have the same sign, centroid is "inside" the angle
            if cross1 * cross2 >= 0:
                angles.append(angle)
            else:
                angles.append(2 * np.pi - angle)

        return shadow_vertices, np.array(angles)

    def _load_cylinder_from_csv(self, centroid: np.ndarray | None = None) -> int:
        """
        Reads cylinder rows from:
        - csv_outputs/{obj_stem}_cylinder_properties.csv (legacy)
        Exports all cylinders into parallel lists:
        self.cylinder_radius: list[float]
        self.cyl_axis_origin: list[np.ndarray(3,)]
        self.cyl_axis_dir:    list[np.ndarray(3,)]
        Returns number of cylinders loaded.
        """
        csv_dir = Path("csv_outputs")
        csv_path = csv_dir / f"{self.obj_stem}_cylinder_properties.csv"
        if not csv_path.is_file():
            return 0

        # initialize destination lists
        self.cylinder_radius = []
        self.cyl_axis_origin = []
        self.cyl_axis_dir = []

        appended = 0
        try:
            with csv_path.open(newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Type", "").strip().lower() != "cylinder":
                        continue

                    def ffloat(key, default=0.0):
                        try:
                            return float(row.get(key, default))
                        except Exception:
                            return float(default)

                    radius = ffloat("Radius")
                    origin = np.array([
                        ffloat("AxisOriginX"), ffloat("AxisOriginY"), ffloat("AxisOriginZ")
                    ], dtype=float)

                    direction = np.array([
                        ffloat("AxisDirX", 0.0), ffloat("AxisDirY", 0.0), ffloat("AxisDirZ", 1.0)
                    ], dtype=float)

                    n = np.linalg.norm(direction)
                    if n > 0:
                        direction = direction / n

                    if centroid is not None:
                        origin = origin - centroid

                    self.cylinder_radius.append(radius)
                    self.cyl_axis_origin.append(origin)
                    self.cyl_axis_dir.append(direction)

                    appended += 1

        except Exception:
            return appended

        return appended


    def _update_axis_pose(
        self,
        quat_xyzw: tuple[float, float, float, float],
        origin: np.ndarray | None = None,
        direction: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Applies quaternion to the given (origin, direction) axis (no translation).
        If origin/direction are None, uses self.cyl_axis_origin / self.cyl_axis_dir (single-axis legacy).
        Returns (rotated_origin, rotated_unit_direction); None if no axis present.
        """
        o_in = self.cyl_axis_origin if origin is None else origin
        d_in = self.cyl_axis_dir    if direction is None else direction
        if o_in is None or d_in is None:
            return None
        r = R.from_quat(quat_xyzw)
        o = r.apply(np.asarray(o_in, float))
        d = r.apply(np.asarray(d_in, float))
        n = np.linalg.norm(d)
        if n > 0:
            d = d / n
        return o, d
    # ---------- end helpers ----------

    def find_candidate_rotations_by_resting_face_normal_alignment(self) -> tuple[list[tuple[int, int, int, tuple[float, float, float, float]]],list[np.ndarray],list[np.ndarray]]:
        """
        Aligns each outward-pointing face normal of the convex hull to the resting face normal (closest to -Z).
        Returns candidate rotations as quaternions and corresponding XY shadows and their angles.
        """

        mesh = self.convex_hull_mesh
        face_normals = mesh.face_normals
        face_centers = mesh.triangles_center
        vertices = mesh.vertices

        # Remove duplicate normals (within tolerance)
        unique_normals = []
        unique_centers = []

        for n in face_normals:
            if not any(np.allclose(n, un, atol=self.tolerance) for un in unique_normals):
                unique_normals.append(n)
                unique_centers.append(face_centers[np.all(face_normals == n, axis=1)][0])

        # Find the face normal closest to -Z
        z_axis = np.array([0, 0, 1])
        resting_face_idx = np.argmin(np.dot(unique_normals, z_axis))
        resting_normal = unique_normals[resting_face_idx]

        # Initialize candidate rotations
        candidate_rotations = [(0, 0, 0, (0.0, 0.0, 0.0, 1.0))]  # Identity rotation as tuple of float
        first_shadow, first_shadow_angles = self._compute_xy_shadow(vertices)
        candidate_shadows = [first_shadow]
        candidate_shadow_angles = [first_shadow_angles]

        candidate_count = 1

        for i, normal in enumerate(unique_normals):
            if i == resting_face_idx:
                continue

            try:
                # Compute rotation that aligns current face normal to the resting normal
                r, _ = R.align_vectors([resting_normal], [normal])
                rotated_normal = r.apply(normal)
                rotated_vertices = r.apply(vertices)
                rotated_face_center = r.apply(unique_centers[i])
                min_z = np.min(rotated_vertices[:, 2])
                quat = r.as_quat()
            except Exception:
                # Handle edge cases (e.g., 180° rotation)
                if np.dot(normal, resting_normal) < -0.9999:
                    ortho = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
                    axis = np.cross(normal, ortho)
                    axis /= np.linalg.norm(axis)
                    quat = R.from_rotvec(np.pi * axis).as_quat()
                else:
                    quat = (0.0, 0.0, 0.0, 1.0)

            rotated_vertices = R.from_quat(quat).apply(vertices)

            # Compute the shadow of the rotated vertices
            shadow, shadow_angles = self._compute_xy_shadow(rotated_vertices)
            candidate_shadows.append(shadow)
            candidate_shadow_angles.append(shadow_angles)

            quat = tuple(np.round(quat, decimals=int(np.log10(1/self.tolerance))))
            quat = tuple(0.0 if abs(x) < self.tolerance else x for x in quat)

            candidate_rotations.append((candidate_count, candidate_count, 0, quat))
            candidate_count += 1

        return candidate_rotations, candidate_shadows, candidate_shadow_angles
    
    def find_candidate_rotations_by_shadow_edge_alignment(self, xy_shadow: np.ndarray = None, xy_shadow_angles: np.ndarray = None) -> tuple[list[tuple[int, int, int, tuple[float, float, float, float]]], list[np.ndarray]]:
        """
        Generates quaternions rotating around the z-axis by cumulatively adding internal angles.
        The starting edge is the one between the leftmost vertex and its next vertex in the CCW-ordered list.

        :param xy_shadow: Tuple of (N, 3) array of 3D shadow points on the x-y plane (z=0) and (N,) array of internal angles.
        :return: List of (index, 0, index, quaternion) and corresponding rotated shadows.
        """
        if xy_shadow is None or xy_shadow_angles is None:
            vertices = self.convex_hull_mesh.vertices
            xy_shadow, xy_shadow_angles = self._compute_xy_shadow(vertices)

        candidate_rotations = []
        candidate_shadows = []

        # Step 1: Find leftmost vertex (min x)
        # Find indices of vertices with minimum x and minimum y
        # Find all indices where x is exactly the minimum x (preserving sign, not just magnitude)
        min_x = np.min(xy_shadow[:, 0])
        leftmost_indices = np.where(xy_shadow[:, 0] == min_x)[0]
        # From these, pick the one with the minimum y value
        bottommost_idx = leftmost_indices[np.argmin(xy_shadow[leftmost_indices, 1])]
        #print(f"Leftmost indices: {leftmost_indices}, Chosen index (min y among leftmost): {bottommost_idx}")

        # Step 2: Define the initial edge from leftmost_idx to (leftmost_idx + 1) % N
        p1 = xy_shadow[bottommost_idx][:2]
        p2 = xy_shadow[(bottommost_idx - 1) % len(xy_shadow)][:2]
        edge_vec = p2 - p1
        edge_dir = edge_vec / np.linalg.norm(edge_vec)
        #print(f"Leftmost vertex: {p1}, Next vertex: {p2}, Edge direction: {edge_dir}")
        
        # Step 3: Compute rotation to align this edge with +Y axis
        y_axis = np.array([0, 1])
        angle = np.arccos(np.clip(np.dot(edge_dir, y_axis), -1.0, 1.0))
        if np.cross(edge_dir, y_axis) < 0:
            angle = -angle

        cumulative_angle = angle
        #print(f"Initial angle to align edge with +Y: {np.degrees(angle)} degrees")

        # Step 4: Apply rotations incrementally using xy_shadow_angles
        n = len(xy_shadow)
        for i in range(n):
            quat = R.from_euler('z', cumulative_angle).as_quat()  # Positive angle for clockwise rotation
            quat = tuple(np.round(quat, decimals=int(np.log10(1/self.tolerance))))
            quat = tuple(0.0 if abs(x) < self.tolerance else x for x in quat)

            #print(f"Rotation quaternion: {quat}")
            rotated_shadow = R.from_quat(quat).apply(xy_shadow)

            quat = tuple(np.round(quat, decimals=int(np.log10(1/self.tolerance))))
            quat = tuple(0.0 if abs(x) < self.tolerance else x for x in quat)

            candidate_rotations.append((i, 0, i, quat))
            candidate_shadows.append(rotated_shadow)

            # Increment angle CCW using precomputed xy_shadow_angles
            idx = (bottommost_idx + i) % n
            cumulative_angle -= np.abs(np.pi - xy_shadow_angles[idx])
            #print(f"Internal Angle: {np.degrees(xy_shadow_angles[idx])} degrees, Cumulative Angle: {np.degrees(cumulative_angle)} degrees, index: {idx}")

        return candidate_rotations, candidate_shadows

    def find_candidate_rotations_by_face_and_shadow_alignment(self) -> tuple[
        list[tuple[int, int, int, tuple[float, float, float, float]]],
        list[tuple],
        list[list[tuple[tuple[float, float, float], tuple[float, float, float]]]]
    ]:
        face_rotations, base_xy_shadows, base_shadow_angles = \
            self.find_candidate_rotations_by_resting_face_normal_alignment()

        combined_rotations: list[tuple[int,int,int,tuple[float,float,float,float]]] = []
        combined_shadows:   list[np.ndarray] = []
        cylinder_axis_params: list[list[tuple[tuple[float,float,float],tuple[float,float,float]]]] = []

        # grab stored global (unrotated) axes lists once
        origins = list(getattr(self, "cyl_axis_origin", []) or [])
        dirs    = list(getattr(self, "cyl_axis_dir", []) or [])
        radii   = list(getattr(self, "cylinder_radius", []) or [])

        candidate_count = 0
        for i, (face_id, _, _, face_quat) in enumerate(face_rotations):
            shadow_rotations, shadow_variants = self.find_candidate_rotations_by_shadow_edge_alignment(
                base_xy_shadows[i], base_shadow_angles[i]
            )

            for j, (edge_id, _, _, shadow_quat) in enumerate(shadow_rotations):
                q_total = (R.from_quat(shadow_quat) * R.from_quat(face_quat)).as_quat()
                q_rounded = tuple(np.round(q_total, decimals=int(np.log10(1/self.tolerance))))
                q_rounded_zero = tuple(0.0 if abs(x) < self.tolerance else x for x in q_rounded)

                combined_rotations.append((candidate_count, face_id, edge_id, q_rounded_zero))
                combined_shadows.append(shadow_variants[j])

                # rotate every cylinder axis with this pose
                if origins and dirs:
                    per_pose_axes: list[tuple[tuple[float,float,float], tuple[float,float,float]]] = []
                    for k, (o0, d0) in enumerate(zip(origins, dirs)):
                        upd = self._update_axis_pose(q_total, origin=o0, direction=d0)
                        if upd is None:
                            continue
                        o_r, d_r = upd
                        # include radius if available as 3rd element (keeps downstream extractors happy)
                        if k < len(radii) and radii[k] is not None and radii[k] > 0:
                            per_pose_axes.append((tuple(o_r.tolist()), tuple(d_r.tolist()), float(radii[k])))
                        else:
                            per_pose_axes.append((tuple(o_r.tolist()), tuple(d_r.tolist())))
                    cylinder_axis_params.append(per_pose_axes)

                candidate_count += 1

        return combined_rotations, combined_shadows, cylinder_axis_params

    
    def symmetry_handler(self, rotations: list[tuple[int, int, int, tuple[float,float,float]]], symmetry_tolerance: float = 0.1) -> list[tuple[int, int, int, tuple[float,float,float]]]:
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
                rotated_vertices -= np.mean(rotated_vertices, axis=0)  # Center the rotated vertices

                for check_index, (_, face_id, edge_id, check_quat) in enumerate(rotations):
                    if assigned_rotations[check_index] == -1:  # Only check unassigned rotations

                        check_rotation = R.from_quat(check_quat)
                        check_rotated_vertices = check_rotation.apply(self.mesh.vertices)
                        check_rotated_vertices -= np.mean(check_rotated_vertices, axis=0)  # Center the rotated vertices

                        tree = cKDTree(rotated_vertices)
                        dists, _ = tree.query(check_rotated_vertices, k=1)
                        print(f"Comparing rotation {index} with rotation {check_index}, max distance: {max(dists)}")

                        if max(dists) < symmetry_tolerance:
                            assigned_rotations[check_index] = assymetric_pose_count

                assymetric_pose_count += 1

        # Return the assigned rotations in the same order as the input
        return [(assigned_rotations[i], rotations[i][1], rotations[i][2], quat) for i, (_, _, _, quat) in enumerate(rotations)]

    def write_candidate_rotations_to_file(self, 
                                          candidate_rotations: list[tuple[int, int, int, tuple[float, float, float, float]]], 
                                          pose_types: list[int], 
                                          cylinder_radius: list[int], 
                                          cylinder_axis_origin: list[list[tuple]], 
                                          cylinder_axis_direction: list[list[tuple]],
                                          cylinder_group_ids: list[list[int]],
                                          output_cylinder_features: bool,
                                          output_file: str):
        """
        Writes the candidate rotations to a file in a readable format.
        :param candidate_rotations: List of candidate rotations.
        :param pose_types: List of pose types corresponding to each rotation.
        :param cylinder_radius: List of cylinder radii for each pose.
        :param cylinder_axis_origin: List of cylinder axis origins (x, y, z) for each pose.
        :param cylinder_axis_direction: List of cylinder axis directions (x, y, z) for each pose.
        :param output_file: Path to the output file.
        """
        with open(output_file, 'w') as f:
            if output_cylinder_features:
                f.write("PoseID,FaceID,EdgeID,PoseType,Cylinder_Radius,CylinderGroupID,AxisOriginX,AxisOriginY,AxisOriginZ,AxisDirX,AxisDirY,AxisDirZ,QuatX,QuatY,QuatZ,QuatW\n")
            else:
                f.write("PoseID,FaceID,EdgeID,PoseType,QuatX,QuatY,QuatZ,QuatW\n")
            for i, rotation in enumerate(candidate_rotations):
                quat = rotation[3]
                #origin = cylinder_axis_origin[i] if i < len(cylinder_axis_origin) else (0.0, 0.0, 0.0)
                #direction = cylinder_axis_direction[i] if i < len(cylinder_axis_direction) else (0.0, 0.0, 1.0)
                origin = cylinder_axis_origin[i]
                direction = cylinder_axis_direction[i]
                if output_cylinder_features:
                    f.write(f"{rotation[0]},{rotation[1]},{rotation[2]},"
                            f"{pose_types[i]},{cylinder_radius[i]},{cylinder_group_ids[i]},"
                            f"{origin[0]},{origin[1]},{origin[2]},"
                            f"{direction[0]},{direction[1]},{direction[2]},"
                            f"{quat[0]},{quat[1]},{quat[2]},{quat[3]}\n")
                else:
                    f.write(f"{rotation[0]},{rotation[1]},{rotation[2]},"
                            f"{pose_types[i]},"
                            f"{quat[0]},{quat[1]},{quat[2]},{quat[3]}\n")

    def write_pose_shadows_to_file(
        self,
        rotations: list[tuple[int, int, int, tuple[float, float, float, float]]],
        shadows: list[np.ndarray] | None = None,
        out_dir: str = "Shadows",
        integer: bool = True,
        decimals: int = 0,
    ) -> list[str]:
        """
        Write the convex-hull XY shadow for each pose to txt files:
        Shadows/<workpieceId>_<poseId>.txt

        File format:
            //Please don't alter its structure or it may become unreadable!
            x1,y1
            x2,y2
            ...

        Args:
            rotations: [(pose_id, face_id, edge_id, quat[x,y,z,w]), ...]
            shadows: optional list of precomputed shadows (same order as rotations).
                    Each is (N,3) on z=const. If None, they are computed here.
            out_dir: output directory.
            integer: if True, write integer coordinates (rounded). If False, write floats.
            decimals: number of decimal places when integer=False (ignored when integer=True).

        Returns:
            List of file paths written.
        """
        import os
        os.makedirs(out_dir, exist_ok=True)

        written = []
        for idx, (pose_id, _face_id, _edge_id, quat) in enumerate(rotations):
            # Use provided shadow if available; otherwise compute from rotated convex hull
            if shadows is not None and idx < len(shadows) and shadows[idx] is not None:
                shadow_xy = np.asarray(shadows[idx], float)[:, :2]
            else:
                rot = R.from_quat(quat)
                verts_rot = rot.apply(self.convex_hull_mesh.vertices)
                shadow, _ = self._compute_xy_shadow(verts_rot)
                shadow_xy = shadow[:, :2]

            # Format coordinates
            if integer:
                xy_out = np.rint(shadow_xy).astype(int)
                def fmt_pair(x, y): return f"{int(x)},{int(y)}"
            else:
                xy_out = np.round(shadow_xy, decimals=decimals)
                def fmt_pair(x, y): return f"{x:.{decimals}f},{y:.{decimals}f}"

            # File path: <stem>_<poseId>.txt
            fname = f"{self.obj_stem}_{pose_id}.txt"
            fpath = os.path.join(out_dir, fname)

            # Write file
            with open(fpath, "w", newline="\n") as f:
                f.write("//This file was created by GripperApplet\n")
                f.write("//Please don't alter its structure or it may become unreadable!\n")
                for x, y in xy_out:
                    f.write(fmt_pair(x, y) + "\n")

            written.append(fpath)

        return written
