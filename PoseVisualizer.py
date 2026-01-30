import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d
import os


class PoseVisualizer:
    def __init__(self, original_obj_file: str = None, convex_hull_obj_file: str = None,
                 valid_rotations: list = None, xy_shadows: list = None,
                 cylinder_axis_params: list | None = None,
                 critical_solid_angle_scores: list | None = None,
                 centroid_solid_angle_scores: list | None = None,
                 stability_scores: list | None = None,
                 font_scale: float = 1.2):
        """
        Initialize the PoseVisualizer.

        cylinder_axis_params accepts:
          1) A single list[dict]: global cylinders for all poses.
             Each dict: {"radius": float, "origin": np.ndarray(3,), "direction": np.ndarray(3,), "entity_id": int|None}
          2) A list[ list[dict] ] parallel to valid_rotations (same length): per-rotation cylinders.

        During plotting, every cylinder axis is rotated with the pose quaternion.
        """
        if any(arg is None for arg in [original_obj_file, convex_hull_obj_file, xy_shadows]):
            raise ValueError("All inputs (original_obj_file, convex_hull_obj_file, xy_shadows) must be provided and not None.")

        self.original_mesh = trimesh.load_mesh(original_obj_file)
        self.convex_hull_mesh = trimesh.load_mesh(convex_hull_obj_file)
        self.xy_shadows = xy_shadows

        self.original_mesh.fix_normals()
        self.convex_hull_mesh.fix_normals()

        centroid = self.convex_hull_mesh.centroid
        self.original_mesh.apply_translation(-centroid)
        self.convex_hull_mesh.apply_translation(-centroid)

        self.valid_rotations = valid_rotations or []

        self.critical_solid_angle_scores = critical_solid_angle_scores 
        self.centroid_solid_angle_scores = centroid_solid_angle_scores
        self.stability_scores = stability_scores

        # --- normalize cylinder input into: rot_idx -> list[dict] -----------------
        self._cyl_axes_by_rot_idx = {}
        if cylinder_axis_params is None:
            # nothing provided
            pass
        else:
            # Case A: a single list of cylinder dicts (global for all rotations)
            is_global_list = (len(cylinder_axis_params) > 0 and isinstance(cylinder_axis_params[0], dict))
            if is_global_list:
                global_cyls = cylinder_axis_params
                for rot_tuple in (valid_rotations or []):
                    rot_idx = rot_tuple[0]  # (pose_id, face_id, edge_id, quat)
                    self._cyl_axes_by_rot_idx[rot_idx] = list(global_cyls)  # shallow copy
            else:
                # Case B: per-rotation lists parallel to valid_rotations
                for rot_tuple, cyl_list in zip((valid_rotations or []), cylinder_axis_params):
                    rot_idx = rot_tuple[0]
                    # accept None or [] gracefully
                    self._cyl_axes_by_rot_idx[rot_idx] = list(cyl_list or [])
        
        # font sizes (scaled)
        self.fs = {
            "tick":      int(9  * font_scale),
            "legend":    int(9 * font_scale),
            "title":     int(9 * font_scale),
            "suptitle":  int(14 * font_scale),
            "coord_lbl": int(11 * font_scale),
        }

    def _rotate_mesh(self, mesh, quat = None):
        """
        Rotates the vertices of the mesh using the given quaternion.
        :param mesh: Trimesh mesh whose vertices are to be rotated.
        :param quat: Quaternion for rotation.
        :return: Rotated vertices and faces of the mesh.
        """
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        
        if quat is not None:
            rotation = R.from_quat(quat)
            vertices = rotation.apply(vertices)
        return vertices, faces

    def _plot_mesh(self, ax, vertices, faces, title):
        """
        Plots a 3D mesh with an optional rotation.
        :param ax: Matplotlib 3D axis.
        :param mesh: Trimesh mesh to plot.
        :param title: Title for the plot.
        :param rotation: Quaternion to rotate the mesh before plotting.
        """
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, color='orange', alpha=0.2, edgecolor='k')
        ax.set_title(title)
        
        # Normalize the axes
        max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(), 
                              vertices[:, 1].max() - vertices[:, 1].min(), 
                              vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0

        mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # bigger tick labels
        ax.tick_params(axis='x', which='both', labelsize=self.fs["tick"])
        ax.tick_params(axis='y', which='both', labelsize=self.fs["tick"])
        ax.tick_params(axis='z', which='both', labelsize=self.fs["tick"])
    
    def _plot_shadow(self, ax, shadow_vertices: np.ndarray, title: str):
        """
        Plot a 2D shadow (z=0) from its ordered vertices using a fan triangulation.
        :param ax: Matplotlib 3D axis.
        :param shadow_vertices: (N, 3) vertices on the x-y plane.
        :param title: Plot title.
        """
        if len(shadow_vertices) < 3:
            return

        faces = [[0, i, i + 1] for i in range(1, len(shadow_vertices) - 1)]
        ax.plot_trisurf(shadow_vertices[:, 0], shadow_vertices[:, 1], shadow_vertices[:, 2],
                        triangles=faces, color='gray', alpha=0.5, edgecolor='k')
        ax.set_title(title, fontsize=self.fs["title"])

    def _add_reference_planes(self, ax, vertices, plane_alpha=0.1):
        """
        Adds two intersecting planes:
        - XY plane (blue): at lowest Z, clipped to axis X/Y limits
        - YZ plane (green): at leftmost X, clipped to axis Y/Z limits
        """

        # Max edge length from convex hull (fallback size)
        hull_edges = self.convex_hull_mesh.edges_unique
        hull_vertices = self.convex_hull_mesh.vertices
        edge_lengths = np.linalg.norm(hull_vertices[hull_edges[:, 0]] - hull_vertices[hull_edges[:, 1]], axis=1)
        max_edge = np.max(edge_lengths)

        # Anchors
        min_z_vertex = vertices[np.argmin(vertices[:, 2])]
        min_x_vertex = vertices[np.argmin(vertices[:, 0])]
        z0 = min_z_vertex[2]
        x0 = min_x_vertex[0]

        # Get axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        # Clip all ranges
        x_range = [max(xlim[0], x0), min(xlim[1], x0 + max_edge)]
        y_range = [max(ylim[0], -max_edge), min(ylim[1], max_edge)]
        z_range = [max(zlim[0], z0), min(zlim[1], z0 + max_edge)]

        # --- XY plane at z = z0 ---
        xy_vertices = np.array([
            [x_range[0], y_range[0], z0],
            [x_range[1], y_range[0], z0],
            [x_range[1], y_range[1], z0],
            [x_range[0], y_range[1], z0],
        ])
        xy_faces = np.array([[0, 1, 2], [0, 2, 3]])
        ax.plot_trisurf(xy_vertices[:, 0], xy_vertices[:, 1], xy_vertices[:, 2],
                        triangles=xy_faces, color='blue', alpha=plane_alpha, edgecolor='k')

        # --- YZ plane at x = x0 ---
        yz_vertices = np.array([
            [x0, y_range[0], z_range[0]],
            [x0, y_range[0], z_range[1]],
            [x0, y_range[1], z_range[1]],
            [x0, y_range[1], z_range[0]],
        ])
        yz_faces = np.array([[0, 1, 2], [0, 2, 3]])
        ax.plot_trisurf(yz_vertices[:, 0], yz_vertices[:, 1], yz_vertices[:, 2],
                        triangles=yz_faces, color='green', alpha=plane_alpha, edgecolor='k')


    def _add_plot_legend(self, ax):
        """
        Adds a custom legend to the 3D plot showing labels for all components.
        """
        legend_elements = [
            Patch(facecolor='blue', edgecolor='k', label='Slide Edge XY'),
            Patch(facecolor='green', edgecolor='k', label='Slide Edge YZ'),
            Patch(facecolor='gray', edgecolor='k', label='Convex Hull Shadow'),
            Patch(facecolor='orange', edgecolor='k', label='Workpiece'),  # assuming mesh color is orange
            Line2D([0],[0], color='m', lw=2, label='Cylinder Axis'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=self.fs["legend"])

    def _plot_centroid(self, ax, vertices: np.ndarray, faces: np.ndarray, title: str = None):
        """
        Plots the centroid of the rotated mesh vertices as a red dot.
        """
        # Plot the volume centroid (center of mass) of the mesh
        centroid = trimesh.Trimesh(vertices=vertices, faces=faces).center_mass
        ax.scatter([centroid[0]], [centroid[1]], [centroid[2]],
                   color='red', s=60, marker='o', edgecolors='black', linewidths=0.5, zorder=100)

    def _save_pose_figure(self, fig, workpiece_name: str, face_id: int):
        """
        Saves the figure to the 'Poses_Found' folder with a standardized filename and closes it.
        Does nothing if the figure is None or has no axes.
        """
        if fig is None or not fig.axes:
            print(f"Skipped saving: no content in figure for face {face_id}")
            return

        output_dir = "Poses_Found"
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{workpiece_name}_poses_on_face_{face_id}.png"
        path = os.path.join(output_dir, filename)

        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path}")

    def _add_coordinate_axes(self, ax, scale=0.10, offset_ratio=0.04, zorder=10):
            """
            Draw XYZ arrows with length = *scale* × max(x‑range, y‑range, z‑range).
            """
            # Current limits after you've called set_xlim / set_ylim / set_zlim
            x0, x1 = ax.get_xlim3d()
            y0, y1 = ax.get_ylim3d()
            z0, z1 = ax.get_zlim3d()

            max_range = max(x1 - x0, y1 - y0, z1 - z0)
            length    = scale * max_range          # arrow length

            origin = np.array([x0, y0, z0]) + offset_ratio * np.array([x1 - x0,
                                                                    y1 - y0,
                                                                    z1 - z0])

            # Three arrows
            ax.quiver(*origin, length, 0,      0,      color='r', linewidth=1.2, zorder=zorder)
            ax.quiver(*origin, 0,      length, 0,      color='g', linewidth=1.2, zorder=zorder)
            ax.quiver(*origin, 0,      0,      length, color='b', linewidth=1.2, zorder=zorder)

            # Labels
            ax.text(*(origin + [length, 0, 0]), 'X', color='r', fontsize=self.fs["coord_lbl"],
                    va='bottom', ha='left', zorder=zorder+1)
            ax.text(*(origin + [0, length, 0]), 'Y', color='g', fontsize=self.fs["coord_lbl"],
                    va='bottom', ha='left', zorder=zorder+1)
            ax.text(*(origin + [0, 0, length]), 'Z', color='b', fontsize=self.fs["coord_lbl"],
                    va='bottom', ha='left', zorder=zorder+1)
            
    def _add_cylinder_axis(self, ax, cylinders, scale=1.0, zorder=12):
        """
        Draw one or many cylinder axes.

        cylinders can be:
          - None
          - a single tuple ((ox,oy,oz),(dx,dy,dz))
          - a dict: {"origin": np.array(3), "direction": np.array(3), ...}
          - a list of either tuples or dicts (any mix)

        Segment length = scale * max_range to match coordinate axes sizing.
        """
        if cylinders is None:
            return

        # Normalize to a list of (origin, direction) pairs
        def as_pair(item):
            if item is None:
                return None
            if isinstance(item, dict):
                o = np.asarray(item.get("origin"), dtype=float)
                d = np.asarray(item.get("direction"), dtype=float)
                return (o, d)
            if isinstance(item, (tuple, list)) and len(item) == 2:
                o = np.asarray(item[0], dtype=float)
                d = np.asarray(item[1], dtype=float)
                return (o, d)
            return None

        if not isinstance(cylinders, (list, tuple)):
            cylinders = [cylinders]

        pairs = []
        for it in cylinders:
            p = as_pair(it)
            if p is not None:
                pairs.append(p)
        if not pairs:
            return

        x0, x1 = ax.get_xlim3d()
        y0, y1 = ax.get_ylim3d()
        z0, z1 = ax.get_zlim3d()
        max_range = max(x1 - x0, y1 - y0, z1 - z0)
        length = scale * max_range

        for (o, d) in pairs:
            n = np.linalg.norm(d)
            if n == 0:
                continue
            d = d / n
            p0 = o - 0.5 * length * d
            p1 = o + 0.5 * length * d
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], linewidth=2.0, color='m', zorder=zorder)
            ax.scatter([o[0]], [o[1]], [o[2]], color='m', s=18, zorder=zorder+1)
        # --- helpers to add inside the class -------------------------------------
        
    def _rotate_axes_for_pose(self, cyl_list, quat):
        """Rotate a list of cylinder axes (origin, direction) by pose quaternion into world frame."""
        if not cyl_list:
            return []
        rot = R.from_quat(quat)
        out = []
        for c in cyl_list:
            if isinstance(c, dict):
                o = np.asarray(c["origin"], float)
                d = np.asarray(c["direction"], float)
            else:
                o = np.asarray(c[0], float)
                d = np.asarray(c[1], float)
            o_r = rot.apply(o)
            d_r = rot.apply(d)
            n = np.linalg.norm(d_r) or 1.0
            out.append((o_r, d_r / n))
        return out

    def _order_indices_by_identity_start(self, indices, axis_dir, rotations):
        """
        For given pose indices and a fixed axis_dir (unit), pick the start pose that has
        minimal twist about axis (closest to identity), then return wrapped order.
        """
        def twist_deg(q, n_ref):
            q = np.asarray(q, float)
            if q[3] < 0:  # same hemisphere as identity
                q = -q
            vx, vy, vz, w = q
            vdot = vx*n_ref[0] + vy*n_ref[1] + vz*n_ref[2]
            ang = 2.0*np.arctan2(vdot, w)
            a = float(np.degrees(ang) % 360.0)
            # distance to 0/360
            return min(a, 360.0 - a)

        scores = [twist_deg(rotations[i][3], axis_dir) for i in indices]
        start = int(np.argmin(scores))
        return indices[start:] + indices[:start]

    def _add_score_box(self, ax, rot_idx):
        """Adds a box with score values to the top-right corner of a plot."""
        if not hasattr(self, "centroid_solid_angle_scores") or not hasattr(self, "critical_solid_angle_scores") or not hasattr(self, "stability_scores"):
            return

        # take the scores at the same index the rot idx appears as pose number first
        indices = [i for i, (r_idx, _, _, _) in enumerate(self.valid_rotations) if r_idx == rot_idx]
        # Only draw one box per symmetric pose group
        critical = self.critical_solid_angle_scores[indices[0]]
        centroid = self.centroid_solid_angle_scores[indices[0]]
        stable = self.stability_scores[indices[0]]

        text = f"CSA: {centroid:.2f}%\n"
        text += f"CSRA: {critical:.2f}%\n"
        text += f"Stability: {stable:.2f}%"


        ax.text2D(0.98, 0.98, text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8))

    def visualize_rotations(self, workpiece_name: str = None):
        """
        Visualizes all rotations grouped by face_id.
        For each pose, draws every cylinder axis (if any) after rotating origin+direction with the pose quaternion.
        """
        zipped = list(zip(self.valid_rotations, self.xy_shadows))
        entries = [(rot_idx, face_id, quat, shadow) for (rot_idx, face_id, edge_id, quat), shadow in zipped]

        face_to_entries = {}
        for rot_idx, face_id, quat, shadow in entries:
            face_to_entries.setdefault(face_id, []).append((rot_idx, quat, shadow))

        rot_idx_groups = {}
        for rot_idx, face_id, quat, shadow in entries:
            rot_idx_groups.setdefault(rot_idx, []).append((face_id, quat, shadow))

        plotted_rot_idxs = set()
        screen_inch_w, screen_inch_h = 16, 9  # fallback

        for face_id, entry_list in face_to_entries.items():
            rot_ids_in_face = sorted({rot_idx for rot_idx, _, _ in entry_list if rot_idx not in plotted_rot_idxs})
            if not rot_ids_in_face:
                continue

            n = len(rot_ids_in_face)
            cols = min(n, 4)
            rows = int(np.ceil(n / cols))

            fig, axes = plt.subplots(rows, cols, figsize=(screen_inch_w, screen_inch_h),
                                     subplot_kw={'projection': '3d'})
            axes = np.array(axes).reshape(-1) if n > 1 else [axes]

            for i, rot_idx in enumerate(rot_ids_in_face):
                ax = axes[i]
                legend_lines = [f"Pose {rot_idx}:"]

                # get cylinders for this pose (list[dict] or empty)
                cyl_list = self._cyl_axes_by_rot_idx.get(rot_idx, [])

                for face_id_i, quat, shadow in rot_idx_groups[rot_idx]:
                    vertices, faces = self._rotate_mesh(self.original_mesh, quat)
                    self._plot_mesh(ax, vertices, faces, None)
                    self._add_reference_planes(ax, vertices, plane_alpha=0.1)
                    if shadow is not None:
                        self._plot_shadow(ax, shadow, title=None)
                    self._plot_centroid(ax, vertices, faces, title=None)
                    self._add_coordinate_axes(ax)

                    # rotate cylinders into this pose
                    #if cyl_list:
                    #    self._add_cylinder_axis(ax, cyl_list, scale=0.10)
                    if cyl_list:
                        rot = R.from_quat(quat)
                        rotated_cyls = []
                        unrotated_cyls = []
                        for c in cyl_list:
                            # tolerate tuple input as well
                            if isinstance(c, dict):
                                o = np.asarray(c["origin"], dtype=float)
                                d = np.asarray(c["direction"], dtype=float)
                            else:
                                o = np.asarray(c[0], dtype=float)
                                d = np.asarray(c[1], dtype=float)
                            o_r = rot.apply(o)
                            d_r = rot.apply(d)
                            rotated_cyls.append((o_r, d_r))
                            unrotated_cyls.append((o, d))

                        self._add_cylinder_axis(ax, unrotated_cyls, scale=0.2, zorder=100)  # unrotated, for reference

                    legend_lines.append(f"Resting Face {face_id_i}")
                    legend_lines.append(f"Quaternion {np.round(quat, 4)}") # [x,y,z,w]
                    self._add_score_box(ax, rot_idx) # <- Add score box
                    print('Added score box for pose', rot_idx)

                ax.set_title("\n".join(legend_lines), fontsize=self.fs["title"])
                fig.subplots_adjust(hspace=0.5)
                self._add_plot_legend(ax)
                plotted_rot_idxs.add(rot_idx)

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            #fig.suptitle(f"Unique and Symmetric Resting Poses of {workpiece_name} on Face {face_id}", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            self._save_pose_figure(fig, workpiece_name, face_id)









