import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.patches import Patch
import tkinter as tk
import os

class PoseVisualizer:
    def __init__(self, original_obj_file: str = None, convex_hull_obj_file: str = None, valid_rotations: list = None, xy_shadows: list = None):
        """
        Initialize the PoseVisualizer with original and convex hull OBJ files and valid rotations.
        :param original_obj_file: Path to the original OBJ file.
        :param convex_hull_obj_file: Path to the convex hull OBJ file.
        :param valid_rotations: List of tuples (index, quaternion) representing valid rotations.
        """
        if any(arg is None for arg in [original_obj_file, convex_hull_obj_file, xy_shadows]):
            raise ValueError("All inputs (original_obj_file, convex_hull_obj_file, xy_shadows) must be provided and not None.")
        
        self.original_mesh = trimesh.load_mesh(original_obj_file)
        self.convex_hull_mesh = trimesh.load_mesh(convex_hull_obj_file)
        self.xy_shadows = xy_shadows

        # fix normals of normal mesh to ensure they point outward
        self.original_mesh.fix_normals()

        # fix normals of convex hull to ensure they point outward
        self.convex_hull_mesh.fix_normals()

        # Ensure the meshes are centered around the centroid of the convex hull
        centroid = self.convex_hull_mesh.centroid
        self.original_mesh.apply_translation(-centroid)
        self.convex_hull_mesh.apply_translation(-centroid)

        # Store valid rotations
        self.valid_rotations = valid_rotations

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
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, color='orange', alpha=0.6, edgecolor='k')
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
        ax.set_title(title)

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
            Patch(facecolor='orange', edgecolor='k', label='Workpiece')  # assuming mesh color is orange
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize='x-small')

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

    def visualize_rotations(self, workpiece_name: str = None):
        """
        Visualizes all rotations grouped by face_id.
        Subplots are laid out to fit the screen size and use maximum space.
        Each rot_idx is only plotted once, in the first face it appears in.
        """

        zipped = list(zip(self.valid_rotations, self.xy_shadows))
        entries = [(rot_idx, face_id, quat, shadow) for (rot_idx, face_id, edge_id, quat), shadow in zipped]

        # face_id → list of (rot_idx, quat, shadow)
        face_to_entries = {}
        for rot_idx, face_id, quat, shadow in entries:
            face_to_entries.setdefault(face_id, []).append((rot_idx, quat, shadow))

        # rot_idx → list of (face_id, quat, shadow)
        rot_idx_groups = {}
        for rot_idx, face_id, quat, shadow in entries:
            rot_idx_groups.setdefault(rot_idx, []).append((face_id, quat, shadow))

        plotted_rot_idxs = set()

        # Try to get screen resolution in inches
        try:
            root = tk.Tk()
            root.withdraw()  # hide the window but keep the application alive
            screen_px_w = root.winfo_screenwidth()
            screen_px_h = root.winfo_screenheight()
            dpi = plt.rcParams['figure.dpi']
            screen_inch_w = screen_px_w / dpi
            screen_inch_h = screen_px_h / dpi
        except:
            screen_inch_w = 16  # fallback
            screen_inch_h = 9

        for face_id, entry_list in face_to_entries.items():
            rot_ids_in_face = sorted({rot_idx for rot_idx, _, _ in entry_list if rot_idx not in plotted_rot_idxs})
            if not rot_ids_in_face:
                continue

            n = len(rot_ids_in_face)
            cols = min(n, 4)
            rows = int(np.ceil(n / cols))

            # Adjust full figure size to fill screen while maintaining aspect ratio
            subplot_w = screen_inch_w / cols
            subplot_h = screen_inch_h / rows
            fig_w = subplot_w * cols
            fig_h = subplot_h * rows

            fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), subplot_kw={'projection': '3d'})
            axes = np.array(axes).reshape(-1) if n > 1 else [axes]

            for i, rot_idx in enumerate(rot_ids_in_face):
                ax = axes[i]
                legend_lines = [f"Pose {rot_idx}:"]
                for face_id_i, quat, shadow in rot_idx_groups[rot_idx]:
                    vertices, faces = self._rotate_mesh(self.original_mesh, quat)
                    self._plot_mesh(ax, vertices, faces, None)
                    self._add_reference_planes(ax, vertices, plane_alpha=0.1)
                    if shadow is not None:
                        self._plot_shadow(ax, shadow, title=None)
                    legend_lines.append(f"Resting Face {face_id_i}")
                    legend_lines.append(f"Quaternion [x,y,z,w] {np.round(quat, 4)}")

                # Move text inside plot area (top-left corner, just below the box edge)
                title_text = "\n".join(legend_lines)
                ax.set_title(title_text, fontsize=8)  # pad in points
                fig.subplots_adjust(hspace=0.5)    
                self._add_plot_legend(ax)
                plotted_rot_idxs.add(rot_idx)

            # Hide unused axes
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            fig.suptitle(f"Unique and Symmetric Resting Poses of {workpiece_name} on Face {face_id}", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            #plt.show()
            self._save_pose_figure(fig, workpiece_name, face_id)









