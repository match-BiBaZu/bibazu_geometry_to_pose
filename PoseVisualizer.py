import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.patches import Patch

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

        # Ensure the meshes are centered around the centroid of the convex hull
        centroid = self.convex_hull_mesh.centroid
        self.original_mesh.apply_translation(-centroid)
        self.convex_hull_mesh.apply_translation(-centroid)

        # Store valid rotations
        self.valid_rotations = valid_rotations

    def plot_mesh(self, ax, mesh, title, quat=None):
        """
        Plots a 3D mesh with an optional rotation.
        :param ax: Matplotlib 3D axis.
        :param mesh: Trimesh mesh to plot.
        :param title: Title for the plot.
        :param rotation: Quaternion to rotate the mesh before plotting.
        """
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        
        if quat is not None:
            rotation = R.from_quat(quat)
            vertices = rotation.apply(vertices)
        
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, alpha=0.6, edgecolor='k')
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
    
    def plot_shadow(self, ax, shadow_vertices: np.ndarray, title: str):
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

    def add_reference_planes(self, ax, mesh, plane_alpha=0.1):
        """
        Adds two reference planes:
        - XY plane aligned with the bottom (min Z) and left-most (min X) vertex of the mesh
        - YZ plane aligned with the left-most vertex and touching the XY plane
        Plane size is based on the largest edge of the convex hull mesh.
        """

        # Get largest convex hull edge
        hull_edges = self.convex_hull_mesh.edges_unique
        hull_vertices = self.convex_hull_mesh.vertices
        edge_lengths = np.linalg.norm(hull_vertices[hull_edges[:, 0]] - hull_vertices[hull_edges[:, 1]], axis=1)
        max_edge = np.max(edge_lengths)
        half = max_edge / 2

        # Get key intersection anchors
        min_z_vertex = mesh.vertices[np.argmin(mesh.vertices[:, 2])]
        min_x_vertex = mesh.vertices[np.argmin(mesh.vertices[:, 0])]
        z_intercept = min_z_vertex[2]
        x_intercept = min_x_vertex[0]

        # --- XY Plane: aligned with min_z and shifted so left edge touches min_x ---
        xy_vertices = np.array([
            [x_intercept,        -half, z_intercept],
            [x_intercept + max_edge, -half, z_intercept],
            [x_intercept + max_edge,  half, z_intercept],
            [x_intercept,         half, z_intercept],
        ])
        xy_faces = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ])
        ax.plot_trisurf(xy_vertices[:, 0], xy_vertices[:, 1], xy_vertices[:, 2],
                        triangles=xy_faces, color='blue', alpha=plane_alpha, edgecolor='k')

        # --- YZ Plane: aligned with min_x and bottom edge at z = min_z ---
        yz_vertices = np.array([
            [x_intercept, -half, z_intercept],
            [x_intercept, -half, z_intercept + max_edge],
            [x_intercept,  half, z_intercept + max_edge],
            [x_intercept,  half, z_intercept],
        ])
        yz_faces = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ])
        ax.plot_trisurf(yz_vertices[:, 0], yz_vertices[:, 1], yz_vertices[:, 2],
                        triangles=yz_faces, color='green', alpha=plane_alpha, edgecolor='k')

    def add_plot_legend(self, ax):
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


    def visualize_rotations(self):
        """
        Visualizes all rotations grouped by rot_idx and face_id.
        Each rotation is shown in its own subplot with a detailed title.
        """

        zipped = list(zip(self.valid_rotations, self.xy_shadows))
        entries = [(rot_idx, quat, shadow, face_id) for (rot_idx, face_id, edge_id, quat), shadow in zipped]

        # Group by rot_idx or face_id
        grouped = []
        used = set()
        for i, (rot_idx_i, _, _, face_id_i) in enumerate(entries):
            if i in used:
                continue
            group = []
            for j, (rot_idx_j, _, _, face_id_j) in enumerate(entries):
                if j in used:
                    continue
                if rot_idx_j == rot_idx_i or face_id_j == face_id_i:
                    group.append(entries[j])
                    used.add(j)
            grouped.append(group)

        for group in grouped:
            n = len(group)
            cols = min(n, 4)
            rows = int(np.ceil(n / cols))

            fig = plt.figure(figsize=(4 * cols, 4 * rows))

            for idx, (rot_idx, quat, shadow, face_id) in enumerate(group):
                ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
                self.plot_mesh(ax, self.original_mesh, None, quat=quat)
                self.add_reference_planes(ax, self.original_mesh)
                if shadow is not None:
                    self.plot_shadow(ax, shadow, title=None)
                
                # Add legend
                self.add_plot_legend(ax)

                # Detailed legend as subplot title
                title_lines = [
                    f"Pose Number {rot_idx}",
                    f"↪ On Resting Face {face_id}",
                    f"↪ Quaternion {np.round(quat, 4)}"
                ]
                ax.set_title("\n".join(title_lines), fontsize=8)

            fig.suptitle('Grouped by Resting Face and Symmetry', fontsize=14)
            plt.tight_layout()
            plt.show()





