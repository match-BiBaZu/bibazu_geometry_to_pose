import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

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

    def visualize_rotations(self):
        """
        Visualizes all rotations grouped by rot_idx and face_id.
        Rotations with the same face_id or same rot_idx are shown in the same figure.
        """

        zipped = list(zip(self.valid_rotations, self.xy_shadows))

        # Build full entry list: each entry = (rot_idx, quat, shadow, face_id)
        entries = [(rot_idx, quat, shadow, face_id) for (rot_idx, face_id, edge_id, quat), shadow, in zipped]

        # Group entries by combined (rot_idx OR face_id)
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
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_axes([0.3, 0.1, 0.65, 0.85], projection='3d')

            # Group legend entries by rot_idx
            rot_groups = {}
            for rot_idx, quat, shadow, face_id in group:
                if rot_idx not in rot_groups:
                    rot_groups[rot_idx] = []
                rot_groups[rot_idx].append((quat, shadow, face_id))

            for rot_idx, entries in rot_groups.items():
                for quat, shadow, _ in entries:
                    self.plot_mesh(ax, self.original_mesh, None, quat=quat)
                    if shadow is not None:
                        self.plot_shadow(ax, shadow, None)

            # Build grouped legend text by rot_idx
            legend_lines = []
            for rot_idx, entries in rot_groups.items():
                legend_lines.append(f"Pose Number {rot_idx}:")
                for quat, _, face_id in entries:
                    legend_lines.append(f"↪ On Resting Face {face_id} Rotated by {np.round(quat, 4)}")

            fig.text(0.02, 0.5, "\n".join(legend_lines), va='center', ha='left',
                    fontsize='small', family='monospace', color='black')

            fig.suptitle('Grouped by Resting Face and Symmetry', color='black')
            plt.tight_layout()
            plt.show()





