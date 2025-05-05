import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class PoseVisualizer:
    def __init__(self, original_obj_file: str, convex_hull_obj_file: str, valid_rotations, xy_shadows: list = None, candidate_ids: list = None):
        """
        Initialize the PoseVisualizer with original and convex hull OBJ files and valid rotations.
        :param original_obj_file: Path to the original OBJ file.
        :param convex_hull_obj_file: Path to the convex hull OBJ file.
        :param valid_rotations: List of tuples (index, quaternion) representing valid rotations.
        """
        self.original_mesh = trimesh.load_mesh(original_obj_file)
        self.convex_hull_mesh = trimesh.load_mesh(convex_hull_obj_file)
        self.xy_shadows = xy_shadows
        self.candidate_ids = candidate_ids

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
        Visualizes all rotations grouped by candidate_id (from self.candidate_ids).
        If a candidate_id appears more than once (i.e., symmetry), highlight in red.
        """

        if self.xy_shadows and len(self.xy_shadows) == len(self.valid_rotations):
            zipped = list(zip(self.valid_rotations, self.xy_shadows, self.candidate_ids))
        else:
            zipped = [(entry, None, cid) for entry, cid in zip(self.valid_rotations, self.candidate_ids)]

        # Detect symmetric candidate_ids manually
        seen = set()
        symmetric_ids = set()
        for (rot_idx, quat) in self.valid_rotations:
            if rot_idx in seen:
                symmetric_ids.add(rot_idx)
            else:
                seen.add(rot_idx)

        # Group entries by candidate_id
        grouped = {}
        for (rot_idx, quat), shadow, (face_id, shadow_id) in zipped:
            if face_id not in grouped:
                grouped[face_id] = []
            grouped[face_id].append((rot_idx, quat, shadow))

        for face_id, entries in grouped.items():
            is_symmetric = face_id in symmetric_ids

            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_axes([0.3, 0.1, 0.65, 0.85], projection='3d')

            legend_text = []
            for i, (rot_idx, quat, shadow) in enumerate(entries):
                self.plot_mesh(ax, self.original_mesh, None, quat=quat)
                if shadow is not None:
                    self.plot_shadow(ax, shadow, None)
                legend_text.append(f"Rot {rot_idx+1}: {np.round(quat, 4)}")

            label_color = 'red' if is_symmetric else 'black'
            fig.text(0.02, 0.5, "\n".join(legend_text), va='center', ha='left',
                    fontsize='small', family='monospace', color=label_color)
            fig.suptitle(f'All valid rotations on natural resting face {face_id}', color=label_color)
            plt.tight_layout()
            plt.show()



