import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class PoseVisualizer:
    def __init__(self, original_obj_file: str, convex_hull_obj_file: str, valid_rotations, xy_shadows: list = None):
        """
        Initialize the PoseVisualizer with original and convex hull OBJ files and valid rotations.
        :param original_obj_file: Path to the original OBJ file.
        :param convex_hull_obj_file: Path to the convex hull OBJ file.
        :param valid_rotations: List of tuples (index, quaternion) representing valid rotations.
        """
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
        Generates one plot per unique pose index, with all associated rotations visualized in the same figure.
        """

        if self.xy_shadows and len(self.xy_shadows) == len(self.valid_rotations):
            zipped = list(zip(self.valid_rotations, self.xy_shadows))
        else:
            zipped = [(entry, None) for entry in self.valid_rotations]

        # Group by pose index
        poses_dict = {}
        for (pose_idx, quat), shadow in zipped:
            if pose_idx not in poses_dict:
                poses_dict[pose_idx] = []
            poses_dict[pose_idx].append((quat, shadow))

        for pose_idx, entries in poses_dict.items():
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_axes([0.3, 0.1, 0.65, 0.85], projection='3d')

            for i, (quat, shadow) in enumerate(entries):
                self.plot_mesh(ax, self.original_mesh, f'Pose {pose_idx} - Rot {i+1}', quat=quat)
                if shadow is not None:
                    self.plot_shadow(ax, shadow, title=f'Shadow {pose_idx} - Rot {i+1}')

            legend_text = "\n".join([f"Rot {i+1}: {np.round(q, 4)}" for i, (q, _) in enumerate(entries)])
            fig.text(0.02, 0.5, legend_text, va='center', ha='left', fontsize='small', family='monospace')

            plt.title(f'All rotations for Pose {pose_idx}')
            plt.tight_layout()
            plt.show()
