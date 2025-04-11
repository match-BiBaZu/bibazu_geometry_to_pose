import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class PoseVisualizer:
    def __init__(self, original_obj_file: str, convex_hull_obj_file: str, valid_rotations):
        """
        Initialize the PoseVisualizer with original and convex hull OBJ files and valid rotations.
        :param original_obj_file: Path to the original OBJ file.
        :param convex_hull_obj_file: Path to the convex hull OBJ file.
        :param valid_rotations: List of tuples (index, quaternion) representing valid rotations.
        """
        self.original_mesh = trimesh.load_mesh(original_obj_file)
        self.convex_hull_mesh = trimesh.load_mesh(convex_hull_obj_file)

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

    def visualize_rotations(self):
        """
        Generates a separate plot for each valid rotation applied to the original and convex hull meshes.
        """
        for index, rot in self.valid_rotations:
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111, projection='3d')
            self.plot_mesh(ax, self.original_mesh, f'Original Model - Rotation {index}', quat=rot)
            plt.show()
