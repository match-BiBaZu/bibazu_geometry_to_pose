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
        self.valid_rotations = valid_rotations

    def plot_mesh(self, ax, mesh, title, rotation=None):
        """
        Plots a 3D mesh with an optional rotation.
        :param ax: Matplotlib 3D axis.
        :param mesh: Trimesh mesh to plot.
        :param title: Title for the plot.
        :param rotation: Quaternion to rotate the mesh before plotting.
        """
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        
        if rotation is not None:
            rot = R.from_quat(rotation)
            vertices = rot.apply(vertices)
        
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, alpha=0.6, edgecolor='k')
        ax.set_title(title)

    def visualize_rotations(self):
        """
        Generates a separate plot for each valid rotation applied to the original and convex hull meshes.
        """
        num_plots = len(self.valid_rotations)
        fig = plt.figure(figsize=(10, num_plots * 5))
        
        for i, (index, quat) in enumerate(self.valid_rotations):
            ax = fig.add_subplot(num_plots, 1, i + 1, projection='3d')
            self.plot_mesh(ax, self.original_mesh, f'Original Model - Rotation {index}', rotation=quat)
            self.plot_mesh(ax, self.convex_hull_mesh, f'Convex Hull - Rotation {index}', rotation=quat)
        
        plt.show()