import os
import glob
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Parameters
path_to_datasets = r"C:/Users/Shieff/Documents/Dasha/BiBaZu/01_Arbeitspakete/AP2 - Simulation/Bestimmung stabiler Bauteillagen/Possible_Pose_Finder/bibazu_geometry_to_pose/Workpieces/"

# Find all OBJ files
files = [f.replace("\\", "/") for f in glob.glob(os.path.join(path_to_datasets, "**/*.obj"), recursive=True)]

# Loop through models
for idx, file in enumerate(files):
    model_name = os.path.splitext(os.path.basename(file))[0]

    # Load 3D model
    try:
        mesh = trimesh.load(file, force="mesh")
    except Exception as e:
        print(f"Error loading {file}: {e}")
        continue  # Skip invalid files

    # Compute bounding box and centroid
    bounding_box = mesh.bounding_box.bounds
    centroid = mesh.centroid
    max_diameter = np.max(np.linalg.norm(mesh.vertices - centroid, axis=1))
    threshold = max_diameter * 0.02

    # Ask for symmetry type
    print(f"{idx+1} / {len(files)} - {model_name}")
    symmetry_type = input("Choose symmetry type: [Rn, b, r, ru, nvidia, n (none)]: ").strip().lower()
    if symmetry_type == "n":
        continue  # Skip this model

    # Initialize symmetry variables
    poses = []
    axis = np.array([0, 0, 1])  # Default axis (Z-axis)
    center = centroid.tolist()
    symmetric = False
    offset = np.array([0, 0, 0])

    # Function to normalize axis vectors safely
    def normalize(vector):
        norm = np.linalg.norm(vector)
        return vector / norm if norm != 0 else vector  # Avoid division by zero

    # Compute symmetry
    if symmetry_type == "Rn":
        # **n-fold discrete rotational symmetry**
        scores = []
        for n in range(2, 9):
            rotation = R.from_euler("z", 360.0 / n, degrees=True).as_matrix()
            scores.append(np.linalg.norm(rotation))

        best_n = np.argmax(scores) + 2
        print(f"Best N for {model_name}: {best_n}")

        rotation = R.from_euler("z", 360.0 / best_n, degrees=True).as_matrix()
        poses.append(rotation.tolist())

    elif symmetry_type == "b":
        # **Box-type symmetry**
        if "ycbv" in model_name and "0007" in model_name:
            tmp_axis = np.array([1, 0.6, 0])
            tmp_axis = normalize(tmp_axis)  # Normalize axis
            rotation = R.from_rotvec(np.pi * tmp_axis).as_matrix()
            poses.append(rotation.tolist())
        else:
            poses.append(R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix().tolist())
            poses.append(R.from_euler("xyz", [0, 180, 0], degrees=True).as_matrix().tolist())

    elif symmetry_type in ["r", "ru"]:
        # **Rotation symmetry with or without upside-down symmetry**
        if "ycbv" in model_name and "0018" in model_name:
            axis = np.array([0, 1, 0])
        else:
            axis = np.array([0, 0, 1])
        axis = normalize(axis)  # Normalize axis
        rotation = R.from_rotvec(axis * np.deg2rad(10)).as_matrix()
        poses.append(rotation.tolist())
        symmetric = True

        if symmetry_type == "ru":
            axis2 = normalize(1 - axis)
            axis2[np.argmax(axis2)] = 0  # Ensure only one axis is flipped
            flip_rotation = R.from_rotvec(axis2 * np.deg2rad(180)).as_matrix()
            poses.append(flip_rotation.tolist())

    elif symmetry_type == "nvidia":
        # **NVIDIA models**
        moments = np.mean(mesh.vertices, axis=0)
        center = moments * (1 - axis)
        offset = center
        rotation = R.from_rotvec(axis * np.deg2rad(10)).as_matrix()
        poses.append(rotation.tolist())

    # Visualization
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Get vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces

    # Plot the 3D model
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, alpha=0.4, color="gray")

    # Draw symmetry axis
    if np.any(axis):
        start_point = centroid - axis * 0.5
        end_point = centroid + axis * 0.5
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]],
                color="red", linewidth=2, label="Symmetry Axis")

    # Plot rotation poses (if available)
    for pose in poses:
        rotation_matrix = np.array(pose)
        rotated_point = np.dot(rotation_matrix, np.array([0.3, 0, 0])) + centroid
        ax.quiver(centroid[0], centroid[1], centroid[2],
                  rotated_point[0] - centroid[0],
                  rotated_point[1] - centroid[1],
                  rotated_point[2] - centroid[2],
                  color="blue", arrow_length_ratio=0.2, label="Rotation")

    # Normalize axes
    max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                          vertices[:, 1].max() - vertices[:, 1].min(),
                          vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0

    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Display plot
    ax.set_title(f"Symmetry Visualization: {model_name}")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.legend()
    plt.show()
