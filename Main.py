from pathlib import Path
from convex_hull_extractor import convex_hull_extractor
import PoseFinder as pf
import PoseVisualizer as pv
#from STLtoOBJConverter import stl_to_obj_converter

# Get the current script's directory
script_dir = Path(__file__).parent

# Get the current script's directory and then add the file path to the folders containing the workpiece stls
workpiece_path =  script_dir / 'Workpieces'

# Get the workpiece name you want to find poses for
workpiece_name = 'Teil_5'

# Convert the STL file to an OBJ file
#stl_to_obj_converter(str(workpiece_path / (workpiece_name + '.STL')), str(workpiece_path / (workpiece_name + '.obj')),0.001)

# Extract the convex hull of the 3D model
convex_hull_extractor(str(workpiece_path / (workpiece_name + '.obj')), str(workpiece_path / (workpiece_name + '_convex_hull.obj')))

# Initialize the PoseFinder with the convex hull OBJ file
pose_finder = pf.PoseFinder(str(workpiece_path / (workpiece_name + '_convex_hull.obj')))

# Find the valid poses of the workpiece
valid_rotations = pose_finder.find_valid_rotations()

# Initialize the PoseVisualizer with the original and convex hull OBJ files and valid rotations
pose_visualizer = pv.PoseVisualizer(str(workpiece_path / (workpiece_name + '.obj')), str(workpiece_path / (workpiece_name + '_convex_hull.obj')), valid_rotations)

# Visualize the valid poses
pose_visualizer.visualize_rotations()