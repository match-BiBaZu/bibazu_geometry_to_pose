from pathlib import Path
from obj_convex_hull_extractor import obj_convex_hull_extractor
from obj_to_ply_converter import obj_to_ply_converter
import PoseFinder as pf
import PoseVisualizer as pv
from stl_to_obj_converter import stl_to_obj_converter

# Get the current script's directory
script_dir = Path(__file__).parent

# Get the current script's directory and then add the file path to the folders containing the workpiece stls
workpiece_path =  script_dir / 'Workpieces'

# Get the workpiece name you want to find poses for
workpiece_name = 'Teil_1'

# Convert the STL file to an OBJ file
stl_to_obj_converter(str(workpiece_path / (workpiece_name + '.STL')), str(workpiece_path / (workpiece_name + '.obj')),1)

# Convert the OBJ file to a PLY file
# obj_to_ply_converter(str(workpiece_path / (workpiece_name + '.obj')), str(workpiece_path / (workpiece_name + '.ply')))

# Extract the convex hull of the 3D model
obj_convex_hull_extractor(str(workpiece_path / (workpiece_name + '.obj')), str(workpiece_path / (workpiece_name + '_convex_hull.obj')))

# Initialize the PoseFinder with the convex hull OBJ file and define a tolerance for grouping rotations
pose_finder = pf.PoseFinder(str(workpiece_path / (workpiece_name + '_convex_hull.obj')), str(workpiece_path / (workpiece_name + '.obj')),1e-5)

# Find candidate rotations based on face normals
#candidate_rotations, xy_shadows = pose_finder.find_candidate_rotations_by_resting_face_normal_alignment()

#candidate_rotations, xy_shadows = pose_finder.find_candidate_rotations_by_shadow_edge_alignment()

candidate_rotations, xy_shadows, rotation_ids = pose_finder.find_candidate_rotations_by_face_and_shadow_alignment()

# Remove duplicate rotations 
#unique_rotations = pose_finder.duplicate_remover(candidate_rotations)

# Find unique poses by considering symmetry
symmetrically_unique_rotations = pose_finder.symmetry_handler(candidate_rotations)

# Initialize the PoseVisualizer with the original and convex hull OBJ files and valid rotations
pose_visualizer = pv.PoseVisualizer(str(workpiece_path / (workpiece_name + '.obj')), str(workpiece_path / (workpiece_name + '_convex_hull.obj')), symmetrically_unique_rotations,xy_shadows,rotation_ids)

# Visualize the valid poses
pose_visualizer.visualize_rotations()