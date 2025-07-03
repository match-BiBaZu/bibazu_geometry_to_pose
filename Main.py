from pathlib import Path
from obj_convex_hull_extractor import obj_convex_hull_extractor
from obj_to_ply_converter import obj_to_ply_converter
from stl_to_obj_converter import stl_to_obj_converter
from PoseFinder import PoseFinder
from PoseEliminator import PoseEliminator
from PoseVisualizer import PoseVisualizer
import os


# Get the current script's directory
script_dir = Path(__file__).parent

# Get the current script's directory and then add the file path to the folders containing the workpiece stls
#workpiece_path =  script_dir / 'Workpieces'

workpiece_path = script_dir / 'Werkstücke_STL_grob'	

workpiece_names = ['Teil_1', 'Teil_2', 'Teil_3', 'Teil_4', 'Teil_5']

#workpiece_names = ['Teil_4']

workpiece_names = ['Df1a','Df2i','Df4a','Dk1i','Dk2a','Dk4i','Dl1a','Dl4a','Kf1i','Kf2a','Kf4i','Kk1a','Kk2i','Kk4a','Kl1i','Kl2a','Kl4i','Qf1i','Qf2a','Qf4i','Qk1a','Qk2i','Qk4a','Ql1i','Ql2a','Ql4i','Rf1a','Rf2i','Rf4i','Rk1a','Rk2i','Rk3a','Rk4i','Rl1a','Rl2i','Rl3a','Rl4i']

workpiece_names = ['Df1a','Df2i','Df4a','Dk1i','Dk2a','Dk4i','Dl1a','Dl4a','Qf1i','Qf2a','Qf4i','Qk1a','Qk2i','Qk4a','Ql1i','Ql2a','Ql4i','Rf1a','Rf2i','Rf4i','Rk1a','Rk2i','Rk3a','Rk4i','Rl1a','Rl2i','Rl3a','Rl4i']

#List of workpieces with rounded features that are likely to appear in the convex hull
#rounded_workpiece_names = ['Dk2a','Kf1i','Kf2a','Kf4i','Kk1a','Kk2i','Kk4a','Kl1i','Kl2a','Kl4i','Qf2a','Qk1a','Ql1i','Ql4i','Rf1a','Rf3a','Rk1a','Rk3a','Rl1a','Rl4i']

workpiece_names =['Df1a','Df4a','Df2i']
workpiece_names =['Rl4i','Ql4i','Qf4i','Df4a','Rk2i']

# Get the workpiece name you want to find poses for
#workpiece_name = 'Teil_2'

for workpiece_name in workpiece_names: 

    # Convert the STL file to an OBJ file
    stl_to_obj_converter(str(workpiece_path / (workpiece_name + '.STL')), str(workpiece_path / (workpiece_name + '.obj')), 1, 1.0)

    # Convert the OBJ file to a PLY file
    # obj_to_ply_converter(str(workpiece_path / (workpiece_name + '.obj')), str(workpiece_path / (workpiece_name + '.ply')))

    # Extract the convex hull of the 3D model
    obj_convex_hull_extractor(str(workpiece_path / (workpiece_name + '.obj')), str(workpiece_path / (workpiece_name + '_convex_hull.obj')))

    # Initialize the PoseFinder with the convex hull OBJ file and define a tolerance for grouping rotations
    pose_finder = PoseFinder(str(workpiece_path / (workpiece_name + '_convex_hull.obj')), str(workpiece_path / (workpiece_name + '.obj')), 1e-5)

    # Find candidate rotations based on face normals and shadow edges
    candidate_rotations, xy_shadows = pose_finder.find_candidate_rotations_by_face_and_shadow_alignment()

    # Initialize the PoseEliminator with the convex hull OBJ file and self OBJ file
    pose_eliminator = PoseEliminator(str(workpiece_path / (workpiece_name + '_convex_hull.obj')), str(workpiece_path / (workpiece_name + '.obj')), 0.01)

    # Remove duplicate rotations (if any) from the candidate rotations
    unique_rotations, unique_shadows = pose_eliminator.remove_duplicates(candidate_rotations, xy_shadows)

    # Remove rotations that are not stable enough by the crude centroid over resting plane detection
    stable_rotations, stable_shadows = pose_eliminator.remove_unstable_poses(unique_rotations,unique_shadows)

    # Find unique poses by considering symmetry with an adjustable tolerance, this is set for workpieces with feature sizes between 0.1 and 0.03 cm
    symmetrically_unique_rotations = pose_finder.symmetry_handler(stable_rotations,1)

    pose_finder.write_candidate_rotations_to_file(symmetrically_unique_rotations, str(workpiece_name + '_candidate_rotations.csv'))

    # Initialize the PoseVisualizer with the original and convex hull OBJ files and valid rotations
    pose_visualizer = PoseVisualizer(str(workpiece_path / (workpiece_name + '.obj')), str(workpiece_path / (workpiece_name + '_convex_hull.obj')), symmetrically_unique_rotations, stable_shadows)

    # Visualize the valid poses
    pose_visualizer.visualize_rotations(workpiece_name)
