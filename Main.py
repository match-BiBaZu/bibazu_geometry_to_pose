from pathlib import Path
from obj_convex_hull_extractor import obj_convex_hull_extractor
from obj_to_ply_converter import obj_to_ply_converter
from stl_to_obj_converter import stl_to_obj_converter
from step_find_largest_cylinder import step_find_largest_cylinder
from step_find_all_cylinders import step_find_all_cylinders
from PoseFinder import PoseFinder
from PoseEliminator import PoseEliminator
from PoseVisualizer import PoseVisualizer


# Get the current script's directory
script_dir = Path(__file__).parent

# Get the current script's directory and then add the file path to the folders containing the workpiece stls
#workpiece_path =  script_dir / 'Workpieces'

workpiece_path = script_dir / 'Werkstücke_STL_grob'	

csv_path = script_dir / 'csv_outputs'

workpiece_names = ['Teil_1', 'Teil_2', 'Teil_3', 'Teil_4', 'Teil_5']

workpiece_names = ['Df1a','Df2i','Df4a','Dk1i','Dk2a','Dk4i','Dl1a','Dl4a','Kf1i','Kf2a','Kf4i','Kk1a','Kk2i','Kk4a','Kl1i','Kl2a','Kl4i','Qf1i','Qf2a','Qf4i','Qk1a','Qk2i','Qk4a','Ql1i','Ql2a','Ql4i','Rf1a','Rf2i','Rf4i','Rk1a','Rk2i','Rk3a','Rk4i','Rl1a','Rl2i','Rl3a','Rl4i']

#workpiece_names = ['Df1a','Df2i','Df4a','Dk1i','Dk2a','Dk4i','Dl1a','Dl4a','Qf1i','Qf2a','Qf4i','Qk1a','Qk2i','Qk4a','Ql1i','Ql2a','Ql4i','Rf1a','Rf2i','Rf4i','Rk1a','Rk2i','Rk3a','Rk4i','Rl1a','Rl2i','Rl3a','Rl4i']

#List of workpieces with rounded features that are likely to appear in the convex hull
rounded_workpiece_names = ['Kf1i','Kf2a','Kf4i','Kk1a','Kk2i','Kk4a','Kl1i','Kl2a','Kl4i','Rf3a','Rl1a']

#workpiece_names =['Df1a','Df4a','Df2i']
#workpiece_names =['Rl4i','Ql4i','Qf4i','Df4a','Rk2i']
#workpiece_names =['Rl2i','Df2i','Dk4i','Dl4a','Qk4a','Rf4i','Rk4i','Rf2i','Dl2i']
#workpiece_names = ['Kl4i','Kl1i','Kl2a','Rl1a']
workpiece_names = ['Kk1a','Rl1a']
#workpiece_names = rounded_workpiece_names

# check if step file is centered or not, if the first letter of the workpiece name is 'k' or 'K' it is centered as it is a circle based part
step_file_centered = 0

# Get the workpiece name you want to find poses for
#workpiece_name = 'Teil_2'

for workpiece_name in workpiece_names:

    # Set is_step_file_centered to 2 if 'k' is detected in the first letter of the workpiece name, 1 if workpiece_name is 'Rl1a', else 0
    step_file_centered = 2 if workpiece_name[0].lower() == 'k' else 1 if workpiece_name == 'Rl1a' else 0
    #print(f"Processing workpiece: {workpiece_name}, is_step_file_centered: {step_file_centered}")    # Use the original STEP file to find the largest cylinder or circle edge
    step_find_all_cylinders(str(workpiece_path / (workpiece_name + '.STEP')), str(csv_path / (workpiece_name + '_cylinder_properties.csv')))

    # Convert the STL file to an OBJ file
    stl_to_obj_converter(str(workpiece_path / (workpiece_name + '.STL')), str(workpiece_path / (workpiece_name + '.obj')), 1, 1.0)

    # Convert the OBJ file to a PLY file
    # obj_to_ply_converter(str(workpiece_path / (workpiece_name + '.obj')), str(workpiece_path / (workpiece_name + '.ply')))

    # Extract the convex hull of the 3D model
    obj_convex_hull_extractor(str(workpiece_path / (workpiece_name + '.obj')), str(workpiece_path / (workpiece_name + '_convex_hull.obj')))

    # Initialize the PoseFinder with the convex hull OBJ file and define a tolerance for grouping rotations
    pose_finder = PoseFinder(str(workpiece_path / (workpiece_name + '_convex_hull.obj')), str(workpiece_path / (workpiece_name + '.obj')), 1e-5, step_file_centered)

    # Find candidate rotations based on face normals and shadow edges
    candidate_rotations, xy_shadows, cylinder_axis_parameters = pose_finder.find_candidate_rotations_by_face_and_shadow_alignment()

    # Initialize the PoseEliminator with the convex hull OBJ file and self OBJ file
    pose_eliminator = PoseEliminator(str(workpiece_path / (workpiece_name + '_convex_hull.obj')), str(workpiece_path / (workpiece_name + '.obj')), 0.01, 1,15)

    # Remove duplicate rotations (if any) from the candidate rotations
    unique_rotations, unique_shadows, unique_axis_parameters = pose_eliminator.remove_duplicates(candidate_rotations, xy_shadows, cylinder_axis_parameters)

    # Remove rotations that are not stable enough by the crude centroid over resting plane detection
    stable_rotations, stable_shadows,stable_axis_parameters = pose_eliminator.remove_unstable_poses(unique_rotations, unique_shadows, unique_axis_parameters)

    # Discretize the cylindrical components of the workpieces according to a step size
    discretized_rotations, discretized_shadows, discretized_axis_parameters = pose_eliminator.discretise_rotations(stable_rotations, stable_shadows,stable_axis_parameters)
    
    # Find unique poses by considering symmetry with an adjustable tolerance, this is set for workpieces with feature sizes between 0.1 and 0.03 cm (I still think this is programmed weirdly)
    #symmetrically_unique_rotations = pose_finder.symmetry_handler(stable_rotations,2)

    pose_finder.write_candidate_rotations_to_file(discretized_rotations, str(csv_path / (workpiece_name + '_candidate_rotations.csv')))

    # Initialize the PoseVisualizer with the original and convex hull OBJ files and valid rotations
    pose_visualizer = PoseVisualizer(str(workpiece_path / (workpiece_name + '.obj')), str(workpiece_path / (workpiece_name + '_convex_hull.obj')),discretized_rotations, discretized_shadows, discretized_axis_parameters)

    # Visualize the valid poses
    pose_visualizer.visualize_rotations(workpiece_name)
