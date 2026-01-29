from pathlib import Path
from obj_convex_hull_extractor import obj_convex_hull_extractor
from obj_to_ply_converter import obj_to_ply_converter
from stl_to_obj_converter import stl_to_obj_converter
from step_find_largest_cylinder import step_find_largest_cylinder
from step_find_all_cylinders import step_find_all_cylinders
from PoseFinder import PoseFinder
from CylinderHandler import CylinderHandler
from PoseEliminator import PoseEliminator
from PoseVisualizer import PoseVisualizer


# Get the current script's directory
script_dir = Path(__file__).parent

# Get the current script's directory and then add the file path to the folders containing the workpiece stls
# PLEASE MAKE SURE ALL WORKPIECES ARE IN MM UNITS FOR CONSISTENCY, THE TOLERANCES ARE FINE TUNED FOR FEATURES BETWEEN 1 AND 100 MM
#workpiece_path =  script_dir / 'Workpieces'

workpiece_path = script_dir / 'Werkstücke_STL_grob'

csv_path = script_dir / 'csv_outputs'

workpiece_names = ['Teil_1', 'Teil_2', 'Teil_3', 'Teil_4', 'Teil_5']

workpiece_names = ['Df1a','Df2i','Df4a','Dk1i','Dk2a','Dk4i','Dl1a','Dl2i','Dl4a','Kf1i','Kf2a','Kf4i','Kk1a','Kk2i','Kk4a','Kl1i','Kl2a','Kl4i','Qf1i','Qf2a','Qf4i','Qk1a','Qk2i','Qk4a','Ql1i','Ql2a','Ql4i','Rf1a','Rf2i','Rf4i','Rf3a','Rk1a','Rk2i','Rk3a','Rk4i','Rl1a','Rl2i','Rl3a','Rl4i']
print(f"Total workpieces to process: {len(workpiece_names)}")
#workpiece_names = ['Df1a','Df2i','Df4a','Dk1i','Dk2a','Dk4i','Dl1a','Dl4a','Qf1i','Qf2a','Qf4i','Qk1a','Qk2i','Qk4a','Ql1i','Ql2a','Ql4i','Rf1a','Rf2i','Rf4i','Rk1a','Rk2i','Rk3a','Rk4i','Rl1a','Rl2i','Rl3a','Rl4i']

#List of workpieces with rounded features that are likely to appear in the convex hull
#rounded_workpiece_names = ['Rf3a','Rl1a']
circular_workpiece_names = ['Kf1i','Kf2a','Kf4i','Kk1a','Kk2i','Kk4a','Kl1i','Kl2a','Kl4i','Rl1a','Rf3a','Rl3a','Dk2a','Rl3a','Qk1a','Qf2a','Rk1a']

#workpiece_names =['Df1a','Df4a','Df2i']
#workpiece_names =['Rl4i','Ql4i','Qf4i','Df4a','Rk2i']
#workpiece_names =['Rl2i','Df2i','Dk4i','Dl4a','Qk4a','Rf4i','Rk4i','Rf2i','Dl2i']
#workpiece_names = ['Kl4i','Kl1i','Kl2a','Rl1a']
#workpiece_names = ['Kk2i','Kl2a','Kf2a']
#workpiece_names = circular_workpiece_names
#workpiece_names = ['Df1a','Df2i','Df4a','Dk1i','Dk4i','Dl1a','Dl4a','Qf1i','Qf4i','Qk2i','Qk4a','Ql2a','Rf2i','Rf4i','Rk2i','Rk4i','Rl2i','Rl3a','Qk1a','Ql1i','Ql4i','Rf1a','Rf3a','Rk1a','Rk3a','Rl4i']
#workpiece_names = ['Kf4i','Kf2a']
#workpiece_names = ['Kk1a']
#workpiece_names = ['Df4a','Df2i']
#workpiece_names = ['Rf3a']
#workpiece_names = ['Kf2a','Kl2a']
#workpiece_names = ['Kl4i','Kk4a','Kl2a','Kl1i']
# is the step file origin at the center of mass of its convex hull? 0 = no, 1 = yes, when no make sure the step files origin is at the corner of a workpiece
step_file_centered = 0

# location of axes points or perpedicular origin distance based cylinder check switch (usually 1, except for kk2i which uses a crude distance check)
axis_based_cylinder_check = 1  # 0 = off, 1 = on

eliminator_stability_tolerance = -1.5 #tolerance for center of mass point in polygon test during stability check, negative value means stricter check

alpha_tilt_angle = 20.0  # degrees this is the perfect aero angle
beta_tilt_angle = 5.0   # degrees this is the biggest angle before the workpiece shifts it's weight onto the other face of the slide in any poses

# Get the workpiece name you want to find poses for
#workpiece_name = 'Teil_2'

for workpiece_name in workpiece_names:

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

    if workpiece_name == 'Kk2i':
        axis_based_cylinder_check = 0  # crude perpendicular distance origin check for kk2i as its convex hull points are all a radius away from the cylinder axis
    else:
        axis_based_cylinder_check = 1  # normal axis based check for other workpieces

    cylinder_handler = CylinderHandler(str(workpiece_path / (workpiece_name + '_convex_hull.obj')), str(workpiece_path / (workpiece_name + '.obj')), 1e-5, 1, 4, axis_based_cylinder_check)

    # Find and classify cylinder poses from the candidate rotations (if cylinders are detected in the workpiece)
    if cylinder_axis_parameters:
        candidate_rotations, xy_shadows, cylinder_axis_parameters, pose_types, pose_cylinder_radius, pose_cylinder_axis_direction, pose_cylinder_axis_origin, pose_cylinder_group = cylinder_handler.find_cylinder_poses(candidate_rotations, xy_shadows, cylinder_axis_parameters,alpha_tilt_angle,beta_tilt_angle)
    else:
        pose_types = [0] * len(candidate_rotations)  # default: non-cylinder
        pose_cylinder_radius = [0] * len(candidate_rotations)
        pose_cylinder_axis_direction = [[0,0,1]] * len(candidate_rotations)
        pose_cylinder_axis_origin = [[0,0,0]]  * len(candidate_rotations)
        pose_cylinder_group = [0] * len(candidate_rotations)


    # Initialize the PoseEliminator with the convex hull OBJ file and self OBJ file
    pose_eliminator = PoseEliminator(
        str(workpiece_path / (workpiece_name + '_convex_hull.obj')),
        str(workpiece_path / (workpiece_name + '.obj')),
        tolerance=1e-2, # tolerance needs to be a bit looser for the cylinder workpieces when doing stability check
        stability_tolerance=eliminator_stability_tolerance,
        stable_rotations=candidate_rotations,
        stable_shadows=xy_shadows,
        stable_axis_parameters=cylinder_axis_parameters,
        pose_types=pose_types,
        pose_cylinder_radius=pose_cylinder_radius,
        pose_cylinder_axis_direction=pose_cylinder_axis_direction,
        pose_cylinder_axis_origin=pose_cylinder_axis_origin,
        pose_cylinder_group=pose_cylinder_group
    )

    # Remove duplicate rotations (if any) from the candidate rotations
    pose_eliminator.remove_duplicates()

    # Remove cylinder poses based on alignment criteria
    pose_eliminator.remove_cylinder_poses()

    # Remove rotations that are not stable enough by the crude centroid over resting plane detection, this also includes a secondary tilt angle check for stability when slide is tilted in alpha tilt angle only
    pose_eliminator.remove_unstable_poses(alpha_tilt_angle,beta_tilt_angle)

    # Find unique poses by considering symmetry with an adjustable tolerance, this is set for workpieces with feature sizes between 0.1 and 0.03 cm (I still think this is programmed weirdly)
    symmetrically_unique_rotations = pose_finder.symmetry_handler(pose_eliminator.get_stable_rotations(),1)

    pose_finder.write_candidate_rotations_to_file(symmetrically_unique_rotations, 
                                                  pose_eliminator.get_pose_types(),
                                                  pose_eliminator.get_pose_cylinder_radius(), 
                                                  pose_eliminator.get_pose_cylinder_axis_origin(),
                                                  pose_eliminator.get_pose_cylinder_axis_direction(),
                                                  pose_eliminator.get_pose_cylinder_group(),
                                                  False, #output_cylinder_features flag set to false but can be set to true if cylinder features are desired in the output
                                                  str(csv_path / (workpiece_name + '_candidate_rotations.csv'))
                                                  )

    #pose_finder.write_pose_shadows_to_file(symmetrically_unique_rotations,xy_shadows)

    # Initialize the PoseVisualizer with the original and convex hull OBJ files and valid rotations
    pose_visualizer = PoseVisualizer(str(workpiece_path / (workpiece_name + '.obj')), str(workpiece_path / (workpiece_name + '_convex_hull.obj')),symmetrically_unique_rotations, pose_eliminator.get_stable_shadows(), pose_eliminator.get_stable_axis_parameters())

    # Visualize the valid poses
    pose_visualizer.visualize_rotations(workpiece_name)
