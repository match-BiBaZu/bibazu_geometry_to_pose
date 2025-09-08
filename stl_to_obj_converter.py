import bpy

def stl_to_obj_converter(stl_filepath, obj_filepath, scale=0.01, simplify_ratio=0.5):	
    """
    Convert an STL file to OBJ format using Blender's Python API.
    :param stl_filepath: Path to the input STL file.
    :param obj_filepath: Path to the output OBJ file.
    :param scale: Scale factor for the conversion (default is 0.01).
    :param simplify_ratio: Ratio for mesh simplification (default is 0.5, i.e., 50% reduction).
    """

    # Clear existing mesh data
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import STL file
    bpy.ops.wm.stl_import(filepath=stl_filepath)

    # Get the imported object (typically the first one in the scene after import)
    obj = bpy.context.selected_objects[0]

    # Ensure the object is selected and set as active
    obj = bpy.data.objects[0]
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Make sure the object is a single user (no shared mesh data)
    bpy.ops.object.make_single_user(object=True, obdata=True)

    # Simplify the mesh using the Decimate modifier
    bpy.ops.object.modifier_add(type='DECIMATE')
    decimate_mod = obj.modifiers["Decimate"]
    decimate_mod.ratio = simplify_ratio
    bpy.ops.object.modifier_apply(modifier=decimate_mod.name)

    # Export as OBJ file
    bpy.ops.wm.obj_export(filepath=obj_filepath, forward_axis='Y', up_axis='Z', global_scale=scale) # Blender like co-ordinate system
