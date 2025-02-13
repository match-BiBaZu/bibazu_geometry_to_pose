import bpy

def stl_to_obj_converter(stl_filepath, obj_filepath, scale=0.01):
    # Clear existing mesh data
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import STL file
    bpy.ops.wm.stl_import(filepath=stl_filepath)

    # Get the imported object (typically the first one in the scene after import)
    obj = bpy.context.selected_objects[0]

    # Ensure the object is selected and set as active
    obj = bpy.data.objects[0]
    obj.select_set(True)

    # Make sure the object is a single user (no shared mesh data)
    bpy.ops.object.make_single_user(object=True, obdata=True)
    
    # Export as OBJ file
    bpy.ops.wm.obj_export(filepath=obj_filepath, forward_axis='NEGATIVE_Y', up_axis='Z', global_scale=scale) #Blender like co-ordinate system
       
