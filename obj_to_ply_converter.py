import trimesh

def obj_to_ply_converter(obj_file_path, ply_file_path):
    
    """
    Convert an OBJ file to PLY format using trimesh.
    """

    # Load the OBJ file
    mesh = trimesh.load(obj_file_path, file_type='obj')
    
    # Export the mesh to PLY format
    mesh.export(ply_file_path, file_type='ply')
