import trimesh
import numpy as np
from scipy.spatial import ConvexHull

def obj_convex_hull_extractor(input_obj_file: str, output_obj_file: str):
    """
    Extracts the convex hull of the 3D model and saves it as an OBJ file.
    """
    # Load the mesh
    mesh = trimesh.load_mesh(input_obj_file)
    
    # Get the vertices of the mesh
    vertices = np.array(mesh.vertices)
    
    # Compute the convex hull
    hull = ConvexHull(vertices)
    
    # Extract convex hull vertices and faces
    hull_vertices = hull.points
    hull_faces = hull.simplices

    # Create a new trimesh object
    convex_hull_mesh = trimesh.Trimesh(vertices=hull_vertices, faces=hull_faces)
    
    # Export the convex hull as an OBJ file
    convex_hull_mesh.export(output_obj_file)
    print(f"Convex hull saved to {output_obj_file}")


    