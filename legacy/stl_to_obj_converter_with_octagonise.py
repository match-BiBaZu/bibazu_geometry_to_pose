import bpy
import bmesh
import math
from mathutils import Vector

def classify_arc_type(face_verts, center, normal, tol=0.1):
    """Classify if the vertex loop is circular or semi-circular."""
    directions = []
    for v in face_verts:
        vec = (v.co - center).normalized()
        projected = vec - vec.dot(normal) * normal  # remove normal component
        directions.append(projected.normalized())

    angles = []
    for i in range(len(directions)):
        a = directions[i]
        b = directions[(i+1) % len(directions)]
        angle = a.angle(b)
        angles.append(angle)

    arc_span = sum(angles)
    if abs(arc_span - 2 * math.pi) < tol:
        return 'full'
    elif abs(arc_span - math.pi) < tol:
        return 'half'
    return 'none'

def create_ngon(bm, center, normal, radius, sides, half=False):
    """Create a (half-)octagon projected in the normal plane."""
    verts = []
    start_angle = math.pi / 2 if half else 0
    step = math.pi / (sides // 2) if half else 2 * math.pi / sides
    for i in range(sides // 2 + 1 if half else sides):
        angle = start_angle + i * step
        offset = Vector((math.cos(angle), math.sin(angle), 0)) * radius
        rot = normal.rotation_difference(Vector((0, 0, 1))).to_matrix().to_4x4()
        pos = center + (rot @ offset)
        verts.append(bm.verts.new(pos))
    if not half:
        bm.faces.new(verts)
    else:
        # close the flat edge
        bm.faces.new(verts + [verts[0]])
    return verts

def stl_to_obj_converter_with_octagonise(stl_filepath, obj_filepath, scale=0.01):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_mesh.stl(filepath=stl_filepath)

    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    faces_to_replace = []

    for face in list(bm.faces):
        if len(face.verts) < 6:
            continue

        center = sum((v.co for v in face.verts), Vector()) / len(face.verts)
        normal = face.normal.normalized()
        distances = [(v.co - center).length for v in face.verts]
        r_mean = sum(distances) / len(distances)
        r_dev = (sum((d - r_mean)**2 for d in distances) / len(distances))**0.5

        if r_dev / r_mean > 0.05:
            continue  # not round enough

        arc_type = classify_arc_type(face.verts, center, normal)

        if arc_type in ['full', 'half']:
            faces_to_replace.append((face, center, normal, r_mean, arc_type))

    # Replace with octagons
    for face, center, normal, radius, arc_type in faces_to_replace:
        bmesh.ops.delete(bm, geom=[face], context='FACES')
        create_ngon(bm, center, normal, radius, 8, half=(arc_type == 'half'))

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    obj.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(scale=True)

    bpy.ops.export_scene.obj(
        filepath=obj_filepath,
        use_selection=True,
        global_scale=1.0,
        forward_axis='NEGATIVE_Y',
        up_axis='Z'
    )
