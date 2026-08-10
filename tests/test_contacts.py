from pathlib import Path

import numpy as np

from chute_pose import build_pose_catalog

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DF1A_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Df1a.STL"
KK1A_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Kk1a.STL"


def test_kk1a_continuous_symmetry_removes_faceted_circle_pose_duplicates() -> None:
    catalog = build_pose_catalog(KK1A_STL)

    assert catalog.continuous_symmetry_axis_part is not None
    assert catalog.continuous_symmetry_edge_edge_exception
    assert len(catalog.support_faces) == 5
    assert len(catalog.poses) == 12
    assert sum(pose.floor_contact_type == "face" for pose in catalog.poses) == 2
    assert sum(pose.wall_contact_type == "face" for pose in catalog.poses) == 2
    assert sum(
        pose.floor_contact_type == pose.wall_contact_type == "edge"
        for pose in catalog.poses
    ) == 8


def test_df1a_catalog_contains_all_convex_support_faces() -> None:
    catalog = build_pose_catalog(DF1A_STL)

    assert len(catalog.support_faces) == 9
    assert len(catalog.poses) == 108
    represented_faces = {
        face_id
        for pose in catalog.poses
        for face_id in pose.floor_face_ids + pose.wall_face_ids
    }
    assert represented_faces == {face.face_id for face in catalog.support_faces}

    contact_type_counts: dict[tuple[int, int], int] = {}
    for pose in catalog.poses:
        key = (pose.floor_contact_dimension, pose.wall_contact_dimension)
        contact_type_counts[key] = contact_type_counts.get(key, 0) + 1
    assert contact_type_counts == {(1, 2): 48, (2, 1): 48, (2, 2): 12}


def test_df1a_catalog_has_only_non_point_isolated_contacts() -> None:
    catalog = build_pose_catalog(DF1A_STL)

    assert catalog.poses
    for pose in catalog.poses:
        assert pose.floor_contact_dimension >= 1
        assert pose.wall_contact_dimension >= 1
        assert max(pose.floor_contact_dimension, pose.wall_contact_dimension) == 2


def test_df1a_pose_rotations_are_unique_and_right_handed() -> None:
    catalog = build_pose_catalog(DF1A_STL)

    quaternions = np.asarray([pose.quaternion_xyzw for pose in catalog.poses])
    for index, pose in enumerate(catalog.poses):
        rotation = np.asarray(pose.rotation_chute_from_part)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-10)
        for other_index in range(index):
            dot = abs(float(np.dot(quaternions[index], quaternions[other_index])))
            distance = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))
            assert distance > catalog.rotation_tolerance_rad


def test_df1a_contacts_can_be_translated_into_the_corner() -> None:
    catalog = build_pose_catalog(DF1A_STL)

    # Contact vertices are indexed on the convex hull used by the catalog.
    import trimesh

    mesh = trimesh.load_mesh(DF1A_STL, force="mesh", process=True)
    hull = mesh.convex_hull
    vertices = np.asarray(hull.vertices) - np.asarray(mesh.center_mass)

    for pose in catalog.poses:
        rotation = np.asarray(pose.rotation_chute_from_part)
        translation = np.asarray(pose.translation_to_corner_mm)
        transformed = (rotation @ vertices.T).T + translation
        assert float(np.min(transformed[:, 1])) >= -1e-7
        assert float(np.min(transformed[:, 2])) >= -1e-7
        np.testing.assert_allclose(
            transformed[list(pose.wall_contact_vertex_indices), 1],
            0.0,
            atol=catalog.contact_tolerance_mm,
        )
        np.testing.assert_allclose(
            transformed[list(pose.floor_contact_vertex_indices), 2],
            0.0,
            atol=catalog.contact_tolerance_mm,
        )
