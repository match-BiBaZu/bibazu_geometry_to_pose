from pathlib import Path

import numpy as np

from chute_pose import (
    build_pose_catalog,
    cluster_practical_contact_poses,
    detect_rotational_symmetry,
    load_solid_mesh,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DF1A_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Df1a.STL"
QK1A_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Qk1a.STL"


def test_practical_clustering_can_apply_an_explicit_part_symmetry() -> None:
    catalog = build_pose_catalog(DF1A_STL)
    symmetry = detect_rotational_symmetry(DF1A_STL, tolerance_mm=0.05)
    mesh = load_solid_mesh(DF1A_STL)
    vertices_centered = np.asarray(mesh.vertices) - np.asarray(mesh.center_mass)

    clustering = cluster_practical_contact_poses(
        catalog,
        vertices_centered,
        [9, 12, 31],
        symmetry=symmetry,
        surface_displacement_tolerance_mm=0.05,
    )

    assert len(clustering.classes) == 1
    assert clustering.classes[0].pose_ids == (9, 12, 31)


def test_qk1a_mesh_facet_variants_merge_when_contact_planes_are_exchanged() -> None:
    catalog = build_pose_catalog(QK1A_STL)
    symmetry = detect_rotational_symmetry(QK1A_STL, tolerance_mm=0.05)
    mesh = load_solid_mesh(QK1A_STL)
    vertices_centered = np.asarray(mesh.vertices) - np.asarray(mesh.center_mass)

    clustering = cluster_practical_contact_poses(
        catalog,
        vertices_centered,
        [
            988,
            989,
            993,
            994,
            1843,
            1844,
            1859,
            1860,
            3291,
            3292,
            3311,
            3312,
            5008,
            5009,
            5016,
            5017,
        ],
        symmetry=symmetry,
        angular_tolerance_deg=1.0,
        surface_displacement_tolerance_mm=0.5,
    )

    assert [pose_class.pose_ids for pose_class in clustering.classes] == [
        (988, 994, 1844, 1860, 3292, 3311, 5008, 5017),
        (989, 993, 1843, 1859, 3291, 3312, 5009, 5016),
    ]
