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


def test_practical_clustering_can_apply_an_explicit_part_symmetry() -> None:
    catalog = build_pose_catalog(DF1A_STL)
    symmetry = detect_rotational_symmetry(DF1A_STL, tolerance_mm=0.5)
    mesh = load_solid_mesh(DF1A_STL)
    vertices_centered = np.asarray(mesh.vertices) - np.asarray(mesh.center_mass)

    clustering = cluster_practical_contact_poses(
        catalog,
        vertices_centered,
        [9, 12, 32],
        symmetry=symmetry,
        surface_displacement_tolerance_mm=0.5,
    )

    assert len(clustering.classes) == 1
    assert clustering.classes[0].pose_ids == (9, 12, 32)
