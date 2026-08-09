from pathlib import Path

import numpy as np

from chute_pose import (
    analyze_rocking_barriers,
    build_pose_catalog,
    cluster_practical_contact_poses,
    detect_rotational_symmetry,
    load_solid_mesh,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PARTS = REPOSITORY_ROOT / "Werkstücke_STL_grob"


def test_qk1a_observed_diagonal_pose_has_deeper_finite_barrier() -> None:
    mesh_path = PARTS / "Qk1a.STL"
    catalog = build_pose_catalog(mesh_path)
    pose_283 = catalog.poses[283]
    assert pose_283.floor_contact_topology == "2-point"
    assert pose_283.wall_contact_topology == "edge+point"
    assert len(pose_283.floor_mesh_contact_vertex_indices) == 2
    assert len(pose_283.wall_mesh_contact_vertex_indices) == 3
    assert len(pose_283.wall_mesh_contact_edges) == 1
    analysis = analyze_rocking_barriers(
        mesh_path,
        pose_ids=[164, 518, 519],
        axis_samples=512,
        angle_steps=10,
        catalog=catalog,
    )
    barriers = {value.pose_id: value for value in analysis.barriers}

    assert barriers[518].barrier_height_mm > 0.25
    assert barriers[519].barrier_height_mm > 0.25
    assert barriers[518].barrier_height_mm > 2.0 * barriers[164].barrier_height_mm
    assert barriers[519].barrier_height_mm > 2.0 * barriers[164].barrier_height_mm

    symmetry = detect_rotational_symmetry(mesh_path, tolerance_mm=0.6)
    mesh = load_solid_mesh(mesh_path)
    clustering = cluster_practical_contact_poses(
        catalog,
        np.asarray(mesh.vertices) - np.asarray(mesh.center_mass),
        [283, 519, 532, 947],
        symmetry=symmetry,
        angular_tolerance_deg=1.0,
        surface_displacement_tolerance_mm=0.6,
    )
    assert len(clustering.classes) == 1
    assert clustering.classes[0].pose_ids == (283, 519, 532, 947)


def test_rocking_analysis_rejects_invalid_sampling_parameters() -> None:
    catalog = build_pose_catalog(PARTS / "Df1a.STL")

    try:
        analyze_rocking_barriers(
            PARTS / "Df1a.STL", pose_ids=[9], axis_samples=7, catalog=catalog
        )
    except ValueError as error:
        assert "axis_samples" in str(error)
    else:
        raise AssertionError("Expected invalid axis_samples to be rejected")


def test_dl1a_observed_edge_pose_has_deeper_barrier_than_rollover_mirror() -> None:
    catalog = build_pose_catalog(PARTS / "Dl1a.STL")
    analysis = analyze_rocking_barriers(
        PARTS / "Dl1a.STL",
        pose_ids=[15, 48],
        axis_samples=128,
        angle_steps=10,
        catalog=catalog,
    )
    barriers = {value.pose_id: value for value in analysis.barriers}

    assert barriers[15].barrier_height_mm > 0.20
    assert barriers[48].barrier_height_mm < 0.02
