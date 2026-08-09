import numpy as np

from chute_pose import ChuteFrame


def test_neutral_frame_is_identity() -> None:
    frame = ChuteFrame(alpha_deg=0.0, beta_deg=0.0)

    np.testing.assert_allclose(frame.rotation_world_from_chute, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(frame.gravity_chute(), [0.0, 0.0, -9.81], atol=1e-12)


def test_df1a_test_angles_drive_part_downhill_and_toward_wall() -> None:
    frame = ChuteFrame(alpha_deg=45.0, beta_deg=20.0)
    gravity = frame.gravity_chute()

    assert gravity[0] > 0.0  # downhill in +X
    assert gravity[1] < 0.0  # toward wall at y=0
    assert gravity[2] < 0.0  # toward floor at z=0
    np.testing.assert_allclose(np.linalg.norm(gravity), 9.81, atol=1e-12)
    np.testing.assert_allclose(
        gravity,
        [3.35521761, -6.51838272, -6.51838272],
        atol=1e-8,
    )


def test_rotation_is_right_handed_and_orthonormal() -> None:
    rotation = ChuteFrame(alpha_deg=45.0, beta_deg=20.0).rotation_world_from_chute

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-12)


def test_floor_wall_seam_is_the_x_axis() -> None:
    frame = ChuteFrame(alpha_deg=45.0, beta_deg=20.0)

    np.testing.assert_array_equal(frame.seam_direction, [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(frame.floor_inward_normal, [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(frame.wall_inward_normal, [0.0, 1.0, 0.0])

