from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import proj3d

from chute_pose.plot_view import apply_pose_view
from chute_pose.visualization import render_pose_sheets


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DF1A_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Df1a.STL"


def test_pose_view_matches_gui_axis_directions() -> None:
    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-1.0, 1.0)
    axis.set_zlim(-1.0, 1.0)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    apply_pose_view(axis)
    figure.canvas.draw()

    projection = axis.get_proj()
    origin = np.asarray(proj3d.proj_transform(0.0, 0.0, 0.0, projection)[:2])
    projected_axes = []
    for endpoint in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        projected = np.asarray(proj3d.proj_transform(*endpoint, projection)[:2])
        direction = projected - origin
        projected_axes.append(direction / np.linalg.norm(direction))
    plt.close(figure)

    np.testing.assert_allclose(projected_axes[0], (np.sqrt(3.0) / 2.0, 0.5), atol=1e-6)
    np.testing.assert_allclose(projected_axes[1], (np.sqrt(3.0) / 2.0, -0.5), atol=1e-6)
    np.testing.assert_allclose(projected_axes[2], (0.0, 1.0), atol=1e-6)


def test_render_selected_df1a_poses(tmp_path: Path) -> None:
    sheets = render_pose_sheets(
        DF1A_STL,
        tmp_path,
        poses_per_sheet=3,
        columns=3,
        dpi=72,
        pose_ids=[0, 1, 2],
    )

    assert sheets
    assert sum(len(sheet.pose_ids) for sheet in sheets) == 3
    for sheet in sheets:
        assert sheet.path.is_file()
        assert sheet.path.stat().st_size > 1_000

