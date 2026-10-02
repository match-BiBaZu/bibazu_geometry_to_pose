from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import proj3d

from chute_pose.plot_view import apply_pose_view, draw_coordinate_axes
from chute_pose.visualization import _draw_contact_set, render_pose_sheets


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

    np.testing.assert_allclose(projected_axes[0], (-np.sqrt(3.0) / 2.0, -0.5), atol=1e-6)
    np.testing.assert_allclose(projected_axes[1], (np.sqrt(3.0) / 2.0, -0.5), atol=1e-6)
    np.testing.assert_allclose(projected_axes[2], (0.0, 1.0), atol=1e-6)


def test_coordinate_axes_draw_xyz_labels() -> None:
    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")

    draw_coordinate_axes(axis)

    labels = [text.get_text() for text in axis.texts if text.get_text()]
    assert set(labels) == {"X", "Y", "Z"}
    assert len(labels) == 3
    x_label = next(text for text in axis.texts if text.get_text() == "X")
    x_position = x_label.get_position()
    assert x_position[0] < 0.80
    assert x_position[1] < 0.80
    plt.close(figure)


def test_contact_markers_are_compact_and_borderless() -> None:
    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")

    _draw_contact_set(
        axis,
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        np.asarray((0, 1), dtype=int),
        ((0, 1),),
        color="#10a64a",
        marker="o",
    )

    marker_collection = axis.collections[0]
    assert marker_collection.get_sizes().tolist() == [34]
    assert marker_collection.get_edgecolors().size == 0
    assert axis.lines[0].get_linewidth() == 2.0
    plt.close(figure)


def test_render_selected_df1a_poses(tmp_path: Path) -> None:
    sheets = render_pose_sheets(
        DF1A_STL,
        tmp_path,
        dpi=72,
        pose_ids=[0, 1, 2],
        formats=("png", "svg"),
        sheet_title="Df1a",
        metric_labels={0: [("Rocking: 0.385 mm", "black")]},
    )

    assert len(sheets) == 6
    assert all(len(sheet.pose_ids) == 1 for sheet in sheets)
    for sheet in sheets:
        assert sheet.path.is_file()
        assert sheet.path.stat().st_size > 1_000
    svg = next(sheet.path for sheet in sheets if sheet.path.suffix == ".svg" and sheet.pose_ids == (0,))
    content = svg.read_text(encoding="utf-8")
    assert "Rocking: 0.385 mm" in content
    assert all(label not in content for label in ("Floor:", "Wall:", "Page ", "Green/circle:"))

