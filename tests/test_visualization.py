from pathlib import Path

from chute_pose.visualization import render_pose_sheets


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DF1A_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Df1a.STL"


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

