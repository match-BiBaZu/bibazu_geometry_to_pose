"""Shared camera convention for generated pose and roadmap plots."""

from __future__ import annotations

import math


# Matplotlib's equivalent of the GUI's orthographic display matrix.  In the
# rendered image Z is vertical, Y points down-right and X points up-right
# between Y and Z.
POSE_VIEW_ELEVATION_DEG = -math.degrees(math.asin(1.0 / math.sqrt(3.0)))
POSE_VIEW_AZIMUTH_DEG = -45.0
POSE_VIEW_ROLL_DEG = 0.0


def apply_pose_view(axis) -> None:
    """Apply the common GUI-compatible orthographic pose camera."""

    axis.set_proj_type("ortho")
    axis.view_init(
        elev=POSE_VIEW_ELEVATION_DEG,
        azim=POSE_VIEW_AZIMUTH_DEG,
        roll=POSE_VIEW_ROLL_DEG,
    )
