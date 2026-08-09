"""Technical rendering of theoretical floor-wall contact poses."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .contacts import ContactPose, PoseCatalog, build_pose_catalog
from .geometry import load_solid_mesh


@dataclass(frozen=True, slots=True)
class RenderedSheet:
    path: Path
    pose_ids: tuple[int, ...]
    contact_group: str


def _contact_group(pose: ContactPose) -> str:
    return f"floor-{pose.floor_contact_type}_wall-{pose.wall_contact_type}"


def _chunks(values: list[ContactPose], chunk_size: int) -> Iterable[list[ContactPose]]:
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def _draw_reference_surfaces(ax, bounds: np.ndarray, margin: float) -> None:
    min_x, _, _ = bounds[0]
    max_x, max_y, max_z = bounds[1]
    x0 = min_x - margin
    x1 = max_x + margin
    y1 = max_y + margin
    z1 = max_z + margin

    floor = [[(x0, 0.0, 0.0), (x1, 0.0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)]]
    wall = [[(x0, 0.0, 0.0), (x1, 0.0, 0.0), (x1, 0.0, z1), (x0, 0.0, z1)]]
    ax.add_collection3d(
        Poly3DCollection(floor, facecolor="#c9d1d9", edgecolor="#7d8590", alpha=0.28)
    )
    ax.add_collection3d(
        Poly3DCollection(wall, facecolor="#e3c9a8", edgecolor="#9a7548", alpha=0.25)
    )
    ax.plot([x0, x1], [0.0, 0.0], [0.0, 0.0], color="#24292f", linewidth=1.8)


def _draw_pose(
    ax,
    pose: ContactPose,
    mesh_vertices_centered: np.ndarray,
    mesh_faces: np.ndarray,
    hull_vertices_centered: np.ndarray,
) -> None:
    rotation = np.asarray(pose.rotation_chute_from_part, dtype=float)
    translation = np.asarray(pose.translation_to_corner_mm, dtype=float)
    mesh_vertices = (rotation @ mesh_vertices_centered.T).T + translation
    hull_vertices = (rotation @ hull_vertices_centered.T).T + translation

    triangles = mesh_vertices[mesh_faces]
    part = Poly3DCollection(
        triangles,
        facecolor="#5b9bd5",
        edgecolor="#244a68",
        linewidth=0.35,
        alpha=0.78,
    )
    ax.add_collection3d(part)

    floor_indices = np.asarray(pose.floor_contact_vertex_indices, dtype=int)
    wall_indices = np.asarray(pose.wall_contact_vertex_indices, dtype=int)
    seam_indices = np.intersect1d(floor_indices, wall_indices)
    floor_only = np.setdiff1d(floor_indices, seam_indices)
    wall_only = np.setdiff1d(wall_indices, seam_indices)

    if len(floor_only):
        points = hull_vertices[floor_only]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="#1a9c50", s=14, depthshade=False)
    if len(wall_only):
        points = hull_vertices[wall_only]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="#e67e22", s=14, depthshade=False)
    if len(seam_indices):
        points = hull_vertices[seam_indices]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="#d62728", s=18, depthshade=False)

    bounds = np.vstack([mesh_vertices.min(axis=0), mesh_vertices.max(axis=0)])
    span = np.maximum(bounds[1] - bounds[0], 1e-9)
    margin = 0.12 * float(np.max(span))
    _draw_reference_surfaces(ax, bounds, margin)

    ax.set_xlim(bounds[0, 0] - margin, bounds[1, 0] + margin)
    ax.set_ylim(-0.05 * margin, bounds[1, 1] + margin)
    ax.set_zlim(-0.05 * margin, bounds[1, 2] + margin)
    ax.set_box_aspect(np.maximum(span, 0.35 * np.max(span)))
    ax.view_init(elev=24.0, azim=-58.0)
    ax.set_axis_off()
    ax.set_title(
        f"Pose {pose.pose_id} | Boden: {pose.floor_contact_type}, Wand: {pose.wall_contact_type}",
        fontsize=8,
        pad=1,
    )


def render_pose_sheets(
    mesh_path: str | Path,
    output_dir: str | Path,
    *,
    poses_per_sheet: int = 24,
    columns: int = 6,
    dpi: int = 180,
    pose_ids: Iterable[int] | None = None,
) -> tuple[RenderedSheet, ...]:
    """Render the theoretical catalog as grouped, high-resolution PNG sheets."""

    if poses_per_sheet <= 0 or columns <= 0 or dpi <= 0:
        raise ValueError("poses_per_sheet, columns and dpi must be positive.")

    catalog: PoseCatalog = build_pose_catalog(mesh_path)
    selected_ids = set(pose_ids) if pose_ids is not None else None
    poses = [
        pose for pose in catalog.poses if selected_ids is None or pose.pose_id in selected_ids
    ]
    if selected_ids is not None:
        missing = selected_ids - {pose.pose_id for pose in poses}
        if missing:
            raise ValueError(f"Unknown pose ids: {sorted(missing)}")

    mesh = load_solid_mesh(mesh_path)
    hull = mesh.convex_hull
    center_mass = np.asarray(mesh.center_mass, dtype=float)
    mesh_vertices_centered = np.asarray(mesh.vertices, dtype=float).copy() - center_mass
    hull_vertices_centered = np.asarray(hull.vertices, dtype=float).copy() - center_mass
    mesh_faces = np.asarray(mesh.faces, dtype=int).copy()

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rendered: list[RenderedSheet] = []

    groups: dict[str, list[ContactPose]] = {}
    for pose in poses:
        groups.setdefault(_contact_group(pose), []).append(pose)

    for group_name, group_poses in sorted(groups.items()):
        for page_index, page_poses in enumerate(_chunks(group_poses, poses_per_sheet), start=1):
            rows = math.ceil(len(page_poses) / columns)
            figure = plt.figure(figsize=(columns * 3.1, rows * 2.8), facecolor="white")
            figure.suptitle(
                "Df1a: theoretische Boden-Wand-Kontaktlagen\n"
                f"{group_name} | Seite {page_index}",
                fontsize=14,
            )
            for plot_index, pose in enumerate(page_poses, start=1):
                axis = figure.add_subplot(rows, columns, plot_index, projection="3d")
                _draw_pose(
                    axis,
                    pose,
                    mesh_vertices_centered,
                    mesh_faces,
                    hull_vertices_centered,
                )

            figure.text(
                0.01,
                0.008,
                "Gruen: Bodenkontakt | Orange: Wandkontakt | Rot: gemeinsame Eckkontakte "
                "| Ecklinie = X-Achse, +X bergab",
                fontsize=9,
            )
            figure.tight_layout(rect=(0.0, 0.025, 1.0, 0.95))
            filename = f"Df1a_{group_name}_page-{page_index:02d}.png"
            path = destination / filename
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(figure)
            rendered.append(
                RenderedSheet(
                    path=path,
                    pose_ids=tuple(pose.pose_id for pose in page_poses),
                    contact_group=group_name,
                )
            )

    return tuple(rendered)
