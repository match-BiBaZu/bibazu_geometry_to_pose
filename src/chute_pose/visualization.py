"""Technical rendering of theoretical floor-wall contact poses."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Mapping

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
    floor = pose.floor_contact_topology.replace("+", "-plus-")
    wall = pose.wall_contact_topology.replace("+", "-plus-")
    return f"floor-{floor}_wall-{wall}"


def _contact_label_de(topology: str) -> str:
    exact = {
        "point": "Punkt",
        "2-point": "2-Punkt",
        "3-point": "3-Punkt",
        "edge": "Kante",
        "edge+point": "Kante + Punkt",
        "face": "Flaeche",
        "face+point": "Flaeche + Punkt",
    }
    if topology in exact:
        return exact[topology]
    return topology.replace("face", "Flaeche").replace("edge", "Kante").replace(
        "point", "Punkt"
    ).replace("+", " + ")


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
        Poly3DCollection(
            floor,
            facecolor="#c9d1d9",
            edgecolor="#7d8590",
            alpha=0.22,
            zorder=1,
        )
    )
    ax.add_collection3d(
        Poly3DCollection(
            wall,
            facecolor="#e3c9a8",
            edgecolor="#9a7548",
            alpha=0.20,
            zorder=1,
        )
    )
    ax.plot(
        [x0, x1],
        [0.0, 0.0],
        [0.0, 0.0],
        color="#24292f",
        linewidth=1.8,
        zorder=2,
    )


def _draw_contact_set(
    ax,
    vertices: np.ndarray,
    vertex_indices: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    *,
    color: str,
    marker: str,
) -> None:
    """Draw all full-mesh contact points and truly connected mesh edges."""

    if len(vertex_indices):
        points = vertices[vertex_indices]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=color,
            edgecolors="white",
            linewidths=0.8,
            marker=marker,
            s=58,
            depthshade=False,
            zorder=20,
        )
    selected = set(int(value) for value in vertex_indices)
    for first, second in edges:
        if first not in selected or second not in selected:
            continue
        segment = vertices[[first, second]]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=color,
            linewidth=3.2,
            solid_capstyle="round",
            zorder=19,
        )


def _draw_pose(
    ax,
    pose: ContactPose,
    mesh_vertices_centered: np.ndarray,
    mesh_faces: np.ndarray,
    pose_label: str | None = None,
) -> None:
    rotation = np.asarray(pose.rotation_chute_from_part, dtype=float)
    translation = np.asarray(pose.translation_to_corner_mm, dtype=float)
    mesh_vertices = (rotation @ mesh_vertices_centered.T).T + translation
    bounds = np.vstack([mesh_vertices.min(axis=0), mesh_vertices.max(axis=0)])
    span = np.maximum(bounds[1] - bounds[0], 1e-9)
    margin = 0.12 * float(np.max(span))
    _draw_reference_surfaces(ax, bounds, margin)

    triangles = mesh_vertices[mesh_faces]
    part = Poly3DCollection(
        triangles,
        facecolor="#5b9bd5",
        edgecolor="#244a68",
        linewidth=0.35,
        alpha=0.62,
        zorder=3,
    )
    ax.add_collection3d(part)

    floor_indices = np.asarray(pose.floor_mesh_contact_vertex_indices, dtype=int)
    wall_indices = np.asarray(pose.wall_mesh_contact_vertex_indices, dtype=int)
    seam_indices = np.intersect1d(floor_indices, wall_indices)
    floor_only = np.setdiff1d(floor_indices, seam_indices)
    wall_only = np.setdiff1d(wall_indices, seam_indices)

    _draw_contact_set(
        ax,
        mesh_vertices,
        floor_only,
        pose.floor_mesh_contact_edges,
        color="#10a64a",
        marker="o",
    )
    _draw_contact_set(
        ax,
        mesh_vertices,
        wall_only,
        pose.wall_mesh_contact_edges,
        color="#f07818",
        marker="D",
    )
    _draw_contact_set(
        ax,
        mesh_vertices,
        seam_indices,
        (),
        color="#d62728",
        marker="s",
    )

    ax.set_xlim(bounds[0, 0] - margin, bounds[1, 0] + margin)
    ax.set_ylim(-0.05 * margin, bounds[1, 1] + margin)
    ax.set_zlim(-0.05 * margin, bounds[1, 2] + margin)
    ax.set_box_aspect(np.maximum(span, 0.35 * np.max(span)))
    ax.view_init(elev=24.0, azim=-58.0)
    ax.set_axis_off()
    ax.set_title(
        f"{pose_label or f'Pose {pose.pose_id}'}\n"
        f"Boden: {_contact_label_de(pose.floor_contact_topology)}, "
        f"Wand: {_contact_label_de(pose.wall_contact_topology)}",
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
    sheet_title: str = "Df1a: theoretische Boden-Wand-Kontaktlagen",
    filename_prefix: str = "Df1a",
    pose_labels: Mapping[int, str] | None = None,
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
    center_mass = np.asarray(mesh.center_mass, dtype=float)
    mesh_vertices_centered = np.asarray(mesh.vertices, dtype=float).copy() - center_mass
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
            first_pose = page_poses[0]
            figure.suptitle(
                f"{sheet_title}\n"
                f"Boden: {_contact_label_de(first_pose.floor_contact_topology)} | "
                f"Wand: {_contact_label_de(first_pose.wall_contact_topology)} | "
                f"Seite {page_index}",
                fontsize=14,
            )
            for plot_index, pose in enumerate(page_poses, start=1):
                axis = figure.add_subplot(
                    rows,
                    columns,
                    plot_index,
                    projection="3d",
                    computed_zorder=False,
                )
                _draw_pose(
                    axis,
                    pose,
                    mesh_vertices_centered,
                    mesh_faces,
                    pose_label=(pose_labels or {}).get(pose.pose_id),
                )

            figure.text(
                0.01,
                0.008,
                "Gruen/Kreis: Bodenpunkte und echte Netzkanten | "
                "Orange/Raute: Wandpunkte und echte Netzkanten | Rot/Quadrat: Eckkontakt "
                "| Ecklinie = X-Achse, +X bergab",
                fontsize=9,
            )
            figure.tight_layout(rect=(0.0, 0.025, 1.0, 0.95))
            filename = f"{filename_prefix}_{group_name}_page-{page_index:02d}.png"
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
