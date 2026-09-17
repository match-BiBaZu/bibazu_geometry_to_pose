"""Compare two measured roadmaps, aligning rows by physical pose identity."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chute_pose.roadmap import _roadmap_metric_ranks, load_roadmap_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output-stem", required=True, type=Path)
    args = parser.parse_args()
    first, second = (load_roadmap_json(path) for path in (args.first, args.second))
    if Path(first.source).stem != Path(second.source).stem:
        raise ValueError("Compare two orientations of the same workpiece.")
    by_id = {node.original_catalog_pose_id: node for node in second.nodes}
    if {node.original_catalog_pose_id for node in first.nodes} != set(by_id):
        raise ValueError("The roadmaps have different representative pose sets.")
    pairs = [(node, by_id[node.original_catalog_pose_id]) for node in first.nodes]
    if any(node.csa_stability_index is None for pair in pairs for node in pair):
        raise ValueError("Both maps must contain applicable CWSA scores for every pose.")
    output = args.output_stem.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    first_ranks = _roadmap_metric_ranks(first, "rocking_barrier_mm")
    second_ranks = _roadmap_metric_ranks(second, "rocking_barrier_mm")
    labels = [f"{first_ranks[old.node_id]} ({second_ranks[new.node_id]})" for old, new in pairs]
    y = np.arange(len(pairs))
    figure, axes = plt.subplots(1, 2, figsize=(14, 11), sharey=True)
    tilt_labels = [f"X={r.alpha_deg:g}°, Y={r.beta_deg:g}°" for r in (first, second)]
    for axis, attr, title in zip(
        axes,
        ("rocking_barrier_mm", "csa_stability_index"),
        ("Rocking barrier (mm)", "Modified CSA / CWSA (dimensionless)"),
        strict=True,
    ):
        old_values = np.array([getattr(old, attr) for old, _ in pairs])
        new_values = np.array([getattr(new, attr) for _, new in pairs])
        axis.hlines(y, np.minimum(old_values, new_values), np.maximum(old_values, new_values),
                    color="#64748b", lw=1.5, zorder=2)
        axis.scatter(old_values, y - 0.10, marker="s", s=42, color="#2563eb",
                     label=tilt_labels[0], zorder=3)
        axis.scatter(new_values, y + 0.10, marker="o", s=38, color="#ea580c",
                     label=tilt_labels[1], zorder=3)
        axis.set_title(title, fontsize=13, pad=13)
        axis.set_xlabel("Higher value = greater reserve in this metric")
        axis.grid(axis="x", color="#e2e8f0")
        axis.set_axisbelow(True)
        axis.set_xlim(left=-0.02 * max(float(max(new_values)), float(max(old_values)), 0.01))
        axis.legend(loc="lower right", fontsize=10)
        for row, (old, _) in enumerate(pairs):
            if old.node_id in (6, 12):
                axis.axhspan(row - 0.45, row + 0.45, color="#fef3c7", zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_ylabel(f"Rocking rank: Y{first.beta_deg:g}° (Y{second.beta_deg:g}°); ties share rank")
    figure.suptitle(f"{Path(first.source).stem}: rocking barrier and CWSA across chute tilts",
                   fontsize=17, y=0.98)
    figure.text(0.5, 0.94, "Tied barriers within 0.000001 mm share a rank. Rows match physical poses; highlights: original map IDs 6 and 12.",
                ha="center", fontsize=10)
    figure.text(0.5, 0.025,
                f"Saved rocking cutoffs: Y{first.beta_deg:g}° = {first.robust_barrier_threshold_mm:g} mm; "
                f"Y{second.beta_deg:g}° = {second.robust_barrier_threshold_mm:g} mm. "
                "Cutoffs change classification, not the measured values.",
                ha="center", fontsize=10)
    figure.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.10, wspace=0.14)
    for extension in (".png", ".svg"):
        path = output.with_suffix(extension)
        figure.savefig(path, dpi=200)
        print(path)
    plt.close(figure)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["catalog_pose_id", "y20_pose_id", "y0_pose_id", "y20_rocking_mm",
                         "y0_rocking_mm", "y20_cwsa", "y0_cwsa", "y20_rocking_rank", "y0_rocking_rank"])
        for old, new in pairs:
            writer.writerow([old.original_catalog_pose_id, old.node_id, new.node_id,
                             old.rocking_barrier_mm, new.rocking_barrier_mm,
                             old.csa_stability_index, new.csa_stability_index,
                             first_ranks[old.node_id], second_ranks[new.node_id]])
    print(csv_path)


if __name__ == "__main__":
    main()
