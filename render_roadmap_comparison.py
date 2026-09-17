"""Add CWSA measurements to an existing roadmap and render a comparison."""

import argparse
from dataclasses import replace
from pathlib import Path

from chute_pose.csa import analyze_contact_wrench_solid_angle
from chute_pose.roadmap import load_roadmap_json, render_pose_roadmap, save_roadmap_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roadmap", type=Path)
    parser.add_argument("--friction-policy", choices=("zero", "range"), default="zero")
    args = parser.parse_args()
    roadmap = load_roadmap_json(args.roadmap)
    print("Computing modified CSA (CWSA) for the existing pose classes...", flush=True)
    print("Existing robust/metastable labels are preserved; rebuild the roadmap to change classification.", flush=True)
    analysis = analyze_contact_wrench_solid_angle(
        roadmap.source,
        pose_ids=tuple(pose_id for node in roadmap.nodes for pose_id in node.pose_ids),
        alpha_deg=roadmap.alpha_deg,
        beta_deg=roadmap.beta_deg,
        cap_half_angle_deg=roadmap.csa_cap_half_angle_deg,
        direction_samples=roadmap.csa_direction_samples,
        friction_policy=args.friction_policy,
    )
    by_id = {value.pose_id: value for value in analysis.poses}
    nodes = []
    for node in roadmap.nodes:
        values = [by_id[pose_id] for pose_id in node.pose_ids]
        applicable = all(value.applicable for value in values)
        nodes.append(replace(
            node,
            csa_applicable=applicable,
            csa_stability_index=min(value.score for value in values) if applicable else None,
            csa_feasible_fraction=(
                min(value.feasible_fraction for value in values) if applicable else None
            ),
            csa_feasible_solid_angle_sr=(
                min(value.feasible_solid_angle_sr for value in values) if applicable else None
            ),
            csa_angular_clearance_deg=(
                min(value.angular_clearance_deg for value in values) if applicable else None
            ),
        ))
    comparison = replace(
        roadmap, nodes=tuple(nodes), csa_load_model=analysis.to_dict()["load_model"]
    )
    # Measurement enrichment must preserve the original pose and route identities.
    assert comparison.edges == roadmap.edges
    assert [(n.node_id, n.pose_ids, n.kind) for n in comparison.nodes] == [
        (n.node_id, n.pose_ids, n.kind) for n in roadmap.nodes
    ]
    output_stem = args.roadmap.with_name(args.roadmap.stem + "_comparison")
    print(save_roadmap_json(comparison, output_stem.with_suffix(".json")), flush=True)
    for stable_only in (False, True):
        stem = output_stem.with_name(output_stem.name + "_stable") if stable_only else output_stem
        paths = render_pose_roadmap(
            comparison, stem, stable_only=stable_only, show_csa_ranks=True
        )
        for path in paths:
            print(path, flush=True)
    for node in comparison.nodes:
        print(
            f"Pose {node.node_id}: rocking={node.rocking_barrier_mm:.6f} mm; "
            f"CWSA={node.csa_stability_index}",
            flush=True,
        )


if __name__ == "__main__":
    main()
