from dataclasses import replace
from pathlib import Path
import json

import pytest

from chute_pose.generate import GenerationConfig, generate_one, robust_subset
from chute_pose.roadmap import build_pose_roadmap, PoseRoadmap
from chute_pose.metrics import metric_lines

PART = Path(__file__).resolve().parents[1] / "Werkstücke_STL_grob" / "Df1a.STL"


@pytest.fixture(scope="module")
def roadmap():
    return build_pose_roadmap(PART, alpha_deg=45, beta_deg=0, robust_barrier_threshold_mm=.2,
                              classical_methods=("standard_csa", "crsa"))


def test_classical_scores_do_not_change_rocking_ids(roadmap):
    baseline = build_pose_roadmap(PART, alpha_deg=45, beta_deg=0, robust_barrier_threshold_mm=.2)
    assert [(n.node_id, n.pose_ids, n.kind) for n in roadmap.nodes] == [(n.node_id, n.pose_ids, n.kind) for n in baseline.nodes]
    for method in ("standard_csa", "crsa"):
        shares = [n.classical_metrics[method]["normalized_share"] for n in roadmap.nodes]
        assert sum(v for v in shares if v is not None) == pytest.approx(1)
    loaded = PoseRoadmap.from_dict(json.loads(json.dumps(roadmap.to_dict())))
    assert loaded.classical_metadata == json.loads(json.dumps(roadmap.classical_metadata))
    assert loaded.nodes[0].classical_metrics == roadmap.nodes[0].classical_metrics


def test_config_requires_explicit_classical_threshold():
    with pytest.raises(ValueError, match="cutoff"):
        GenerationConfig("out", methods=("crsa",), ranking="crsa", classifier="crsa").validate()
    with pytest.raises(ValueError): GenerationConfig("out", outputs=()).validate()
    with pytest.raises(ValueError): GenerationConfig("out", beta=float("nan")).validate()


def test_only_requested_files_and_robust_nodes_are_exported(tmp_path, monkeypatch, roadmap):
    monkeypatch.setattr("chute_pose.generate.build_pose_roadmap", lambda *a, **k: roadmap)
    config = GenerationConfig(str(tmp_path), outputs=("yaml",), methods=("rocking", "standard_csa", "crsa"))
    status, paths = generate_one(PART, config)
    assert status == "completed"
    assert len(paths) == 1 and Path(paths[0]).suffix == ".yaml"
    assert list(tmp_path.rglob("*.png")) == []
    assert "stability: metastable" not in Path(paths[0]).read_text()
    assert generate_one(PART, config)[0] == "skipped"


def test_robust_filter_keeps_valid_edges_and_metric_identity(roadmap):
    filtered = robust_subset(roadmap)
    ids = {n.node_id for n in filtered.nodes}
    catalogue = {pid for n in filtered.nodes for pid in n.pose_ids}
    assert all(e.source in ids and e.target in ids and set(e.settling_pose_ids) <= catalogue for e in filtered.edges)
    assert all(n.kind == "robust" for n in filtered.nodes)
    lines = metric_lines(roadmap, ("rocking", "standard_csa", "crsa"))
    assert all(len(v) == 3 for v in lines.values())
    assert all("rank" in text or "N/A" in text for values in lines.values() for text, color in values)


def test_new_ranking_keeps_ids_and_requires_selected_threshold(roadmap):
    from chute_pose.roadmap import _relabel_roadmap
    # IDs of an already rocking-numbered roadmap stay the same across ordering.
    nodes, edges, _ = _relabel_roadmap(list(roadmap.nodes), list(roadmap.edges), (), pose_ranking_method="standard_csa")
    assert {n.original_catalog_pose_id: n.node_id for n in nodes} == {n.original_catalog_pose_id: n.node_id for n in roadmap.nodes}
    scores = [n.classical_metrics["standard_csa"]["raw_score"] for n in nodes if n.classical_metrics["standard_csa"]["raw_score"] is not None]
    assert scores == sorted(scores, reverse=True)
