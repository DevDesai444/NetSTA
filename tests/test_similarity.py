"""Circuit similarity index + search tests."""

import numpy as np
import pytest


@pytest.fixture
def circuit_index(tmp_chroma_dir, untrained_model, small_dataset):
    """Per-test CircuitIndex backed by a tmp Chroma persist dir."""
    from timingnet.similarity.circuit_index import CircuitIndex
    idx = CircuitIndex(
        model=untrained_model,
        persist_dir=tmp_chroma_dir,
        collection_name="test_similarity_index",
        device="cpu",
    )
    idx.build(small_dataset, force=True, verbose=False)
    return idx


def test_index_count_matches_dataset_size(circuit_index, small_dataset):
    assert circuit_index.count() == len(small_dataset)


def test_index_metadata_has_required_fields(circuit_index):
    bundle = circuit_index.get_all()
    assert len(bundle["metadatas"]) == circuit_index.count()
    required = {
        "circuit_type", "num_gates", "max_congestion",
        "critical_path_length", "avg_slack",
        "avg_gbw_score", "avg_parasitic", "_schema",
    }
    for meta in bundle["metadatas"]:
        missing = required - meta.keys()
        assert not missing, f"row missing keys: {missing}"


def test_find_similar_self_returns_self_first(
    circuit_index, untrained_model, small_dataset,
):
    from timingnet.similarity.search import find_similar
    anchor = small_dataset[0]
    hits = find_similar(anchor, untrained_model, circuit_index, top_k=3)
    assert hits, "expected at least one hit"
    # Top hit should be a near-perfect match (cosine ≈ 1.0).
    assert hits[0]["similarity"] is not None
    assert hits[0]["similarity"] > 0.999


def test_compare_identical_circuits_is_one(untrained_model, small_dataset):
    from timingnet.similarity.search import compare_circuits
    data = small_dataset[0]
    result = compare_circuits(data, data, untrained_model)
    assert abs(result["cosine_similarity"] - 1.0) < 1e-4
    # Per-metric deltas must be zero on identical inputs.
    for key, delta in result["deltas"].items():
        assert delta == 0, f"non-zero delta on identical inputs for {key}"


def test_compare_different_circuits_returns_valid_similarity(
    untrained_model, small_dataset,
):
    from timingnet.similarity.search import compare_circuits
    a, b = small_dataset[0], small_dataset[1]
    result = compare_circuits(a, b, untrained_model)
    sim = result["cosine_similarity"]
    assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6


def test_find_by_property_metadata_filter(circuit_index):
    """num_gates filter must return only matching metadata."""
    from timingnet.similarity.search import find_by_property
    target_specs = {"num_gates": {"min": 1, "max": 1000}}
    hits = find_by_property(target_specs, circuit_index, top_k=10)
    assert hits, "filter should match at least one row"
    for h in hits:
        n = h["metadata"]["num_gates"]
        assert 1 <= n <= 1000


def test_index_schema_tag_consistent(circuit_index):
    from timingnet.similarity.circuit_index import INDEX_SCHEMA_TAG
    assert circuit_index.schema_tag() == INDEX_SCHEMA_TAG
    assert circuit_index.needs_rebuild() is False
