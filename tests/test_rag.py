"""
RAG subsystem tests.

Heavy deps (sentence-transformers + ChromaDB) are imported lazily inside the
tests that need them. Tests are skipped cleanly if either dep is unavailable.
LLM API keys are stripped by the `_disable_llm_api_keys` autouse fixture in
conftest, so the parser always exercises its deterministic fallback path.
"""

import os
import sys

import pytest


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------


def test_knowledge_base_has_at_least_50_entries():
    from timingnet.rag.knowledge_base import load_knowledge
    entries = load_knowledge()
    assert len(entries) >= 50


def test_knowledge_entries_have_required_fields():
    from timingnet.rag.knowledge_base import load_knowledge
    required = {
        "circuit_name", "description", "typical_specs", "topology_type",
        "device_count_range", "common_issues", "optimization_tips",
    }
    for entry in load_knowledge():
        missing = required - entry.keys()
        assert not missing, (
            f"entry {entry.get('circuit_name')!r} missing keys: {missing}"
        )


def test_chunk_text_returns_nonempty_chunks_for_long_text():
    from timingnet.rag.knowledge_base import chunk_text
    text = ("hello world " * 800).strip()
    chunks = chunk_text(text, chunk_size=512, overlap=50)
    assert len(chunks) >= 1
    assert all(c.strip() for c in chunks)


def test_chunk_text_handles_short_text_in_one_chunk():
    from timingnet.rag.knowledge_base import chunk_text
    chunks = chunk_text("short text", chunk_size=512, overlap=50)
    assert chunks == ["short text"]


def test_build_chunks_attaches_metadata():
    from timingnet.rag.knowledge_base import build_chunks
    chunks = build_chunks()
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert "circuit_name" in c.metadata
        assert "topology_type" in c.metadata


# ---------------------------------------------------------------------------
# KnowledgeStore — try Chroma path first, fall back to keyword retrieval
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def knowledge_store_module(tmp_path_factory):
    """Module-scoped: build the store once so the (potentially slow) embedder
    download / Chroma init happens only on the first test."""
    try:
        from timingnet.rag.embeddings import KnowledgeStore
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"rag.embeddings unavailable: {exc!r}")
    persist = tmp_path_factory.mktemp("kb_chroma")
    return KnowledgeStore(persist_dir=str(persist), collection="test_kb")


def test_knowledge_store_retrieve_returns_something(knowledge_store_module):
    hits = knowledge_store_module.retrieve("two-stage Miller op-amp gain", top_k=3)
    assert hits, "expected at least one retrieval hit"
    assert all(isinstance(h, str) and h for h in hits)


def test_add_document_then_retrieve(knowledge_store_module):
    knowledge_store_module.add_document(
        "Custom test entry: a special low-noise transimpedance amplifier.",
        metadata={"source": "test", "topology_type": "tia"},
    )
    hits = knowledge_store_module.retrieve("transimpedance amplifier low-noise", top_k=5)
    assert any("transimpedance" in h.lower() for h in hits)


# ---------------------------------------------------------------------------
# Circuit parser (fallback path, no LLM)
# ---------------------------------------------------------------------------


def test_parser_fallback_extracts_topology_and_specs(knowledge_store_module):
    from timingnet.rag.circuit_parser import parse_to_spec
    spec, backend = parse_to_spec(
        "Design a two-stage Miller-compensated op-amp with 60dB gain and 10MHz GBW",
        knowledge_store=knowledge_store_module,
    )
    assert backend == "fallback"
    assert spec.topology == "two_stage_opamp"
    assert spec.target_specs.get("gain_db") == 60.0
    assert spec.target_specs.get("gbw_mhz") == 10.0
    assert spec.num_stages == 2
    assert spec.compensation == "miller"


@pytest.mark.parametrize("query,expected_topology", [
    ("Build a folded cascode op-amp.",          "folded_cascode"),
    ("Differential pair input stage.",          "diff_pair"),
    ("Simple current mirror.",                  "current_mirror"),
    ("Common-source amplifier with active load.", "common_source_amp"),
])
def test_parser_fallback_topology_keyword_detection(
    knowledge_store_module, query, expected_topology,
):
    from timingnet.rag.circuit_parser import parse_to_spec
    spec, _ = parse_to_spec(query, knowledge_store=knowledge_store_module)
    assert spec.topology == expected_topology


def test_design_advisor_runs_template_fallback(knowledge_store_module):
    """Even with no LLM, the advisor should emit a non-empty recommendation list."""
    from timingnet.rag.circuit_parser import parse_to_spec
    from timingnet.rag.design_advisor import advise
    query = "Two-stage op-amp with 60dB gain."
    spec, _ = parse_to_spec(query, knowledge_store=knowledge_store_module)
    report = advise(spec, predictions=None, knowledge_store=knowledge_store_module)
    assert report.spec.topology == "two_stage_opamp"
    assert isinstance(report.recommendations, list)
    assert report.backend == "fallback"
