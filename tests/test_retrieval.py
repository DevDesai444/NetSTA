"""Tests for the hybrid retrieval layer (FAISS + knowledge graph + fusion)."""

from netsta.retrieval import FaissTextIndex, HybridRetriever, KnowledgeGraph


def test_faiss_index_builds_and_searches():
    idx = FaissTextIndex()
    st = idx.stats()
    assert st["num_chunks"] > 0
    # backend is faiss-* when deps are present, else keyword — both must search.
    hits = idx.search_text("setup timing violation on the critical path", top_k=3)
    assert len(hits) > 0
    assert all(isinstance(h, str) for h in hits)


def test_knowledge_graph_queries():
    kg = KnowledgeGraph()
    st = kg.stats()
    assert st["nodes"] > 0 and st["relationships"] > 0

    fixes = [f.obj for f in kg.fixes_for_violation("setup_violation")]
    assert "gate_upsizing" in fixes and "buffer_insertion" in fixes

    outcomes = [f.obj for f in kg.outcomes_for_fix("gate_upsizing")]
    assert "slack_improved" in outcomes

    conflicts = [f.obj for f in kg.conflicts_for_fix("gate_upsizing")]
    assert "cell_spreading" in conflicts  # the cross-task conflict edge


def test_knowledge_graph_process_node_filter():
    kg = KnowledgeGraph()
    # guard_ring is 130nm-only; should not appear for a 45nm query.
    fixes_45 = [f.obj for f in kg.fixes_for_violation("parasitic_coupling", process_node="45nm")]
    fixes_130 = [f.obj for f in kg.fixes_for_violation("parasitic_coupling", process_node="130nm")]
    assert "guard_ring" in fixes_130
    assert "guard_ring" not in fixes_45


def test_hybrid_retrieve_fuses_sources():
    hr = HybridRetriever()
    ctx = hr.retrieve(
        "setup violation on the critical path",
        topology="pipelined_datapath",
        violation="setup_violation",
        process_node="45nm",
    )
    assert ctx.facts, "expected knowledge-graph facts"
    assert "gate_upsizing" in ctx.fix_names()
    prompt = ctx.as_prompt()
    assert "Knowledge-graph facts" in prompt
