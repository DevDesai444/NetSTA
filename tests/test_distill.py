"""Tests for the distillation pipeline (roles, scenarios, teacher worker)."""

import json
import os
import tempfile
import threading
from unittest.mock import patch

import pytest

from netsta.distill.roles import ROLES
from netsta.distill.teacher import (
    FALLBACK_MODELS, KeyPool, _EXHAUSTED_MODELS,
    _extract_json, _mark_exhausted, _next_model_after,
    _scenario_to_user_prompt, load_keys,
)


@pytest.fixture(autouse=True)
def _reset_exhausted():
    """Each test starts with a clean exhausted-models set."""
    _EXHAUSTED_MODELS.clear()
    yield
    _EXHAUSTED_MODELS.clear()


def test_four_roles_defined():
    assert set(ROLES) == {"supervisor", "timing", "drc", "optimization"}
    for role in ROLES.values():
        assert role.system_prompt.strip()
        assert role.output_schema_hint.strip()


def test_extract_json_handles_fenced_output():
    assert _extract_json("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert _extract_json("Here's the JSON:\n{\"a\": 2}") == {"a": 2}
    assert _extract_json("not json at all") is None


def test_scenario_user_prompt_mentions_json_for_response_format():
    """gpt-oss models reject json_object mode unless the message contains 'json'."""
    role = ROLES["timing"]
    sc = {
        "circuit_name": "demo", "topology": "pipelined_datapath",
        "process_node": "45nm", "num_nodes": 100, "num_edges": 200,
        "predictions_summary": {},
        "bottlenecks": [],
        "retrieved_facts": [], "retrieved_text": [], "peer_findings": [],
    }
    prompt = _scenario_to_user_prompt(role, sc)
    assert "json" in prompt.lower()


def test_key_pool_round_robin():
    pool = KeyPool(["k1", "k2", "k3"])
    states = [pool.acquire() for _ in range(6)]
    seen = {s.key for s in states}
    assert seen == {"k1", "k2", "k3"}
    for s in states:
        pool.release(s)


def test_key_pool_cooldown_skips_unavailable():
    pool = KeyPool(["k1", "k2"])
    s1 = pool.acquire()
    pool.release(s1, cool_for=10.0)  # cool k1
    # next acquisition should pick k2 since k1 is cooling.
    s2 = pool.acquire(max_wait_s=1.0)
    assert s2.key == "k2"
    pool.release(s2)


def test_next_model_promotes_to_next_in_ladder():
    nxt = _next_model_after("openai/gpt-oss-120b", FALLBACK_MODELS)
    assert nxt == "openai/gpt-oss-20b"


def test_next_model_skips_exhausted():
    _mark_exhausted("openai/gpt-oss-20b")
    nxt = _next_model_after("openai/gpt-oss-120b", FALLBACK_MODELS)
    assert nxt == "llama-3.3-70b-versatile"


def test_next_model_returns_none_when_all_exhausted():
    for m in FALLBACK_MODELS:
        _mark_exhausted(m)
    assert _next_model_after("openai/gpt-oss-120b", FALLBACK_MODELS) is None


def test_load_keys_reads_numbered_env(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    for i in range(1, 6):
        monkeypatch.delenv(f"GROQ_API_KEY_{i}", raising=False)
    monkeypatch.setenv("GROQ_API_KEY_1", "k1")
    monkeypatch.setenv("GROQ_API_KEY_2", "k2")
    keys = load_keys()
    assert keys == ["k1", "k2"]
