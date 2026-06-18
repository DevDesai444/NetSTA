"""Smoke tests for the AutoGen → vLLM client routing.

These tests don't require an actual running vLLM endpoint — they exercise the
client-construction logic and the per-agent LoRA mapping. The full end-to-end
GroupChat is exercised by manual integration tests once vLLM is deployed.
"""

import os
from unittest.mock import patch

import pytest

from netsta.agents.autogen_backend import _model_client


def test_vllm_url_takes_precedence(monkeypatch):
    """When NETSTA_VLLM_URL is set, prefer vLLM over any cloud API key."""
    monkeypatch.setenv("NETSTA_VLLM_URL", "http://fake:8000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-xxx")
    # autogen_ext.models.openai imports lazily; we patch the constructor.
    try:
        from autogen_ext.models.openai import OpenAIChatCompletionClient
    except ImportError:
        pytest.skip("autogen_ext not installed")

    with patch.object(OpenAIChatCompletionClient, "__init__", return_value=None) as init:
        client = _model_client(role_lora="timing")
    assert client is not None
    # The vLLM path passes model=role_lora (vLLM uses the model field for
    # adapter routing).
    kwargs = init.call_args.kwargs
    assert kwargs.get("model") == "timing"
    assert kwargs.get("base_url") == "http://fake:8000/v1"


def test_role_lora_defaults_to_supervisor(monkeypatch):
    monkeypatch.setenv("NETSTA_VLLM_URL", "http://fake:8000/v1")
    try:
        from autogen_ext.models.openai import OpenAIChatCompletionClient
    except ImportError:
        pytest.skip("autogen_ext not installed")
    with patch.object(OpenAIChatCompletionClient, "__init__", return_value=None) as init:
        client = _model_client()  # no role -> supervisor
    assert client is not None
    assert init.call_args.kwargs.get("model") == "supervisor"


def test_falls_through_when_no_creds(monkeypatch):
    monkeypatch.delenv("NETSTA_VLLM_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = _model_client(role_lora="drc")
    assert client is None
