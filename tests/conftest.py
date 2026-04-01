"""
Shared pytest fixtures for the NetSTA test suite.

Fixtures are intentionally tiny (small graphs, low hidden dims) so the whole
suite finishes in seconds even on CI runners with no GPU.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

# Make sure project root is on sys.path when pytest is invoked from anywhere.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return _PROJECT_ROOT


# ---------------------------------------------------------------------------
# Circuits
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_digital_circuit():
    """Small deterministic digital circuit."""
    from timingnet.circuit_gen import generate_circuit
    return generate_circuit(
        num_inputs=4, num_gates=10, num_outputs=2,
        seed=42, name="test_digital",
    )


@pytest.fixture
def sample_analog_circuit():
    """Differential-pair analog circuit (has matched-pair symmetry groups)."""
    from timingnet.analog_circuit_gen import generate_analog_circuit
    return generate_analog_circuit(seed=42, topology="diff_pair")


@pytest.fixture
def sample_pyg_data(sample_digital_circuit):
    from timingnet.graph_builder import circuit_to_pyg
    from timingnet.sta import run_sta
    return circuit_to_pyg(sample_digital_circuit, run_sta(sample_digital_circuit))


@pytest.fixture
def small_dataset():
    """Three small in-memory digital circuits as PyG Data objects."""
    from timingnet.circuit_gen import generate_circuit
    from timingnet.graph_builder import circuit_to_pyg
    from timingnet.sta import run_sta
    out = []
    for seed in (1, 2, 3):
        c = generate_circuit(
            num_inputs=3, num_gates=6, num_outputs=2,
            seed=seed, name=f"ds_{seed}",
        )
        out.append(circuit_to_pyg(c, run_sta(c)))
    return out


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_config():
    """Fast-to-construct config used by every forward-pass test."""
    from timingnet.config import NetSTAConfig
    return NetSTAConfig(
        node_feature_dim=31,
        edge_feature_dim=5,
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        active_tasks=(
            "slack", "critical_path", "congestion", "drc", "analog_performance",
        ),
        task_weights={
            "slack": 0.2, "critical_path": 0.2, "congestion": 0.2,
            "drc": 0.2, "analog_performance": 0.2,
        },
    )


@pytest.fixture
def untrained_model(tiny_config):
    """Fresh NetSTAModel — random init, no checkpoint dependency."""
    from timingnet.model import NetSTAModel
    torch.manual_seed(0)
    return NetSTAModel(tiny_config)


# ---------------------------------------------------------------------------
# Persistent storage
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_chroma_dir(tmp_path) -> str:
    """Per-test tmp directory for ChromaDB persistent collections."""
    p = tmp_path / "chroma"
    p.mkdir()
    return str(p)


@pytest.fixture(autouse=True)
def _disable_llm_api_keys(monkeypatch):
    """Force the RAG fallback path during tests — never accidentally hit the
    real OpenAI / Anthropic APIs from CI."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
