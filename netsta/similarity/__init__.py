"""netsta.similarity — shim re-exporting timingnet.similarity."""

from timingnet.similarity import (  # noqa: F401
    CircuitIndex,
    compare_circuits,
    find_by_property,
    find_similar,
)
from timingnet.similarity.circuit_index import embed_circuit  # noqa: F401

__all__ = [
    "CircuitIndex", "compare_circuits", "embed_circuit",
    "find_by_property", "find_similar",
]
