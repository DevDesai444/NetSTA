"""Circuit similarity subpackage: graph-embedding index + retrieval."""

from .circuit_index import CircuitIndex
from .search import compare_circuits, find_by_property, find_similar

__all__ = ["CircuitIndex", "compare_circuits", "find_by_property", "find_similar"]
