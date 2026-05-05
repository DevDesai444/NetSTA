"""NetSTA RAG subpackage: knowledge base, vector store, parser, advisor."""

from .circuit_parser import (
    CircuitSpec,
    DeviceSpec,
    generate_from_spec,
    parse_to_spec,
    run_prediction,
)
from .design_advisor import Bottleneck, DesignReport, advise
from .embeddings import KnowledgeStore
from .knowledge_base import (
    KnowledgeChunk,
    build_chunks,
    chunk_text,
    load_knowledge,
)

__all__ = [
    "CircuitSpec", "DeviceSpec", "DesignReport", "Bottleneck",
    "KnowledgeChunk", "KnowledgeStore",
    "advise", "build_chunks", "chunk_text", "generate_from_spec",
    "load_knowledge", "parse_to_spec", "run_prediction",
]
