"""
Hybrid retrieval for the NetSTA design-advisory agents.

Three complementary sources, each capturing a signal the others miss:

  - FaissTextIndex   semantic search over chunked EDA text (timing-closure
                     guides, DRC notes, analog optimisation tips).
  - KnowledgeGraph   structured topology -> violation -> fix -> outcome
                     relationships (real Neo4j when NEO4J_URI is set, else an
                     in-memory NetworkX graph with the same query surface).
  - CircuitIndex     k-NN over the GNN's own graph embeddings (lives in
                     netsta.similarity; pulled in by HybridRetriever).

`HybridRetriever` fuses the first two (and, when a built CircuitIndex is
handed in, the third) behind one `retrieve()` call.
"""

from .faiss_index import FaissTextIndex
from .knowledge_graph import KnowledgeGraph, GraphFact
from .hybrid import HybridRetriever, FusedContext

__all__ = [
    "FaissTextIndex",
    "KnowledgeGraph",
    "GraphFact",
    "HybridRetriever",
    "FusedContext",
]
