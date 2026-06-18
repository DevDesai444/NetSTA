"""
Hybrid retrieval: fuse FAISS text search, the Neo4j/NetworkX knowledge graph,
and (optionally) the ChromaDB circuit-embedding index into one call.

Each source answers a different question:
  - FAISS      "what does the literature say about this?"   (semantic text)
  - graph      "what fixes resolve this, and what do they cost/conflict with?"
  - ChromaDB   "which previously-analysed circuits look like this one?"

A diagnosis agent calls retrieve() once and gets a FusedContext it can hand to
an LLM (as a grounding prompt) or consume directly in the deterministic path.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .faiss_index import FaissTextIndex
from .knowledge_graph import GraphFact, KnowledgeGraph


@dataclass
class FusedContext:
    """Retrieved evidence from all three sources for one query."""
    query: str
    text: List[str] = field(default_factory=list)
    facts: List[GraphFact] = field(default_factory=list)
    circuits: List[Dict[str, Any]] = field(default_factory=list)

    def as_prompt(self, max_text: int = 4, max_facts: int = 10) -> str:
        """Format the evidence as a grounding block for an LLM."""
        lines: List[str] = []
        if self.facts:
            lines.append("Knowledge-graph facts:")
            lines += [f"  - {f.as_text()}" for f in self.facts[:max_facts]]
        if self.text:
            lines.append("Relevant EDA knowledge:")
            lines += [f"  - {t[:300]}" for t in self.text[:max_text]]
        if self.circuits:
            lines.append("Similar past circuits:")
            for c in self.circuits[:3]:
                meta = c.get("metadata", {})
                sim = c.get("similarity")
                sim_s = f"{sim:.2f}" if isinstance(sim, (int, float)) else "?"
                lines.append(
                    f"  - {meta.get('circuit_name', c.get('id', '?'))} "
                    f"(sim {sim_s}, {meta.get('num_gates', '?')} gates)"
                )
        return "\n".join(lines) if lines else "(no retrieved context)"

    def fix_names(self) -> List[str]:
        """Distinct FixStrategy names surfaced by the graph (RESOLVED_BY edges)."""
        out = []
        for f in self.facts:
            if f.relation == "RESOLVED_BY" and f.obj not in out:
                out.append(f.obj)
        return out


class HybridRetriever:
    def __init__(
        self,
        faiss_index: Optional[FaissTextIndex] = None,
        kg: Optional[KnowledgeGraph] = None,
        circuit_index: Optional[Any] = None,
        build: bool = True,
    ):
        self.faiss = faiss_index if faiss_index is not None or not build else FaissTextIndex()
        self.kg = kg if kg is not None or not build else KnowledgeGraph()
        self.circuit_index = circuit_index  # optional; supplied when a model exists

    def retrieve(
        self,
        query: str,
        topology: Optional[str] = None,
        violation: Optional[str] = None,
        process_node: Optional[str] = None,
        circuit_embedding=None,
        top_k: int = 5,
    ) -> FusedContext:
        ctx = FusedContext(query=query)

        # 1. Semantic text.
        if self.faiss is not None:
            try:
                ctx.text = self.faiss.search_text(query, top_k=top_k)
            except Exception:
                ctx.text = []

        # 2. Structured graph facts.
        if self.kg is not None:
            facts: List[GraphFact] = []
            if violation:
                fixes = self.kg.fixes_for_violation(violation, process_node=process_node)
                facts += fixes
                for f in fixes:
                    facts += self.kg.outcomes_for_fix(f.obj)
                    facts += self.kg.conflicts_for_fix(f.obj)
            if topology:
                facts += self.kg.violations_for_topology(topology)
                if not violation:
                    facts += self.kg.fixes_for_topology(topology, process_node=process_node)
            ctx.facts = facts

        # 3. Circuit-embedding similarity (only if an index was supplied).
        if self.circuit_index is not None and circuit_embedding is not None:
            try:
                ctx.circuits = self.circuit_index.query_by_embedding(
                    circuit_embedding, top_k=3
                )
            except Exception:
                ctx.circuits = []

        return ctx

    def stats(self) -> dict:
        return {
            "faiss": self.faiss.stats() if self.faiss else None,
            "kg": self.kg.stats() if self.kg else None,
            "circuit_index": bool(self.circuit_index),
        }
