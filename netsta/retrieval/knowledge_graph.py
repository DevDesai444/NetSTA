"""
EDA design-pattern knowledge graph.

Models the structured relationships flat vector search can't traverse:

    Topology   -COMMONLY_EXHIBITS->  ViolationType
    ViolationType -RESOLVED_BY->     FixStrategy
    FixStrategy  -PRODUCES->         Outcome
    FixStrategy  -CONSTRAINED_BY->   DesignRule
    FixStrategy  -CONFLICTS_WITH->   FixStrategy

The CONFLICTS_WITH edges are what let the Optimization agent reason across
tasks ("this timing fix conflicts with that DRC fix"). Two interchangeable
backends behind one query surface:

  - Neo4j  when NEO4J_URI (+ NEO4J_USER / NEO4J_PASSWORD) is set and the driver
    connects — the graph is (idempotently) loaded and queried with Cypher.
  - NetworkX in-memory otherwise — same seed data, same queries, no server.

`self.backend` reports which one is live.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_SEED_PATH = os.path.join(os.path.dirname(__file__), "eda_knowledge_graph.json")


@dataclass
class GraphFact:
    """One retrieved relationship, backend-agnostic."""
    subject: str
    relation: str
    obj: str
    props: Dict = field(default_factory=dict)

    def as_text(self) -> str:
        extra = ""
        if self.props.get("action"):
            extra = f" — {self.props['action']}"
        elif self.props.get("description"):
            extra = f" — {self.props['description']}"
        return f"{self.subject} {self.relation} {self.obj}{extra}"


def load_seed(path: str = _SEED_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


class KnowledgeGraph:
    def __init__(self, seed: Optional[dict] = None, prefer_neo4j: bool = True):
        self.seed = seed or load_seed()
        self.backend = "networkx"
        self._g = None          # networkx graph
        self._driver = None     # neo4j driver
        self._node_props: Dict[str, dict] = {}
        self._node_label: Dict[str, str] = {}
        for label, rows in self.seed["nodes"].items():
            for row in rows:
                self._node_props[row["name"]] = row
                self._node_label[row["name"]] = label

        if prefer_neo4j and os.getenv("NEO4J_URI"):
            self._try_neo4j()
        if self._driver is None:
            self._build_networkx()

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _try_neo4j(self) -> None:
        try:
            from neo4j import GraphDatabase

            uri = os.environ["NEO4J_URI"]
            user = os.getenv("NEO4J_USER", "neo4j")
            pwd = os.getenv("NEO4J_PASSWORD", "neo4j")
            driver = GraphDatabase.driver(uri, auth=(user, pwd))
            driver.verify_connectivity()
            self._driver = driver
            self.backend = "neo4j"
            self._load_neo4j()
        except Exception as exc:  # pragma: no cover - needs a live server
            print(f"[KnowledgeGraph] Neo4j unavailable, using NetworkX: {exc!r}")
            self._driver = None

    def _load_neo4j(self) -> None:
        """Idempotently MERGE the seed graph into Neo4j."""
        with self._driver.session() as s:
            for label, rows in self.seed["nodes"].items():
                for row in rows:
                    s.run(
                        f"MERGE (n:{label} {{name:$name}}) SET n += $props",
                        name=row["name"], props=row,
                    )
            for rel in self.seed["relationships"]:
                s.run(
                    f"MATCH (a {{name:$f}}), (b {{name:$t}}) "
                    f"MERGE (a)-[:{rel['type']}]->(b)",
                    f=rel["from"], t=rel["to"],
                )

    def _build_networkx(self) -> None:
        import networkx as nx

        g = nx.MultiDiGraph()
        for name, props in self._node_props.items():
            g.add_node(name, label=self._node_label[name], **props)
        for rel in self.seed["relationships"]:
            g.add_edge(rel["from"], rel["to"], key=rel["type"], rel=rel["type"])
        self._g = g

    # ------------------------------------------------------------------
    # Internal edge traversal (works on either backend)
    # ------------------------------------------------------------------

    def _out(self, node: str, rel: str) -> List[str]:
        """Neighbours reachable from `node` via relation `rel`."""
        if self.backend == "neo4j":
            with self._driver.session() as s:
                rows = s.run(
                    f"MATCH (a {{name:$n}})-[:{rel}]->(b) RETURN b.name AS name",
                    n=node,
                )
                return [r["name"] for r in rows]
        out = []
        if self._g is not None and node in self._g:
            for _u, v, k in self._g.out_edges(node, keys=True):
                if k == rel:
                    out.append(v)
        return out

    def _fact(self, subj: str, rel: str, obj: str) -> GraphFact:
        return GraphFact(subj, rel, obj, props=dict(self._node_props.get(obj, {})))

    # ------------------------------------------------------------------
    # Public query surface (used by the agent tools)
    # ------------------------------------------------------------------

    def violations_for_topology(self, topology: str) -> List[GraphFact]:
        return [self._fact(topology, "COMMONLY_EXHIBITS", v)
                for v in self._out(topology, "COMMONLY_EXHIBITS")]

    def fixes_for_violation(
        self, violation: str,
        process_node: Optional[str] = None,
    ) -> List[GraphFact]:
        """FixStrategies that resolve `violation`, optionally filtered by node."""
        facts = []
        for fix in self._out(violation, "RESOLVED_BY"):
            props = self._node_props.get(fix, {})
            nodes = props.get("applicable_process_nodes")
            if process_node and nodes and process_node not in nodes:
                continue
            facts.append(self._fact(violation, "RESOLVED_BY", fix))
        return facts

    def outcomes_for_fix(self, fix: str) -> List[GraphFact]:
        return [self._fact(fix, "PRODUCES", o) for o in self._out(fix, "PRODUCES")]

    def rules_for_fix(self, fix: str) -> List[GraphFact]:
        return [self._fact(fix, "CONSTRAINED_BY", r)
                for r in self._out(fix, "CONSTRAINED_BY")]

    def conflicts_for_fix(self, fix: str) -> List[GraphFact]:
        """Symmetric CONFLICTS_WITH lookup (seed lists each edge once)."""
        out = set(self._out(fix, "CONFLICTS_WITH"))
        # also catch the reverse direction
        for rel in self.seed["relationships"]:
            if rel["type"] == "CONFLICTS_WITH" and rel["to"] == fix:
                out.add(rel["from"])
        return [self._fact(fix, "CONFLICTS_WITH", o) for o in sorted(out)]

    def fixes_for_topology(
        self, topology: str, process_node: Optional[str] = None,
    ) -> List[GraphFact]:
        """One-hop convenience: topology -> its violations -> their fixes."""
        facts = []
        seen = set()
        for v in self._out(topology, "COMMONLY_EXHIBITS"):
            for f in self.fixes_for_violation(v, process_node=process_node):
                if f.obj not in seen:
                    seen.add(f.obj)
                    facts.append(f)
        return facts

    def stats(self) -> dict:
        return {
            "backend": self.backend,
            "nodes": len(self._node_props),
            "relationships": len(self.seed["relationships"]),
            "node_types": {k: len(v) for k, v in self.seed["nodes"].items()},
        }

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
