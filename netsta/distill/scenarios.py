"""
Generate diverse grounded scenarios for distillation.

Each scenario is a self-contained problem the teacher LLM reacts to:
    {
      "role": "timing|drc|optimization|supervisor",
      "circuit_name": ...,
      "circuit_summary": "...",        # process node, topology, sizes
      "predictions_summary": {...},    # model's per-task numbers, summarised
      "bottlenecks": [...],            # from deterministic tools.py
      "retrieved_facts": [...],        # KG facts (RESOLVED_BY, CONFLICTS_WITH, ...)
      "retrieved_text": [...],         # FAISS hits
      "peer_findings": [...],          # for OptimizationAgent: other agents' fixes
    }

We draw scenarios from the actual real-netlist graph dataset that Block B
trained on, so the LLM is reacting to data that looks like inference traffic.
"""

import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from netsta.agents.tools import (
    classify_timing_violation,
    rank_congestion,
    rank_critical_nodes,
    rank_drc_hotspots,
)
from netsta.real_dataset import load_dataset
from netsta.retrieval import HybridRetriever


def _logit(label: np.ndarray, scale: float = 8.0) -> np.ndarray:
    return (np.asarray(label, dtype=float) - 0.5) * scale


def _predictions_from_graph(g) -> Dict[str, np.ndarray]:
    """Synthesise per-node 'predictions' from a labelled real-graph cone.

    For distillation, the bottleneck-ranking + retrieval pipeline only cares
    that the values are plausible model outputs. Using the STA ground-truth on
    the graph (lightly perturbed) gives realistic per-task distributions to
    surface bottlenecks the teacher can reason about, without needing a trained
    checkpoint loaded here.
    """
    rng = np.random.default_rng(int(hash(getattr(g, "circuit_name", "")) & 0xFFFFFFFF))

    slack = g.y_slack.numpy()
    crit = _logit(g.y_critical.numpy())
    cong = g.y_congestion.numpy()
    drc = _logit(g.y_drc.numpy())

    # Add small Gaussian noise so "predictions" don't exactly equal labels.
    slack = slack + rng.normal(0.0, 0.05, size=slack.shape).astype(np.float32)
    cong = np.clip(cong + rng.normal(0.0, 0.03, size=cong.shape), 0.0, 1.0)
    return {"slack": slack, "critical_path": crit, "congestion": cong, "drc": drc}


def _node_ids_from_graph(g) -> List[str]:
    """Use a generic ID list; real circuits use proper names but tools.py is
    name-agnostic, and the LLM doesn't need real names — only structure."""
    return [f"n{i}" for i in range(int(g.x.size(0)))]


@dataclass
class Scenario:
    role: str
    circuit_name: str
    process_node: str
    topology: str
    num_nodes: int
    num_edges: int
    predictions_summary: Dict
    bottlenecks: List[Dict] = field(default_factory=list)
    retrieved_facts: List[str] = field(default_factory=list)
    retrieved_text: List[str] = field(default_factory=list)
    peer_findings: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "circuit_name": self.circuit_name,
            "process_node": self.process_node,
            "topology": self.topology,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "predictions_summary": self.predictions_summary,
            "bottlenecks": self.bottlenecks,
            "retrieved_facts": self.retrieved_facts,
            "retrieved_text": self.retrieved_text,
            "peer_findings": self.peer_findings,
        }


def _summarise_preds(preds: Dict[str, np.ndarray]) -> Dict:
    out: Dict = {}
    if "slack" in preds:
        s = preds["slack"].reshape(-1)
        out["slack"] = {
            "min_ns": float(s.min()),
            "mean_ns": float(s.mean()),
            "violation_count": int((s <= 0).sum()),
            "node_count": int(s.size),
        }
    if "critical_path" in preds:
        p = 1.0 / (1.0 + np.exp(-preds["critical_path"].reshape(-1)))
        out["critical_path"] = {
            "max_prob": float(p.max()),
            "flagged_count": int((p > 0.5).sum()),
        }
    if "congestion" in preds:
        c = preds["congestion"].reshape(-1)
        out["congestion"] = {"max": float(c.max()), "mean": float(c.mean())}
    if "drc" in preds:
        d = 1.0 / (1.0 + np.exp(-preds["drc"].reshape(-1)))
        out["drc"] = {"max_prob": float(d.max()), "flagged_count": int((d > 0.5).sum())}
    return out


def _bottlenecks_for(role: str, preds, node_ids):
    """Surface the bottlenecks each role's agent would see."""
    out = []
    if role == "timing":
        b = rank_critical_nodes(preds, node_ids)
        if b:
            out.append(b)
    elif role == "drc":
        for b in (rank_drc_hotspots(preds, node_ids), rank_congestion(preds, node_ids)):
            if b:
                out.append(b)
    elif role == "supervisor":
        # Sees everything.
        for b in (
            rank_critical_nodes(preds, node_ids),
            rank_drc_hotspots(preds, node_ids),
            rank_congestion(preds, node_ids),
        ):
            if b:
                out.append(b)
    # Optimization role: bottlenecks come via peer_findings, not directly.
    return [
        {
            "task": b.task,
            "violation_type": b.violation_type,
            "severity": b.severity,
            "location": b.location,
            "summary": b.summary,
        }
        for b in out
    ]


def _process_node_for(circuit_name: str) -> str:
    """Heuristic: OpenABC's bench files are technology-agnostic AIGs; map them
    plus everything else to 45nm so KG retrieval uses the Nangate45 design rules
    that match our STA labels."""
    return "45nm"


def _topology_for(circuit_name: str) -> str:
    """Pick a topology label that maps onto a knowledge-graph Topology node."""
    n = circuit_name.lower()
    # OpenABC industrial designs are mostly pipelined datapaths or memory arrays.
    if any(k in n for k in ("bp_", "rocket", "ariane", "core", "fetch", "exec")):
        return "pipelined_datapath"
    if any(k in n for k in ("aes", "des", "jpeg", "fft", "dft", "idft", "fpu", "mult")):
        return "pipelined_datapath"
    if any(k in n for k in ("eth", "i2c", "spi", "uart", "vga", "tlb", "dma")):
        return "finite_state_machine"
    if any(k in n for k in ("ram", "rom", "mem", "cache")):
        return "memory_array"
    if any(k in n for k in ("clk", "pll", "ckdiv")):
        return "clock_distribution"
    return "combinational_logic"


def build_scenarios(
    dataset_path: str = "data_real/graphs.pt",
    n_per_role: int = 200,
    seed: int = 42,
    retriever: Optional[HybridRetriever] = None,
) -> Dict[str, List[Scenario]]:
    """Produce n_per_role scenarios for each of the 4 roles, grounded in
    real circuit graphs + the deterministic retrieval pipeline.
    """
    graphs, sources, _meta = load_dataset(dataset_path)
    rng = random.Random(seed)
    indices = list(range(len(graphs)))
    rng.shuffle(indices)

    retriever = retriever or HybridRetriever()
    out: Dict[str, List[Scenario]] = {r: [] for r in ("supervisor", "timing", "drc", "optimization")}

    # Round-robin a shuffled stream of graphs through all 4 roles; restart the
    # stream if we run out before hitting the per-role target.
    target_total = 4 * n_per_role
    cursor = 0
    while sum(len(v) for v in out.values()) < target_total:
        if cursor >= len(indices):
            cursor = 0
            rng.shuffle(indices)
        g_idx = indices[cursor]
        cursor += 1
        g = graphs[g_idx]
        circuit_name = sources[g_idx]
        node_ids = _node_ids_from_graph(g)
        preds = _predictions_from_graph(g)

        # Pick a role whose bucket isn't full yet.
        for role in ("supervisor", "timing", "drc", "optimization"):
            if len(out[role]) >= n_per_role:
                continue

            bottlenecks = _bottlenecks_for(role, preds, node_ids)
            # Skip empty scenarios for specialist roles — there must be at
            # least one violation to discuss. Supervisor and Optimization can
            # handle "all clean" but it's a less interesting training signal.
            if not bottlenecks and role in ("timing", "drc"):
                break

            topology = _topology_for(circuit_name)
            process_node = _process_node_for(circuit_name)

            # Retrieve grounding context.
            if bottlenecks:
                violation = bottlenecks[0]["violation_type"]
            else:
                violation = "setup_violation"  # default for retrieval
            ctx = retriever.retrieve(
                query=f"{topology} with {violation} on the critical path",
                topology=topology,
                violation=violation,
                process_node=process_node,
                top_k=4,
            )
            retrieved_facts = [f.as_text() for f in ctx.facts[:12]]
            retrieved_text = [t[:400] for t in ctx.text[:4]]

            # For the optimization role, fabricate plausible peer findings from
            # the same retriever so the LLM has fixes to reconcile.
            peer_findings: List[Dict] = []
            if role == "optimization":
                for kg_role, viol in (("TimingAgent", "setup_violation"), ("DRCAgent", "drc_density")):
                    fixes = retriever.kg.fixes_for_violation(viol, process_node=process_node)
                    for f in fixes[:3]:
                        peer_findings.append({
                            "agent": kg_role,
                            "fix": f.obj,
                            "action": f.props.get("action", f.obj),
                            "outcomes": [o.obj for o in retriever.kg.outcomes_for_fix(f.obj)],
                            "conflicts": [c.obj for c in retriever.kg.conflicts_for_fix(f.obj)],
                            "effort": f.props.get("estimated_effort"),
                        })

            out[role].append(
                Scenario(
                    role=role,
                    circuit_name=circuit_name,
                    process_node=process_node,
                    topology=topology,
                    num_nodes=int(g.x.size(0)),
                    num_edges=int(g.edge_index.size(1)),
                    predictions_summary=_summarise_preds(preds),
                    bottlenecks=bottlenecks,
                    retrieved_facts=retrieved_facts,
                    retrieved_text=retrieved_text,
                    peer_findings=peer_findings,
                )
            )
            break  # one role per graph cycle

    return out


def save_scenarios(out: Dict[str, List[Scenario]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for role, scs in out.items():
        path = os.path.join(out_dir, f"{role}.jsonl")
        with open(path, "w") as f:
            for s in scs:
                f.write(json.dumps(s.to_dict()) + "\n")
        print(f"  {role}: {len(scs)} scenarios -> {path}")
