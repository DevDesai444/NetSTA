"""
Supervisor + 4-agent diagnosis pipeline.

The Supervisor receives the GNN predictions, routes them round-robin to the
Timing and DRC specialists, hands both result sets to the Optimization agent
for cross-task reconciliation, and aggregates everything into a DesignReport.

The deterministic path here always runs. When AutoGen + an LLM model client are
available, `use_autogen="auto"` routes through the real RoundRobinGroupChat
(autogen_backend) and falls back here on any failure — same LLM-optional
pattern as the rest of the codebase.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ..retrieval import HybridRetriever
from .agents import DRCAgent, OptimizationAgent, TimingAgent
from .schemas import AgentTurn, DesignReport


def _summarize_predictions(preds: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "slack" in preds:
        s = np.asarray(preds["slack"], dtype=float).reshape(-1)
        if s.size:
            out["slack"] = {"min": float(s.min()), "mean": float(s.mean()),
                            "violations": int((s <= 0).sum()), "n": int(s.size)}
    for t in ("critical_path", "drc"):
        if t in preds:
            p = 1.0 / (1.0 + np.exp(-np.asarray(preds[t], dtype=float).reshape(-1)))
            if p.size:
                out[t] = {"max_prob": float(p.max()), "flagged": int((p > 0.5).sum())}
    if "congestion" in preds:
        c = np.asarray(preds["congestion"], dtype=float).reshape(-1)
        if c.size:
            out["congestion"] = {"max": float(c.max()), "mean": float(c.mean())}
    return out


class Orchestrator:
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        use_autogen: str = "auto",  # "auto" | "never" | "force"
    ):
        self.retriever = retriever or HybridRetriever()
        self.use_autogen = use_autogen
        self.timing = TimingAgent()
        self.drc = DRCAgent()
        self.opt = OptimizationAgent()

    def diagnose(
        self,
        prediction_output: Dict[str, Any],
        circuit_name: str = "circuit",
        topology: Optional[str] = None,
        process_node: str = "45nm",
        circuit_embedding=None,
    ) -> DesignReport:
        node_ids = prediction_output.get("node_ids", []) or []
        preds = prediction_output.get("predictions", prediction_output)

        # Optional: route the real AutoGen GroupChat when asked and available.
        if self.use_autogen in ("auto", "force"):
            report = self._maybe_autogen(
                preds, node_ids, circuit_name, topology, process_node
            )
            if report is not None:
                return report
            if self.use_autogen == "force":
                raise RuntimeError("AutoGen backend requested but unavailable.")

        # Deterministic pipeline.
        supervisor = AgentTurn(
            agent="SupervisorAgent",
            summary=(f"Routing predictions for '{circuit_name}' to Timing and DRC "
                     f"agents, then Optimization for cross-task reconciliation."),
        )
        timing_turn = self.timing.diagnose(
            preds, node_ids, self.retriever, topology, process_node
        )
        drc_turn = self.drc.diagnose(
            preds, node_ids, self.retriever, topology, process_node
        )
        opt_turn = self.opt.reconcile([timing_turn, drc_turn], self.retriever)

        bottlenecks = timing_turn.bottlenecks + drc_turn.bottlenecks
        recommendations = (
            timing_turn.recommendations + drc_turn.recommendations + opt_turn.recommendations
        )
        confidence = {bn.task: round(bn.severity, 3) for bn in bottlenecks}

        return DesignReport(
            circuit_name=circuit_name,
            backend="deterministic",
            bottlenecks=sorted(bottlenecks, key=lambda b: -b.severity),
            recommendations=sorted(recommendations, key=lambda r: -r.confidence),
            transcript=[supervisor, timing_turn, drc_turn, opt_turn],
            predictions_summary=_summarize_predictions(preds),
            confidence_scores=confidence,
        )

    def _maybe_autogen(self, preds, node_ids, circuit_name, topology, process_node):
        try:
            from .autogen_backend import run_autogen_groupchat
        except Exception:
            return None
        try:
            return run_autogen_groupchat(
                preds, node_ids, self.retriever,
                circuit_name=circuit_name, topology=topology, process_node=process_node,
                agents=(self.timing, self.drc, self.opt),
            )
        except Exception as exc:  # pragma: no cover - needs LLM creds
            print(f"[Orchestrator] AutoGen path failed, using deterministic: {exc!r}")
            return None


def diagnose(
    prediction_output: Dict[str, Any],
    retriever: Optional[HybridRetriever] = None,
    circuit_name: str = "circuit",
    topology: Optional[str] = None,
    process_node: str = "45nm",
    use_autogen: str = "auto",
) -> DesignReport:
    """One-call entry: build an orchestrator and diagnose."""
    return Orchestrator(retriever=retriever, use_autogen=use_autogen).diagnose(
        prediction_output, circuit_name=circuit_name,
        topology=topology, process_node=process_node,
    )
