"""
The specialist diagnosis agents.

Each agent reads the GNN predictions, flags bottlenecks (tools.py), grounds
fixes in the hybrid retriever (knowledge graph + FAISS text), and emits typed
Recommendations. The OptimizationAgent then reasons across the others' fixes
using the graph's CONFLICTS_WITH edges — the cross-task PPA step a single
sequential advisor can't do.

These role classes are backend-agnostic: the deterministic orchestrator calls
them directly, and the AutoGen backend wraps the same methods as agent tools.
"""

from typing import Any, Dict, List, Optional

from ..retrieval import HybridRetriever
from .schemas import AgentTurn, Bottleneck, Recommendation
from .tools import (
    classify_timing_violation,
    rank_congestion,
    rank_critical_nodes,
    rank_drc_hotspots,
)


def _confidence(severity: float, effort: Optional[str]) -> float:
    """Higher when the problem is severe and the fix is cheap."""
    c = 0.4 + 0.4 * float(severity)
    c += {"low": 0.1, "medium": 0.0, "high": -0.1}.get(effort or "medium", 0.0)
    return float(max(0.05, min(0.95, c)))


def _recommendations(
    kg, violation: str, bottleneck: Bottleneck,
    text_evidence: List[str], agent: str, process_node: Optional[str],
) -> List[Recommendation]:
    """Turn the graph's fixes for `violation` into grounded recommendations."""
    recs: List[Recommendation] = []
    for fix_fact in kg.fixes_for_violation(violation, process_node=process_node):
        fix = fix_fact.obj
        props = fix_fact.props
        outcomes = [f.obj for f in kg.outcomes_for_fix(fix)]
        conflicts = [f.obj for f in kg.conflicts_for_fix(fix)]
        rules = [f.obj for f in kg.rules_for_fix(fix)]
        evidence = list(text_evidence[:2])
        if rules:
            evidence.append("constrained by design rule(s): " + ", ".join(rules[:2]))
        recs.append(Recommendation(
            bottleneck_task=bottleneck.task,
            fix=fix,
            action=props.get("action", fix),
            rationale=f"Resolves {violation} at {bottleneck.location or 'the flagged nodes'}.",
            evidence=evidence,
            outcomes=outcomes,
            conflicts=conflicts,
            effort=props.get("estimated_effort"),
            confidence=_confidence(bottleneck.severity, props.get("estimated_effort")),
            agent=agent,
        ))
    return recs


class TimingAgent:
    name = "TimingAgent"
    role = "Diagnose timing (slack / critical-path) violations and recommend closure fixes."

    def diagnose(
        self, predictions: Dict[str, Any], node_ids: List[str],
        retriever: HybridRetriever, topology: Optional[str] = None,
        process_node: Optional[str] = None,
    ) -> AgentTurn:
        bn = rank_critical_nodes(predictions, node_ids)
        if bn is None:
            return AgentTurn(agent=self.name, summary="No timing violations flagged.")
        violation = classify_timing_violation(predictions)
        bn.violation_type = violation
        ctx = retriever.retrieve(
            f"{violation} on the critical path: {bn.summary}",
            topology=topology, violation=violation, process_node=process_node,
        )
        recs = _recommendations(
            retriever.kg, violation, bn, ctx.text, self.name, process_node
        )
        return AgentTurn(
            agent=self.name,
            summary=(f"Flagged {violation} ({bn.severity:.2f} severity) at "
                     f"{bn.location}; {len(recs)} candidate fixes."),
            bottlenecks=[bn], recommendations=recs,
        )


class DRCAgent:
    name = "DRCAgent"
    role = "Diagnose DRC hotspots and routing congestion; recommend layout fixes."

    def diagnose(
        self, predictions: Dict[str, Any], node_ids: List[str],
        retriever: HybridRetriever, topology: Optional[str] = None,
        process_node: Optional[str] = None,
    ) -> AgentTurn:
        bottlenecks: List[Bottleneck] = []
        recs: List[Recommendation] = []
        for bn in (rank_drc_hotspots(predictions, node_ids),
                   rank_congestion(predictions, node_ids)):
            if bn is None:
                continue
            bottlenecks.append(bn)
            ctx = retriever.retrieve(
                f"{bn.violation_type}: {bn.summary}",
                topology=topology, violation=bn.violation_type, process_node=process_node,
            )
            recs += _recommendations(
                retriever.kg, bn.violation_type, bn, ctx.text, self.name, process_node
            )
        if not bottlenecks:
            return AgentTurn(agent=self.name, summary="No DRC/congestion hotspots flagged.")
        return AgentTurn(
            agent=self.name,
            summary=f"Flagged {len(bottlenecks)} DRC/congestion issue(s); {len(recs)} fixes.",
            bottlenecks=bottlenecks, recommendations=recs,
        )


class OptimizationAgent:
    name = "OptimizationAgent"
    role = "Cross-task PPA reasoning: reconcile timing vs DRC fixes that conflict."

    def reconcile(self, turns: List[AgentTurn], retriever: HybridRetriever) -> AgentTurn:
        all_recs: List[Recommendation] = [r for t in turns for r in t.recommendations]
        proposed = {r.fix for r in all_recs}
        notes: List[str] = []
        adjusted = 0
        for r in all_recs:
            clashing = [c for c in r.conflicts if c in proposed]
            if clashing:
                # A fix from one task conflicts with a fix proposed for another.
                r.confidence = max(0.05, r.confidence - 0.15)
                notes.append(
                    f"{r.fix} (for {r.bottleneck_task}) conflicts with {', '.join(clashing)} "
                    f"— applying both risks trading one violation for another."
                )
                adjusted += 1
        summary = (
            f"Reconciled {len(all_recs)} fixes across {len(turns)} agents; "
            f"{adjusted} flagged for cross-task conflict."
            if adjusted else
            f"Reviewed {len(all_recs)} fixes; no cross-task conflicts among the proposed set."
        )
        # Surface the conflict notes as low-confidence advisory recommendations.
        advisories = [
            Recommendation(
                bottleneck_task="ppa", fix="reconcile_conflict", action=n,
                rationale="Cross-task interaction detected via knowledge-graph conflict edges.",
                agent=self.name, confidence=0.5,
            )
            for n in notes
        ]
        return AgentTurn(agent=self.name, summary=summary, recommendations=advisories)
