"""Pydantic schemas for the agent diagnosis pipeline.

Every agent emits type-validated structures so the supervisor and the final
report receive consistent data whether they were produced by the deterministic
orchestrator or by an LLM in the AutoGen path.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Bottleneck(BaseModel):
    """A flagged problem region surfaced from the GNN predictions."""
    task: str = Field(description="slack | critical_path | congestion | drc | ...")
    violation_type: Optional[str] = Field(
        default=None, description="knowledge-graph ViolationType key, if classified"
    )
    severity: float = Field(ge=0.0, le=1.0)
    location: Optional[str] = Field(default=None, description="top node id(s)")
    node_ids: List[str] = Field(default_factory=list)
    summary: str = ""


class Recommendation(BaseModel):
    """A grounded fix for a bottleneck, with evidence and trade-offs."""
    bottleneck_task: str
    fix: str = Field(description="FixStrategy key, e.g. gate_upsizing")
    action: str = Field(description="human-readable instruction")
    rationale: str = ""
    evidence: List[str] = Field(default_factory=list, description="retrieved support")
    outcomes: List[str] = Field(default_factory=list, description="expected effects")
    conflicts: List[str] = Field(
        default_factory=list, description="fixes this one fights with (cross-task)"
    )
    effort: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    agent: str = "TimingAgent"


class AgentTurn(BaseModel):
    """One agent's contribution to the GroupChat transcript."""
    agent: str
    summary: str
    bottlenecks: List[Bottleneck] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)


class DesignReport(BaseModel):
    """Final aggregated output of the 4-agent pipeline."""
    circuit_name: str = "circuit"
    backend: str = "deterministic"  # or "autogen"
    bottlenecks: List[Bottleneck] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    transcript: List[AgentTurn] = Field(default_factory=list)
    predictions_summary: Dict[str, Any] = Field(default_factory=dict)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)

    def top_recommendations(self, k: int = 5) -> List[Recommendation]:
        return sorted(self.recommendations, key=lambda r: -r.confidence)[:k]
