"""
Multi-agent design-advisory layer.

Four agents turn the GNN's per-node predictions into grounded, per-violation
fix recommendations:

  - SupervisorAgent     routes predictions to the specialists and aggregates
  - TimingAgent         diagnoses slack / critical-path violations
  - DRCAgent            diagnoses DRC + routing-congestion hotspots
  - OptimizationAgent   cross-task PPA reasoning over the other two's fixes

Recommendations are grounded in hybrid retrieval (FAISS text + knowledge graph
+ circuit similarity), never invented. The deterministic orchestrator always
runs (offline / CI); when AutoGen + an LLM backend are available, the same
agents run as a real RoundRobinGroupChat (see autogen_backend).
"""

from .schemas import Bottleneck, Recommendation, DesignReport, AgentTurn
from .orchestrator import diagnose, Orchestrator

__all__ = [
    "Bottleneck",
    "Recommendation",
    "DesignReport",
    "AgentTurn",
    "diagnose",
    "Orchestrator",
]
