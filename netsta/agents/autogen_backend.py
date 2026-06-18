"""
Real AutoGen GroupChat backend (optional).

When `autogen-agentchat` + a model client (`autogen-ext`) + an LLM key are all
present, the four roles run as genuine AutoGen `AssistantAgent`s in a
`RoundRobinGroupChat`. The structured facts (bottlenecks, candidate fixes,
retrieved evidence) still come from the deterministic tools — the LLM agents
reason and communicate *over that grounded data* rather than inventing slack
numbers — and their discussion is captured in the report transcript.

Everything autogen-specific is imported lazily inside run_autogen_groupchat so
this module imports fine when autogen isn't installed; the orchestrator catches
any failure and falls back to the deterministic pipeline.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

from .schemas import AgentTurn, DesignReport


_SYSTEM = {
    "SupervisorAgent": (
        "You are the supervisor of a VLSI design-review panel. You receive GNN "
        "predictions and grounded findings, route them to the Timing and DRC "
        "specialists, then ask the Optimization agent to reconcile conflicts. "
        "Keep the discussion tight and end with a short ranked action list."
    ),
    "TimingAgent": (
        "You are a timing-closure specialist. Given the flagged slack/critical "
        "bottlenecks and the retrieved fixes, explain which fixes to apply and "
        "why, grounded ONLY in the provided facts. Do not invent numbers."
    ),
    "DRCAgent": (
        "You are a DRC/congestion specialist. Given the flagged hotspots and "
        "retrieved fixes, recommend layout remedies grounded in the provided "
        "facts and process node."
    ),
    "OptimizationAgent": (
        "You are a cross-task PPA reviewer. Inspect the timing and DRC fixes for "
        "conflicts (a timing fix that worsens DRC, or vice versa) using the "
        "provided conflict facts, and produce the final reconciled ranking."
    ),
}


def _model_client():
    """Build an autogen-ext model client from env, or return None."""
    try:
        if os.getenv("OPENAI_API_KEY"):
            from autogen_ext.models.openai import OpenAIChatCompletionClient

            return OpenAIChatCompletionClient(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            )
        if os.getenv("ANTHROPIC_API_KEY"):
            from autogen_ext.models.anthropic import AnthropicChatCompletionClient

            return AnthropicChatCompletionClient(
                model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
            )
    except Exception as exc:
        print(f"[autogen_backend] no model client: {exc!r}")
    return None


def _grounding_task(turns: List[AgentTurn], circuit_name: str) -> str:
    lines = [f"Circuit under review: {circuit_name}", ""]
    for t in turns:
        lines.append(f"## {t.agent} findings")
        lines.append(t.summary)
        for b in t.bottlenecks:
            lines.append(f"- bottleneck: {b.task} ({b.violation_type}) "
                         f"severity {b.severity:.2f} at {b.location}")
        for r in t.recommendations:
            cf = f" [conflicts: {', '.join(r.conflicts)}]" if r.conflicts else ""
            lines.append(f"- fix: {r.fix} — {r.action} (outcomes: "
                         f"{', '.join(r.outcomes) or 'n/a'}){cf}")
        lines.append("")
    lines.append("Discuss and produce a final ranked, conflict-aware action list.")
    return "\n".join(lines)


def run_autogen_groupchat(
    predictions: Dict[str, Any],
    node_ids: List[str],
    retriever,
    circuit_name: str = "circuit",
    topology: Optional[str] = None,
    process_node: str = "45nm",
    agents=None,
) -> Optional[DesignReport]:
    """Run the real AutoGen RoundRobinGroupChat. Returns None if unavailable."""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.teams import RoundRobinGroupChat

    client = _model_client()
    if client is None:
        return None

    # 1. Ground the discussion with the deterministic specialists' typed output.
    timing_agent, drc_agent, opt_agent = agents
    timing_turn = timing_agent.diagnose(predictions, node_ids, retriever, topology, process_node)
    drc_turn = drc_agent.diagnose(predictions, node_ids, retriever, topology, process_node)
    opt_turn = opt_agent.reconcile([timing_turn, drc_turn], retriever)
    grounded_turns = [timing_turn, drc_turn, opt_turn]

    # 2. Real AutoGen agents discuss over that grounded context.
    ag_agents = [
        AssistantAgent(name=name, model_client=client, system_message=msg)
        for name, msg in _SYSTEM.items()
    ]
    team = RoundRobinGroupChat(
        ag_agents, termination_condition=MaxMessageTermination(max_messages=8)
    )
    task = _grounding_task(grounded_turns, circuit_name)

    async def _go():
        return await team.run(task=task)

    result = asyncio.run(_go())

    # 3. Capture the LLM discussion as transcript turns on top of the typed data.
    transcript: List[AgentTurn] = [
        AgentTurn(agent="SupervisorAgent",
                  summary=f"Convened AutoGen panel for '{circuit_name}'.")
    ]
    for msg in getattr(result, "messages", []) or []:
        src = getattr(msg, "source", "agent")
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            transcript.append(AgentTurn(agent=src, summary=content.strip()[:1500]))

    bottlenecks = timing_turn.bottlenecks + drc_turn.bottlenecks
    recommendations = (timing_turn.recommendations + drc_turn.recommendations
                       + opt_turn.recommendations)
    return DesignReport(
        circuit_name=circuit_name,
        backend="autogen",
        bottlenecks=sorted(bottlenecks, key=lambda b: -b.severity),
        recommendations=sorted(recommendations, key=lambda r: -r.confidence),
        transcript=transcript,
        confidence_scores={b.task: round(b.severity, 3) for b in bottlenecks},
    )
