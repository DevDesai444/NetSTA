"""
Post-prediction design advisor.

Given the parser's CircuitSpec + the GNN's per-node predictions, this module
identifies the most concerning regions of the circuit ("bottlenecks") and
generates human-readable recommendations. It uses the same KnowledgeStore as
circuit_parser to retrieve relevant optimisation tips, then optionally asks
an LLM to phrase them more fluently. When no LLM backend is available the
recommendations are produced by a small templating rule set.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from .circuit_parser import CircuitSpec
from .embeddings import KnowledgeStore


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Bottleneck(BaseModel):
    task: str
    severity: float = Field(ge=0.0, le=1.0)
    location: Optional[str] = None
    summary: str


class DesignReport(BaseModel):
    spec: CircuitSpec
    # Per-task: List[float] for 1-D predictions, List[List[float]] for 2-D
    # heads (e.g. analog_performance is [N, 2]).
    predictions: Dict[str, Any] = Field(default_factory=dict)
    bottlenecks: List[Bottleneck] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    backend: str = "fallback"


# ---------------------------------------------------------------------------
# Bottleneck identification
# ---------------------------------------------------------------------------


def _as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _top_nodes(values: np.ndarray, node_ids: List[str], k: int = 3) -> List[str]:
    if values.size == 0:
        return []
    k = min(k, values.size)
    idx = np.argsort(-values)[:k]
    return [node_ids[i] for i in idx if i < len(node_ids)]


def identify_bottlenecks(
    predictions: Dict[str, Any], node_ids: List[str]
) -> List[Bottleneck]:
    """Rank per-task predictions and surface the worst-3 areas.

    Heuristic thresholds map normalized scores to a 0..1 severity. Tasks
    absent from predictions are skipped silently — the model only reports
    on what the loaded checkpoint actually predicted.
    """
    bottlenecks: List[Bottleneck] = []
    if not predictions:
        return bottlenecks

    pred_map = predictions.get("predictions", {})

    # Congestion: high RUDY-aggregated values indicate routing pressure.
    if "congestion" in pred_map:
        vals = _as_array(pred_map["congestion"])
        mx = float(vals.max()) if vals.size else 0.0
        if mx > 0.3:
            locs = _top_nodes(vals, node_ids, 3)
            bottlenecks.append(Bottleneck(
                task="congestion",
                severity=min(1.0, mx),
                location=", ".join(locs) if locs else None,
                summary=(
                    f"Predicted routing congestion peaks at {mx:.2f} (normalized) "
                    f"near {', '.join(locs) or 'unknown'}."
                ),
            ))

    # DRC: predicted hotspot probability.
    if "drc" in pred_map:
        logits = _as_array(pred_map["drc"])
        if logits.size:
            probs = 1.0 / (1.0 + np.exp(-logits))
            mx = float(probs.max())
            if mx > 0.5:
                locs = _top_nodes(probs, node_ids, 3)
                bottlenecks.append(Bottleneck(
                    task="drc",
                    severity=mx,
                    location=", ".join(locs) if locs else None,
                    summary=(
                        f"DRC-hotspot probability up to {mx:.2f} on "
                        f"{', '.join(locs) or 'unknown'}."
                    ),
                ))

    # Analog performance: low gbw_score or high parasitic_impact in the
    # 2-column prediction array (shape [N, 2]).
    if "analog_performance" in pred_map:
        ap = _as_array(pred_map["analog_performance"])
        if ap.ndim == 2 and ap.size:
            gbw = ap[:, 0]
            parasitic = ap[:, 1]
            min_gbw = float(gbw.min())
            max_par = float(parasitic.max())
            if min_gbw < 0.3:
                worst = _top_nodes(-gbw, node_ids, 3)
                bottlenecks.append(Bottleneck(
                    task="analog_performance:gbw",
                    severity=min(1.0, 1.0 - max(min_gbw, 0.0)),
                    location=", ".join(worst) if worst else None,
                    summary=(
                        f"Predicted GBW score drops to {min_gbw:.2f} at "
                        f"{', '.join(worst) or 'unknown'} -- bandwidth-limited."
                    ),
                ))
            if max_par > 0.5:
                hot = _top_nodes(parasitic, node_ids, 3)
                bottlenecks.append(Bottleneck(
                    task="analog_performance:parasitic",
                    severity=max_par,
                    location=", ".join(hot) if hot else None,
                    summary=(
                        f"Parasitic impact reaches {max_par:.2f} near "
                        f"{', '.join(hot) or 'unknown'} -- consider isolating coupling."
                    ),
                ))

    # Sort by severity descending and cap the list length.
    bottlenecks.sort(key=lambda b: -b.severity)
    return bottlenecks[:5]


# ---------------------------------------------------------------------------
# Recommendation generation
# ---------------------------------------------------------------------------


_TEMPLATE_TIPS = {
    "congestion": [
        "Spread densely-placed devices apart along the bottleneck axis.",
        "Promote wide nets to higher metal layers to relieve local routing demand.",
        "Re-pin matched-pair drains to shorten the longest connecting nets.",
    ],
    "drc": [
        "Increase spacing between the flagged cells before routing locks in.",
        "Add a small filler region or via array to break up dense hotspots.",
        "Bias the placer to give the flagged nodes lower density priority.",
    ],
    "analog_performance:gbw": [
        "Increase tail current to raise gm on the slow nodes.",
        "Use a wider M1/M2 input pair (or shorter L) to boost gm.",
        "Reduce parasitic loading on the slow nodes via shielding or shorter nets.",
    ],
    "analog_performance:parasitic": [
        "Move the affected device away from high-activity neighbours.",
        "Add a guard ring to reduce substrate coupling.",
        "Reduce Cgd by shortening the gate-to-drain wire on the offender.",
    ],
}


def _template_recommendations(
    bottlenecks: List[Bottleneck], spec: CircuitSpec, knowledge: List[str],
) -> List[str]:
    out: List[str] = []
    for b in bottlenecks:
        tips = _TEMPLATE_TIPS.get(b.task, [])
        if not tips:
            continue
        # Per-bottleneck tip: pick the first generic template + supplement
        # with the closest knowledge hit if it mentions the task family.
        tip = tips[0]
        for k in knowledge:
            short = k.lower()
            if any(token in short for token in (b.task.replace(":", " "), "compensation", "miller")):
                tip = tip + " (See: " + k[: min(160, len(k))] + "...)"
                break
        out.append(f"[{b.task}] {tip}")
    if not out:
        out.append(
            "GNN predictions are uniformly low-severity in this synthesis run; "
            "no specific bottleneck stands out."
        )
    return out


_ADVICE_SYSTEM_PROMPT = """You are a senior analog/digital design reviewer.
Given a circuit specification, a list of model-predicted bottlenecks, and
retrieved design knowledge, produce a JSON object:

{
  "recommendations": ["<concise actionable bullet>", ...],
  "confidence_scores": {"<bottleneck.task>": <0.0..1.0>}
}

Reply with JSON only, no markdown fences."""


def _try_llm_advice(
    spec: CircuitSpec, bottlenecks: List[Bottleneck], knowledge: List[str],
) -> Optional[Dict[str, Any]]:
    """Best-effort LLM call to refine recommendations; None on any failure."""
    bn = [
        {"task": b.task, "severity": b.severity, "location": b.location,
         "summary": b.summary}
        for b in bottlenecks
    ]
    user_msg = (
        f"Circuit spec:\n{spec.model_dump_json(indent=2)}\n\n"
        f"Predicted bottlenecks:\n{json.dumps(bn, indent=2)}\n\n"
        f"Retrieved knowledge:\n" + "\n".join(f"- {k}" for k in knowledge)
        + "\n\nReturn JSON only."
    )
    # Try OpenAI -> Anthropic -> Ollama, exactly mirroring the parser.
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            resp = OpenAI().chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "system", "content": _ADVICE_SYSTEM_PROMPT},
                          {"role": "user", "content": user_msg}],
                temperature=0.0,
            )
            return _parse_advice_json(resp.choices[0].message.content)
        except Exception as exc:
            print(f"[design_advisor] OpenAI failed: {exc!r}")
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from anthropic import Anthropic
            resp = Anthropic().messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
                max_tokens=1024,
                system=_ADVICE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            return _parse_advice_json(resp.content[0].text if resp.content else None)
        except Exception as exc:
            print(f"[design_advisor] Anthropic failed: {exc!r}")
    try:
        import ollama
        ollama.list()
        resp = ollama.chat(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            messages=[{"role": "system", "content": _ADVICE_SYSTEM_PROMPT},
                      {"role": "user", "content": user_msg}],
            options={"temperature": 0.0},
        )
        return _parse_advice_json(resp.get("message", {}).get("content"))
    except Exception:
        return None


def _parse_advice_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def advise(
    spec: CircuitSpec,
    predictions: Optional[Dict[str, Any]],
    knowledge_store: Optional[KnowledgeStore] = None,
) -> DesignReport:
    """Build a DesignReport from the parsed spec + GNN predictions."""
    store = knowledge_store or KnowledgeStore()
    node_ids = (predictions or {}).get("node_ids", []) or []
    bottlenecks = identify_bottlenecks(predictions or {}, node_ids)

    # Retrieve knowledge relevant to the worst bottlenecks for the LLM/template.
    queries = [b.summary for b in bottlenecks] or [spec.raw_query]
    retrieved: List[str] = []
    seen = set()
    for q in queries[:3]:
        for chunk in store.retrieve(q, top_k=3):
            if chunk not in seen:
                retrieved.append(chunk)
                seen.add(chunk)

    backend = "fallback"
    recommendations: List[str] = []
    confidence: Dict[str, float] = {}
    llm_out = _try_llm_advice(spec, bottlenecks, retrieved)
    if llm_out:
        recommendations = list(llm_out.get("recommendations", []))
        confidence = dict(llm_out.get("confidence_scores", {}))
        backend = "llm"
    if not recommendations:
        recommendations = _template_recommendations(bottlenecks, spec, retrieved)
        # Crude confidence: severity-derived. Higher severity → lower confidence
        # in the spec, which prompts more aggressive remediation.
        for b in bottlenecks:
            confidence[b.task] = round(1.0 - 0.5 * b.severity, 2)

    # Pack a small numpy-->list view of predictions for the report payload.
    pred_payload: Dict[str, List[float]] = {}
    if predictions and "predictions" in predictions:
        for task, arr in predictions["predictions"].items():
            try:
                np_arr = np.asarray(arr, dtype=float)
                # Flatten 2-D (analog_performance) into list of lists.
                if np_arr.ndim == 2:
                    pred_payload[task] = np_arr.tolist()
                else:
                    pred_payload[task] = np_arr.tolist()
            except Exception:
                continue

    return DesignReport(
        spec=spec,
        predictions=pred_payload,
        bottlenecks=bottlenecks,
        recommendations=recommendations,
        confidence_scores=confidence,
        backend=backend,
    )
