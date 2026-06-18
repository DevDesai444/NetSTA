"""
Diagnosis primitives the agents call to turn raw GNN predictions into ranked,
typed bottlenecks. Pure functions over (predictions, node_ids) so they're
trivially testable and reusable by both the deterministic orchestrator and the
AutoGen tool-calling path.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .schemas import Bottleneck


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _names(node_ids: List[str], idx: np.ndarray) -> List[str]:
    return [node_ids[i] for i in idx if 0 <= i < len(node_ids)]


def rank_critical_nodes(
    predictions: Dict[str, Any],
    node_ids: List[str],
    top_k: int = 5,
    slack_threshold_ns: float = 0.0,
) -> Optional[Bottleneck]:
    """Flag the worst-slack nodes as a timing bottleneck.

    Uses the slack head when present (lower = worse), else the critical-path
    head's probability. Severity scales with how negative the worst slack is
    and how many nodes fall at/below the threshold.
    """
    if "slack" in predictions:
        slack = _arr(predictions["slack"])
        if slack.size == 0:
            return None
        order = np.argsort(slack)  # ascending: worst (lowest) first
        worst = _names(node_ids, order[:top_k])
        n_viol = int((slack <= slack_threshold_ns).sum())
        worst_slack = float(slack.min())
        # Severity: violation fraction blended with worst-slack magnitude.
        frac = n_viol / max(slack.size, 1)
        mag = min(1.0, abs(min(worst_slack, 0.0)) / 0.2)
        severity = float(min(1.0, 0.5 * frac + 0.5 * mag)) if n_viol else float(0.3 * frac)
        return Bottleneck(
            task="slack",
            violation_type="setup_violation",
            severity=max(severity, 0.05 if n_viol else 0.0),
            location=", ".join(worst[:3]) or None,
            node_ids=worst,
            summary=(
                f"Worst slack {worst_slack:+.3f} ns; {n_viol}/{slack.size} nodes "
                f"at or below {slack_threshold_ns:.2f} ns."
            ),
        )
    if "critical_path" in predictions:
        prob = _sigmoid(_arr(predictions["critical_path"]))
        if prob.size == 0:
            return None
        order = np.argsort(-prob)
        hot = _names(node_ids, order[:top_k])
        mx = float(prob.max())
        if mx < 0.5:
            return None
        return Bottleneck(
            task="critical_path", violation_type="setup_violation",
            severity=mx, location=", ".join(hot[:3]) or None, node_ids=hot,
            summary=f"Critical-path probability up to {mx:.2f}.",
        )
    return None


def rank_drc_hotspots(
    predictions: Dict[str, Any], node_ids: List[str],
    top_k: int = 5, prob_threshold: float = 0.5,
) -> Optional[Bottleneck]:
    if "drc" not in predictions:
        return None
    prob = _sigmoid(_arr(predictions["drc"]))
    if prob.size == 0 or float(prob.max()) < prob_threshold:
        return None
    order = np.argsort(-prob)
    hot = _names(node_ids, order[:top_k])
    mx = float(prob.max())
    return Bottleneck(
        task="drc", violation_type="drc_density",
        severity=mx, location=", ".join(hot[:3]) or None, node_ids=hot,
        summary=f"DRC-hotspot probability up to {mx:.2f} on {len(hot)} cells.",
    )


def rank_congestion(
    predictions: Dict[str, Any], node_ids: List[str],
    top_k: int = 5, threshold: float = 0.3,
) -> Optional[Bottleneck]:
    if "congestion" not in predictions:
        return None
    cong = _arr(predictions["congestion"])
    if cong.size == 0 or float(cong.max()) < threshold:
        return None
    order = np.argsort(-cong)
    hot = _names(node_ids, order[:top_k])
    mx = float(cong.max())
    return Bottleneck(
        task="congestion", violation_type="routing_congestion",
        severity=min(1.0, mx), location=", ".join(hot[:3]) or None, node_ids=hot,
        summary=f"Routing congestion peaks at {mx:.2f} (normalized).",
    )


def classify_timing_violation(predictions: Dict[str, Any]) -> str:
    """Coarse setup/hold classification from the slack distribution.

    Most failing paths in a combinational/inter-register graph are setup
    (path too slow). A cluster of small-magnitude negative slacks with many
    short paths can indicate hold issues; we keep a simple, honest heuristic
    and default to setup_violation.
    """
    if "slack" not in predictions:
        return "setup_violation"
    slack = _arr(predictions["slack"])
    if slack.size == 0:
        return "setup_violation"
    neg = slack[slack < 0]
    if neg.size and float(np.mean(neg)) > -0.02 and neg.size > 0.3 * slack.size:
        # Many shallow negatives -> more likely hold-ish on this synthetic STA.
        return "hold_violation"
    return "setup_violation"
