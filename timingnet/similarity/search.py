"""
Top-level similarity-search API.

Three functions cover the common use-cases:
  - find_similar(circuit_data, model, index, top_k=5)
        embed a new circuit and return its nearest neighbours
  - find_by_property(target_specs, index, top_k=5)
        filter the index by metadata, optionally re-rank by an embedding
  - compare_circuits(circuit_a, circuit_b, model)
        return cosine similarity + a side-by-side metric diff
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..model import NetSTAModel
from .circuit_index import CircuitIndex, _summarise, embed_circuit


# ChromaDB metadata filter operators. We map common keys to range queries.
_RANGE_KEYS = {"num_gates", "max_congestion", "critical_path_length", "avg_slack"}


def _build_where(target_specs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map a target_specs dict to a Chroma `where` filter.

    Supports two value forms per key:
      - scalar -> exact match (Chroma's $eq)
      - {"min": x, "max": y} -> range query (Chroma's $gte/$lte)
    `circuit_type` is special-cased as an exact match on the canonical
    'digital' / 'analog' tags.
    """
    if not target_specs:
        return None
    clauses = []
    for key, value in target_specs.items():
        if key == "circuit_type" and isinstance(value, str):
            clauses.append({"circuit_type": {"$eq": value}})
            continue
        if isinstance(value, dict):
            # ChromaDB requires exactly one operator per expression, so a
            # min+max range must be split into two clauses combined by $and.
            if "min" in value:
                clauses.append({key: {"$gte": value["min"]}})
            if "max" in value:
                clauses.append({key: {"$lte": value["max"]}})
            continue
        if key in _RANGE_KEYS:
            clauses.append({key: {"$eq": value}})
        else:
            clauses.append({key: value})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def find_similar(
    circuit_data,
    model: NetSTAModel,
    index: CircuitIndex,
    top_k: int = 5,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Embed `circuit_data` (a PyG Data) and return its nearest neighbours.

    `where` accepts the same shape as `find_by_property`'s target_specs.
    """
    emb = embed_circuit(model, circuit_data, device=index.device)
    chroma_where = _build_where(where or {}) if where else None
    return index.query_by_embedding(emb, top_k=top_k, where=chroma_where)


def find_by_property(
    target_specs: Dict[str, Any],
    index: CircuitIndex,
    top_k: int = 5,
    anchor_circuit=None,
    model: Optional[NetSTAModel] = None,
) -> List[Dict[str, Any]]:
    """Metadata-filtered retrieval.

    If `anchor_circuit` and `model` are supplied, the metadata-filtered
    candidates are additionally re-ranked by embedding similarity to the
    anchor. Otherwise the function returns up to `top_k` matches from the
    metadata filter alone.
    """
    where = _build_where(target_specs)
    if anchor_circuit is not None and model is not None:
        emb = embed_circuit(model, anchor_circuit, device=index.device)
        return index.query_by_embedding(emb, top_k=top_k, where=where)
    return index.query_by_metadata(where or {}, limit=top_k)


def compare_circuits(
    circuit_a,
    circuit_b,
    model: NetSTAModel,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Pairwise comparison: cosine sim + per-metric delta."""
    emb_a = embed_circuit(model, circuit_a, device=device)
    emb_b = embed_circuit(model, circuit_b, device=device)
    na = float(np.linalg.norm(emb_a))
    nb = float(np.linalg.norm(emb_b))
    if na > 0 and nb > 0:
        cos = float(np.dot(emb_a, emb_b) / (na * nb))
    else:
        cos = float("nan")
    meta_a = _summarise(circuit_a)
    meta_b = _summarise(circuit_b)
    deltas = {}
    for key in ("num_gates", "max_congestion", "critical_path_length", "avg_slack"):
        a_val = meta_a.get(key, 0)
        b_val = meta_b.get(key, 0)
        if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
            deltas[key] = b_val - a_val
    return {
        "cosine_similarity": cos,
        "metadata_a": meta_a,
        "metadata_b": meta_b,
        "deltas": deltas,
    }
