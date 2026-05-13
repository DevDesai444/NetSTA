"""
Graph-embedding index for circuit similarity search.

Extracts the backbone's global-pooling output (mean+max concatenated) for
every circuit in a dataset and stores those vectors in a persistent
ChromaDB collection ("circuit_embeddings"). Each entry carries a small
metadata dict so callers can filter by circuit_type, gate count, peak
congestion, critical-path length, and average slack.
"""

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from ..model import NetSTAModel


DEFAULT_PERSIST_DIR = "./circuit_embeddb"
DEFAULT_COLLECTION = "circuit_embeddings"
# Tag stored on every row's metadata. Bump when the metadata schema changes
# so callers know to rebuild the index.
INDEX_SCHEMA_TAG = "v2"


def _scalar(x) -> float:
    """Coerce a torch/numpy/plain scalar to a float."""
    if isinstance(x, torch.Tensor):
        return float(x.item())
    return float(x)


def _summarise(data) -> Dict[str, Any]:
    """Extract the per-circuit metadata stored alongside each embedding."""
    is_analog = bool(getattr(data, "is_analog", False))
    num_nodes = int(data.x.size(0))
    if hasattr(data, "y_congestion") and data.y_congestion.numel() > 0:
        max_cong = _scalar(data.y_congestion.max())
    else:
        max_cong = 0.0
    if hasattr(data, "y_critical") and data.y_critical.numel() > 0:
        cp_len = int(data.y_critical.sum().item())
    else:
        cp_len = 0
    if hasattr(data, "y_slack") and data.y_slack.numel() > 0:
        avg_slack = _scalar(data.y_slack.mean())
    else:
        avg_slack = 0.0
    # Analog perf is stored as a [N, 2] tensor (gbw_score, parasitic_impact).
    # Digital circuits zero-fill this, so the resulting averages are 0.
    avg_gbw = 0.0
    avg_parasitic = 0.0
    ap = getattr(data, "y_analog_performance", None)
    if ap is not None and ap.numel() > 0 and ap.dim() == 2 and ap.size(1) == 2:
        avg_gbw = _scalar(ap[:, 0].mean())
        avg_parasitic = _scalar(ap[:, 1].mean())
    return {
        "circuit_type": "analog" if is_analog else "digital",
        "num_gates": num_nodes,
        "max_congestion": max_cong,
        "critical_path_length": cp_len,
        "avg_slack": avg_slack,
        "avg_gbw_score": avg_gbw,
        "avg_parasitic": avg_parasitic,
        "circuit_name": str(getattr(data, "circuit_name", "")),
        "_schema": INDEX_SCHEMA_TAG,
    }


@torch.no_grad()
def embed_circuit(model: NetSTAModel, data, device: str = "cpu") -> np.ndarray:
    """Return the graph-level embedding (1-D numpy array) for one PyG Data.

    Uses the backbone's `_graph_emb` key from the forward dict (mean+max pool).
    """
    model.eval()
    data = data.to(device)
    batch = getattr(data, "batch", None)
    if batch is None:
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
    preds = model(data.x, data.edge_index, edge_attr=data.edge_attr, batch=batch)
    g = preds["_graph_emb"].detach().cpu().numpy()
    # _graph_emb is [B, D]; single-graph case yields [1, D]. Squeeze.
    return g.reshape(g.shape[-1])


class CircuitIndex:
    """ChromaDB-backed nearest-neighbour index over circuit graph embeddings."""

    def __init__(
        self,
        model: NetSTAModel,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(persist_dir, exist_ok=True)

        import chromadb
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:
            return 0

    def schema_tag(self) -> Optional[str]:
        """Return the schema tag of the first row, or None if empty."""
        try:
            row = self._collection.get(limit=1, include=["metadatas"])
            metas = row.get("metadatas") or []
            return metas[0].get("_schema") if metas else None
        except Exception:
            return None

    def needs_rebuild(self) -> bool:
        """True if the index is empty or its rows predate INDEX_SCHEMA_TAG."""
        if self.count() == 0:
            return True
        return self.schema_tag() != INDEX_SCHEMA_TAG

    def reset(self) -> None:
        """Wipe and recreate the collection (e.g. when the model changes)."""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @torch.no_grad()
    def build(
        self,
        dataset,
        batch_size: int = 16,
        id_prefix: str = "ckt",
        force: bool = False,
        verbose: bool = True,
    ) -> int:
        """Embed every entry in `dataset` and write to the collection.

        Returns the count of vectors after insertion. Pass `force=True` to
        wipe and rebuild even if the collection already has data.
        """
        should_rebuild = force or self.needs_rebuild()
        if should_rebuild:
            if self.count() > 0:
                if verbose and not force:
                    print(f"[CircuitIndex] schema_tag mismatch (have "
                          f"{self.schema_tag()!r}, want {INDEX_SCHEMA_TAG!r}); rebuilding")
                self.reset()
        else:
            if verbose:
                print(f"[CircuitIndex] collection has {self.count()} rows; "
                      "pass force=True to rebuild")
            return self.count()

        self.model.eval()
        device = torch.device(self.device)
        loader = DataLoader(dataset, batch_size=batch_size)

        all_ids: List[str] = []
        all_embeds: List[List[float]] = []
        all_meta: List[Dict[str, Any]] = []
        cursor = 0

        for batch in loader:
            batch = batch.to(device)
            preds = self.model(
                batch.x, batch.edge_index,
                edge_attr=batch.edge_attr, batch=batch.batch,
            )
            graph_emb = preds["_graph_emb"].detach().cpu().numpy()
            # PyG's Batch object lets us recover per-graph slices.
            data_list = batch.to_data_list()
            for i, data in enumerate(data_list):
                emb = graph_emb[i]
                meta = _summarise(data)
                all_embeds.append(emb.tolist())
                all_meta.append(meta)
                all_ids.append(f"{id_prefix}_{cursor:06d}")
                cursor += 1

        if all_ids:
            self._collection.add(
                ids=all_ids,
                embeddings=all_embeds,
                metadatas=all_meta,
                # Provide a stable "document" placeholder so Chroma doesn't
                # complain; the metadata carries the real per-circuit info.
                documents=[m["circuit_name"] or i for i, m in zip(all_ids, all_meta)],
            )
        if verbose:
            print(f"[CircuitIndex] indexed {len(all_ids)} circuits")
        return self.count()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_by_embedding(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return up to top_k nearest neighbours given an embedding vector."""
        kwargs = {"query_embeddings": [embedding.tolist()], "n_results": top_k}
        if where:
            kwargs["where"] = where
        res = self._collection.query(**kwargs)
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[None] * len(ids)])[0]
        metas = res.get("metadatas", [[]])[0]
        out = []
        for i, d, m in zip(ids, dists, metas):
            out.append({"id": i, "distance": d, "similarity": _to_similarity(d), "metadata": m})
        return out

    def query_by_metadata(
        self,
        where: Dict[str, Any],
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Pure metadata filter (no embedding query). Returns up to `limit` matches."""
        res = self._collection.get(where=where, limit=limit, include=["metadatas"])
        ids = res.get("ids", [])
        metas = res.get("metadatas", [])
        return [{"id": i, "metadata": m} for i, m in zip(ids, metas)]

    def get_all(self) -> Dict[str, Any]:
        """Return every row's id, embedding, and metadata.

        Used by visualisations that want to render the full embedding space
        (e.g. t-SNE / UMAP projection in the Streamlit tab).
        """
        res = self._collection.get(include=["embeddings", "metadatas"])
        ids = list(res.get("ids", []) or [])
        # ChromaDB ≥1.0 returns numpy arrays for embeddings; `array or []` is
        # ambiguous, so handle the empty case explicitly.
        raw = res.get("embeddings")
        if raw is None or len(raw) == 0:
            embeds = np.zeros((0, 0))
        else:
            embeds = np.asarray(raw, dtype=float)
        metas = list(res.get("metadatas") or [])
        return {"ids": ids, "embeddings": embeds, "metadatas": metas}


def _to_similarity(distance: Optional[float]) -> Optional[float]:
    """Cosine distance -> cosine similarity in [-1, 1]."""
    if distance is None:
        return None
    return 1.0 - float(distance)
