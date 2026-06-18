"""
FAISS index over chunked EDA text for the diagnosis agents.

The agents need fast semantic search over unstructured design knowledge —
timing-closure techniques, DRC mitigation notes, analog optimisation tips.
This wraps FAISS with sentence-transformer embeddings and degrades to the
keyword store from rag.embeddings when either dependency is missing, so the
pipeline still answers offline.

Index choice is corpus-size-aware: a flat inner-product index for small
corpora (exact, no training needed) and an IVF-Flat index once the corpus is
large enough for the coarse quantiser to help. Embeddings are L2-normalised
so inner product == cosine similarity.

Public API:
    idx = FaissTextIndex()                  # builds from the bundled KB
    idx.search("setup violation on the critical path", top_k=5)
        -> [(text, metadata, score), ...]
    idx.search_text(query, top_k)           -> [text, ...]
    idx.stats()                             -> {"backend", "ntotal", "dim", ...}
"""

import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..rag.knowledge_base import KnowledgeChunk, build_chunks
from ..rag.embeddings import KeywordStore


DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = "./netsta_faiss"
# Below this many vectors an exact flat index is both faster and more accurate
# than IVF (IVF training needs a comfortable multiple of nlist points).
IVF_MIN_VECTORS = 1000


@dataclass
class _Hit:
    text: str
    metadata: dict
    score: float


class FaissTextIndex:
    """Semantic text index with a keyword fallback.

    backend is one of:
      "faiss-ivf"  IVF-Flat (large corpus)
      "faiss-flat" exact inner-product (small corpus)
      "keyword"    no faiss / sentence-transformers — overlap scoring
    """

    def __init__(
        self,
        chunks: Optional[Sequence[KnowledgeChunk]] = None,
        embed_model: str = DEFAULT_EMBED_MODEL,
        extra_docs: Optional[Sequence[Tuple[str, dict]]] = None,
    ):
        self.embed_model_name = embed_model
        self.backend = "keyword"
        self._encoder = None
        self._index = None
        self._dim = 0

        # Corpus: bundled EDA knowledge + any caller-supplied extra docs.
        self._chunks: List[KnowledgeChunk] = list(
            chunks if chunks is not None else build_chunks()
        )
        for text, meta in (extra_docs or []):
            self._chunks.append(KnowledgeChunk(text=text, metadata=dict(meta or {})))

        # Always keep a keyword index around as the safety net.
        self._keyword = KeywordStore()
        for c in self._chunks:
            self._keyword.add(c)

        self._build_vector_index()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_vector_index(self) -> None:
        try:
            import faiss  # noqa: F401
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - depends on env
            print(f"[FaissTextIndex] faiss/sentence-transformers unavailable: {exc!r}")
            return

        if not self._chunks:
            return

        import faiss
        import numpy as np

        self._encoder = SentenceTransformer(self.embed_model_name)
        embs = self._encoder.encode(
            [c.text for c in self._chunks],
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype("float32")
        self._dim = int(embs.shape[1])
        n = embs.shape[0]

        if n >= IVF_MIN_VECTORS:
            # nlist ~ sqrt(n) is the usual rule of thumb; cap at 256 cells.
            nlist = max(1, min(256, int(math.sqrt(n))))
            quantizer = faiss.IndexFlatIP(self._dim)
            index = faiss.IndexIVFFlat(quantizer, self._dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embs)
            index.add(embs)
            # nprobe controls the recall/latency trade at query time.
            index.nprobe = min(16, nlist)
            self.backend = "faiss-ivf"
        else:
            index = faiss.IndexFlatIP(self._dim)
            index.add(embs)
            self.backend = "faiss-flat"

        self._index = index

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, dict, float]]:
        """Return up to top_k (text, metadata, score) hits for `query`.

        Scores are cosine similarity in [-1, 1] on the faiss paths and raw
        overlap counts on the keyword path; callers should treat them as a
        relative ranking, not an absolute scale.
        """
        if not query or not query.strip():
            return []
        if self._index is not None and self._encoder is not None:
            try:
                import numpy as np

                q = self._encoder.encode(
                    [query], normalize_embeddings=True
                ).astype("float32")
                k = min(top_k, len(self._chunks))
                scores, idxs = self._index.search(q, k)
                hits: List[Tuple[str, dict, float]] = []
                for score, i in zip(scores[0], idxs[0]):
                    if i < 0:
                        continue
                    c = self._chunks[i]
                    hits.append((c.text, c.metadata, float(score)))
                if hits:
                    return hits
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[FaissTextIndex] faiss query failed, falling back: {exc!r}")
        # Keyword fallback.
        return self._keyword.query(query, top_k)

    def search_text(self, query: str, top_k: int = 5) -> List[str]:
        return [text for (text, _m, _s) in self.search(query, top_k)]

    # ------------------------------------------------------------------
    # Persistence (optional; the corpus is small enough to rebuild cheaply)
    # ------------------------------------------------------------------

    def save(self, index_dir: str = DEFAULT_INDEX_DIR) -> None:
        if self._index is None:
            return
        import faiss

        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(self._index, os.path.join(index_dir, "text.index"))
        with open(os.path.join(index_dir, "chunks.json"), "w") as f:
            json.dump(
                [{"text": c.text, "metadata": c.metadata} for c in self._chunks], f
            )

    def stats(self) -> dict:
        return {
            "backend": self.backend,
            "ntotal": int(self._index.ntotal) if self._index is not None else 0,
            "num_chunks": len(self._chunks),
            "dim": self._dim,
            "embed_model": self.embed_model_name if self._encoder else None,
        }
