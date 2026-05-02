"""
Vector store for NetSTA RAG: sentence-transformers + ChromaDB primary path,
keyword-overlap fallback if either dep is unavailable.

Public API:
  KnowledgeStore.retrieve(query: str, top_k: int = 5) -> List[str]
  KnowledgeStore.add_document(text: str, metadata: dict) -> None

The store builds itself from `knowledge_base.build_chunks()` on first
construction and persists to `./netsta_vectordb/`. Subsequent runs skip the
embedding step if the collection already has rows.
"""

import os
import re
from collections import Counter
from typing import List, Optional, Tuple

from .knowledge_base import KnowledgeChunk, build_chunks


DEFAULT_VECTORDB_DIR = "./netsta_vectordb"
DEFAULT_COLLECTION = "netsta_circuits"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


class KeywordStore:
    """Inverted-index TF-style fallback. No external deps."""

    def __init__(self):
        self._chunks: List[KnowledgeChunk] = []
        self._toks: List[Counter] = []

    def add(self, chunk: KnowledgeChunk) -> None:
        self._chunks.append(chunk)
        self._toks.append(Counter(_tokenize(chunk.text)))

    def query(self, text: str, top_k: int) -> List[Tuple[str, dict, float]]:
        if not self._chunks:
            return []
        q = Counter(_tokenize(text))
        if not q:
            return []
        scored: List[Tuple[float, int]] = []
        for i, doc in enumerate(self._toks):
            # Sum of min(q[w], doc[w]) — simple overlap score.
            score = sum(min(q[w], doc[w]) for w in q)
            if score > 0:
                scored.append((float(score), i))
        scored.sort(reverse=True)
        out = []
        for score, i in scored[:top_k]:
            out.append((self._chunks[i].text, self._chunks[i].metadata, score))
        return out


class KnowledgeStore:
    """Retrieval-augmented circuit knowledge store with graceful degradation.

    Tries (1) sentence-transformers + ChromaDB; (2) sentence-transformers
    only with in-memory cosine; (3) pure keyword overlap. The choice is
    made at construction and exposed via `self.backend`.
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_VECTORDB_DIR,
        collection: str = DEFAULT_COLLECTION,
        embed_model: str = DEFAULT_EMBED_MODEL,
        bootstrap: bool = True,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection
        self.embed_model_name = embed_model

        self.backend: str = "keyword"  # may be upgraded below
        self._encoder = None
        self._collection = None
        self._keyword = KeywordStore()

        # Try sentence-transformers.
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(embed_model)
            self.backend = "embeddings_only"
        except Exception as exc:
            print(f"[KnowledgeStore] sentence-transformers unavailable: {exc!r}")
            self._encoder = None

        # Try ChromaDB.
        if self._encoder is not None:
            try:
                import chromadb
                os.makedirs(persist_dir, exist_ok=True)
                client = chromadb.PersistentClient(path=persist_dir)
                self._collection = client.get_or_create_collection(
                    name=collection,
                    metadata={"hnsw:space": "cosine"},
                )
                self.backend = "chroma"
            except Exception as exc:
                print(f"[KnowledgeStore] chromadb unavailable: {exc!r}")
                self._collection = None

        if bootstrap:
            self._bootstrap_from_knowledge_base()

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def _existing_chroma_count(self) -> int:
        if self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def _bootstrap_from_knowledge_base(self) -> None:
        chunks = build_chunks()
        # Always keep an in-memory keyword index so fallback works
        # alongside the vector path.
        for c in chunks:
            self._keyword.add(c)

        # Populate ChromaDB only if it's empty (avoid re-embedding on every run).
        if self.backend == "chroma" and self._existing_chroma_count() == 0:
            self._index_chunks(chunks)
        elif self.backend == "embeddings_only":
            # In-memory cosine path — pre-embed everything once.
            self._cache_embeddings(chunks)

    def _index_chunks(self, chunks):
        if not chunks or self._collection is None or self._encoder is None:
            return
        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [f"chunk_{i:04d}" for i in range(len(chunks))]
        embs = self._encoder.encode(
            texts, batch_size=32, show_progress_bar=False
        ).tolist()
        # ChromaDB add() will reject ID collisions; we already checked count==0.
        self._collection.add(
            documents=texts, metadatas=metadatas, ids=ids, embeddings=embs,
        )

    def _cache_embeddings(self, chunks):
        if self._encoder is None:
            return
        self._chunks_cache = chunks
        self._chunk_embeddings = self._encoder.encode(
            [c.text for c in chunks], batch_size=32, show_progress_bar=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Return up to top_k document chunks most relevant to `query`."""
        if not query.strip():
            return []
        if self.backend == "chroma":
            try:
                q_emb = self._encoder.encode([query]).tolist()
                res = self._collection.query(query_embeddings=q_emb, n_results=top_k)
                docs = res.get("documents", [[]])[0]
                return list(docs)
            except Exception as exc:
                print(f"[KnowledgeStore] chroma query failed, falling back: {exc!r}")
        if self.backend == "embeddings_only" and self._encoder is not None:
            import numpy as np
            q = self._encoder.encode([query])
            sims = (q @ self._chunk_embeddings.T)[0]
            order = sims.argsort()[::-1][:top_k]
            return [self._chunks_cache[i].text for i in order]
        # Pure keyword fallback.
        hits = self._keyword.query(query, top_k)
        return [text for (text, _meta, _score) in hits]

    def retrieve_with_metadata(self, query: str, top_k: int = 5):
        """Same as retrieve but yields (text, metadata, score-or-None) tuples."""
        if self.backend == "chroma":
            try:
                q_emb = self._encoder.encode([query]).tolist()
                res = self._collection.query(query_embeddings=q_emb, n_results=top_k)
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                dists = res.get("distances", [[None] * len(docs)])[0]
                return list(zip(docs, metas, dists))
            except Exception:
                pass
        hits = self._keyword.query(query, top_k)
        return [(text, meta, score) for (text, meta, score) in hits]

    def add_document(self, text: str, metadata: Optional[dict] = None) -> None:
        """Index a new document at runtime."""
        meta = dict(metadata or {})
        meta.setdefault("source", "user")
        chunk = KnowledgeChunk(text=text, metadata=meta)
        self._keyword.add(chunk)
        if self.backend == "chroma" and self._encoder is not None:
            try:
                emb = self._encoder.encode([text]).tolist()
                # ID by content hash to avoid collisions on repeated adds.
                doc_id = f"user_{abs(hash(text)) % (10 ** 10):010d}"
                self._collection.add(
                    documents=[text], metadatas=[meta], ids=[doc_id], embeddings=emb,
                )
            except Exception as exc:
                print(f"[KnowledgeStore] chroma add_document failed: {exc!r}")
