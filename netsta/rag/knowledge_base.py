"""
Curated circuits knowledge base + chunking for retrieval.

The JSON corpus contains 50+ entries spanning op-amps, comparators, bandgaps,
LDOs, oscillators, ADC blocks, standard cells, and timing paths. Each entry
is flattened into a single document string, then split into ~512-token
chunks with 50-token overlap so the vector store sees retrievable units.

Token counts are approximated by word count (1 word ≈ 1.3 tokens). For the
small entries here the approximation is harmless; if you swap in a real
tokenizer later the chunker only needs a different `_token_count`.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List


_KB_PATH = os.path.join(os.path.dirname(__file__), "circuits_knowledge.json")
DEFAULT_CHUNK_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 50


@dataclass
class KnowledgeChunk:
    text: str
    metadata: Dict[str, str]


def load_knowledge(path: str = _KB_PATH) -> List[Dict]:
    """Load the raw entries list from the bundled JSON corpus."""
    with open(path) as f:
        kb = json.load(f)
    return kb["entries"]


def entry_to_text(entry: Dict) -> str:
    """Flatten one structured entry into a single retrievable paragraph."""
    parts = [
        f"Circuit: {entry['circuit_name']} ({entry['topology_type']}).",
        f"Description: {entry['description']}",
    ]
    specs = entry.get("typical_specs") or {}
    if specs:
        spec_line = ", ".join(f"{k}={v}" for k, v in specs.items())
        parts.append(f"Typical specs: {spec_line}.")
    rng = entry.get("device_count_range")
    if rng:
        parts.append(f"Typical device count: {rng[0]}-{rng[1]}.")
    issues = entry.get("common_issues") or []
    if issues:
        parts.append("Common issues: " + "; ".join(issues) + ".")
    tips = entry.get("optimization_tips") or []
    if tips:
        parts.append("Optimization tips: " + "; ".join(tips) + ".")
    return " ".join(parts)


def _token_count(text: str) -> int:
    """Approximate token count via whitespace tokens × 1.3."""
    return max(1, int(len(text.split()) * 1.3))


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_TOKENS,
    overlap: int = DEFAULT_OVERLAP_TOKENS,
) -> List[str]:
    """Split `text` into chunks of ~chunk_size tokens with `overlap` token reuse.

    Operates on word boundaries. Returns the original text in a single chunk
    if it already fits.
    """
    words = text.split()
    if not words:
        return []
    # words-per-chunk derived from the 1 word ≈ 1.3 tokens approximation.
    wpc = max(1, int(chunk_size / 1.3))
    overlap_w = max(0, int(overlap / 1.3))
    if len(words) <= wpc:
        return [" ".join(words)]
    chunks: List[str] = []
    step = max(1, wpc - overlap_w)
    i = 0
    while i < len(words):
        window = words[i : i + wpc]
        chunks.append(" ".join(window))
        if i + wpc >= len(words):
            break
        i += step
    return chunks


def build_chunks(
    entries: Iterable[Dict] = None,
    chunk_size: int = DEFAULT_CHUNK_TOKENS,
    overlap: int = DEFAULT_OVERLAP_TOKENS,
) -> List[KnowledgeChunk]:
    """Return one or more KnowledgeChunk per entry, with attached metadata."""
    entries = list(entries) if entries is not None else load_knowledge()
    out: List[KnowledgeChunk] = []
    for entry in entries:
        text = entry_to_text(entry)
        for chunk_idx, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            out.append(KnowledgeChunk(
                text=chunk,
                metadata={
                    "circuit_name": entry["circuit_name"],
                    "topology_type": entry["topology_type"],
                    "chunk_index": str(chunk_idx),
                },
            ))
    return out
