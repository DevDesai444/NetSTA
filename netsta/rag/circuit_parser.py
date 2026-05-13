"""
Natural-language circuit parser.

Pipeline:
  user English description
    -> [LLM with RAG context]  (OpenAI / Anthropic / Ollama; optional)
    -> CircuitSpec (pydantic)
    -> analog circuit generation
    -> trained-GNN inference (optional)
    -> {spec, circuit, predictions}

The LLM step is strictly an enhancement. When no API key is present and no
local Ollama server is reachable, a deterministic keyword+regex parser
extracts the spec. RAG retrieval still happens in this fallback so the
returned spec is annotated with the closest matching knowledge chunks.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .embeddings import KnowledgeStore


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class DeviceSpec(BaseModel):
    """One device in the parsed circuit specification."""
    role: str = Field(description="Functional role, e.g. 'input_pair', 'tail', 'mirror'")
    device_type: str = Field(description="One of NMOS, PMOS, R, C, CURRENT_MIRROR, DIFF_PAIR")
    W_um: Optional[float] = None
    L_nm: Optional[float] = None
    notes: Optional[str] = None


class CircuitSpec(BaseModel):
    """Parsed structured representation of a user circuit request."""
    topology: str = Field(description="Topology key (e.g. two_stage_opamp)")
    target_specs: Dict[str, float] = Field(default_factory=dict)
    process_node: str = "130nm"
    num_stages: int = 1
    compensation: str = "none"
    devices: List[DeviceSpec] = Field(default_factory=list)
    raw_query: str = ""
    retrieved_knowledge: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Topology / spec keyword extraction (deterministic fallback)
# ---------------------------------------------------------------------------


_TOPOLOGY_KEYWORDS = [
    ("two_stage_opamp",   ["two-stage", "two stage", "miller", "miller-compensated"]),
    ("folded_cascode",    ["folded cascode", "folded-cascode"]),
    ("diff_pair",         ["differential pair", "diff pair", "diff-pair"]),
    ("current_mirror",    ["current mirror", "current-mirror"]),
    ("common_source_amp", ["common-source", "common source", "cs amp", "cs amplifier"]),
]

_SPEC_PATTERNS = {
    "gain_db": re.compile(r"(\d+(?:\.\d+)?)\s*dB\s+(?:of\s+)?gain", re.I),
    "gain_db_alt": re.compile(r"gain\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*dB", re.I),
    "gbw_mhz": re.compile(r"(\d+(?:\.\d+)?)\s*MHz\s+(?:gbw|gain[-\s]bandwidth|bandwidth)", re.I),
    "gbw_mhz_alt": re.compile(r"gbw\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*MHz", re.I),
    "gbw_ghz": re.compile(r"(\d+(?:\.\d+)?)\s*GHz\s+(?:gbw|gain[-\s]bandwidth|bandwidth)", re.I),
    "phase_margin_deg": re.compile(r"(\d+(?:\.\d+)?)\s*°?\s*(?:degrees?\s+)?phase\s+margin", re.I),
    "slew_rate_v_us": re.compile(r"(\d+(?:\.\d+)?)\s*V/?μ?u?s\s+slew", re.I),
    "power_mw": re.compile(r"(\d+(?:\.\d+)?)\s*mW\s+(?:of\s+)?power", re.I),
}

_PROCESS_PATTERN = re.compile(r"(\d+)\s*nm(?:\s+process|\s+CMOS|\b)", re.I)
_STAGE_PATTERN = re.compile(r"(\d+)[-\s]stage", re.I)


def _keyword_topology(text: str) -> str:
    lowered = text.lower()
    for topo, keywords in _TOPOLOGY_KEYWORDS:
        if any(kw in lowered for kw in keywords):
            return topo
    return "two_stage_opamp"


def _extract_specs(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    # Match the most specific patterns first; only set if not already set.
    for key, pat in _SPEC_PATTERNS.items():
        m = pat.search(text)
        if m:
            canonical = key.replace("_alt", "")
            if canonical == "gbw_ghz":
                out["gbw_mhz"] = float(m.group(1)) * 1000.0
                continue
            out.setdefault(canonical, float(m.group(1)))
    return out


def _infer_stages(text: str, topology: str) -> int:
    m = _STAGE_PATTERN.search(text)
    if m:
        return int(m.group(1))
    if "two-stage" in text.lower() or "two stage" in text.lower():
        return 2
    if topology == "two_stage_opamp":
        return 2
    if topology in ("folded_cascode", "current_mirror", "diff_pair", "common_source_amp"):
        return 1
    return 1


def _infer_compensation(text: str) -> str:
    lower = text.lower()
    if "miller" in lower:
        return "miller"
    if "nested" in lower:
        return "nested miller"
    if "cascode comp" in lower:
        return "cascode"
    if "no compensation" in lower or "uncompensated" in lower:
        return "none"
    return "none"


def _process_node(text: str) -> str:
    m = _PROCESS_PATTERN.search(text)
    if m:
        return f"{m.group(1)}nm"
    return "130nm"


def fallback_parse(text: str, knowledge: List[str]) -> CircuitSpec:
    """Deterministic NL parser — runs when no LLM backend is available."""
    topology = _keyword_topology(text)
    specs = _extract_specs(text)
    stages = _infer_stages(text, topology)
    compensation = _infer_compensation(text)
    process = _process_node(text)
    return CircuitSpec(
        topology=topology,
        target_specs=specs,
        process_node=process,
        num_stages=stages,
        compensation=compensation,
        devices=[],
        raw_query=text,
        retrieved_knowledge=knowledge,
    )


# ---------------------------------------------------------------------------
# LLM clients (best-effort; absent / unreachable backends are silent skips).
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """You are an analog/digital circuit-design assistant. Given a user
request and retrieved knowledge snippets, output a JSON object matching this schema:

{
  "topology": "<one of: two_stage_opamp, folded_cascode, diff_pair, current_mirror, common_source_amp>",
  "target_specs": {"<key>": <numeric value>, ...},
  "process_node": "<e.g. 130nm, 180nm>",
  "num_stages": <integer>,
  "compensation": "<e.g. miller, cascode, none>",
  "devices": [
    {"role": "<role>", "device_type": "<NMOS|PMOS|R|C|CURRENT_MIRROR|DIFF_PAIR>",
     "W_um": <float|null>, "L_nm": <float|null>, "notes": "<string|null>"}, ...
  ]
}

Reply with ONLY the JSON object, no commentary, no markdown fences."""


def _build_user_prompt(user_text: str, knowledge: List[str]) -> str:
    knowledge_block = "\n\n".join(f"- {k}" for k in knowledge) or "(no matches)"
    return (
        f"User request:\n{user_text}\n\n"
        f"Retrieved circuit knowledge (top matches):\n{knowledge_block}\n\n"
        "Output JSON only."
    )


def _try_openai(user_text: str, knowledge: List[str]) -> Optional[str]:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(user_text, knowledge)},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content
    except Exception as exc:
        print(f"[circuit_parser] OpenAI call failed: {exc!r}")
        return None


def _try_anthropic(user_text: str, knowledge: List[str]) -> Optional[str]:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        from anthropic import Anthropic
        client = Anthropic()
        resp = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(user_text, knowledge)}],
        )
        if resp.content and getattr(resp.content[0], "text", None):
            return resp.content[0].text
    except Exception as exc:
        print(f"[circuit_parser] Anthropic call failed: {exc!r}")
    return None


def _try_ollama(user_text: str, knowledge: List[str]) -> Optional[str]:
    try:
        import ollama
        # Probe to confirm a server is listening.
        ollama.list()
        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(user_text, knowledge)},
            ],
            options={"temperature": 0.0},
        )
        return resp.get("message", {}).get("content")
    except Exception as exc:
        # Most common case: no local server running. Stay quiet.
        return None


def _llm_response_to_spec(
    response: str, user_text: str, knowledge: List[str],
) -> Optional[CircuitSpec]:
    if not response:
        return None
    # Allow models that wrap JSON in fences.
    cleaned = response.strip().strip("`")
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()
    # Extract the first top-level JSON object.
    match = re.search(r"\{.*\}", cleaned, re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        print(f"[circuit_parser] LLM JSON parse failed: {exc!r}")
        return None
    payload.setdefault("raw_query", user_text)
    payload["retrieved_knowledge"] = knowledge
    try:
        return CircuitSpec(**payload)
    except Exception as exc:
        print(f"[circuit_parser] LLM spec validation failed: {exc!r}")
        return None


# ---------------------------------------------------------------------------
# Top-level parse + generate + predict
# ---------------------------------------------------------------------------


def parse_to_spec(
    user_text: str,
    knowledge_store: Optional[KnowledgeStore] = None,
    top_k: int = 5,
) -> Tuple[CircuitSpec, str]:
    """Return (spec, backend_used). Backend is one of openai/anthropic/ollama/fallback."""
    store = knowledge_store or KnowledgeStore()
    knowledge = store.retrieve(user_text, top_k=top_k)

    for name, fn in [("openai", _try_openai), ("anthropic", _try_anthropic),
                     ("ollama", _try_ollama)]:
        response = fn(user_text, knowledge)
        spec = _llm_response_to_spec(response, user_text, knowledge) if response else None
        if spec is not None:
            return spec, name
    return fallback_parse(user_text, knowledge), "fallback"


def generate_from_spec(spec: CircuitSpec, seed: int = 42):
    """Map a CircuitSpec to an analog_circuit_gen topology call."""
    from ..analog_circuit_gen import generate_analog_circuit, ANALOG_TOPOLOGIES
    topo_names = {fn.__name__ for fn in ANALOG_TOPOLOGIES}
    topology = spec.topology if spec.topology in topo_names else "two_stage_opamp"
    return generate_analog_circuit(seed=seed, topology=topology,
                                   name=f"{topology}_from_nl_{seed}")


def _find_checkpoint() -> Optional[str]:
    """Pick the most appropriate checkpoint for analog inference."""
    for candidate in [
        "checkpoints/analog/best_model.pt",
        "checkpoints/mixed/best_model.pt",
        "checkpoints/best_model.pt",
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


def run_prediction(circuit) -> Optional[Dict[str, Any]]:
    """Run the trained GNN on the generated circuit.

    Returns the predict_circuit() dict, or None if no usable checkpoint
    exists or the loaded model's feature dim doesn't match the schema.
    """
    ckpt = _find_checkpoint()
    if ckpt is None:
        return None
    try:
        from ..predict import load_model, predict_circuit
        model = load_model(ckpt, device="cpu")
        return predict_circuit(model, circuit, device="cpu")
    except Exception as exc:
        print(f"[circuit_parser] prediction skipped: {exc!r}")
        return None
