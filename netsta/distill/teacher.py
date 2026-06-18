"""
Groq Llama-3.3-70B teacher: turn grounded scenarios into expert role responses.

  - Round-robin across all configured GROQ_API_KEY_N env vars (1..N).
  - On 429 rate-limit, mark the key as cooling and IMMEDIATELY try the next.
  - Cooling keys are retried only after the Retry-After window passes.
  - Threaded worker pool: tens of in-flight requests across the key fleet so
    we burn the combined tokens/min budget, not one key's at a time.

Output per scenario: a structured JSON expert response matching the role's
output_schema_hint. We validate every response is well-formed JSON; malformed
ones are retried once, then dropped (logged) — we don't want garbage in the
student's training set.
"""

import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

from .roles import ROLES, Role


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Primary teacher: strong reasoning model with 200K TPD free-tier quota.
DEFAULT_MODEL = "openai/gpt-oss-120b"
# Fallback ladder: when the primary model hits its daily-token cap, the worker
# automatically promotes the next model. Ordered by reasoning quality desc;
# llama-3.1-8b has the highest TPD (500K) so we can almost always fall through.
FALLBACK_MODELS = [
    "openai/gpt-oss-120b",       # 200K TPD, strongest available
    "openai/gpt-oss-20b",        # 200K TPD, decent reasoning
    "llama-3.3-70b-versatile",   # 100K TPD, often exhausted quickly
    "llama-3.1-8b-instant",      # 500K TPD, fastest fallback
]
KEY_ENV_VAR = "GROQ_API_KEY_"   # GROQ_API_KEY_1, _2, _3, _4


def load_keys() -> List[str]:
    keys: List[str] = []
    # Support an optional `~/.netsta_secrets/groq.env` style file via env preload.
    i = 1
    while True:
        v = os.environ.get(f"{KEY_ENV_VAR}{i}")
        if not v:
            break
        keys.append(v)
        i += 1
    if not keys and os.environ.get("GROQ_API_KEY"):
        keys.append(os.environ["GROQ_API_KEY"])
    return keys


@dataclass
class _KeyState:
    key: str
    cooling_until: float = 0.0
    in_flight: int = 0


class KeyPool:
    """Thread-safe round-robin pool with rate-limit-aware failover."""

    def __init__(self, keys: List[str]):
        if not keys:
            raise RuntimeError("no Groq keys found in env (GROQ_API_KEY_1, ...)")
        self._states = [_KeyState(k) for k in keys]
        self._lock = threading.Lock()
        self._counter = 0

    def acquire(self, max_wait_s: float = 30.0) -> _KeyState:
        """Pick the next available (not-cooling) key. Sleep if all are cooling."""
        deadline = time.time() + max_wait_s
        while True:
            with self._lock:
                now = time.time()
                ready = [s for s in self._states if s.cooling_until <= now]
                if ready:
                    # Pick the readiest key with the fewest in-flight requests.
                    s = min(ready, key=lambda s: (s.in_flight, self._counter))
                    s.in_flight += 1
                    self._counter += 1
                    return s
                soonest = min(s.cooling_until for s in self._states)
            wait = max(0.05, min(soonest - time.time(), deadline - time.time()))
            if wait <= 0:
                # Even cooling — give up and return the soonest-ready anyway.
                with self._lock:
                    s = min(self._states, key=lambda s: s.cooling_until)
                    s.in_flight += 1
                    return s
            time.sleep(wait)

    def release(self, state: _KeyState, cool_for: float = 0.0):
        with self._lock:
            state.in_flight = max(0, state.in_flight - 1)
            if cool_for > 0:
                state.cooling_until = max(state.cooling_until, time.time() + cool_for)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _scenario_to_user_prompt(role: Role, scenario: Dict) -> str:
    """Frame the scenario as the user message the teacher reacts to."""
    parts = [f"# Circuit under review: {scenario['circuit_name']}"]
    parts.append(f"Topology: {scenario['topology']}   Process: {scenario['process_node']}")
    parts.append(f"Size: {scenario['num_nodes']} nodes, {scenario['num_edges']} edges")
    parts.append("")
    parts.append("## GNN per-task predictions (summary)")
    parts.append(json.dumps(scenario["predictions_summary"], indent=2))

    if scenario.get("bottlenecks"):
        parts.append("")
        parts.append("## Flagged bottlenecks (your input)")
        for b in scenario["bottlenecks"]:
            parts.append(
                f"  - [{b['task']}] severity={b['severity']:.2f} "
                f"violation_type={b['violation_type']} at {b['location']}: {b['summary']}"
            )

    if scenario.get("peer_findings"):
        parts.append("")
        parts.append("## Peer agent findings (for cross-task reconciliation)")
        for p in scenario["peer_findings"]:
            parts.append(
                f"  - {p['agent']} proposes fix={p['fix']} ({p['action']}), "
                f"outcomes={p['outcomes']}, conflicts={p['conflicts']}, "
                f"effort={p['effort']}"
            )

    if scenario.get("retrieved_facts"):
        parts.append("")
        parts.append("## Retrieved knowledge-graph facts (the ONLY fixes you may recommend)")
        for f in scenario["retrieved_facts"]:
            parts.append(f"  - {f}")

    if scenario.get("retrieved_text"):
        parts.append("")
        parts.append("## Retrieved EDA literature")
        for t in scenario["retrieved_text"]:
            parts.append(f"  - {t}")

    parts.append("")
    parts.append("## Task")
    parts.append(
        f"As {role.display_name}, produce your structured analysis as a single JSON "
        f"object. Schema:\n  {role.output_schema_hint}"
    )
    parts.append("Reply with the JSON object ONLY — no commentary, no markdown fences.")
    return "\n".join(parts)


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first top-level JSON object out of the model's reply."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        # strip ``` and optional language tag
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# HTTP call
# ---------------------------------------------------------------------------


_EXHAUSTED_MODELS: set = set()
_exhausted_lock = threading.Lock()


def _next_model_after(current: str, ladder: List[str]) -> Optional[str]:
    """Return any non-exhausted model in the ladder (preferring downstream).

    Tries models after `current` first (the natural fallback direction), then
    wraps to earlier ones in case `current` itself was a fallback that has now
    become available again.
    """
    with _exhausted_lock:
        try:
            i = ladder.index(current)
        except ValueError:
            i = -1
        ordered = ladder[i + 1 :] + ladder[: max(0, i + 1)]
        for m in ordered:
            if m != current and m not in _EXHAUSTED_MODELS:
                return m
    return None


def _mark_exhausted(model: str) -> None:
    with _exhausted_lock:
        if model not in _EXHAUSTED_MODELS:
            _EXHAUSTED_MODELS.add(model)
            print(f"  [teacher] model {model} exhausted for the day; promoting next fallback")


def _call_groq(
    pool: KeyPool, role: Role, scenario: Dict,
    model: str = DEFAULT_MODEL, max_attempts: int = 4,
    fallback_models: Optional[List[str]] = None,
) -> Optional[dict]:
    """One scenario -> structured teacher response, with key+model failover.

    On 429 errors the error message tells us whether it's a per-minute (TPM)
    rate limit or a per-day (TPD) quota exhaustion. TPM = cool the key and
    retry. TPD = promote the next model in the fallback ladder.
    """
    fallback_models = fallback_models or FALLBACK_MODELS
    user_msg = _scenario_to_user_prompt(role, scenario)
    current_model = model
    last_err = None
    for attempt in range(max_attempts):
        if current_model in _EXHAUSTED_MODELS:
            nxt = _next_model_after(current_model, fallback_models)
            if nxt is None:
                return None  # full ladder exhausted
            current_model = nxt
        payload = {
            "model": current_model,
            "messages": [
                {"role": "system", "content": role.system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.2,
            "max_tokens": 1500,
        }
        # JSON-mode is supported by the gpt-oss models; for llama it's flaky
        # and produces validation errors, so we rely on the prompt + extract.
        if current_model.startswith("openai/gpt-oss"):
            payload["response_format"] = {"type": "json_object"}
        state = pool.acquire()
        try:
            resp = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {state.key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=90,
            )
            if resp.status_code == 429:
                body = resp.text.lower()
                retry_after = float(resp.headers.get("Retry-After", "5"))
                # Daily-quota exhaustion vs per-minute rate limit:
                #   TPD body contains "tokens per day" or "(tpd)" verbatim.
                #   TPM body says "tokens per minute" or "(tpm)" and Retry-After
                #   is typically < 60s — we cool the key for that duration.
                is_tpd = ("tokens per day" in body or "(tpd)" in body
                          or "per day" in body and "limit" in body)
                if is_tpd:
                    pool.release(state, cool_for=0.5)
                    _mark_exhausted(current_model)
                    nxt = _next_model_after(current_model, fallback_models)
                    if nxt is None:
                        return None
                    current_model = nxt
                else:
                    # TPM — just cool the key briefly and retry the same model.
                    pool.release(state, cool_for=min(retry_after, 30.0))
                continue
            if resp.status_code >= 500:
                pool.release(state, cool_for=2.0)
                continue
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                pool.release(state)
                continue
            pool.release(state)
            j = resp.json()
            content = j["choices"][0]["message"]["content"]
            parsed = _extract_json(content)
            if parsed is None:
                last_err = f"unparseable JSON: {content[:200]!r}"
                continue
            return parsed
        except requests.RequestException as exc:
            pool.release(state, cool_for=1.0)
            last_err = repr(exc)
            time.sleep(0.5)
    return None


# ---------------------------------------------------------------------------
# Top-level batch
# ---------------------------------------------------------------------------


@dataclass
class DistillPair:
    """One supervised training example for a student LoRA."""
    role: str
    system_prompt: str
    user_prompt: str
    assistant_json: dict
    circuit_name: str = ""
    scenario_id: int = 0


def distill_role(
    role_name: str,
    scenarios: List[dict],
    pool: KeyPool,
    workers: int = 8,
    model: str = DEFAULT_MODEL,
    progress_every: int = 25,
) -> List[DistillPair]:
    """Run the teacher over every scenario in parallel; return valid pairs."""
    role = ROLES[role_name]
    pairs: List[DistillPair] = []
    failed = 0

    def _work(i_sc):
        i, sc = i_sc
        resp = _call_groq(pool, role, sc, model=model)
        return i, sc, resp

    print(f"[{role_name}] distilling {len(scenarios)} scenarios with {workers} workers ...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_work, (i, sc)) for i, sc in enumerate(scenarios)]
        for done, fut in enumerate(as_completed(futures), 1):
            i, sc, resp = fut.result()
            if resp is None:
                failed += 1
                continue
            pairs.append(DistillPair(
                role=role_name,
                system_prompt=role.system_prompt,
                user_prompt=_scenario_to_user_prompt(role, sc),
                assistant_json=resp,
                circuit_name=sc.get("circuit_name", ""),
                scenario_id=i,
            ))
            if done % progress_every == 0 or done == len(scenarios):
                rate = done / (time.time() - t0)
                eta = (len(scenarios) - done) / max(rate, 1e-3)
                print(
                    f"  [{role_name} {done}/{len(scenarios)}] "
                    f"valid={len(pairs)} failed={failed} "
                    f"{rate:.1f}/s eta {eta/60:.1f}min"
                )
    print(f"[{role_name}] done: {len(pairs)} valid pairs, {failed} failures, "
          f"{(time.time() - t0)/60:.1f}min wall-clock")
    return pairs


def save_pairs(pairs: List[DistillPair], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps({
                "role": p.role,
                "system": p.system_prompt,
                "user": p.user_prompt,
                "assistant": json.dumps(p.assistant_json, ensure_ascii=False),
                "circuit_name": p.circuit_name,
                "scenario_id": p.scenario_id,
            }) + "\n")
