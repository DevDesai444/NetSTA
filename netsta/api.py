"""
FastAPI backend for NetSTA.

Thin HTTP layer over netsta.service — the same core the CLI uses. Serves the
React frontend's needs: build a circuit (digital / analog / natural-language),
run the GNN + 4-agent advisory, return the graph + per-node predictions + the
design report as JSON.

    uvicorn netsta.api:app --reload --port 8000
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .service import (
    DEFAULT_CKPT,
    analyze_circuit,
    build_analog,
    build_digital,
    build_from_nl,
)

app = FastAPI(title="NetSTA API", version="0.1.0")
# The React dev server runs on a different origin; allow it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_ANALOG_TOPOLOGIES = [
    "two_stage_opamp", "folded_cascode", "diff_pair",
    "current_mirror", "common_source_amp",
]
_TOPOLOGY_HINT = {t: t for t in _ANALOG_TOPOLOGIES}


class DiagnoseRequest(BaseModel):
    kind: str = "digital"          # digital | analog | nl
    inputs: int = 8
    gates: int = 40
    outputs: int = 4
    topology: str = "two_stage_opamp"
    query: str = ""
    seed: int = 42
    process_node: str = "45nm"
    use_autogen: str = "auto"      # auto | never | force


@app.get("/api/health")
def health():
    vllm_url = os.getenv("NETSTA_VLLM_URL")
    return {
        "status": "ok",
        "checkpoint_present": bool(DEFAULT_CKPT and os.path.exists(DEFAULT_CKPT)),
        "checkpoint_path": DEFAULT_CKPT,
        "lora_endpoint": vllm_url,
        "lora_active": bool(vllm_url),
    }


@app.get("/api/topologies")
def topologies():
    return {"analog": _ANALOG_TOPOLOGIES}


@app.post("/api/diagnose")
def diagnose_endpoint(req: DiagnoseRequest):
    try:
        topology: Optional[str] = None
        meta = {}
        if req.kind == "digital":
            circuit = build_digital(req.inputs, req.gates, req.outputs, req.seed)
            topology = "combinational_logic"
        elif req.kind == "analog":
            if req.topology not in _ANALOG_TOPOLOGIES:
                raise HTTPException(400, f"unknown topology '{req.topology}'")
            circuit = build_analog(req.topology, req.seed)
            topology = _TOPOLOGY_HINT.get(req.topology)
        elif req.kind == "nl":
            if not req.query.strip():
                raise HTTPException(400, "query required for kind='nl'")
            circuit, spec, backend = build_from_nl(req.query)
            topology = _TOPOLOGY_HINT.get(getattr(spec, "topology", None))
            meta = {"parser_backend": backend, "parsed_topology": getattr(spec, "topology", None)}
        else:
            raise HTTPException(400, f"unknown kind '{req.kind}'")

        result = analyze_circuit(
            circuit, topology=topology,
            process_node=req.process_node, use_autogen=req.use_autogen,
        )
        result["meta"] = meta
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"analysis failed: {exc!r}")
