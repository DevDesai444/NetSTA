# NetSTA

A graph neural network that predicts gate-level timing from real circuit
netlists, paired with four specialist LLM agents that read those predictions
and recommend concrete fixes.

Static timing analysis is exact but gives no guidance on *what to change*; it
just reports numbers. NetSTA does two things on top of a netlist: (1) a 5M-param
hybrid GNN (directional STA prior + GraphGPS transformer) predicts per-node
slack, arrival/required time, critical-path membership, routing congestion and
DRC hotspots in one pass, and (2) a panel of four LoRA-specialized agents
distilled from a strong teacher LLM turns those predictions into ranked,
grounded recommendations — *"this path is slow, here are the fixes, and here's
the one that would create a DRC problem if you also apply it."*

The model is trained and evaluated on **real benchmark netlists** — ITC'99,
ISCAS-85, EPFL, and OpenABC-D's 47 large industrial designs (AES, ethernet,
JPEG, RISC-V cores, FPU).

---

## How it works

```
netlist (.bench / .v)
   │  parse → Nangate45 cells, cut flip-flops at register boundaries
   ▼
Circuit ── STA / RUDY congestion / DRC labelling ──► PyG graph (17-dim nodes, 5-dim edges)
   │
   ▼
Big GNN: directional STA prior + GraphGPS transformer (5M params)
   ├─ STA branch: forward (max-agg AT) + backward (min-agg RT) directional sweeps
   ├─ GPS branch: Laplacian PE + local GINE + global self-attention × N
   └─ fused: 6 heads (compositional slack, arrival, required, critical, congestion, DRC)
   │
   ▼
4 LoRA-specialized agents (Qwen2.5-7B + per-role adapter on vLLM)
   ├─ Supervisor       routes predictions, aggregates final report
   ├─ TimingAgent      diagnoses slack / critical-path, recommends closure fixes
   ├─ DRCAgent         diagnoses DRC / congestion, recommends layout fixes
   └─ OptimizationAgent cross-task PPA reasoning, reconciles conflicts
   │
   ▼  grounded in hybrid retrieval (FAISS text + Neo4j/NetworkX KG + ChromaDB)
DesignReport: ranked bottlenecks + per-violation fixes + cross-task conflicts
```

### Data — real netlists at industrial scale

The synthetic-DAG generator that shipped earlier produced graphs with little
long-range path structure, so a graph-blind MLP kept pace with the GNN. Real
netlists have genuine depth and reconvergence, which is where message passing
matters. `benchmark_import.py` parses three formats and maps every gate onto
the Nangate45 cell library:

- **ITC'99** `.bench` (b01..b22, sequential designs, DFFs cut at register boundaries)
- **ISCAS-85** gate-primitive `.v` (10 classic combinational circuits)
- **EPFL** flattened-`assign` `.v` (arithmetic + random/control benchmarks)
- **OpenABC-D** AIG `.bench` (47 large industrial designs from real chips:
  AES, DES, JPEG, ethernet, FPU, RISC-V BlackParrot, Rocket cores, ariane)

Each netlist is windowed into fan-in cones rooted at endpoints (real POs +
DFF-D capture points), and every cone is labelled under multiple sampled clock
targets so the slack distribution spans timing-met and timing-violated regimes
across all sizes. Build: **11,580 graphs from 231 source circuits**, sizes
8–7,600 nodes, 8.3 M total nodes. Splits are **by source circuit**, so test
topologies are genuinely unseen.

> **What the labels are.** Ground truth comes from this repo's own STA + RUDY
> congestion + DRC estimators — a fast deterministic surrogate, not a
> commercial signoff flow. A high score means *"accurate learned surrogate for
> STA on real netlists"*, which is what this model is for. Real OpenSTA
> signoff labels are a planned next step, not a current claim.

### Model — GraphGPS + STA prior

The 5M-parameter backbone runs two branches in parallel:

- **STA prior**: a directional message-passing module that mirrors the STA
  relaxation — a forward sweep accumulates arrival time with max-aggregation,
  a backward sweep propagates required time with min-aggregation (seeded from
  a per-graph arrival-time pool, supervised by an auxiliary clock-period loss).
  Provides the physics inductive bias.
- **GraphGPS transformer**: Laplacian positional encoding (top-k eigenvectors
  of the symmetric normalized Laplacian) + N stacked blocks of `(local GINE
  message-passing → global multi-head self-attention → FFN)`. Captures the
  long-range structure standard GNNs can't.

The two branches are fused into a shared per-node embedding the heads read.
The slack head is compositional — it computes `required − arrival` and adds a
zero-init residual — so the STA identity holds by construction. A raw-feature
residual keeps the per-node Liberty features available to every head, so the
model is provably at least as expressive as the MLP baseline.

---

## Results

Real netlists (schema v9), evaluated on **held-out source circuits** — the
split is by circuit, so no test topology is seen in training. Full detail in
[`results/MODEL_RESULTS.md`](results/MODEL_RESULTS.md).

| Task | Metric | Value |
|---|---|---|
| Arrival time | R² | `0.70` |
| Required time | R² | `0.73` |
| Slack | R² | `0.66` |
| Critical path | AUC | `0.80` |
| DRC hotspot | AUC | `0.89` |
| Congestion | R² | `0.34` |

### Held-out named benchmarks

Famous circuits excluded from training entirely:

| Circuit | Slack R² | Arrival R² | Critical AUC | DRC AUC |
|---|---|---|---|---|
| ISCAS-85 `c6288` (16×16 multiplier) | `-0.22` | `0.48` | `0.49` | `0.73` |
| EPFL `multiplier` (64×64) | `0.13` | `0.30` | `0.57` | `0.61` |
| ITC'99 `b19` (≈259K gates) | `0.14` | `0.50` | `0.47` | `0.78` |

---

## Design advisory — 4 LoRA-distilled specialist agents

Four agents turn predictions into recommendations. Each agent is a separate
LoRA adapter distilled from a strong teacher LLM (Groq Llama-3.3-70B + GPT-OSS)
onto a shared Qwen2.5-7B base, served via vLLM with per-request adapter
hot-swap:

- **Supervisor** routes predictions to the specialists and aggregates the report.
- **Timing agent** ranks worst-slack / critical nodes, classifies the violation,
  pulls closure fixes (sizing, buffering, restructuring, useful skew).
- **DRC agent** ranks DRC/congestion hotspots and pulls layout fixes (spreading,
  layer promotion, filler), filtered by process node.
- **Optimization agent** does the cross-task step: it checks whether a timing
  fix would *create* a DRC problem (or vice-versa) and reconciles the two.

Every recommendation is grounded in **hybrid retrieval**, not generated from
thin air:

- **FAISS** over chunked EDA text — the "why/how".
- **Knowledge graph** (`topology → violation → fix → outcome`, plus
  `CONFLICTS_WITH` edges) — the structured relationships and the conflict
  reasoning. Uses Neo4j when `NEO4J_URI` is set, otherwise an in-memory
  NetworkX graph with the same queries.
- **ChromaDB** over the GNN's own circuit embeddings — empirical precedent.

### How the agents are made distinct

Generic role-prompts on one model produces four prompts, not four specialists.
We do real **task-specific knowledge distillation**:

1. Generate ~100 grounded scenarios per role from the real-netlist dataset
   (each scenario = circuit + GNN predictions + retrieved KG facts + bottlenecks)
2. Teacher LLM (Llama-3.3-70B / GPT-OSS-120B via Groq, with multi-key
   round-robin and TPD-aware model failover) emits a high-quality structured
   role response per scenario
3. SFT a separate Qwen2.5-7B + LoRA adapter on each role's (scenario,
   response) pairs via PEFT + TRL on Modal A100
4. Deploy vLLM with `--enable-lora` and `--max-loras 4`; the AutoGen
   `RoundRobinGroupChat` routes each agent to its own adapter via the model
   field

When the vLLM endpoint isn't reachable, a deterministic orchestrator runs the
same agents and tools (CI-tested) and produces the same typed `DesignReport`.

---

## Serving

- **CLI** — `python3 -m netsta.diagnose_cli --kind digital --gates 40`
- **API** — `uvicorn netsta.api:app --port 8000` (`POST /api/diagnose`)
- **Web** — a Vite + React dashboard in `web/` (circuit graph coloured by metric,
  bottlenecks, recommendations, agent transcript with LoRA-adapter tags)

```bash
# backend
uvicorn netsta.api:app --reload --port 8000
# frontend (separate shell)
cd web && npm install && npm run dev   # proxies /api → :8000
# point AutoGen at your vLLM endpoint (per-agent LoRA routing)
export NETSTA_VLLM_URL=https://<account>--netsta-vllm-serve.modal.run/v1
```

---

## Setup

```bash
pip install -e ".[retrieval,api,demo]"      # add ,agents for the LLM panel
bash scripts/fetch_benchmarks.sh            # ITC'99 + ISCAS + EPFL + OpenABC
python3 scripts/build_real_dataset.py --bench-root benchmarks --out data_real/graphs.pt
```

Training runs on a GPU via Modal (`scripts/modal_train.py`):

```bash
# big GraphGPS + STA model (5M params) on A100
python3 -m modal run scripts/modal_train.py \
    --backbone graphgps_sta --hidden 64 --num-layers 8 \
    --split-mode circuit --epochs 80

# held-out named benchmarks
python3 scripts/eval_named.py --checkpoint checkpoints_real/bignet/best_model.pt

# distill 4 LoRA students
export GROQ_API_KEY_1=... GROQ_API_KEY_2=...   # multi-key round-robin
python3 scripts/run_distillation.py --n-per-role 100
python3 -m modal run scripts/train_lora_students.py
python3 -m modal deploy scripts/serve_vllm_loras.py
```

---

## Repository layout

```
netsta/
  benchmark_import.py   .bench / Verilog → Circuit (cell mapping, flop cutting, cones)
  real_dataset.py       real-netlist graph dataset + circuit-level splits
  big_model.py          GraphGPS + STA-prior backbone (5M params)
  model.py              base directional STA backbone + 6 heads + registry
  train.py              training loop (warmup→cosine, soft-temp anneal, AMP)
  sta.py / graph_builder.py / congestion.py / drc.py   labelling
  retrieval/            FAISS index + knowledge graph + hybrid fusion
  agents/               4-agent pipeline (deterministic + AutoGen→vLLM backends)
  distill/              roles + scenario builder + Groq teacher worker
  service.py / api.py / diagnose_cli.py   serving core, REST, CLI
web/                    React + Vite frontend
scripts/                fetch_benchmarks, build_real_dataset, modal_train,
                        eval_named, run_distillation, train_lora_students,
                        serve_vllm_loras
tests/                  importer, STA, model, big_model, retrieval, agents,
                        distill, RAG, similarity
```

## Earlier experiments

Baseline comparisons (MLP / GCN / GraphSAGE), backbone ablations, and the
original synthetic-data prototype live on the
[`research`](https://github.com/DevDesai444/NetSTA/tree/research) branch.

---

MIT License.
