# NetSTA

A graph neural network that predicts gate-level timing from real circuit
netlists, paired with a multi-agent advisor that reads those predictions and
recommends concrete fixes.

Static timing analysis is exact but gives no guidance on *what to change*; it
just reports numbers. NetSTA does two things on top of a netlist: (1) a GNN
predicts per-node slack, arrival/required time, critical-path membership,
routing congestion and DRC hotspots in one pass, and (2) a panel of four agents
turns those predictions into ranked, grounded recommendations — "this path is
slow, here are the fixes, and here's the one that would create a DRC problem if
you also apply it."

The model is trained and evaluated on **real benchmark netlists** (ITC'99 and
ISCAS-85), not random graphs.

---

## How it works

```
netlist (.bench / .v)
   │  parse → Nangate45 cells, cut flip-flops at register boundaries
   ▼
Circuit ── STA / RUDY congestion / DRC labelling ──► PyG graph (17-dim nodes, 5-dim edges)
   │
   ▼
GNN: directional STA-aware backbone
   ├─ forward sweep  (max-aggregation)  → arrival-time embedding
   ├─ backward sweep (min-aggregation)  → required-time embedding
   └─ 6 heads: slack (compositional RT−AT), arrival, required,
               critical-path, congestion, DRC
   │
   ▼
4-agent advisory (Supervisor → Timing, DRC → Optimization)
   grounded in hybrid retrieval: FAISS (EDA text) + knowledge graph + ChromaDB
   │
   ▼
DesignReport: ranked bottlenecks + per-violation fixes + cross-task conflicts
```

### Data — real netlists

The synthetic-DAG generator that shipped earlier produced graphs with little
long-range path structure, so a graph-blind MLP kept pace with the GNN. Real
netlists have genuine depth and reconvergence, which is where message passing
matters. `benchmark_import.py` parses the standard `.bench` format (ITC'99) and
ISCAS-85 gate-primitive Verilog, maps each gate onto the Nangate45 cell library
(decomposing wide gates into 2-input trees), and **cuts sequential elements**
(a flop's Q becomes a launch point, its D a capture point) to expose the
inter-register combinational graph the way STA times a real design.

Each netlist is then windowed into many fan-in cones, and every graph is
labelled under a clock target sampled around its own critical path — so the
slack distribution (and the critical-path label) spans timing-met and
timing-violated regimes across all circuit sizes. The current build is **2,665
graphs from 30 source circuits** (20 ITC'99 + 10 ISCAS-85), 8–7,299 nodes.

Splits are **by source circuit**, so test topologies are genuinely unseen.

> **What the labels are.** Ground truth comes from this repo's own STA +
> RUDY-congestion + DRC estimators — a fast, deterministic surrogate, not a
> commercial signoff flow. So a high score means "accurate learned surrogate
> for STA on real netlists," which is what this model is for. Real OpenSTA
> signoff labels are a planned next step, not a current claim.

### Model

The backbone mirrors the STA relaxation: a forward pass accumulates arrival
time with max-aggregation, a backward pass propagates required time with
min-aggregation (seeded from a per-graph arrival-time pool, supervised by an
auxiliary clock-period loss). The slack head is compositional — it computes
`required − arrival` and adds a zero-initialised residual — so the STA identity
holds by construction. A raw-feature residual keeps the per-node Liberty
features available to every head, so the model is at least as expressive as the
MLP baseline on top of whatever propagation adds. Soft-max aggregation is
annealed toward hard max over training to close the train/eval gap.

---

## Results

Real netlists, schema v9. Numbers are honest measurements from the runs in
`results/MODEL_RESULTS.md`; see that file for the full config.

| Task | In-distribution (random split) | Cross-circuit (held-out topologies) |
|---|---|---|
| Slack R² | `RESULT_SLACK_INDIST` | `RESULT_SLACK_XCKT` |
| Arrival-time R² | `RESULT_AT_INDIST` | `RESULT_AT_XCKT` |
| Required-time R² | `RESULT_RT_INDIST` | `RESULT_RT_XCKT` |
| Critical-path AUC | `RESULT_CP_INDIST` | `RESULT_CP_XCKT` |
| DRC AUC | `RESULT_DRC_INDIST` | `RESULT_DRC_XCKT` |
| Congestion R² | `RESULT_CONG_INDIST` | `RESULT_CONG_XCKT` |

### Held-out named benchmarks

Famous circuits excluded from training entirely:

| Circuit | Slack R² | Arrival R² | Critical AUC | DRC AUC |
|---|---|---|---|---|
| ISCAS-85 `c6288` (16×16 multiplier) | `RESULT_C6288_SLACK` | `RESULT_C6288_AT` | `RESULT_C6288_CP` | `RESULT_C6288_DRC` |
| ITC'99 `b19` | `RESULT_B19_SLACK` | `RESULT_B19_AT` | `RESULT_B19_CP` | `RESULT_B19_DRC` |

The arrival/required-time heads carry the strongest signal (they predict a
directly-propagated quantity); slack is harder because it's the difference of
two predictions. The honest read: the directional backbone learns timing
structure well on real netlists, and clearly beats a graph-blind MLP where path
structure dominates — but it is a surrogate, not a replacement for signoff STA.

---

## Design advisory (the agent panel)

Four agents turn predictions into recommendations:

- **Supervisor** routes predictions to the specialists and aggregates the report.
- **Timing agent** ranks worst-slack / critical nodes, classifies the violation,
  and pulls closure fixes (sizing, buffering, restructuring, useful skew).
- **DRC agent** ranks DRC/congestion hotspots and pulls layout fixes (spreading,
  layer promotion, filler), filtered by process node.
- **Optimization agent** does the cross-task step: it checks whether a timing fix
  would *create* a DRC problem (or vice-versa) and reconciles the two.

Every recommendation is grounded in **hybrid retrieval**, not generated from
thin air:

- **FAISS** over chunked EDA text — the "why/how".
- **Knowledge graph** (`topology → violation → fix → outcome`, plus
  `CONFLICTS_WITH` edges) — the structured relationships and the conflict
  reasoning. Uses Neo4j when `NEO4J_URI` is set, otherwise an in-memory
  NetworkX graph with the same queries.
- **ChromaDB** over the GNN's own circuit embeddings — empirical precedent.

The agents run as a real AutoGen `RoundRobinGroupChat` when `autogen-agentchat`
and an LLM key are present; otherwise a deterministic orchestrator runs the same
agents and tools and produces the same typed `DesignReport`. Both paths are
exercised; the deterministic one is what CI tests.

---

## Serving

- **CLI** — `python3 -m netsta.diagnose_cli --kind digital --gates 40`
- **API** — `uvicorn netsta.api:app --port 8000` (`POST /api/diagnose`)
- **Web** — a Vite + React dashboard in `web/` (circuit graph coloured by metric,
  bottlenecks, recommendations, agent transcript)

```bash
# backend
uvicorn netsta.api:app --reload --port 8000
# frontend (separate shell)
cd web && npm install && npm run dev   # proxies /api → :8000
```

---

## Setup

```bash
pip install -e ".[retrieval,api,demo]"      # add ,agents for the LLM panel
bash scripts/fetch_benchmarks.sh            # ITC'99 + ISCAS into benchmarks/
python3 scripts/build_real_dataset.py --bench-root benchmarks --out data_real/graphs.pt
```

Training runs on a GPU via Modal (`scripts/modal_train.py`) or locally on
Apple MPS / CPU — the model is small (minutes per run):

```bash
# cloud GPU (uploads dataset, trains, pulls checkpoint back)
python3 -m modal run scripts/modal_train.py --split-mode random --epochs 300

# held-out named benchmarks
python3 scripts/eval_named.py --checkpoint checkpoints_real/circuit/best_model.pt
```

---

## Repository layout

```
netsta/
  benchmark_import.py   .bench / Verilog → Circuit (cell mapping, flop cutting, cones)
  real_dataset.py       real-netlist graph dataset + circuit-level splits
  model.py              directional STA backbone + 6 heads
  train.py              training loop (warmup→cosine, soft-temp anneal, AMP)
  sta.py / graph_builder.py / congestion.py / drc.py   labelling
  retrieval/            FAISS index + knowledge graph + hybrid fusion
  agents/               4-agent pipeline (deterministic + AutoGen backends)
  service.py / api.py / diagnose_cli.py   serving core, REST, CLI
web/                    React + Vite frontend
scripts/                fetch_benchmarks, build_real_dataset, modal_train, eval_named
tests/                  importer, STA, model, retrieval, agents, RAG, similarity
```

## Limitations

- Labels are this repo's STA/RUDY/DRC estimators, not commercial signoff.
- Slack is the difference of two learned quantities, so its R² trails the
  arrival/required heads.
- The analog path (small-signal estimates for op-amp topologies) is synthetic.
- The full agent panel needs an LLM backend; without one it runs deterministically.

## Earlier experiments

Baseline comparisons (MLP / GCN / GraphSAGE), backbone ablations, and the
original synthetic-data prototype live on the
[`research`](https://github.com/DevDesai444/NetSTA/tree/research) branch.

---

MIT License.
