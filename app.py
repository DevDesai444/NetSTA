"""
NetSTA — interactive demo for the multi-task GNN + RAG pipeline.

Layout:
  - Landing header (title, subtitle, 4 metric cards from results/, arch diagram)
  - Sidebar: circuit type (Digital / Analog / Natural Language), per-mode
    controls, checkpoint selector, task visibility checkboxes
  - 6 tabs: Timing, Routability, Analog Performance, Circuit Search,
    Design Advisor, Model Info
  - Footer

Top-level imports are kept light. Heavy deps (chromadb, sentence_transformers,
umap-learn, sklearn.manifold) are imported lazily inside the tabs that need
them so first paint stays fast.
"""

import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from netsta.circuit_gen import Circuit, generate_circuit
from netsta.congestion import compute_demand_grid
from netsta.predict import load_model, predict_circuit


# ---------------------------------------------------------------------------
# Page config + dark EDA-styled CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NetSTA — AI for EDA",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Color palette: deep navy background, green/blue/orange accents.
_CSS = """
<style>
  :root {
    --eda-bg: #0e1117;
    --eda-panel: #161b27;
    --eda-panel-2: #1d2433;
    --eda-fg: #e5e9f0;
    --eda-muted: #8a93a6;
    --eda-green: #00d4aa;
    --eda-blue:  #3aa0ff;
    --eda-orange:#f59e0b;
    --eda-red:   #f43f5e;
  }
  .stApp { background-color: var(--eda-bg) !important; }
  .main .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
  h1, h2, h3, h4 { color: var(--eda-fg); letter-spacing: 0.2px; }
  .netsta-title {
    font-size: 2.1rem; font-weight: 700; margin: 0;
    background: linear-gradient(90deg, var(--eda-green) 0%, var(--eda-blue) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .netsta-subtitle {
    color: var(--eda-muted); font-size: 1.0rem; margin: 0.2rem 0 1.2rem 0;
  }
  .metric-card {
    background: linear-gradient(135deg, var(--eda-panel) 0%, var(--eda-panel-2) 100%);
    border-left: 4px solid var(--eda-green);
    padding: 0.8rem 1rem; border-radius: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.35);
  }
  .metric-card.blue   { border-left-color: var(--eda-blue); }
  .metric-card.orange { border-left-color: var(--eda-orange); }
  .metric-card.red    { border-left-color: var(--eda-red); }
  .metric-card .label {
    color: var(--eda-muted); font-size: 0.78rem; text-transform: uppercase;
    letter-spacing: 0.6px;
  }
  .metric-card .value {
    color: var(--eda-fg); font-size: 1.7rem; font-weight: 700; margin-top: 0.2rem;
  }
  .metric-card .delta { color: var(--eda-muted); font-size: 0.75rem; }
  .arch-box {
    background: var(--eda-panel); color: var(--eda-fg);
    border: 1px solid #2a3142; border-radius: 6px;
    padding: 0.55rem 0.7rem; text-align: center; font-family: 'SF Mono','Menlo',monospace;
    font-size: 0.85rem;
  }
  .arch-arrow {
    color: var(--eda-green); text-align: center; font-size: 1.4rem; line-height: 2rem;
  }
  .netsta-footer {
    color: var(--eda-muted); border-top: 1px solid #2a3142; padding-top: 0.8rem;
    margin-top: 1.5rem; font-size: 0.83rem; text-align: center;
  }
  /* Tabs */
  div[data-baseweb="tab-list"] button[aria-selected="true"] {
    color: var(--eda-green) !important; border-bottom-color: var(--eda-green) !important;
  }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_model(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        return None
    try:
        return load_model(checkpoint_path, device="cpu")
    except Exception as exc:
        st.warning(f"Failed to load `{checkpoint_path}`: {exc!r}")
        return None


def _list_checkpoints() -> List[str]:
    return sorted(glob.glob("checkpoints/**/*.pt", recursive=True))


def _read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _format_pct(v) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "--"
    return f"{100 * v:.1f}%"


def _format_float(v, digits=3) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "--"
    return f"{v:.{digits}f}"


def _read_metric_cards() -> List[dict]:
    """Pull headline numbers from whatever benchmark JSONs exist on disk.

    Falls back to the verification metrics from results/verification/.
    Every absent source becomes a 'Run benchmarks to populate' placeholder.
    """
    cards: List[dict] = []

    # Card 1: Slack R²
    slack_r2 = None
    source = None
    bench = _read_json("results/baseline_comparison.json") or {}
    ours = (bench.get("results") or {}).get("NetSTA") or {}
    m = (ours.get("metrics") or {}).get("slack") or {}
    if "r2" in m:
        slack_r2 = m["r2"]
        source = "baseline_comparison.json"
    if slack_r2 is None:
        ver = _read_json("results/verification/metrics.json") or {}
        m = (ver.get("metrics") or {}).get("slack") or {}
        if "r2" in m:
            slack_r2 = m["r2"]
            source = "verification/metrics.json"
    cards.append({
        "label": "Slack Regression R²",
        "value": _format_float(slack_r2),
        "delta": source or "Run benchmarks to populate",
        "accent": "blue",
    })

    # Card 2: Critical-path F1
    cp_f1 = None
    source = None
    if "ours" in locals() and ours:
        cp = (ours.get("metrics") or {}).get("critical_path") or {}
        if "f1" in cp:
            cp_f1, source = cp["f1"], "baseline_comparison.json"
    if cp_f1 is None:
        ver = _read_json("results/verification/metrics.json") or {}
        cp = (ver.get("metrics") or {}).get("critical_path") or {}
        if "f1" in cp:
            cp_f1, source = cp["f1"], "verification/metrics.json"
    cards.append({
        "label": "Critical-Path F1",
        "value": _format_float(cp_f1),
        "delta": source or "Run benchmarks to populate",
        "accent": "green",
    })

    # Card 3: Robustness ± std (Slack R²)
    rob = _read_json("results/robustness_analysis.json") or {}
    agg = rob.get("aggregate") or {}
    if "slack_r2" in agg:
        mean, std = agg["slack_r2"]
        cards.append({
            "label": "Slack R² (5-seed mean ± σ)",
            "value": f"{mean:.3f}",
            "delta": f"± {std:.3f}",
            "accent": "orange",
        })
    else:
        cards.append({
            "label": "Slack R² (5-seed)",
            "value": "--",
            "delta": "Run benchmarks to populate",
            "accent": "orange",
        })

    # Card 4: Speedup vs classical STA at largest tested size
    scl = _read_json("results/scaling_analysis.json") or {}
    timings = [r for r in (scl.get("inference_scaling") or []) if "error" not in r]
    if timings:
        biggest = max(timings, key=lambda r: r["n_gates"])
        sp = biggest.get("speedup")
        if isinstance(sp, (int, float)):
            label = f"GNN vs Classical STA @ {biggest['n_gates']} gates"
            if sp >= 1.0:
                value, accent = f"{sp:.1f}× faster", "green"
            else:
                value, accent = f"{1.0/sp:.1f}× slower", "red"
            cards.append({"label": label, "value": value,
                          "delta": "scaling_analysis.json", "accent": accent})
        else:
            cards.append({"label": "Inference speedup", "value": "--",
                          "delta": "Run benchmarks to populate", "accent": "red"})
    else:
        cards.append({
            "label": "Inference Speedup",
            "value": "--",
            "delta": "Run benchmarks to populate",
            "accent": "red",
        })

    return cards


def _render_metric_cards():
    cards = _read_metric_cards()
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            st.markdown(
                f"<div class='metric-card {card['accent']}'>"
                f"<div class='label'>{card['label']}</div>"
                f"<div class='value'>{card['value']}</div>"
                f"<div class='delta'>{card['delta']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_architecture_diagram():
    arch = [
        ("Circuit graph", "PyG Data\n31-dim nodes\n5-dim edges"),
        ("Backbone", "GATv2 / SymmetryAwareAttn\n4 layers · 4 heads\nResidual + BN"),
        ("Pooling", "Mean ⊕ Max\nGraph embedding\n[B, 512]"),
        ("Task heads", "Slack · CritPath\nCongestion · DRC\nAnalog Perf"),
    ]
    cols = st.columns([2, 1, 2, 1, 2, 1, 2])
    for i, (label, body) in enumerate(arch):
        with cols[i * 2]:
            st.markdown(
                f"<div class='arch-box'><b>{label}</b><br/>"
                + body.replace("\n", "<br/>") + "</div>",
                unsafe_allow_html=True,
            )
        if i < len(arch) - 1:
            with cols[i * 2 + 1]:
                st.markdown("<div class='arch-arrow'>→</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plot helpers (preserved from earlier versions of app.py)
# ---------------------------------------------------------------------------


def _layered_dag_positions(circuit: Circuit) -> Dict[str, tuple]:
    if circuit.positions:
        return {nid: (p[0] * 60, p[1] * 60) for nid, p in circuit.positions.items()}
    layers: Dict[int, List[str]] = {}
    layers[0] = list(circuit.primary_inputs)
    depth = 1
    for gid in circuit.gate_ids:
        layers.setdefault(depth, []).append(gid)
        depth = 1 + (depth % 5)
    layers[6] = list(circuit.primary_outputs)
    pos: Dict[str, tuple] = {}
    for d, nodes in layers.items():
        for i, nid in enumerate(nodes):
            pos[nid] = (d * 100, i * 50 - len(nodes) * 25)
    return pos


def plot_timing_dag(circuit, sta_results, predictions, color_mode):
    pos = _layered_dag_positions(circuit)
    node_timing = sta_results.get("node_timing", {}) if sta_results else {}
    node_order = circuit.primary_inputs + circuit.gate_ids + circuit.primary_outputs

    edge_x, edge_y = [], []
    for net in circuit.nets.values():
        if net.driver not in pos:
            continue
        x0, y0 = pos[net.driver]
        for sink in net.sinks:
            if sink not in pos:
                continue
            x1, y1 = pos[sink]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.7, color="rgba(120,120,120,0.55)"),
        hoverinfo="none",
    )

    nx_, ny_, colors, texts = [], [], [], []
    for i, nid in enumerate(node_order):
        x, y = pos.get(nid, (0, 0))
        nx_.append(x); ny_.append(y)
        t = node_timing.get(nid, {})
        slack = t.get("slack", 0.0)
        is_crit_gt = t.get("is_critical", False)
        node_type = circuit.nodes[nid].node_type
        if color_mode == "slack":
            colors.append(slack)
        elif color_mode == "critical_gt":
            colors.append(1.0 if is_crit_gt else 0.0)
        else:
            colors.append(
                float(predictions["predicted_critical_binary"][i])
                if predictions and "predicted_critical_binary" in predictions else 0.0
            )
        hover = (
            f"<b>{nid}</b><br>Type: {node_type}<br>"
            f"Slack: {slack:.4f}<br>Critical (GT): {is_crit_gt}"
        )
        if predictions and "predicted_slack" in predictions:
            hover += f"<br>Pred slack (norm): {predictions['predicted_slack'][i]:.3f}"
        texts.append(hover)

    if color_mode == "slack":
        colorscale = "RdYlGn"; cb_title = "Slack"
    else:
        colorscale = [[0, "#2ecc71"], [1, "#e74c3c"]]; cb_title = "Critical"
    node_trace = go.Scatter(
        x=nx_, y=ny_, mode="markers", hoverinfo="text", text=texts,
        marker=dict(
            size=12, color=colors, colorscale=colorscale,
            colorbar=dict(title=cb_title, thickness=12),
            line=dict(width=1, color="black"),
        ),
    )
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        showlegend=False, hovermode="closest",
        margin=dict(b=10, l=10, r=10, t=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=460, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161b27",
    ))
    return fig


def _aggregate_to_grid(values, positions, node_order, grid_size):
    gw, gh = grid_size
    if gw <= 0 or gh <= 0:
        return np.zeros((1, 1)), np.zeros((1, 1))
    sum_grid = np.zeros((gh, gw))
    count_grid = np.zeros((gh, gw))
    for i, nid in enumerate(node_order):
        pos = positions.get(nid)
        if pos is None:
            continue
        x, y = pos
        if 0 <= x < gw and 0 <= y < gh:
            sum_grid[y, x] += float(values[i])
            count_grid[y, x] += 1
    mean_grid = np.where(count_grid > 0, sum_grid / np.maximum(count_grid, 1), np.nan)
    return mean_grid, count_grid


def plot_heatmap(grid, title, colorscale="RdBu_r", zmin=None, zmax=None,
                 overlay_bboxes=None, height=420):
    fig = go.Figure(data=go.Heatmap(
        z=grid, colorscale=colorscale, zmin=zmin, zmax=zmax,
        hovertemplate="x=%{x}<br>y=%{y}<br>val=%{z:.3f}<extra></extra>",
        colorbar=dict(thickness=12),
    ))
    if overlay_bboxes:
        for (xmin, ymin, xmax, ymax) in overlay_bboxes:
            fig.add_shape(
                type="rect", x0=xmin - 0.5, y0=ymin - 0.5,
                x1=xmax + 0.5, y1=ymax + 0.5,
                line=dict(width=0.6, color="rgba(255,255,255,0.35)"),
                fillcolor="rgba(0,0,0,0)",
            )
    fig.update_layout(
        title=title, height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(title="X", scaleanchor="y", constrain="domain"),
        yaxis=dict(title="Y", autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161b27",
        font=dict(color="#e5e9f0"),
    )
    return fig


def _net_bboxes(circuit, cap=80):
    pos = circuit.positions
    out = []
    for net in circuit.nets.values():
        coords = [pos[p] for p in ([net.driver] + list(net.sinks)) if p in pos]
        if len(coords) < 2:
            continue
        xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
        out.append((min(xs), min(ys), max(xs), max(ys)))
        if len(out) >= cap:
            break
    return out


def plot_confusion(pred, target, title="Confusion matrix"):
    tp = int(((pred == 1) & (target == 1)).sum())
    fp = int(((pred == 1) & (target == 0)).sum())
    fn = int(((pred == 0) & (target == 1)).sum())
    tn = int(((pred == 0) & (target == 0)).sum())
    cm = np.array([[tn, fp], [fn, tp]])
    fig = go.Figure(data=go.Heatmap(
        z=cm, colorscale="Blues",
        x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
        text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}", textfont=dict(size=18),
        hoverinfo="z", showscale=False,
    ))
    fig.update_layout(
        title=title, height=320,
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161b27",
        font=dict(color="#e5e9f0"),
    )
    return fig, dict(tp=tp, fp=fp, fn=fn, tn=tn)


# ---------------------------------------------------------------------------
# Landing header
# ---------------------------------------------------------------------------

st.markdown(
    "<div class='netsta-title'>NetSTA: AI-Powered Circuit Analysis for EDA</div>"
    "<div class='netsta-subtitle'>Multi-task GNN for timing, routability, and "
    "analog performance prediction with LLM-assisted design.</div>",
    unsafe_allow_html=True,
)

_render_metric_cards()

st.markdown("&nbsp;")  # spacer
_render_architecture_diagram()
st.divider()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Circuit input")
    mode = st.radio(
        "Mode", ["Digital", "Analog", "Natural Language"], index=0, horizontal=False,
    )

    if mode == "Digital":
        st.caption("Generate a synthetic Nangate45 netlist.")
        num_inputs = st.slider("Primary inputs", 4, 24, 8)
        num_gates = st.slider("Logic gates", 10, 100, 30)
        num_outputs = st.slider("Primary outputs", 2, 12, 4)
        seed = st.number_input("Seed", 0, 99999, 42)
        analog_topology = None
        nl_query = None
    elif mode == "Analog":
        st.caption("Pick a hand-curated analog topology.")
        analog_topology = st.selectbox(
            "Topology",
            ["two_stage_opamp", "folded_cascode", "diff_pair",
             "current_mirror", "common_source_amp"],
            index=0,
        )
        seed = st.number_input("Seed", 0, 99999, 42)
        num_inputs = num_gates = num_outputs = None
        nl_query = None
    else:  # Natural Language
        st.caption("Describe the circuit in English; LLM/RAG turns it into a spec.")
        nl_query = st.text_area(
            "Design request",
            value="Design a two-stage Miller-compensated op-amp with 60dB "
                  "gain and 10MHz GBW in 180nm process.",
            height=110,
        )
        seed = st.number_input("Seed", 0, 99999, 42)
        parse_btn = st.button("Parse & Predict", type="primary")
        if parse_btn:
            st.session_state["nl_trigger"] = True
        analog_topology = None
        num_inputs = num_gates = num_outputs = None

    st.divider()
    st.header("Model")
    ckpts = _list_checkpoints()
    if ckpts:
        ckpt_path = st.selectbox("Checkpoint", ckpts, index=0)
    else:
        ckpt_path = "checkpoints/best_model.pt"
        st.info("No checkpoints found. Run `python3 -m netsta.train`.")

    st.divider()
    st.header("Tabs to show")
    show_timing = st.checkbox("Timing", True)
    show_routability = st.checkbox("Routability", True)
    show_analog = st.checkbox("Analog Performance", True)
    show_search = st.checkbox("Circuit Search", True)
    show_advisor = st.checkbox("Design Advisor", True)
    show_modelinfo = st.checkbox("Model Info", True)


# ---------------------------------------------------------------------------
# Build circuit + run predictions
# ---------------------------------------------------------------------------

def _build_circuit_digital(seed, n_in, n_gates, n_out):
    return generate_circuit(
        num_inputs=n_in, num_gates=n_gates, num_outputs=n_out,
        seed=int(seed), name=f"digital_seed{seed}",
    )


def _build_circuit_analog(seed, topology):
    from netsta.analog_circuit_gen import generate_analog_circuit
    return generate_analog_circuit(seed=int(seed), topology=topology)


def _build_circuit_from_nl(query, seed):
    """Run the RAG pipeline. Returns (circuit, spec, parser_backend)."""
    from netsta.rag import KnowledgeStore, parse_to_spec, generate_from_spec
    store = KnowledgeStore()
    spec, backend = parse_to_spec(query, knowledge_store=store)
    circuit = generate_from_spec(spec, seed=int(seed))
    return circuit, spec, backend


# Cache the resulting circuit/predictions on inputs we control via the sidebar.
def _cache_key():
    return (mode, ckpt_path,
            num_inputs, num_gates, num_outputs, analog_topology,
            (nl_query if mode == "Natural Language" else None),
            int(seed),
            # NL bump key — increments on Parse & Predict click.
            st.session_state.get("nl_count", 0))


# Increment NL trigger when button was clicked above.
if st.session_state.pop("nl_trigger", False):
    st.session_state["nl_count"] = st.session_state.get("nl_count", 0) + 1

cache_key = _cache_key()
if st.session_state.get("cache_key") != cache_key:
    st.session_state["cache_key"] = cache_key
    st.session_state.pop("predictions", None)
    st.session_state.pop("circuit", None)
    st.session_state.pop("spec", None)
    st.session_state.pop("parser_backend", None)

if "circuit" not in st.session_state:
    try:
        if mode == "Digital":
            circuit = _build_circuit_digital(seed, num_inputs, num_gates, num_outputs)
        elif mode == "Analog":
            circuit = _build_circuit_analog(seed, analog_topology)
        else:
            # Natural language: only build on first run *and* when triggered.
            if st.session_state.get("nl_count", 0) == 0:
                st.info("Click **Parse & Predict** in the sidebar to run the RAG pipeline.")
                st.stop()
            with st.spinner("Parsing natural-language query (this may build the vector DB)..."):
                circuit, spec, parser_backend = _build_circuit_from_nl(nl_query, seed)
            st.session_state["spec"] = spec
            st.session_state["parser_backend"] = parser_backend
        st.session_state["circuit"] = circuit
    except Exception as exc:
        st.error(f"Failed to build circuit: {exc!r}")
        st.stop()

circuit: Circuit = st.session_state["circuit"]

model = _get_model(ckpt_path)
predictions = None
if model is not None and "predictions" not in st.session_state:
    try:
        with st.spinner("Running GNN inference..."):
            predictions = predict_circuit(model, circuit, device="cpu")
        st.session_state["predictions"] = predictions
    except Exception as exc:
        st.error(f"Prediction failed (likely feature-dim mismatch): {exc!r}")
elif model is not None:
    predictions = st.session_state["predictions"]
elif model is None:
    st.warning(f"Could not load checkpoint `{ckpt_path}`.")

# STA / analog-STA results carried on predictions; fall back to running STA.
sta_results = predictions["sta_results"] if predictions else None
if sta_results is None:
    if getattr(circuit, "is_analog", False):
        from netsta.analog_sta import run_analog_sta
        sta_results = run_analog_sta(circuit)
    else:
        from netsta.sta import run_sta as _run_sta
        sta_results = _run_sta(circuit)


# ---------------------------------------------------------------------------
# Compose the visible tab list
# ---------------------------------------------------------------------------

tab_defs = []
if show_timing:      tab_defs.append("Timing")
if show_routability: tab_defs.append("Routability")
if show_analog:      tab_defs.append("Analog Performance")
if show_search:      tab_defs.append("Circuit Search")
if show_advisor:     tab_defs.append("Design Advisor")
if show_modelinfo:   tab_defs.append("Model Info")

if not tab_defs:
    st.info("Enable at least one tab in the sidebar.")
    st.stop()

tabs = dict(zip(tab_defs, st.tabs(tab_defs)))


# ---- Tab: Timing ----------------------------------------------------------
if "Timing" in tabs:
    with tabs["Timing"]:
        if getattr(circuit, "is_analog", False):
            st.info("Timing tab is for digital circuits. Switch to Digital mode "
                    "in the sidebar, or open the Analog Performance tab.")
        else:
            st.subheader("Slack on the circuit DAG")
            color_options = ["slack", "critical_gt"]
            if predictions and "predicted_critical_binary" in predictions:
                color_options.append("critical_pred")
            color_mode = st.radio(
                "Color nodes by", color_options,
                format_func=lambda x: {
                    "slack": "Slack (GT)",
                    "critical_gt": "Critical path (GT)",
                    "critical_pred": "Critical path (predicted)",
                }[x], horizontal=True,
            )
            st.plotly_chart(
                plot_timing_dag(circuit, sta_results, predictions, color_mode),
                use_container_width=True,
            )

            if predictions and "predicted_slack" in predictions:
                gt = np.asarray(predictions["ground_truth_slack"])
                pr = np.asarray(predictions["predicted_slack"])
                mse = float(np.mean((gt - pr) ** 2))
                mae = float(np.mean(np.abs(gt - pr)))
                ss_tot = float(((gt - gt.mean()) ** 2).sum())
                r2 = 1 - float(((gt - pr) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
                a, b, c = st.columns(3)
                a.metric("Slack MSE", f"{mse:.5f}")
                b.metric("Slack R²", f"{r2:.3f}")
                c.metric("Slack MAE", f"{mae:.4f}")

                if "predicted_critical_binary" in predictions:
                    gtc = np.asarray(predictions["ground_truth_critical"]).astype(int)
                    prc = np.asarray(predictions["predicted_critical_binary"]).astype(int)
                    acc = float((gtc == prc).mean())
                    tp = int(((prc == 1) & (gtc == 1)).sum())
                    fp = int(((prc == 1) & (gtc == 0)).sum())
                    fn = int(((prc == 0) & (gtc == 1)).sum())
                    prec = tp / max(tp + fp, 1)
                    rec = tp / max(tp + fn, 1)
                    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
                    a, b, c, d = st.columns(4)
                    a.metric("CP Accuracy", _format_pct(acc))
                    b.metric("CP Precision", _format_pct(prec))
                    c.metric("CP Recall", _format_pct(rec))
                    d.metric("CP F1", _format_float(f1))


# ---- Tab: Routability -----------------------------------------------------
if "Routability" in tabs:
    with tabs["Routability"]:
        if predictions is None:
            st.info("Load a checkpoint to see predicted routability.")
        else:
            node_order = circuit.primary_inputs + circuit.gate_ids + circuit.primary_outputs
            gw, gh = circuit.grid_size

            st.subheader("Congestion: predicted vs ground-truth RUDY")
            preds_map = predictions.get("predictions", {}) or {}
            if "congestion" in preds_map:
                pred_vals = np.asarray(preds_map["congestion"])
                pred_grid, _ = _aggregate_to_grid(pred_vals, circuit.positions, node_order, (gw, gh))
                demand_grid, _ = compute_demand_grid(circuit)
                demand_np = np.array(demand_grid).T
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(
                        plot_heatmap(
                            pred_grid, "GNN prediction (mean per cell)",
                            colorscale="RdBu_r", zmin=0.0, zmax=1.0,
                            overlay_bboxes=_net_bboxes(circuit),
                        ),
                        use_container_width=True,
                    )
                with c2:
                    st.plotly_chart(
                        plot_heatmap(demand_np, "Raw RUDY demand", colorscale="RdBu_r"),
                        use_container_width=True,
                    )

                gt_vals = np.asarray(predictions["ground_truth"]["congestion"])
                cong_mse = float(np.mean((gt_vals - pred_vals) ** 2))
                cong_mae = float(np.mean(np.abs(gt_vals - pred_vals)))
                ss_tot = float(((gt_vals - gt_vals.mean()) ** 2).sum())
                cong_r2 = 1 - float(((gt_vals - pred_vals) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
                m1, m2, m3 = st.columns(3)
                m1.metric("Congestion MSE", f"{cong_mse:.5f}")
                m2.metric("Congestion MAE", f"{cong_mae:.4f}")
                m3.metric("Congestion R²", f"{cong_r2:.3f}")
            else:
                st.caption("Loaded model does not include the congestion head.")

            st.markdown("&nbsp;")
            st.subheader("DRC hotspots")
            if "drc" in preds_map:
                drc_logits = np.asarray(preds_map["drc"])
                drc_prob = 1.0 / (1.0 + np.exp(-drc_logits))
                threshold = st.slider("Hotspot threshold", 0.0, 1.0, 0.5, 0.05)
                drc_pred = (drc_prob >= threshold).astype(int)
                drc_target = np.asarray(predictions["ground_truth"]["drc"]).astype(int)
                prob_grid, _ = _aggregate_to_grid(drc_prob, circuit.positions, node_order, (gw, gh))
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.plotly_chart(
                        plot_heatmap(prob_grid, "Hotspot probability",
                                     colorscale="Reds", zmin=0.0, zmax=1.0),
                        use_container_width=True,
                    )
                with c2:
                    fig_cm, cm = plot_confusion(drc_pred, drc_target, title="DRC confusion")
                    st.plotly_chart(fig_cm, use_container_width=True)
                acc = float((drc_pred == drc_target).mean())
                prec = cm["tp"] / max(cm["tp"] + cm["fp"], 1)
                rec = cm["tp"] / max(cm["tp"] + cm["fn"], 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-8)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", _format_pct(acc))
                m2.metric("Precision", _format_pct(prec))
                m3.metric("Recall", _format_pct(rec))
                m4.metric("F1", _format_float(f1))
            else:
                st.caption("Loaded model does not include the DRC head.")


# ---- Tab: Analog Performance ---------------------------------------------
if "Analog Performance" in tabs:
    with tabs["Analog Performance"]:
        is_analog = bool(getattr(circuit, "is_analog", False))
        if not is_analog:
            st.info("This circuit is digital. Switch to Analog mode in the sidebar "
                    "to see analog-perf predictions.")
        elif predictions is None:
            st.info("Load a checkpoint to see analog-perf predictions.")
        else:
            preds_map = predictions.get("predictions", {}) or {}
            node_order = circuit.primary_inputs + circuit.gate_ids + circuit.primary_outputs
            if "analog_performance" in preds_map:
                ap = np.asarray(preds_map["analog_performance"])
                if ap.ndim == 2 and ap.shape[1] == 2:
                    gbw = ap[:, 0]
                    parasitic = ap[:, 1]

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Predicted GBW score per node**")
                        fig = go.Figure(data=go.Bar(
                            x=node_order, y=gbw,
                            marker_color=["#00d4aa" if v >= 0.5 else "#f43f5e" for v in gbw],
                        ))
                        fig.update_layout(
                            height=320, paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="#161b27", font=dict(color="#e5e9f0"),
                            margin=dict(l=20, r=20, t=10, b=60),
                            yaxis_title="GBW score (norm)",
                            xaxis=dict(tickangle=-45),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        st.markdown("**Predicted parasitic impact per node**")
                        fig = go.Figure(data=go.Bar(
                            x=node_order, y=parasitic,
                            marker_color=["#f59e0b" if v >= 0.5 else "#3aa0ff" for v in parasitic],
                        ))
                        fig.update_layout(
                            height=320, paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="#161b27", font=dict(color="#e5e9f0"),
                            margin=dict(l=20, r=20, t=10, b=60),
                            yaxis_title="Parasitic impact (norm)",
                            xaxis=dict(tickangle=-45),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Symmetry analysis from circuit.symmetry_groups.
                    st.markdown("**Symmetry / matched-pair groups**")
                    groups = circuit.symmetry_groups or {}
                    if not groups:
                        st.caption("No matched groups in this topology.")
                    else:
                        by_group: Dict[int, List[str]] = {}
                        for nid, g in groups.items():
                            by_group.setdefault(int(g), []).append(nid)
                        rows = []
                        for g, members in sorted(by_group.items()):
                            rows.append({
                                "group": g,
                                "members": ", ".join(members),
                                "device_types": ", ".join(
                                    circuit.nodes[m].node_type for m in members
                                ),
                            })
                        st.dataframe(rows, use_container_width=True, height=180)
                else:
                    st.caption(
                        "analog_performance prediction is not the expected [N, 2] shape — "
                        "this checkpoint may not have been trained with the analog head."
                    )
            else:
                st.caption(
                    "Loaded checkpoint has no analog_performance head. "
                    "Train with `--tasks ...,analog_performance` to enable this view."
                )


# ---- Tab: Circuit Search --------------------------------------------------
if "Circuit Search" in tabs:
    with tabs["Circuit Search"]:
        st.subheader("Vector similarity over GNN graph embeddings")
        if model is None:
            st.warning("Load a checkpoint to run similarity search.")
        else:
            from netsta.similarity.circuit_index import CircuitIndex, embed_circuit
            from netsta.similarity.search import find_by_property, find_similar
            from netsta.dataset import (
                AnalogCircuitDataset, MixedCircuitDataset, NetSTADataset,
            )

            top_l, top_r = st.columns([1, 2])
            with top_l:
                search_type = st.selectbox(
                    "Index dataset",
                    ["digital", "analog", "mixed"],
                    index=(1 if getattr(circuit, "is_analog", False) else 0),
                )
                num_index = st.slider("Index size", 10, 100, 20, step=5)
                rebuild = st.button("Build / refresh index")
            with top_r:
                st.caption(
                    "The index embeds a small dataset of the chosen type using the "
                    "loaded model's graph-pooling output. Cached at `./circuit_embeddb/`."
                )

            @st.cache_resource
            def _get_index(circuit_type, n, ckpt):
                if circuit_type == "digital":
                    ds = NetSTADataset(root="data", num_circuits=n, seed=42)
                elif circuit_type == "analog":
                    ds = AnalogCircuitDataset(root="data_analog", num_circuits=n, seed=42)
                else:
                    ds = MixedCircuitDataset(root="data_mixed", num_circuits=n, seed=42)
                idx = CircuitIndex(
                    model=model, device="cpu",
                    collection_name=f"circuit_embeddings_{circuit_type}",
                )
                idx.build(ds, force=False, verbose=False)
                return ds, idx

            try:
                if rebuild:
                    _get_index.clear()
                with st.spinner("Building / loading similarity index..."):
                    idx_ds, ckt_index = _get_index(search_type, num_index, ckpt_path)
                st.success(f"Index ready — {ckt_index.count()} circuits.")
            except Exception as exc:
                st.error(f"Index build failed: {exc!r}")
                ckt_index = None
                idx_ds = None

            if ckt_index is not None:
                search_mode = st.radio(
                    "Mode", ["Generate and search", "Filter by target specs"],
                    horizontal=True,
                )
                top_k = st.slider("Top-k", 1, 20, 5)
                anchor_emb = None
                results = []
                if search_mode == "Generate and search":
                    if predictions is not None:
                        anchor_emb = embed_circuit(model, predictions["data"], device="cpu")
                        results = find_similar(
                            predictions["data"], model, ckt_index, top_k=top_k,
                        )
                    else:
                        st.caption("Run a prediction in the sidebar first.")
                else:
                    f1, f2 = st.columns(2)
                    with f1:
                        f_type = st.selectbox("Type filter", ["any", "digital", "analog"])
                        f_nodes = st.slider("Node range", 4, 100, (4, 100))
                    with f2:
                        f_cong = st.slider("Max congestion ≤", 0.0, 1.0, 1.0, 0.05)
                        f_gain = st.slider("Min avg_gbw_score", 0.0, 1.0, 0.0, 0.05)
                    target_specs = {
                        "num_gates": {"min": f_nodes[0], "max": f_nodes[1]},
                        "max_congestion": {"max": f_cong},
                        "avg_gbw_score": {"min": f_gain},
                    }
                    if f_type != "any":
                        target_specs["circuit_type"] = f_type
                    if predictions is not None:
                        anchor_emb = embed_circuit(model, predictions["data"], device="cpu")
                        results = find_by_property(
                            target_specs, ckt_index, top_k=top_k,
                            anchor_circuit=predictions["data"], model=model,
                        )
                    else:
                        results = find_by_property(target_specs, ckt_index, top_k=top_k)

                if results:
                    rows = [
                        {
                            "rank": i,
                            "id": h.get("id"),
                            "similarity": (
                                f"{h['similarity']:+.4f}" if h.get("similarity") is not None else "--"
                            ),
                            "type": (h.get("metadata") or {}).get("circuit_type", "-"),
                            "nodes": (h.get("metadata") or {}).get("num_gates", "-"),
                            "max_cong": round((h.get("metadata") or {}).get("max_congestion", 0), 3),
                            "cp_len": (h.get("metadata") or {}).get("critical_path_length", "-"),
                            "avg_slack": round((h.get("metadata") or {}).get("avg_slack", 0), 3),
                            "avg_gbw_score": round((h.get("metadata") or {}).get("avg_gbw_score", 0), 3),
                            "name": (h.get("metadata") or {}).get("circuit_name", ""),
                        }
                        for i, h in enumerate(results, start=1)
                    ]
                    st.dataframe(rows, use_container_width=True, height=260)
                else:
                    st.info("No matches.")

                # Embedding-space projection.
                st.markdown("#### Embedding space (anchor circled)")
                try:
                    bundle = ckt_index.get_all()
                    embs = bundle["embeddings"]
                    metas = bundle["metadatas"]
                    if len(embs) >= 3:
                        use_umap = False; reducer = None
                        try:
                            import umap
                            reducer = umap.UMAP(
                                n_components=2,
                                n_neighbors=min(15, max(2, len(embs) - 1)),
                                min_dist=0.1, random_state=42,
                            )
                            coords = reducer.fit_transform(embs)
                            use_umap = True
                        except Exception:
                            from sklearn.manifold import TSNE
                            perplexity = min(30, max(2, (len(embs) - 1) // 3))
                            coords = TSNE(
                                n_components=2, perplexity=perplexity,
                                random_state=42, init="random",
                            ).fit_transform(embs)
                        text = [
                            f"{m.get('circuit_name', '')} "
                            f"({m.get('circuit_type', '-')}, n={m.get('num_gates', 0)})"
                            for m in metas
                        ]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=coords[:, 0], y=coords[:, 1],
                            mode="markers",
                            marker=dict(
                                size=10,
                                color=[0 if m.get("circuit_type") == "digital" else 1
                                       for m in metas],
                                colorscale=[[0, "#3aa0ff"], [1, "#f43f5e"]],
                                line=dict(width=0.5, color="white"),
                            ),
                            text=text,
                            hovertemplate="%{text}<extra></extra>",
                            name="circuits",
                        ))
                        if anchor_emb is not None:
                            try:
                                if use_umap and reducer is not None:
                                    a = reducer.transform(anchor_emb.reshape(1, -1))
                                else:
                                    from sklearn.manifold import TSNE as _TSNE
                                    combined = np.vstack([embs, anchor_emb.reshape(1, -1)])
                                    perp = min(30, max(2, (len(combined) - 1) // 3))
                                    a = _TSNE(
                                        n_components=2, perplexity=perp,
                                        random_state=42, init="random",
                                    ).fit_transform(combined)[-1:]
                                fig.add_trace(go.Scatter(
                                    x=a[:, 0], y=a[:, 1], mode="markers",
                                    marker=dict(
                                        size=20, color="rgba(255,215,0,0)",
                                        line=dict(width=3, color="#f59e0b"),
                                        symbol="circle-open",
                                    ),
                                    name="anchor",
                                    hovertemplate="anchor<extra></extra>",
                                ))
                            except Exception as ex:
                                st.caption(f"(anchor projection skipped: {ex})")
                        fig.update_layout(
                            height=420,
                            margin=dict(l=40, r=40, t=20, b=40),
                            xaxis_title=("UMAP-1" if use_umap else "t-SNE-1"),
                            yaxis_title=("UMAP-2" if use_umap else "t-SNE-2"),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161b27",
                            font=dict(color="#e5e9f0"),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.caption(
                            f"Index has only {len(embs)} row(s); need ≥3 for a "
                            "meaningful projection."
                        )
                except Exception as exc:
                    st.error(f"Scatter plot failed: {exc!r}")


# ---- Tab: Design Advisor --------------------------------------------------
if "Design Advisor" in tabs:
    with tabs["Design Advisor"]:
        st.subheader("LLM/RAG-driven design recommendations")
        st.caption(
            "Pulls relevant tips from the curated knowledge base "
            "(50 entries, ChromaDB vector store), ranks GNN-predicted "
            "bottlenecks, and renders an actionable report. LLMs (OpenAI / "
            "Anthropic / Ollama) are used when their env vars / local "
            "servers are configured; otherwise a deterministic template "
            "fallback runs offline."
        )

        spec_from_nl = st.session_state.get("spec")
        parser_backend = st.session_state.get("parser_backend")

        if spec_from_nl is None:
            st.info(
                "Use **Natural Language** mode in the sidebar to parse a "
                "free-text query, or run the advisor on the current circuit "
                "below (best-effort spec from circuit metadata)."
            )
            if st.button("Run advisor on current circuit (no NL spec)"):
                from netsta.rag import KnowledgeStore, advise, CircuitSpec
                spec = CircuitSpec(
                    topology=("two_stage_opamp" if getattr(circuit, "is_analog", False)
                              else "digital_netlist"),
                    target_specs={},
                    process_node="130nm",
                    num_stages=1,
                    compensation="none",
                    raw_query=f"Current circuit '{circuit.name}'",
                )
                with st.spinner("Running advisor..."):
                    report = advise(spec, predictions, knowledge_store=KnowledgeStore())
                st.session_state["last_report"] = report
        else:
            from netsta.rag import KnowledgeStore, advise
            if st.button("Re-run advisor"):
                with st.spinner("Running advisor..."):
                    report = advise(spec_from_nl, predictions, knowledge_store=KnowledgeStore())
                st.session_state["last_report"] = report
            elif "last_report" not in st.session_state:
                with st.spinner("Running advisor..."):
                    report = advise(spec_from_nl, predictions, knowledge_store=KnowledgeStore())
                st.session_state["last_report"] = report

        report = st.session_state.get("last_report")
        if report is None:
            st.stop()

        a, b, c = st.columns(3)
        a.metric("Parser backend", parser_backend or "n/a")
        b.metric("Advisor backend", report.backend)
        c.metric("Bottlenecks found", len(report.bottlenecks))

        st.markdown("##### Parsed CircuitSpec")
        st.code(report.spec.model_dump_json(indent=2), language="json")

        st.markdown("##### Bottlenecks")
        if report.bottlenecks:
            st.dataframe(
                [{
                    "task": b.task, "severity": round(b.severity, 3),
                    "location": b.location, "summary": b.summary,
                } for b in report.bottlenecks],
                use_container_width=True, height=180,
            )
        else:
            st.caption("None above threshold.")

        st.markdown("##### Recommendations")
        for r in report.recommendations:
            st.markdown(f"- {r}")

        if report.confidence_scores:
            st.markdown("##### Confidence")
            st.json(report.confidence_scores)


# ---- Tab: Model Info ------------------------------------------------------
if "Model Info" in tabs:
    with tabs["Model Info"]:
        st.subheader("Architecture, training, and benchmark snapshots")

        if model is None:
            st.warning("No checkpoint loaded.")
        else:
            cfg = model.config
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            arch_rows = [
                {"field": "active_tasks",       "value": list(cfg.active_tasks)},
                {"field": "hidden_dim",         "value": cfg.hidden_dim},
                {"field": "num_layers",         "value": cfg.num_layers},
                {"field": "num_heads",          "value": cfg.num_heads},
                {"field": "node_feature_dim",   "value": cfg.node_feature_dim},
                {"field": "edge_feature_dim",   "value": cfg.edge_feature_dim},
                {"field": "use_attention",      "value": cfg.use_attention},
                {"field": "use_residual",       "value": cfg.use_residual},
                {"field": "use_symmetry_attn",  "value": cfg.use_symmetry_attention},
                {"field": "task_weights",       "value": cfg.task_weights},
                {"field": "parameters",         "value": f"{n_params:,}"},
            ]
            st.dataframe(arch_rows, use_container_width=True, height=320)

        # Training curves if available.
        st.markdown("##### Training curves")
        tlog = _read_json("results/training_log.json")
        if tlog and tlog.get("history"):
            history = tlog["history"]
            epochs = [h["epoch"] for h in history]
            train_loss = [h["train"]["loss"] for h in history]
            val_loss   = [h["val"]["loss"] for h in history]
            best_ep = tlog.get("best_epoch")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="train", line=dict(color="#3aa0ff")))
            fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="val", line=dict(color="#00d4aa")))
            if best_ep:
                fig.add_vline(x=best_ep, line=dict(color="#f59e0b", dash="dash"),
                              annotation_text=f"best epoch {best_ep}", annotation_position="top")
            fig.update_layout(
                height=320, margin=dict(l=30, r=30, t=20, b=40),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161b27",
                font=dict(color="#e5e9f0"),
                xaxis_title="Epoch", yaxis_title="Loss",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("`results/training_log.json` not found — run "
                       "`python3 scripts/benchmark_training_curves.py` to populate.")

        st.markdown("##### Per-task metrics (verification run)")
        ver = _read_json("results/verification/metrics.json")
        if ver:
            mrows = []
            for task, m in (ver.get("metrics") or {}).items():
                row = {"task": task}
                for k, v in m.items():
                    row[k] = round(float(v), 4) if isinstance(v, (int, float)) else v
                mrows.append(row)
            st.dataframe(mrows, use_container_width=True, height=200)
        else:
            st.caption("`results/verification/metrics.json` not found — "
                       "run `python3 -m netsta.evaluate`.")

        st.markdown("##### Saved confusion / ROC plots")
        plots = sorted(glob.glob("results/**/*.png", recursive=True))
        if plots:
            cols = st.columns(min(3, len(plots)))
            for i, p in enumerate(plots):
                with cols[i % len(cols)]:
                    st.image(p, caption=p, use_container_width=True)
        else:
            st.caption("No plots in `results/`. Run "
                       "`python3 scripts/run_all_benchmarks.sh` to populate.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    "<div class='netsta-footer'>Built with PyTorch Geometric, GATv2, ChromaDB, "
    "and LLM APIs &nbsp;|&nbsp; Targeting VLSI Place &amp; Route Optimization</div>",
    unsafe_allow_html=True,
)
