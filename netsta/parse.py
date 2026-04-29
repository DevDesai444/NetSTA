"""
CLI entry point: parse English -> circuit -> prediction -> design report.

Usage:
  python3 -m netsta.parse "Design a two-stage Miller-compensated op-amp ..."

Optional flags:
  --top-k          (RAG retrieval depth, default 5)
  --seed           (circuit generation seed, default 42)
  --json           (emit a single JSON payload instead of pretty text)
"""

import argparse
import json
import sys
from textwrap import indent

from timingnet.rag import (
    KnowledgeStore,
    advise,
    generate_from_spec,
    parse_to_spec,
    run_prediction,
)


def _print_section(title: str, body: str = "") -> None:
    print(f"\n=== {title} ===")
    if body:
        print(body)


def _format_text_report(report, backend, spec, prediction_was_run):
    out = []
    out.append(f"Backend (parser): {backend}")
    out.append(f"Backend (advisor): {report.backend}")
    out.append("")
    out.append("--- CircuitSpec ---")
    out.append(spec.model_dump_json(indent=2))
    if not prediction_was_run:
        out.append("")
        out.append("--- GNN predictions ---")
        out.append("Skipped: no compatible checkpoint found under checkpoints/.")
    else:
        out.append("")
        out.append("--- GNN predictions (per-task summary) ---")
        for task, vals in report.predictions.items():
            arr = vals
            if isinstance(arr, list) and arr and isinstance(arr[0], list):
                # 2-D
                import numpy as np
                a = np.asarray(arr)
                out.append(
                    f"  {task}: shape={a.shape} "
                    f"col0 min/mean/max={a[:,0].min():.3f}/{a[:,0].mean():.3f}/{a[:,0].max():.3f} "
                    f"col1 min/mean/max={a[:,1].min():.3f}/{a[:,1].mean():.3f}/{a[:,1].max():.3f}"
                )
            else:
                import numpy as np
                a = np.asarray(arr, dtype=float)
                if a.size == 0:
                    out.append(f"  {task}: <empty>")
                else:
                    out.append(
                        f"  {task}: n={a.size} min={a.min():.3f} "
                        f"mean={a.mean():.3f} max={a.max():.3f}"
                    )
    out.append("")
    out.append("--- Bottlenecks ---")
    if not report.bottlenecks:
        out.append("  (none above threshold)")
    for b in report.bottlenecks:
        out.append(f"  [{b.task}] severity={b.severity:.2f}  loc={b.location}")
        out.append(indent(b.summary, "    "))
    out.append("")
    out.append("--- Recommendations ---")
    for r in report.recommendations:
        out.append(f"  - {r}")
    if report.confidence_scores:
        out.append("")
        out.append("--- Confidence scores ---")
        for task, score in report.confidence_scores.items():
            out.append(f"  {task}: {score:.2f}")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Parse a natural-language circuit description and run the NetSTA pipeline.",
    )
    parser.add_argument("query", help="English description of the circuit to design.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="RAG retrieval depth (default 5).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for circuit generation (default 42).")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON instead of pretty text.")
    args = parser.parse_args(argv)

    print(f"User query: {args.query}")
    _print_section("Building knowledge store")
    store = KnowledgeStore()
    print(f"  retrieval backend = {store.backend}")

    _print_section("Parsing query → CircuitSpec")
    spec, parser_backend = parse_to_spec(args.query, knowledge_store=store, top_k=args.top_k)
    print(f"  parser backend    = {parser_backend}")
    print(f"  topology          = {spec.topology}")
    print(f"  target_specs      = {spec.target_specs}")
    print(f"  process_node      = {spec.process_node}")
    print(f"  num_stages        = {spec.num_stages}")
    print(f"  compensation      = {spec.compensation}")

    _print_section("Generating circuit from spec")
    circuit = generate_from_spec(spec, seed=args.seed)
    print(f"  circuit name      = {circuit.name}")
    print(f"  nodes             = {len(circuit.nodes)}")
    print(f"  nets              = {len(circuit.nets)}")
    print(f"  grid_size         = {circuit.grid_size}")
    if circuit.symmetry_groups:
        print(f"  matched groups    = {len(set(circuit.symmetry_groups.values()))}")

    _print_section("Running GNN inference")
    predictions = run_prediction(circuit)
    if predictions is None:
        print("  No compatible checkpoint at checkpoints/. Running advice on spec only.")
    else:
        print(f"  active tasks      = {predictions.get('active_tasks')}")
        print(f"  num node predictions = {len(predictions.get('node_ids', []))}")

    _print_section("Building design report")
    report = advise(spec, predictions, knowledge_store=store)

    if args.json:
        payload = {
            "parser_backend": parser_backend,
            "advisor_backend": report.backend,
            "spec": spec.model_dump(),
            "predictions": report.predictions,
            "bottlenecks": [b.model_dump() for b in report.bottlenecks],
            "recommendations": report.recommendations,
            "confidence_scores": report.confidence_scores,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_text_report(report, parser_backend, spec, predictions is not None))


if __name__ == "__main__":
    sys.exit(main() or 0)
