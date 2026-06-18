"""
Command-line design advisory.

    python3 -m netsta.diagnose_cli --kind digital --gates 40
    python3 -m netsta.diagnose_cli --kind analog --topology two_stage_opamp
    python3 -m netsta.diagnose_cli --kind nl --query "two-stage miller opamp, 60dB"
    python3 -m netsta.diagnose_cli --kind bench --path benchmarks/itc99/i99t/b14/b14.bench

Runs STA + the GNN (or STA ground truth if no checkpoint) and the 4-agent
advisory pipeline, then prints the ranked bottlenecks and recommendations.
"""

import argparse
import json

from .service import (
    analyze_circuit,
    build_analog,
    build_digital,
    build_from_bench,
    build_from_nl,
)

_TOPOLOGY_HINT = {
    "two_stage_opamp": "two_stage_opamp", "folded_cascode": "folded_cascode",
    "diff_pair": "diff_pair", "current_mirror": "current_mirror",
    "common_source_amp": "common_source_amp",
}


def main():
    ap = argparse.ArgumentParser(description="NetSTA design advisory")
    ap.add_argument("--kind", choices=["digital", "analog", "nl", "bench"], default="digital")
    ap.add_argument("--gates", type=int, default=40)
    ap.add_argument("--inputs", type=int, default=8)
    ap.add_argument("--outputs", type=int, default=4)
    ap.add_argument("--topology", default="two_stage_opamp")
    ap.add_argument("--query", default="")
    ap.add_argument("--path", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--process-node", default="45nm")
    ap.add_argument("--checkpoint", default=None, help="override checkpoint path")
    ap.add_argument("--json", action="store_true", help="emit raw JSON")
    args = ap.parse_args()

    topology = None
    if args.kind == "digital":
        circuit = build_digital(args.inputs, args.gates, args.outputs, args.seed)
        topology = "combinational_logic"
    elif args.kind == "analog":
        circuit = build_analog(args.topology, args.seed)
        topology = _TOPOLOGY_HINT.get(args.topology)
    elif args.kind == "nl":
        circuit, spec, backend = build_from_nl(args.query or "two-stage miller opamp")
        topology = _TOPOLOGY_HINT.get(getattr(spec, "topology", None))
        print(f"[parser backend: {backend}] topology={getattr(spec,'topology',None)}")
    else:
        circuit = build_from_bench(args.path, args.seed)
        topology = "combinational_logic"

    kwargs = {"topology": topology, "process_node": args.process_node}
    if args.checkpoint:
        kwargs["checkpoint_path"] = args.checkpoint
    result = analyze_circuit(circuit, **kwargs)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    r = result["report"]
    print(f"\nCircuit: {result['circuit_name']}  "
          f"({result['num_nodes']} nodes, {result['num_edges']} edges, "
          f"clk {result['clock_period_ns']:.3f} ns)")
    print(f"Predictions from: {result['prediction_source']}   "
          f"Advisory backend: {r['backend']}")

    print("\nBottlenecks:")
    for b in r["bottlenecks"]:
        print(f"  [{b['task']}/{b['violation_type']}] severity {b['severity']:.2f} "
              f"@ {b['location']}\n      {b['summary']}")
    if not r["bottlenecks"]:
        print("  (none flagged)")

    print("\nRecommendations:")
    for rec in r["recommendations"][:8]:
        cf = f"  conflicts: {', '.join(rec['conflicts'])}" if rec["conflicts"] else ""
        print(f"  ({rec['agent']}) {rec['fix']} — {rec['action']}")
        print(f"      conf {rec['confidence']:.2f}  effort {rec['effort']}  "
              f"outcomes {rec['outcomes']}{cf}")

    print("\nPanel transcript:")
    for t in r["transcript"]:
        print(f"  {t['agent']}: {t['summary']}")


if __name__ == "__main__":
    main()
