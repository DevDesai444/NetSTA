"""Run the full distillation pipeline.

  1. Generate diverse grounded scenarios from the real-netlist dataset.
  2. Round-robin all 4 Groq keys to call Llama-3.3-70B as teacher.
  3. Save (system, user, assistant_json) pairs per role for SFT.

    python3 scripts/run_distillation.py --n-per-role 200

Env: GROQ_API_KEY_1..4 must be set (or source ~/.netsta_secrets/groq.env first).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from netsta.distill.scenarios import build_scenarios, save_scenarios
from netsta.distill.teacher import KeyPool, distill_role, load_keys, save_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data_real/graphs.pt")
    ap.add_argument("--scenarios-dir", default="data_real/distill/scenarios")
    ap.add_argument("--pairs-dir", default="data_real/distill/pairs")
    ap.add_argument("--n-per-role", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12,
                    help="parallel HTTP workers across the key pool")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="openai/gpt-oss-120b",
                    help="primary teacher; falls back through the model ladder")
    ap.add_argument("--roles", default="supervisor,timing,drc,optimization",
                    help="comma-separated roles to distill (skip the others)")
    args = ap.parse_args()

    print("=== Phase 1: building scenarios ===")
    scenarios = build_scenarios(
        dataset_path=args.dataset, n_per_role=args.n_per_role, seed=args.seed,
    )
    save_scenarios(scenarios, args.scenarios_dir)

    keys = load_keys()
    print(f"\n=== Phase 2: teacher distillation (pool of {len(keys)} keys) ===")
    if not keys:
        print("FATAL: no GROQ_API_KEY_n env vars set. Source ~/.netsta_secrets/groq.env first.")
        sys.exit(2)
    pool = KeyPool(keys)

    selected = [r.strip() for r in args.roles.split(",") if r.strip()]
    os.makedirs(args.pairs_dir, exist_ok=True)
    summary = {}
    for role in selected:
        scs = [s.to_dict() for s in scenarios.get(role, [])]
        if not scs:
            print(f"  [skip] no scenarios for {role}")
            continue
        pairs = distill_role(role, scs, pool, workers=args.workers, model=args.model)
        out_path = os.path.join(args.pairs_dir, f"{role}.jsonl")
        save_pairs(pairs, out_path)
        summary[role] = {"scenarios": len(scs), "pairs": len(pairs), "path": out_path}

    print("\n=== Distillation summary ===")
    print(json.dumps(summary, indent=2))
    with open(os.path.join(args.pairs_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
