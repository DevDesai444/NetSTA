#!/usr/bin/env bash
# Fetch the open benchmark netlists the real dataset is built from.
# ITC'99 ships as .bench (sequential); ISCAS-85 as gate-primitive Verilog.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/benchmarks"
cd "$ROOT/benchmarks"

if [ ! -d itc99 ]; then
  git clone --depth 1 https://github.com/cad-polito-it/I99T.git itc99
fi
if [ ! -d iscas ]; then
  git clone --depth 1 https://github.com/santoshsmalagi/Benchmarks.git iscas
fi

echo "Fetched: ITC'99 (.bench) + ISCAS (.v) under benchmarks/"
echo "Next: python3 scripts/build_real_dataset.py --bench-root benchmarks --out data_real/graphs.pt"
