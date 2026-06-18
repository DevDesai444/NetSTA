import { useMemo, useState } from "react";
import type { GraphNode } from "../api";

type Metric = "slack" | "critical" | "congestion" | "drc";

const METRICS: { key: Metric; label: string }[] = [
  { key: "slack", label: "Slack" },
  { key: "critical", label: "Critical path" },
  { key: "congestion", label: "Congestion" },
  { key: "drc", label: "DRC" },
];

// Slack: green (high/safe) -> red (low/violating). Others: green (low) -> red (high).
function colorFor(metric: Metric, v: number | null, lo: number, hi: number): string {
  if (v === null || Number.isNaN(v)) return "#3a4356";
  const span = hi - lo || 1;
  let t = (v - lo) / span; // 0..1
  if (metric === "slack") t = 1 - t; // low slack = bad = red
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(40 + t * 200);
  const g = Math.round(200 - t * 160);
  return `rgb(${r},${g},90)`;
}

export function CircuitGraph({
  nodes,
  edges,
}: {
  nodes: GraphNode[];
  edges: [number, number][];
}) {
  const [metric, setMetric] = useState<Metric>("slack");

  const { pts, w, h, lo, hi } = useMemo(() => {
    const xs = nodes.map((n) => n.x);
    const ys = nodes.map((n) => n.y);
    const maxX = Math.max(1, ...xs);
    const maxY = Math.max(1, ...ys);
    const W = 720;
    const H = 460;
    const pad = 28;
    const pts = nodes.map((n) => ({
      x: pad + (n.x / maxX) * (W - 2 * pad),
      y: pad + (n.y / maxY) * (H - 2 * pad),
      n,
    }));
    const vals = nodes
      .map((n) => n[metric])
      .filter((v): v is number => v !== null && !Number.isNaN(v));
    const lo = vals.length ? Math.min(...vals) : 0;
    const hi = vals.length ? Math.max(...vals) : 1;
    return { pts, w: W, h: H, lo, hi };
  }, [nodes, metric]);

  const showEdges = edges.length <= 1200;

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Circuit graph</h3>
        <div className="seg">
          {METRICS.map((m) => (
            <button
              key={m.key}
              className={metric === m.key ? "seg-on" : ""}
              onClick={() => setMetric(m.key)}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="graph">
        {showEdges &&
          edges.map(([s, t], i) => {
            const a = pts[s];
            const b = pts[t];
            if (!a || !b) return null;
            return (
              <line
                key={i}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                stroke="#2a3346"
                strokeWidth={0.6}
              />
            );
          })}
        {pts.map((p, i) => (
          <circle
            key={i}
            cx={p.x}
            cy={p.y}
            r={p.n.type === "PI" || p.n.type === "PO" ? 3.5 : 4.5}
            fill={colorFor(metric, p.n[metric], lo, hi)}
            stroke="#0d1117"
            strokeWidth={0.5}
          >
            <title>
              {p.n.id} ({p.n.type}) — {metric}:{" "}
              {p.n[metric] === null ? "n/a" : p.n[metric]!.toFixed(3)}
            </title>
          </circle>
        ))}
      </svg>
      <div className="legend">
        <span className="muted">
          {metric === "slack" ? "red = low slack (critical)" : "red = high"} ·{" "}
          {nodes.length} nodes · {edges.length} edges
          {!showEdges && " · edges hidden (large graph)"}
        </span>
      </div>
    </div>
  );
}
