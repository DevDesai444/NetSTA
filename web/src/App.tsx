import { useEffect, useState } from "react";
import { diagnose, health, type AnalyzeResult, type DiagnoseRequest } from "./api";
import { CircuitGraph } from "./components/CircuitGraph";
import { Report } from "./components/Report";

const ANALOG = [
  "two_stage_opamp",
  "folded_cascode",
  "diff_pair",
  "current_mirror",
  "common_source_amp",
];

export default function App() {
  const [kind, setKind] = useState<DiagnoseRequest["kind"]>("digital");
  const [gates, setGates] = useState(40);
  const [topology, setTopology] = useState(ANALOG[0]);
  const [query, setQuery] = useState("two-stage miller opamp, 60dB gain, 10MHz GBW");
  const [seed, setSeed] = useState(42);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ckpt, setCkpt] = useState<boolean | null>(null);
  const [loraActive, setLoraActive] = useState<boolean>(false);

  useEffect(() => {
    health()
      .then((h) => {
        setCkpt(h.checkpoint_present);
        setLoraActive(!!h.lora_active);
      })
      .catch(() => setCkpt(null));
  }, []);

  async function run() {
    setLoading(true);
    setError(null);
    try {
      const req: DiagnoseRequest = { kind, seed };
      if (kind === "digital") req.gates = gates;
      if (kind === "analog") req.topology = topology;
      if (kind === "nl") req.query = query;
      setResult(await diagnose(req));
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>NetSTA</h1>
        <span className="sub">
          GNN timing/DRC prediction + multi-agent design advisory
        </span>
        <span className="src">
          {ckpt === null ? "" : ckpt ? "GNN checkpoint loaded" : "STA fallback (no checkpoint)"}
          {loraActive && <span className="lora-on">· 4 LoRA students live</span>}
        </span>
      </header>

      <div className="controls panel">
        <div className="seg">
          {(["digital", "analog", "nl"] as const).map((k) => (
            <button key={k} className={kind === k ? "seg-on" : ""} onClick={() => setKind(k)}>
              {k === "nl" ? "natural language" : k}
            </button>
          ))}
        </div>

        {kind === "digital" && (
          <label>
            gates
            <input
              type="range"
              min={10}
              max={120}
              value={gates}
              onChange={(e) => setGates(+e.target.value)}
            />
            <span className="val">{gates}</span>
          </label>
        )}
        {kind === "analog" && (
          <label>
            topology
            <select value={topology} onChange={(e) => setTopology(e.target.value)}>
              {ANALOG.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </label>
        )}
        {kind === "nl" && (
          <label className="grow">
            request
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="describe a circuit..."
            />
          </label>
        )}
        <label>
          seed
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(+e.target.value)}
            style={{ width: 64 }}
          />
        </label>
        <button className="run" onClick={run} disabled={loading}>
          {loading ? "analyzing…" : "Analyze"}
        </button>
      </div>

      {error && <div className="error">⚠ {error}</div>}

      {result && (
        <>
          <div className="summary muted">
            {result.circuit_name} · {result.num_nodes} nodes · {result.num_edges} edges ·
            clk {result.clock_period_ns.toFixed(3)} ns · predictions:{" "}
            {result.prediction_source}
            {result.meta?.parser_backend && ` · parser: ${result.meta.parser_backend}`}
          </div>
          <div className="grid">
            <CircuitGraph nodes={result.graph.nodes} edges={result.graph.edges} />
            <Report report={result.report} />
          </div>
        </>
      )}

      {!result && !error && (
        <p className="muted hint">
          Pick a circuit and hit Analyze — the GNN predicts per-node slack,
          critical-path, congestion and DRC, then the agent panel recommends fixes.
        </p>
      )}
    </div>
  );
}
