import type { DesignReport } from "../api";

function pct(x: number) {
  return `${Math.round(x * 100)}%`;
}

// Map agent names -> the LoRA adapter that answered (visible only when the
// AutoGen backend is wired to our vLLM specialist students).
const LORA_FOR_AGENT: Record<string, string> = {
  SupervisorAgent: "supervisor-lora",
  TimingAgent: "timing-lora",
  DRCAgent: "drc-lora",
  OptimizationAgent: "optimization-lora",
};

export function Report({ report }: { report: DesignReport }) {
  const isAutogen = report.backend === "autogen";
  return (
    <div className="report">
      <div className="panel">
        <div className="panel-head">
          <h3>Bottlenecks</h3>
          <span className={`badge badge-${report.backend}`}>
            {isAutogen ? "AutoGen + 4 LoRA students" : report.backend}
          </span>
        </div>
        {report.bottlenecks.length === 0 && (
          <p className="muted">No violations flagged for this circuit.</p>
        )}
        {report.bottlenecks.map((b, i) => (
          <div key={i} className="bn">
            <div className="bn-head">
              <span className="tag">{b.task}</span>
              {b.violation_type && <span className="tag tag-dim">{b.violation_type}</span>}
              <span className="sev">sev {pct(b.severity)}</span>
            </div>
            <div className="muted">{b.summary}</div>
            {b.location && <div className="loc">@ {b.location}</div>}
          </div>
        ))}
      </div>

      <div className="panel">
        <div className="panel-head">
          <h3>Recommendations</h3>
        </div>
        {report.recommendations.slice(0, 10).map((r, i) => (
          <div key={i} className="rec">
            <div className="rec-head">
              <span className="fix">{r.fix}</span>
              <span className="agent">{r.agent}</span>
              <span className="conf" title="confidence">
                {pct(r.confidence)}
              </span>
            </div>
            <div className="action">{r.action}</div>
            <div className="rec-meta">
              {r.effort && <span className="chip">effort: {r.effort}</span>}
              {r.outcomes.map((o) => (
                <span key={o} className="chip chip-good">
                  {o}
                </span>
              ))}
              {r.conflicts.map((c) => (
                <span key={c} className="chip chip-bad">
                  conflicts: {c}
                </span>
              ))}
            </div>
            {r.evidence.length > 0 && (
              <details className="evidence">
                <summary>evidence</summary>
                <ul>
                  {r.evidence.map((e, j) => (
                    <li key={j}>{e}</li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        ))}
      </div>

      <div className="panel">
        <div className="panel-head">
          <h3>Agent panel transcript</h3>
        </div>
        {report.transcript.map((t, i) => (
          <div key={i} className="turn">
            <span className="who">
              {t.agent}
              {isAutogen && LORA_FOR_AGENT[t.agent] && (
                <span className="lora-tag">{LORA_FOR_AGENT[t.agent]}</span>
              )}
            </span>
            <span className="said">{t.summary}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
