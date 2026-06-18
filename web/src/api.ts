// Typed client for the NetSTA FastAPI backend.

export interface GraphNode {
  id: string;
  type: string;
  x: number;
  y: number;
  slack: number | null;
  critical: number | null;
  congestion: number | null;
  drc: number | null;
}

export interface Bottleneck {
  task: string;
  violation_type: string | null;
  severity: number;
  location: string | null;
  node_ids: string[];
  summary: string;
}

export interface Recommendation {
  bottleneck_task: string;
  fix: string;
  action: string;
  rationale: string;
  evidence: string[];
  outcomes: string[];
  conflicts: string[];
  effort: string | null;
  confidence: number;
  agent: string;
}

export interface AgentTurn {
  agent: string;
  summary: string;
  bottlenecks: Bottleneck[];
  recommendations: Recommendation[];
}

export interface DesignReport {
  circuit_name: string;
  backend: string;
  bottlenecks: Bottleneck[];
  recommendations: Recommendation[];
  transcript: AgentTurn[];
  predictions_summary: Record<string, any>;
  confidence_scores: Record<string, number>;
}

export interface AnalyzeResult {
  circuit_name: string;
  prediction_source: string;
  num_nodes: number;
  num_edges: number;
  clock_period_ns: number;
  graph: { nodes: GraphNode[]; edges: [number, number][] };
  report: DesignReport;
  meta?: Record<string, any>;
}

export interface DiagnoseRequest {
  kind: "digital" | "analog" | "nl";
  inputs?: number;
  gates?: number;
  outputs?: number;
  topology?: string;
  query?: string;
  seed?: number;
  process_node?: string;
}

export async function diagnose(req: DiagnoseRequest): Promise<AnalyzeResult> {
  const res = await fetch("/api/diagnose", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `request failed (${res.status})`);
  }
  return res.json();
}

export async function health(): Promise<{ checkpoint_present: boolean }> {
  const res = await fetch("/api/health");
  return res.json();
}
