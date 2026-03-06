const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface JobStatus {
    job_id: string;
    status: "pending" | "processing" | "done" | "failed";
    progress: number;
    message: string;
    created_at: string;
}

export interface UtilityMetric {
    column: string;
    utility_score: number;
    original_mean: number;
    noisy_mean: number;
    relative_error: number;
    std_change_pct: number;
    mae: number;
}

export interface JobResult {
    job_id: string;
    headers: string[];
    original_preview: Record<string, string>[];
    anonymized_preview: Record<string, string>[];
    anonymized_data: Record<string, unknown>[];
    column_types: Record<string, string>;
    budget_used: number;
    budget_total: number;
    risk_level: "LOW" | "MODERATE" | "CRITICAL";
    risk_report: string;
    utility_report: string;
    utility_metrics: UtilityMetric[];
    row_count: number;
    total_dataset_rows?: number;
    max_rows_selected?: number;
    processed_rows?: number;
    ai_active: boolean;
    preprocessing_report: Record<string, unknown>;
    bias_report?: string;
    bias_analysis?: {
        health_score: number;
        findings: { type: string; severity: string; message: string; column?: string; metric?: string }[];
        target_impacts?: Record<string, number>;
        metrics: {
            total_rows: number;
            duplicate_count: number;
            duplicate_pct: number;
            null_counts: Record<string, number>;
            imbalances: { column: string; ratio: number }[];
            associations: { column: string; score: number; label: string }[];
        };
    };
}

// ── Per-column config types ─────────────────────────────────────────────────

export interface ColumnAnalysis {
    name: string;
    detected_type: string;
    mechanism: string;
    sample_values: string[];
    stats: Record<string, number>;
}

export interface AnalyzeResult {
    headers: string[];
    row_count: number;
    columns: ColumnAnalysis[];
    health_score?: number;
    bias_findings?: { type: string; severity: string; message: string; column?: string }[];
}

export interface ColumnConfig {
    epsilon: number;
    method: string;
    enabled: boolean;        // false = excluded from anonymization
    type_override: string;   // user can override detected type
}

// ── API functions ───────────────────────────────────────────────────────────

export async function analyzeFile(params: {
    file: File;
    maxRows: number;
}): Promise<AnalyzeResult> {
    const form = new FormData();
    form.append("file", params.file);
    form.append("max_rows", String(params.maxRows));

    const res = await fetch(`${API_BASE}/api/v1/analyze`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`Analysis failed: ${res.statusText}`);
    return res.json();
}

export async function uploadFile(params: {
    file: File;
    epsilon: number;
    purpose: string;
    seed: number | null;
    maxRows: number;
    excludedColumns: string[];
    columnConfigs?: Record<string, Partial<ColumnConfig>>;
    typeOverrides?: Record<string, string>;
}): Promise<{ job_id: string }> {
    const form = new FormData();
    form.append("file", params.file);
    form.append("epsilon", String(params.epsilon));
    form.append("purpose", params.purpose);
    form.append("max_rows", String(params.maxRows));
    form.append("excluded_columns", params.excludedColumns.join(","));
    if (params.seed !== null) form.append("seed", String(params.seed));

    // Per-column configs — send as JSON strings
    if (params.columnConfigs && Object.keys(params.columnConfigs).length > 0) {
        // Build the backend-compatible column configs (epsilon + method per column)
        const backendConfigs: Record<string, Record<string, unknown>> = {};
        for (const [col, cfg] of Object.entries(params.columnConfigs)) {
            if (cfg.enabled === false) continue; // excluded columns handled separately
            backendConfigs[col] = {
                epsilon: cfg.epsilon ?? 0.2,
                method: cfg.method ?? "laplace",
            };
        }
        form.append("column_configs", JSON.stringify(backendConfigs));
    }

    if (params.typeOverrides && Object.keys(params.typeOverrides).length > 0) {
        form.append("type_overrides", JSON.stringify(params.typeOverrides));
    }

    const res = await fetch(`${API_BASE}/api/v1/upload`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
    return res.json();
}

export async function pollJobStatus(jobId: string): Promise<JobStatus> {
    const res = await fetch(`${API_BASE}/api/v1/jobs/${jobId}/status`);
    if (!res.ok) throw new Error(`Status check failed: ${res.statusText}`);
    return res.json();
}

export async function getJobResult(jobId: string): Promise<JobResult> {
    const res = await fetch(`${API_BASE}/api/v1/jobs/${jobId}/result`);
    if (!res.ok) throw new Error(`Failed to get result: ${res.statusText}`);
    return res.json();
}

export function downloadUrl(jobId: string): string {
    return `${API_BASE}/api/v1/jobs/${jobId}/download`;
}

export const COL_TYPE_META: Record<string, { emoji: string; label: string; color: string }> = {
    age: { emoji: "🎂", label: "Age", color: "#f59e0b" },
    year: { emoji: "📅", label: "Year", color: "#8b5cf6" },
    monetary: { emoji: "💰", label: "Monetary", color: "#10b981" },
    numeric: { emoji: "📊", label: "Numeric", color: "#22d3ee" },
    count: { emoji: "🔢", label: "Count", color: "#6366f1" },
    boolean: { emoji: "✅", label: "Boolean", color: "#ec4899" },
    id: { emoji: "🆔", label: "ID", color: "#94a3b8" },
    string: { emoji: "🏷️", label: "String", color: "#64748b" },
};

export const VALID_COL_TYPES = Object.keys(COL_TYPE_META);

export const MECHANISM_OPTIONS: Record<string, string[]> = {
    age: ["bounded_laplace", "laplace"],
    year: ["bounded_laplace", "laplace"],
    monetary: ["laplace", "bounded_laplace"],
    numeric: ["laplace", "bounded_laplace"],
    count: ["discrete_laplace", "laplace"],
    boolean: ["randomized_response"],
    id: ["hash"],
    string: ["mask", "hash"],
};
