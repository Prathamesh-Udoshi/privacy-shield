"use client";
import { Fragment, useEffect, useState, useMemo } from "react";
import { useParams, useRouter } from "next/navigation";
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import {
    Shield, ArrowLeft, Download, AlertTriangle, CheckCircle, AlertCircle,
    TrendingUp, TrendingDown, Target, Users, Info, Activity, Database, Sparkles
} from "lucide-react";
import { getJobResult, downloadUrl, COL_TYPE_META, type JobResult } from "@/lib/api";

type Tab = "analysis" | "preview" | "reports" | "bias";

const RISK_CONFIG = {
    LOW: { color: "#10b981", bg: "rgba(16,185,129,0.1)", border: "rgba(16,185,129,0.3)", icon: <CheckCircle size={20} color="#10b981" />, label: "Low Risk", desc: "Strong protection — matches are likely statistical coincidences." },
    MODERATE: { color: "#f59e0b", bg: "rgba(245,158,11,0.1)", border: "rgba(245,158,11,0.3)", icon: <AlertCircle size={20} color="#f59e0b" />, label: "Moderate Risk", desc: "Probabilistic privacy — some records may be linkable by determined attackers." },
    CRITICAL: { color: "#ef4444", bg: "rgba(239,68,68,0.1)", border: "rgba(239,68,68,0.3)", icon: <AlertTriangle size={20} color="#ef4444" />, label: "Critical Risk", desc: "High linkage — noise is too low relative to data density." },
};

/* ---------- Report Parsers ---------- */

interface UtilityColumn {
    name: string;
    sampleSize: string;
    originalMean: string;
    noisyMean: string;
    errorAbs: string;
    errorPct: string;
    originalStd: string;
    noisyStd: string;
    stdChangePct: string;
    mae: string;
    utilityScore: string;
}

interface UtilityParsed {
    columns: UtilityColumn[];
    summary: { columnsAnalyzed: string; avgScore: string; interpretation: string };
}

function parseUtilityReport(raw: string): UtilityParsed | null {
    if (!raw) return null;
    try {
        const lines = raw.split("\n");
        const columns: UtilityColumn[] = [];
        let current: Partial<UtilityColumn> = {};
        const summary = { columnsAnalyzed: "", avgScore: "", interpretation: "" };

        for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith("Column:")) {
                if (current.name) columns.push(current as UtilityColumn);
                current = { name: trimmed.replace("Column:", "").trim() };
            } else if (trimmed.startsWith("Sample Size:")) {
                current.sampleSize = trimmed.replace("Sample Size:", "").trim();
            } else if (trimmed.startsWith("Original:") && !current.originalStd) {
                current.originalMean = trimmed.replace("Original:", "").trim();
            } else if (trimmed.startsWith("Noisy:") && !current.noisyStd) {
                current.noisyMean = trimmed.replace("Noisy:", "").trim();
            } else if (trimmed.startsWith("Error:")) {
                const parts = trimmed.replace("Error:", "").trim().split("(");
                current.errorAbs = parts[0]?.trim() ?? "";
                current.errorPct = parts[1]?.replace(")", "").trim() ?? "";
            } else if (trimmed.startsWith("Original:") && current.originalMean) {
                current.originalStd = trimmed.replace("Original:", "").trim();
            } else if (trimmed.startsWith("Noisy:") && current.noisyMean) {
                current.noisyStd = trimmed.replace("Noisy:", "").trim();
            } else if (trimmed.startsWith("Change:")) {
                current.stdChangePct = trimmed.replace("Change:", "").trim();
            } else if (trimmed.startsWith("Mean Absolute Error:")) {
                current.mae = trimmed.replace("Mean Absolute Error:", "").trim();
            } else if (trimmed.startsWith("Utility Score:")) {
                current.utilityScore = trimmed.replace("Utility Score:", "").trim();
            } else if (trimmed.startsWith("Columns Analyzed:")) {
                summary.columnsAnalyzed = trimmed.replace("Columns Analyzed:", "").trim();
            } else if (trimmed.startsWith("Average Utility Score:")) {
                summary.avgScore = trimmed.replace("Average Utility Score:", "").trim();
            } else if (trimmed.startsWith("Interpretation:")) {
                summary.interpretation = trimmed.replace("Interpretation:", "").trim();
            }
        }
        if (current.name) columns.push(current as UtilityColumn);
        return { columns, summary };
    } catch (e) {
        console.error("Error parsing utility report:", e);
        return null;
    }
}

interface RiskParsed {
    uniquenessReduction: string;
    membershipInference: string;
    riskLevelLinking: string;
    kAnonymity: string | null;
    overallRisk: string;
    interpretation: string;
}

function parseRiskReport(raw: string): RiskParsed | null {
    if (!raw) return null;
    try {
        const lines = raw.split("\n");
        const data: RiskParsed = {
            uniquenessReduction: "",
            membershipInference: "",
            riskLevelLinking: "",
            kAnonymity: null,
            overallRisk: "",
            interpretation: "",
        };
        for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith("Uniqueness Reduction:")) data.uniquenessReduction = trimmed.split(":").slice(1).join(":").trim();
            else if (trimmed.startsWith("Membership Inference:")) data.membershipInference = trimmed.split(":").slice(1).join(":").trim();
            else if (trimmed.startsWith("Risk Level (Linking):")) data.riskLevelLinking = trimmed.split(":").slice(1).join(":").trim();
            else if (trimmed.includes("Noisy Data:")) data.kAnonymity = trimmed.split(":").slice(1).join(":").trim();
            else if (trimmed.startsWith("Overall Risk Category:")) data.overallRisk = trimmed.split(":").slice(1).join(":").trim();
            else if (trimmed.startsWith("Interpretation:")) data.interpretation = trimmed.split(":").slice(1).join(":").trim();
        }
        return data;
    } catch (e) {
        console.error("Error parsing risk report:", e);
        return null;
    }
}

function parseBiasReport(raw: string) {
    if (!raw) return null;
    try {
        const lines = raw.split("\n");
        const findings: { type: string; severity: string; message: string }[] = [];
        const targetImpacts: { col: string; impact: number }[] = [];
        let score = "0";
        let status = "UNKNOWN";
        let aiInsights = "";

        let inImpacts = false;
        let inInsights = false;

        for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.includes("Health Score:")) {
                const parts = trimmed.split(":")[1].trim().split(" ");
                score = parts[0].split("/")[0];
                status = parts[1]?.replace("(", "").replace(")", "") ?? "";
            } else if (trimmed.startsWith("- 🔴") || trimmed.startsWith("- 🟡") || trimmed.startsWith("- 🔵")) {
                const severity = trimmed.includes("🔴") ? "high" : trimmed.includes("🟡") ? "medium" : "low";
                const type = trimmed.split("**[")[1]?.split("]**")[0] ?? "info";
                const message = trimmed.split("]**")[1]?.trim() ?? trimmed.slice(4).trim();
                findings.push({ severity, message, type });
            } else if (trimmed.startsWith("### 📊 Feature Predictive Strength")) {
                inImpacts = true;
                inInsights = false;
            } else if (trimmed.startsWith("### 💡 AI Insider Insights")) {
                inInsights = true;
                inImpacts = false;
            } else if (inImpacts && trimmed.startsWith("- ")) {
                const parts = trimmed.replace("- ", "").split(":");
                const col = parts[0]?.trim();
                const valPart = parts[1]?.trim().split(" ")[0];
                const impact = parseFloat(valPart ?? "0");
                if (col && !isNaN(impact)) targetImpacts.push({ col, impact });
            } else if (inInsights && !trimmed.startsWith("#") && trimmed.length > 0) {
                aiInsights += line + "\n";
            }
        }
        return { score, status, findings, targetImpacts, aiInsights: aiInsights.trim() };
    } catch (e) {
        console.error("Error parsing bias report:", e);
        return null;
    }
}


/* ---------- Helper Components ---------- */

function ScoreBar({ value, max = 100, colorFn }: { value: number; max?: number; colorFn?: (v: number) => string }) {
    const pct = Math.min(100, Math.max(0, (value / max) * 100));
    const color = colorFn ? colorFn(value) : (value >= 70 ? "#10b981" : value >= 40 ? "#f59e0b" : "#ef4444");
    return (
        <div style={{ display: "flex", alignItems: "center", gap: 10, width: "100%" }}>
            <div style={{ flex: 1, height: 6, borderRadius: 99, background: "rgba(255,255,255,0.06)" }}>
                <div style={{ width: `${pct}%`, height: "100%", borderRadius: 99, background: color, transition: "width 0.6s ease" }} />
            </div>
            <span style={{ color, fontSize: 13, fontWeight: 600, minWidth: 40, textAlign: "right" }}>{value.toFixed(0)}</span>
        </div>
    );
}

/* ---------- Main Component ---------- */

export default function ResultsPage() {
    const { jobId } = useParams<{ jobId: string }>();
    const router = useRouter();
    const [result, setResult] = useState<JobResult | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [tab, setTab] = useState<Tab>("analysis");
    const [expandedRow, setExpandedRow] = useState<number | null>(0);

    useEffect(() => {
        console.log("JobResults page mounted triggered for jobId:", jobId);
        if (!jobId) {
            setError("No job ID provided in URL");
            setLoading(false);
            return;
        }

        getJobResult(jobId)
            .then((data) => {
                console.log("Successfully fetched job data:", data);
                setResult(data);
            })
            .catch((e) => {
                console.error("Failed to fetch job result:", e);
                setError(e.message || "Failed to load job results. Please try again.");
            })
            .finally(() => setLoading(false));
    }, [jobId]);

    const utilityParsed = useMemo(() => result ? parseUtilityReport(result.utility_report) : null, [result]);
    const riskParsed = useMemo(() => result ? parseRiskReport(result.risk_report) : null, [result]);
    const biasParsed = useMemo(() => result ? parseBiasReport(result.bias_report ?? "") : null, [result]);

    if (loading) return (
        <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <div style={{ textAlign: "center" }}>
                <Shield size={40} color="#8b5cf6" style={{ margin: "0 auto 16px" }} />
                <p style={{ color: "#64748b" }}>Loading results…</p>
            </div>
        </div>
    );

    if (error || !result) return (
        <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <div className="glass" style={{ padding: 40, textAlign: "center", maxWidth: 400 }}>
                <AlertTriangle size={40} color="#ef4444" style={{ margin: "0 auto 16px" }} />
                <p style={{ color: "#ef4444", marginBottom: 16 }}>{error ?? "Result not found"}</p>
                <button className="btn-brand" onClick={() => router.push("/")}>Start over</button>
            </div>
        </div>
    );

    const risk = RISK_CONFIG[result.risk_level] ?? RISK_CONFIG.LOW;
    const avgUtility = (result.utility_metrics && result.utility_metrics.length)
        ? result.utility_metrics.reduce((s, m) => s + m.utility_score, 0) / result.utility_metrics.length
        : null;

    return (
        <div style={{ minHeight: "100vh" }}>
            {/* Nav */}
            <nav className="nav">
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <button className="btn-ghost" onClick={() => router.push("/")} style={{ gap: 6 }}>
                        <ArrowLeft size={14} /> New Analysis
                    </button>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginLeft: 8 }}>
                        <Shield size={18} color="#8b5cf6" />
                        <span style={{ fontWeight: 700, fontSize: 16 }}>Privacy Shield</span>
                    </div>
                </div>
                <a href={downloadUrl(jobId)} download>
                    <button className="btn-brand" style={{ padding: "9px 20px", fontSize: 14 }}>
                        <Download size={14} /> Download CSV
                    </button>
                </a>
            </nav>

            <div className="section" style={{ width: "100%", padding: "32px 40px 80px" }}>
                {/* Header */}
                <div style={{ maxWidth: 1400, margin: "0 auto" }}>
                    <h1 style={{ fontSize: 30, fontWeight: 800, marginBottom: 8 }}>
                        Privacy <span className="gradient-text">Results</span>
                    </h1>
                    <p style={{ color: "#475569", fontSize: 14, marginBottom: 32 }}>
                        Job <code style={{ color: "#8b5cf6", background: "rgba(139,92,246,0.1)", padding: "1px 6px", borderRadius: 4 }}>{jobId.slice(0, 8)}</code>
                        &nbsp;— {result.processed_rows?.toLocaleString() ?? result.row_count} rows anonymized
                        {result.ai_active && <span style={{ marginLeft: 8, color: "#22d3ee", fontSize: 12 }}>⚡ AI-Enhanced</span>}
                    </p>
                </div>

                {/* Metric cards - full width */}
                <div style={{ maxWidth: 1400, margin: "0 auto 32px" }}>
                    <div className="results-metric-grid">
                        {/* Risk card */}
                        <div className="metric-card" style={{ borderColor: risk.border, background: risk.bg }}>
                            <p className="metric-label">Linkage Risk</p>
                            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                                {risk.icon}
                                <span className="metric-value" style={{ color: risk.color, fontSize: 22 }}>{risk.label}</span>
                            </div>
                            <p style={{ fontSize: 12, color: "#64748b", lineHeight: 1.5 }}>{risk.desc}</p>
                        </div>

                        {/* Utility */}
                        <div className="metric-card">
                            <p className="metric-label">Avg Utility Score</p>
                            <p className="metric-value" style={{ color: avgUtility && avgUtility >= 70 ? "#10b981" : "#f59e0b", marginBottom: 6 }}>
                                {avgUtility !== null ? `${avgUtility.toFixed(1)}/100` : "N/A"}
                            </p>
                            {avgUtility !== null && <ScoreBar value={avgUtility} />}
                        </div>

                        {/* Budget */}
                        <div className="metric-card">
                            <p className="metric-label">Privacy Budget Used</p>
                            <p className="metric-value" style={{ color: "#8b5cf6" }}>ε = {result.budget_used.toFixed(3)}</p>
                            <p style={{ fontSize: 11, color: "#475569", marginTop: 4 }}>of {result.budget_total.toFixed(3)} total</p>
                        </div>

                        {/* Rows */}
                        <div className="metric-card">
                            <p className="metric-label">Processed / Total Dataset</p>
                            <p className="metric-value" style={{ fontSize: 22 }}>
                                {result.processed_rows?.toLocaleString() ?? result.row_count.toLocaleString()}
                                <span style={{ fontSize: 13, color: "#64748b", fontWeight: 500, marginLeft: 6 }}>
                                    / {result.total_dataset_rows?.toLocaleString() ?? "—"}
                                </span>
                            </p>
                            <p style={{ fontSize: 11, color: "#475569", marginTop: 4 }}>
                                Anonymization Limit: {result.max_rows_selected?.toLocaleString() ?? "—"} rows
                            </p>
                        </div>
                    </div>
                </div>

                {/* Column types */}
                <div style={{ maxWidth: 1400, margin: "0 auto 28px" }}>
                    <div className="glass" style={{ padding: "16px 24px" }}>
                        <p style={{ fontSize: 12, color: "#64748b", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12 }}>
                            Detected Column Types
                        </p>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                            {Object.entries(result.column_types).map(([col, type]) => {
                                const meta = COL_TYPE_META[type] ?? { emoji: "📝", label: type, color: "#64748b" };
                                return (
                                    <span key={col} className="col-badge" style={{ borderColor: `${meta.color}33` }}>
                                        <span>{meta.emoji}</span>
                                        <span style={{ color: "#94a3b8" }}>{col}</span>
                                        <span style={{ color: meta.color, fontSize: 11 }}>{meta.label}</span>
                                    </span>
                                );
                            })}
                        </div>
                    </div>
                </div>

                {/* Tabs */}
                <div style={{ maxWidth: 1400, margin: "0 auto 28px" }}>
                    <div className="tab-list">
                        {(["analysis", "preview", "reports", "bias"] as Tab[]).map((t) => (
                            <button key={t} className={`tab-btn ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
                                {{ analysis: "📊 Analysis", preview: "🔍 Data Preview", reports: "📋 Reports", bias: "⚖️ Bias & Health" }[t]}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Tab: Analysis */}
                <div style={{ maxWidth: 1400, margin: "0 auto" }}>
                    {tab === "analysis" && (
                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                            {result.utility_metrics.length > 0 ? (
                                <>
                                    <div className="glass" style={{ padding: 24 }}>
                                        <p style={{ fontSize: 14, fontWeight: 600, color: "#94a3b8", marginBottom: 16 }}>Utility Score per Column <span style={{ color: "#475569", fontWeight: 400 }}>(higher = better)</span></p>
                                        <ResponsiveContainer width="100%" height={280}>
                                            <BarChart data={result.utility_metrics} margin={{ left: -10 }}>
                                                <XAxis dataKey="column" tick={{ fontSize: 11, fill: "#64748b" }} />
                                                <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: "#64748b" }} />
                                                <Tooltip
                                                    contentStyle={{ background: "#0d1117", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, fontSize: 12 }}
                                                    formatter={(v: number) => [`${v.toFixed(1)}/100`, "Utility"]}
                                                />
                                                <Bar dataKey="utility_score" radius={[6, 6, 0, 0]}>
                                                    {result.utility_metrics.map((m, i) => (
                                                        <Cell key={i} fill={m.utility_score >= 70 ? "#10b981" : m.utility_score >= 40 ? "#f59e0b" : "#ef4444"} />
                                                    ))}
                                                </Bar>
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>

                                    <div className="glass" style={{ padding: 24 }}>
                                        <p style={{ fontSize: 14, fontWeight: 600, color: "#94a3b8", marginBottom: 16 }}>Relative Error % <span style={{ color: "#475569", fontWeight: 400 }}>(lower = better)</span></p>
                                        <ResponsiveContainer width="100%" height={280}>
                                            <BarChart data={result.utility_metrics} margin={{ left: -10 }}>
                                                <XAxis dataKey="column" tick={{ fontSize: 11, fill: "#64748b" }} />
                                                <YAxis tick={{ fontSize: 11, fill: "#64748b" }} />
                                                <Tooltip
                                                    contentStyle={{ background: "#0d1117", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, fontSize: 12 }}
                                                    formatter={(v: number) => [`${v.toFixed(2)}%`, "Error"]}
                                                />
                                                <Bar dataKey="relative_error" radius={[6, 6, 0, 0]}>
                                                    {result.utility_metrics.map((m, i) => (
                                                        <Cell key={i} fill={m.relative_error <= 5 ? "#10b981" : m.relative_error <= 25 ? "#22d3ee" : "#f59e0b"} />
                                                    ))}
                                                </Bar>
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>

                                    {/* Detailed metrics table */}
                                    <div className="glass" style={{ padding: 24, gridColumn: "1 / -1" }}>
                                        <p style={{ fontSize: 14, fontWeight: 600, color: "#94a3b8", marginBottom: 16 }}>Detailed Column Statistics</p>
                                        <div style={{ overflowX: "auto" }}>
                                            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                                                <thead>
                                                    <tr>
                                                        {["Column", "Original Mean", "Noisy Mean", "MAE", "Relative Error", "Std Δ", "Utility Score"].map((h) => (
                                                            <th key={h} style={{ padding: "10px 14px", textAlign: "left", color: "#475569", fontWeight: 600, fontSize: 12, borderBottom: "1px solid rgba(255,255,255,0.07)", textTransform: "uppercase", letterSpacing: "0.04em" }}>{h}</th>
                                                        ))}
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {result.utility_metrics.map((m) => (
                                                        <tr key={m.column} style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                                                            <td style={{ padding: "10px 14px", color: "#a78bfa", fontWeight: 600 }}>{m.column}</td>
                                                            <td style={{ padding: "10px 14px", color: "#94a3b8", fontFamily: "monospace", fontSize: 12 }}>{m.original_mean.toFixed(3)}</td>
                                                            <td style={{ padding: "10px 14px", color: "#94a3b8", fontFamily: "monospace", fontSize: 12 }}>{m.noisy_mean.toFixed(3)}</td>
                                                            <td style={{ padding: "10px 14px", color: "#94a3b8", fontFamily: "monospace", fontSize: 12 }}>{m.mae.toFixed(3)}</td>
                                                            <td style={{ padding: "10px 14px" }}>
                                                                <span style={{
                                                                    padding: "2px 8px", borderRadius: 6, fontSize: 12, fontWeight: 600, fontFamily: "monospace",
                                                                    color: m.relative_error > 25 ? "#f59e0b" : m.relative_error > 10 ? "#22d3ee" : "#10b981",
                                                                    background: m.relative_error > 25 ? "rgba(245,158,11,0.1)" : m.relative_error > 10 ? "rgba(34,211,238,0.1)" : "rgba(16,185,129,0.1)",
                                                                }}>{m.relative_error.toFixed(2)}%</span>
                                                            </td>
                                                            <td style={{ padding: "10px 14px" }}>
                                                                <span style={{
                                                                    display: "inline-flex", alignItems: "center", gap: 4,
                                                                    color: Math.abs(m.std_change_pct) > 20 ? "#f59e0b" : "#10b981", fontSize: 12, fontFamily: "monospace",
                                                                }}>
                                                                    {m.std_change_pct > 0 ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                                                                    {m.std_change_pct > 0 ? "+" : ""}{m.std_change_pct.toFixed(1)}%
                                                                </span>
                                                            </td>
                                                            <td style={{ padding: "10px 14px", minWidth: 160 }}>
                                                                <ScoreBar value={m.utility_score} />
                                                            </td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </>
                            ) : (
                                <div className="glass" style={{ padding: 32, textAlign: "center", gridColumn: "1/-1", color: "#475569" }}>
                                    No numeric columns found for utility analysis.
                                </div>
                            )}
                        </div>
                    )}

                    {/* Tab: Preview */}
                    {tab === "preview" && (
                        <div>
                            <p style={{ color: "#64748b", fontSize: 13, marginBottom: 16 }}>
                                Row-by-row comparison of original vs anonymized data (first {result.original_preview.length} rows).
                            </p>
                            {/* Table-based preview */}
                            <div className="glass" style={{ overflow: "hidden" }}>
                                <div style={{ overflowX: "auto" }}>
                                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                                        <thead>
                                            <tr style={{ background: "rgba(255,255,255,0.04)" }}>
                                                <th style={{ padding: "10px 14px", textAlign: "left", color: "#475569", fontWeight: 600, fontSize: 12, borderBottom: "1px solid rgba(255,255,255,0.07)", position: "sticky", left: 0, background: "rgba(8,13,26,0.95)", zIndex: 2 }}>Row</th>
                                                {result.headers.map((h) => (
                                                    <th key={h} colSpan={2} style={{ padding: "10px 14px", textAlign: "center", color: "#64748b", fontWeight: 600, fontSize: 12, borderBottom: "1px solid rgba(255,255,255,0.07)", borderLeft: "1px solid rgba(255,255,255,0.04)" }}>
                                                        {h}
                                                    </th>
                                                ))}
                                            </tr>
                                            <tr style={{ background: "rgba(255,255,255,0.02)" }}>
                                                <th style={{ padding: "6px 14px", borderBottom: "1px solid rgba(255,255,255,0.07)", position: "sticky", left: 0, background: "rgba(8,13,26,0.95)", zIndex: 2 }} />
                                                {result.headers.map((h) => (
                                                    <Fragment key={h}>
                                                        <th style={{ padding: "6px 10px", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.1em", color: "#475569", fontWeight: 600, borderBottom: "1px solid rgba(255,255,255,0.07)", borderLeft: "1px solid rgba(255,255,255,0.04)" }}>Original</th>
                                                        <th style={{ padding: "6px 10px", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.1em", color: "#7c3aed", fontWeight: 600, borderBottom: "1px solid rgba(255,255,255,0.07)" }}>Anon</th>
                                                    </Fragment>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {result.original_preview.map((orig, i) => {
                                                const anon = result.anonymized_preview[i];
                                                return (
                                                    <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                                                        <td style={{ padding: "8px 14px", fontWeight: 600, color: "#64748b", fontSize: 12, position: "sticky", left: 0, background: "rgba(8,13,26,0.95)", zIndex: 1 }}>#{i + 1}</td>
                                                        {result.headers.map((h) => {
                                                            const origVal = String(orig?.[h] ?? "—");
                                                            const anonVal = String(anon?.[h] ?? "—");
                                                            const changed = origVal !== anonVal;
                                                            return (
                                                                <Fragment key={h}>
                                                                    <td style={{ padding: "8px 10px", color: "#94a3b8", fontSize: 12, fontFamily: "monospace", borderLeft: "1px solid rgba(255,255,255,0.04)", whiteSpace: "nowrap" }}>{origVal}</td>
                                                                    <td style={{
                                                                        padding: "8px 10px", fontSize: 12, fontFamily: "monospace", whiteSpace: "nowrap",
                                                                        color: changed ? "#a78bfa" : "#64748b",
                                                                        background: changed ? "rgba(139,92,246,0.04)" : undefined,
                                                                    }}>{anonVal}</td>
                                                                </Fragment>
                                                            );
                                                        })}
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Tab: Reports */}
                    {tab === "reports" && (
                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>

                            {/* ===== UTILITY REPORT ===== */}
                            <div className="glass report-card" style={{ padding: 0, overflow: "hidden" }}>
                                <div className="report-card-header" style={{ background: "linear-gradient(135deg, rgba(34,211,238,0.1), rgba(34,211,238,0.03))", borderBottom: "1px solid rgba(34,211,238,0.15)" }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                                        <TrendingUp size={18} color="#22d3ee" />
                                        <span style={{ fontSize: 15, fontWeight: 700, color: "#22d3ee" }}>Utility Preservation Report</span>
                                    </div>
                                </div>

                                <div style={{ padding: "20px 24px" }}>
                                    {utilityParsed ? (
                                        <>
                                            {/* Summary banner */}
                                            {utilityParsed.summary.avgScore && (
                                                <div style={{
                                                    padding: "14px 18px", borderRadius: 12, marginBottom: 20,
                                                    background: "rgba(34,211,238,0.06)", border: "1px solid rgba(34,211,238,0.15)",
                                                }}>
                                                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12 }}>
                                                        <div>
                                                            <p style={{ fontSize: 12, color: "#64748b", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>Average Utility Score</p>
                                                            <p style={{ fontSize: 24, fontWeight: 800, color: "#22d3ee" }}>{utilityParsed.summary.avgScore}</p>
                                                        </div>
                                                        <div style={{ textAlign: "right" }}>
                                                            <p style={{ fontSize: 12, color: "#64748b", marginBottom: 2 }}>{utilityParsed.summary.columnsAnalyzed} columns analyzed</p>
                                                            <span style={{
                                                                fontSize: 12, fontWeight: 600, padding: "3px 10px", borderRadius: 6,
                                                                background: utilityParsed.summary.interpretation.includes("EXCELLENT") ? "rgba(16,185,129,0.15)" :
                                                                    utilityParsed.summary.interpretation.includes("GOOD") ? "rgba(34,211,238,0.15)" :
                                                                        utilityParsed.summary.interpretation.includes("FAIR") ? "rgba(245,158,11,0.15)" : "rgba(239,68,68,0.15)",
                                                                color: utilityParsed.summary.interpretation.includes("EXCELLENT") ? "#10b981" :
                                                                    utilityParsed.summary.interpretation.includes("GOOD") ? "#22d3ee" :
                                                                        utilityParsed.summary.interpretation.includes("FAIR") ? "#f59e0b" : "#ef4444",
                                                            }}>{utilityParsed.summary.interpretation}</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            )}

                                            {/* Per-column breakdown */}
                                            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                                                {utilityParsed.columns.map((col) => (
                                                    <div key={col.name} style={{
                                                        padding: "14px 16px", borderRadius: 10,
                                                        background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
                                                    }}>
                                                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                                                            <span style={{ fontSize: 14, fontWeight: 700, color: "#a78bfa" }}>{col.name}</span>
                                                            <span style={{ fontSize: 12, color: "#64748b" }}>{col.sampleSize} samples</span>
                                                        </div>
                                                        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 8 }}>
                                                            <div>
                                                                <p style={{ fontSize: 10, color: "#475569", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 2 }}>Mean</p>
                                                                <p style={{ fontSize: 12, color: "#94a3b8", fontFamily: "monospace" }}>{col.originalMean} → {col.noisyMean}</p>
                                                            </div>
                                                            <div>
                                                                <p style={{ fontSize: 10, color: "#475569", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 2 }}>Error</p>
                                                                <p style={{ fontSize: 12, color: "#f59e0b", fontFamily: "monospace" }}>{col.errorAbs} ({col.errorPct})</p>
                                                            </div>
                                                            <div>
                                                                <p style={{ fontSize: 10, color: "#475569", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 2 }}>Std Change</p>
                                                                <p style={{ fontSize: 12, color: "#94a3b8", fontFamily: "monospace" }}>{col.stdChangePct}</p>
                                                            </div>
                                                            <div>
                                                                <p style={{ fontSize: 10, color: "#475569", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 2 }}>Utility</p>
                                                                <p style={{ fontSize: 13, fontWeight: 700, color: parseFloat(col.utilityScore) >= 70 ? "#10b981" : parseFloat(col.utilityScore) >= 40 ? "#f59e0b" : "#ef4444" }}>{col.utilityScore}</p>
                                                            </div>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </>
                                    ) : (
                                        <pre style={{ fontSize: 12, color: "#64748b", whiteSpace: "pre-wrap", lineHeight: 1.7, fontFamily: "monospace" }}>{result.utility_report}</pre>
                                    )}
                                </div>
                            </div>

                            {/* ===== RISK REPORT ===== */}
                            <div className="glass report-card" style={{ padding: 0, overflow: "hidden" }}>
                                <div className="report-card-header" style={{ background: "linear-gradient(135deg, rgba(139,92,246,0.1), rgba(139,92,246,0.03))", borderBottom: "1px solid rgba(139,92,246,0.15)" }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                                        <Target size={18} color="#a78bfa" />
                                        <span style={{ fontSize: 15, fontWeight: 700, color: "#a78bfa" }}>Re-identification Risk Assessment</span>
                                    </div>
                                </div>

                                <div style={{ padding: "20px 24px" }}>
                                    {riskParsed ? (
                                        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                                            {/* Overall risk banner */}
                                            <div style={{
                                                padding: "16px 20px", borderRadius: 12,
                                                background: risk.bg, border: `1px solid ${risk.border}`,
                                            }}>
                                                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
                                                    {risk.icon}
                                                    <div>
                                                        <p style={{ fontSize: 12, color: "#64748b", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 2 }}>Overall Risk Category</p>
                                                        <p style={{ fontSize: 22, fontWeight: 800, color: risk.color }}>{riskParsed.overallRisk}</p>
                                                    </div>
                                                </div>
                                                <p style={{ fontSize: 13, color: "#94a3b8", lineHeight: 1.5, paddingLeft: 32 }}>{riskParsed.interpretation}</p>
                                            </div>

                                            {/* Risk factors */}
                                            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                                                {/* Membership inference */}
                                                <div style={{
                                                    padding: "14px 16px", borderRadius: 10,
                                                    background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
                                                }}>
                                                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                                                        <Users size={14} color="#a78bfa" />
                                                        <span style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0" }}>Membership Inference Attack</span>
                                                    </div>
                                                    <p style={{ fontSize: 13, fontFamily: "monospace", color: "#94a3b8", marginBottom: 4 }}>
                                                        {riskParsed.membershipInference}
                                                    </p>
                                                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                                        <span style={{ fontSize: 12, color: "#64748b" }}>Linking Risk Level:</span>
                                                        <span style={{
                                                            fontSize: 12, fontWeight: 700, padding: "2px 8px", borderRadius: 6,
                                                            color: riskParsed.riskLevelLinking === "HIGH" ? "#ef4444" : riskParsed.riskLevelLinking === "MEDIUM" ? "#f59e0b" : "#10b981",
                                                            background: riskParsed.riskLevelLinking === "HIGH" ? "rgba(239,68,68,0.1)" : riskParsed.riskLevelLinking === "MEDIUM" ? "rgba(245,158,11,0.1)" : "rgba(16,185,129,0.1)",
                                                        }}>{riskParsed.riskLevelLinking}</span>
                                                    </div>
                                                </div>

                                                {/* Uniqueness */}
                                                <div style={{
                                                    padding: "14px 16px", borderRadius: 10,
                                                    background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
                                                }}>
                                                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                                                        <Info size={14} color="#22d3ee" />
                                                        <span style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0" }}>Uniqueness Reduction</span>
                                                    </div>
                                                    <p style={{ fontSize: 20, fontWeight: 700, color: "#22d3ee", marginBottom: 4 }}>{riskParsed.uniquenessReduction}</p>
                                                    <p style={{ fontSize: 12, color: "#64748b" }}>
                                                        Higher uniqueness reduction means fewer records are individually identifiable.
                                                    </p>
                                                </div>

                                                {/* K-Anonymity */}
                                                {riskParsed.kAnonymity && (
                                                    <div style={{
                                                        padding: "14px 16px", borderRadius: 10,
                                                        background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
                                                    }}>
                                                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                                                            <Shield size={14} color="#f59e0b" />
                                                            <span style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0" }}>K-Anonymity</span>
                                                        </div>
                                                        <p style={{ fontSize: 13, fontFamily: "monospace", color: "#94a3b8" }}>
                                                            {riskParsed.kAnonymity}
                                                        </p>
                                                        <p style={{ fontSize: 12, color: "#64748b", marginTop: 4 }}>
                                                            k ≥ 5 is generally considered safe. k = 1 means some records are unique.
                                                        </p>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    ) : (
                                        <pre style={{ fontSize: 12, color: "#64748b", whiteSpace: "pre-wrap", lineHeight: 1.7, fontFamily: "monospace" }}>{result.risk_report}</pre>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}
                    {tab === "bias" && (
                        <div style={{ maxWidth: 1000, margin: "0 auto" }}>
                            <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 24 }}>
                                <div className="glass" style={{ padding: 32, textAlign: "center", height: "fit-content" }}>
                                    <div style={{ marginBottom: 24 }}>
                                        <div style={{
                                            width: 120, height: 120, borderRadius: "50%", border: "8px solid rgba(255,255,255,0.05)",
                                            margin: "0 auto", display: "flex", alignItems: "center", justifyContent: "center", position: "relative"
                                        }}>
                                            <div style={{ textAlign: "center" }}>
                                                <p style={{ fontSize: 32, fontWeight: 800, color: (parseFloat(biasParsed?.score ?? "0")) >= 80 ? "#10b981" : (parseFloat(biasParsed?.score ?? "0")) >= 50 ? "#f59e0b" : "#ef4444" }}>
                                                    {biasParsed?.score ?? 0}
                                                </p>
                                                <p style={{ fontSize: 10, color: "#64748b", fontWeight: 700 }}>HEALTH SCORE</p>
                                            </div>
                                        </div>
                                    </div>
                                    <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 8 }}>{biasParsed?.status}</h3>
                                    <p style={{ fontSize: 13, color: "#64748b", lineHeight: 1.6 }}>
                                        This score represents the objective statistical integrity and health of your dataset based on distribution patterns.
                                    </p>

                                    <div style={{ marginTop: 24, textAlign: "left", padding: "16px", borderRadius: 12, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.05)" }}>
                                        <p style={{ fontSize: 11, fontWeight: 800, color: "#94a3b8", textTransform: "uppercase", marginBottom: 10 }}>Dataset Stats</p>
                                        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                                            <div style={{ display: "flex", justifyContent: "space-between" }}>
                                                <span style={{ fontSize: 11, color: "#64748b" }}>TOTAL ROWS</span>
                                                <span style={{ fontSize: 12, color: "#e2e8f0", fontWeight: 600 }}>{result.bias_analysis?.metrics?.total_rows ?? "—"}</span>
                                            </div>
                                            <div style={{ display: "flex", justifyContent: "space-between" }}>
                                                <span style={{ fontSize: 11, color: "#64748b" }}>DUPLICATES</span>
                                                <span style={{ fontSize: 12, color: "#e2e8f0", fontWeight: 600 }}>{result.bias_analysis?.metrics?.duplicate_pct ?? "—"}%</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                                    <div className="glass" style={{ padding: 24 }}>
                                        <p style={{ fontSize: 14, fontWeight: 700, color: "#94a3b8", marginBottom: 20, display: "flex", alignItems: "center", gap: 8 }}>
                                            <AlertCircle size={16} /> Dataset Integrity Findings
                                        </p>
                                        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                                            {biasParsed?.findings.length ? biasParsed.findings.map((f, i) => (
                                                <div key={i} style={{
                                                    padding: "16px 20px", borderRadius: 12,
                                                    border: `1px solid ${f.severity === "high" ? "rgba(239,68,68,0.2)" : f.severity === "medium" ? "rgba(245,158,11,0.2)" : "rgba(16,185,129,0.2)"}`,
                                                    background: f.severity === "high" ? "rgba(239,68,68,0.05)" : f.severity === "medium" ? "rgba(245,158,11,0.05)" : "rgba(16,185,129,0.05)",
                                                    display: "flex", gap: 16, alignItems: "center"
                                                }}>
                                                    {f.severity === "high" ? <AlertTriangle color="#ef4444" size={24} /> : f.severity === "medium" ? <AlertCircle color="#f59e0b" size={24} /> : <CheckCircle color="#10b981" size={24} />}
                                                    <div>
                                                        <p style={{ fontSize: 11, fontWeight: 800, color: f.severity === "high" ? "#ef4444" : f.severity === "medium" ? "#f59e0b" : "#10b981", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4 }}>
                                                            {f.type.toUpperCase()}
                                                        </p>
                                                        <p style={{ fontSize: 14, color: "#e2e8f0", lineHeight: 1.4 }}>{f.message}</p>
                                                    </div>
                                                </div>
                                            )) : (
                                                <div style={{ textAlign: "center", padding: "32px 0", color: "#64748b" }}>
                                                    <CheckCircle size={32} style={{ margin: "0 auto 12px", opacity: 0.5 }} />
                                                    <p>No significant dataset integrity issues detected.</p>
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    {biasParsed?.targetImpacts && biasParsed.targetImpacts.length > 0 && (
                                        <div className="glass" style={{ padding: 24 }}>
                                            <p style={{ fontSize: 14, fontWeight: 700, color: "#94a3b8", marginBottom: 20, display: "flex", alignItems: "center", gap: 8 }}>
                                                <TrendingUp size={16} /> Feature Predictive Strength
                                            </p>
                                            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                                                {biasParsed.targetImpacts.slice(0, 10).map((imp) => (
                                                    <div key={imp.col}>
                                                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                                                            <span style={{ fontSize: 13, color: "#e2e8f0", fontWeight: 600 }}>{imp.col}</span>
                                                            <span style={{ fontSize: 12, color: "#64748b" }}>Strength: {imp.impact.toFixed(3)}</span>
                                                        </div>
                                                        <ScoreBar value={imp.impact} max={1} colorFn={(v) => v >= 0.95 ? "#ef4444" : v >= 0.7 ? "#8b5cf6" : "#10b981"} />
                                                        <p style={{ fontSize: 11, color: "#475569", marginTop: 4 }}>
                                                            {imp.impact >= 0.95
                                                                ? "Potential target leakage detected (extreme correlation)."
                                                                : imp.impact >= 0.7
                                                                    ? "Strong predictive signal for the target variable."
                                                                    : "Informational predictor with moderate signal strength."}
                                                        </p>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                                        {biasParsed?.aiInsights && (
                                            <div className="glass" style={{ padding: 20, gridColumn: "span 2", background: "rgba(139,92,246,0.05)", border: "1px solid rgba(139,92,246,0.2)" }}>
                                                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                                                    <Sparkles size={16} color="#a78bfa" />
                                                    <span style={{ fontSize: 13, fontWeight: 700, color: "#a78bfa" }}>AI Insight & Recommendation</span>
                                                </div>
                                                <div style={{ fontSize: 13, color: "#e2e8f0", lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
                                                    {biasParsed.aiInsights}
                                                </div>
                                            </div>
                                        )}

                                        <div className="glass" style={{ padding: 20 }}>
                                            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                                                <Activity size={16} color="#22d3ee" />
                                                <span style={{ fontSize: 13, fontWeight: 700, color: "#94a3b8" }}>Association Signals</span>
                                            </div>
                                            <p style={{ fontSize: 20, fontWeight: 700, color: (result.bias_analysis?.metrics?.associations?.length ?? 0) > 0 ? "#8b5cf6" : "#10b981" }}>
                                                {result.bias_analysis?.metrics?.associations?.length ?? 0} Sensitive Detectors
                                            </p>
                                            <div style={{ fontSize: 11, color: "#475569", marginTop: 4 }}>
                                                Columns with significant predictive or leakage potential identified via statistical tests.
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
