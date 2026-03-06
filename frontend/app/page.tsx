"use client";
import { useState, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import { useDropzone } from "react-dropzone";
import {
    Shield, Upload, Lock, ChevronDown, ChevronUp,
    Zap, AlertTriangle, BarChart2, GitBranch, Share2,
    ArrowRight, Eye, FileText, Database, ShieldCheck, TrendingUp, CheckCircle,
    Settings2, ToggleLeft, ToggleRight,
} from "lucide-react";
import {
    uploadFile, pollJobStatus, analyzeFile,
    COL_TYPE_META, VALID_COL_TYPES, MECHANISM_OPTIONS,
    ColumnAnalysis, ColumnConfig,
} from "@/lib/api";

const PURPOSES = [
    { key: "general", label: "General", epsilon: 1.0, icon: <Shield size={14} /> },
    { key: "qa_testing", label: "QA Testing", epsilon: 1.5, icon: <Zap size={14} /> },
    { key: "model_retraining", label: "ML Training", epsilon: 0.5, icon: <BarChart2 size={14} /> },
    { key: "analytics", label: "Analytics", epsilon: 0.8, icon: <GitBranch size={14} /> },
    { key: "data_sharing", label: "Data Sharing", epsilon: 0.3, icon: <Share2 size={14} /> },
];

const FEATURES = [
    {
        icon: <ShieldCheck size={28} />,
        title: "Mathematically Proven Privacy",
        desc: "Built on differential privacy, the gold standard used by Apple, Google, and the US Census Bureau. Your data gets provable anonymization guarantees, not just heuristic masking.",
        color: "#8b5cf6",
    },
    {
        icon: <TrendingUp size={28} />,
        title: "Preserve Data Utility",
        desc: "Smart noise calibration per column type ensures aggregate statistics (means, distributions, correlations) survive anonymization. Get detailed utility scores for every column.",
        color: "#22d3ee",
    },
    {
        icon: <Eye size={28} />,
        title: "Re-identification Risk Analysis",
        desc: "Automated membership inference simulations and k-anonymity analysis tell you exactly how safe your data is. Catch risks before they become breaches.",
        color: "#10b981",
    },
    {
        icon: <Database size={28} />,
        title: "Automatic Column Detection",
        desc: "Upload any CSV and Privacy Shield auto-detects ages, monetary values, IDs, booleans, and more. Each type gets purpose-tuned noise for optimal privacy and utility tradeoff.",
        color: "#f59e0b",
    },
    {
        icon: <FileText size={28} />,
        title: "Comprehensive Reports",
        desc: "Get full utility preservation and risk assessment reports. Column-by-column breakdown with mean preservation, standard deviation change, MAE, and overall utility scores.",
        color: "#ec4899",
    },
    {
        icon: <Lock size={28} />,
        title: "Configurable Privacy Budget",
        desc: "Fine-tune your privacy-utility tradeoff with the epsilon (ε) slider. Choose from purpose presets like ML Training, Analytics, QA Testing, or Data Sharing.",
        color: "#6366f1",
    },
];

const HOW_TO_STEPS = [
    {
        step: "01",
        title: "Upload Your CSV",
        desc: "Drag and drop any CSV file with sensitive data. The system automatically detects column types (age, monetary, boolean, IDs, etc.) and shows you a preview.",
        icon: <Upload size={24} />,
    },
    {
        step: "02",
        title: "Configure Privacy Settings",
        desc: "Choose a data purpose preset (General, QA Testing, ML Training, Analytics, Data Sharing) or manually set the privacy budget ε. Lower ε = stronger privacy, higher ε = better utility.",
        icon: <Zap size={24} />,
    },
    {
        step: "03",
        title: "Anonymize with One Click",
        desc: "Hit 'Anonymize Data' and differential privacy noise is applied column-by-column with type-specific mechanisms. Laplace for numeric, randomized response for booleans.",
        icon: <Shield size={24} />,
    },
    {
        step: "04",
        title: "Analyze & Download",
        desc: "Review interactive charts, detailed utility scores, re-identification risk analysis, and row-by-row comparisons. Then download your provably anonymous CSV.",
        icon: <BarChart2 size={24} />,
    },
];

export default function Home() {
    const router = useRouter();
    const toolRef = useRef<HTMLDivElement>(null);
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<{ headers: string[]; rows: Record<string, string>[] } | null>(null);
    const [epsilon, setEpsilon] = useState(1.0);
    const [purpose, setPurpose] = useState("general");
    const [seed, setSeed] = useState<number | null>(null);
    const [maxRows, setMaxRows] = useState(5000);
    const [excluded, setExcluded] = useState<string[]>([]);
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [processing, setProcessing] = useState(false);
    const [progress, setProgress] = useState(0);
    const [progressMsg, setProgressMsg] = useState("");
    const [error, setError] = useState<string | null>(null);

    // Column config dashboard state
    const [analyzing, setAnalyzing] = useState(false);
    const [columnAnalysis, setColumnAnalysis] = useState<ColumnAnalysis[] | null>(null);
    const [columnConfigs, setColumnConfigs] = useState<Record<string, ColumnConfig>>({});
    const [auditSummary, setAuditSummary] = useState<{ score: number; findings: { type: string; severity: string; message: string }[] } | null>(null);
    const [showColumnConfig, setShowColumnConfig] = useState(false);

    const scrollToTool = () => {
        toolRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    // Parse CSV client-side for preview only
    const parsePreview = useCallback((f: File) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const text = (e.target?.result as string) || "";
            const lines = text.split("\n").filter(Boolean);
            if (!lines.length) return;
            const headers = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""));
            const rows = lines.slice(1, 6).map((line) => {
                const vals = line.split(",").map((v) => v.trim().replace(/^"|"$/g, ""));
                return Object.fromEntries(headers.map((h, i) => [h, vals[i] ?? ""]));
            });
            setPreview({ headers, rows });
        };
        reader.readAsText(f);
    }, []);

    const onDrop = useCallback((accepted: File[]) => {
        const f = accepted[0];
        if (!f) return;
        setFile(f);
        setError(null);
        parsePreview(f);

        // Call /analyze to get column info for the config dashboard
        setAnalyzing(true);
        setColumnAnalysis(null);
        setColumnConfigs({});
        analyzeFile({ file: f, maxRows })
            .then((result) => {
                setColumnAnalysis(result.columns);
                if (result.health_score !== undefined) {
                    setAuditSummary({
                        score: result.health_score,
                        findings: result.bias_findings || []
                    });
                }
                // Initialize per-column configs with defaults
                const configs: Record<string, ColumnConfig> = {};
                for (const col of result.columns) {
                    configs[col.name] = {
                        epsilon: 0.2,
                        method: col.mechanism,
                        enabled: true,
                        type_override: col.detected_type,
                    };
                }
                setColumnConfigs(configs);
            })
            .catch((err) => { console.warn("Column analysis failed:", err); })
            .finally(() => setAnalyzing(false));
    }, [parsePreview, maxRows]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop, accept: { "text/csv": [".csv"] }, multiple: false,
    });

    const handleAnonymize = async () => {
        if (!file) return;
        setProcessing(true); setError(null); setProgress(5);
        setProgressMsg("Uploading file…");
        try {
            // Build excluded columns from configs
            const disabledCols = Object.entries(columnConfigs)
                .filter(([, cfg]) => !cfg.enabled)
                .map(([name]) => name);
            const allExcluded = [...new Set([...excluded, ...disabledCols])];

            // Build type overrides
            const typeOverrides: Record<string, string> = {};
            if (columnAnalysis) {
                for (const col of columnAnalysis) {
                    const cfg = columnConfigs[col.name];
                    if (cfg && cfg.type_override !== col.detected_type) {
                        typeOverrides[col.name] = cfg.type_override;
                    }
                }
            }

            // Build per-column epsilon/method configs
            const colConfigs: Record<string, Partial<ColumnConfig>> = {};
            for (const [name, cfg] of Object.entries(columnConfigs)) {
                if (cfg.enabled) {
                    colConfigs[name] = { epsilon: cfg.epsilon, method: cfg.method };
                }
            }

            const { job_id } = await uploadFile({
                file, epsilon, purpose, seed, maxRows,
                excludedColumns: allExcluded,
                columnConfigs: Object.keys(colConfigs).length > 0 ? colConfigs : undefined,
                typeOverrides: Object.keys(typeOverrides).length > 0 ? typeOverrides : undefined,
            });
            let done = false;
            while (!done) {
                await new Promise((r) => setTimeout(r, 600));
                const status = await pollJobStatus(job_id);
                setProgress(status.progress);
                setProgressMsg(status.message);
                if (status.status === "done") {
                    done = true;
                    router.push(`/results/${job_id}`);
                } else if (status.status === "failed") {
                    throw new Error(status.message);
                }
            }
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : "Unknown error");
            setProcessing(false);
        }
    };

    const updateColumnConfig = (colName: string, updates: Partial<ColumnConfig>) => {
        setColumnConfigs(prev => ({
            ...prev,
            [colName]: { ...prev[colName], ...updates },
        }));
    };

    const toggleExclude = (col: string) =>
        setExcluded((prev) => prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]);

    return (
        <div style={{ minHeight: "100vh" }}>
            {/* Nav */}
            <nav className="nav">
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <Shield size={22} color="#8b5cf6" />
                    <span style={{ fontWeight: 700, fontSize: 17, color: "#e2e8f0" }}>Privacy Shield</span>
                    <span style={{
                        marginLeft: 8, padding: "2px 8px", borderRadius: 6,
                        background: "rgba(139,92,246,0.15)", color: "#a78bfa",
                        fontSize: 11, fontWeight: 600, border: "1px solid rgba(139,92,246,0.3)"
                    }}>v2.0</span>
                </div>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <button className="btn-ghost" onClick={scrollToTool}>
                        <Upload size={14} /> Try Now
                    </button>
                    <a href="/api/docs" target="_blank">
                        <button className="btn-ghost">API Docs</button>
                    </a>
                </div>
            </nav>

            {/* ===== LANDING SECTION ===== */}

            {/* Hero */}
            <section className="section landing-hero">
                <div style={{ position: "relative", zIndex: 1, textAlign: "center", maxWidth: 800, margin: "0 auto" }}>
                    <div style={{
                        display: "inline-flex", alignItems: "center", gap: 6, marginBottom: 28,
                        padding: "6px 16px", borderRadius: 99,
                        background: "rgba(139,92,246,0.12)", border: "1px solid rgba(139,92,246,0.3)"
                    }}>
                        <Lock size={12} color="#a78bfa" />
                        <span style={{ fontSize: 12, color: "#a78bfa", fontWeight: 600 }}>
                            Mathematically guaranteed differential privacy
                        </span>
                    </div>
                    <h1 style={{ fontSize: "clamp(40px,7vw,72px)", fontWeight: 800, lineHeight: 1.08, marginBottom: 20 }}>
                        <span className="gradient-text"><i>Protect Data.</i><br /><i>Preserve Insights.</i></span>
                    </h1>
                    <p style={{ fontSize: 19, color: "#94a3b8", maxWidth: 620, margin: "0 auto 40px", lineHeight: 1.6 }}>
                        Upload any CSV with sensitive data, configure your privacy budget, and download a
                        provably anonymous dataset, complete with utility scores and risk reports, in seconds.
                    </p>
                    <div style={{ display: "flex", gap: 14, justifyContent: "center", flexWrap: "wrap" }}>
                        <button className="btn-brand" onClick={scrollToTool} style={{ fontSize: 16, padding: "16px 36px" }}>
                            <Shield size={18} /> Anonymize Your Data
                            <ArrowRight size={16} />
                        </button>
                        <a href="#how-it-works">
                            <button className="btn-ghost" style={{ fontSize: 15, padding: "15px 28px" }}>
                                Learn How It Works
                            </button>
                        </a>
                    </div>
                </div>
            </section>

            {/* What is this tool? */}
            <section className="section" style={{ padding: "80px 32px" }}>
                <div style={{ maxWidth: 1200, margin: "0 auto" }}>
                    <div style={{ textAlign: "center", marginBottom: 56 }}>
                        <p style={{
                            fontSize: 12, fontWeight: 700, color: "#8b5cf6", textTransform: "uppercase",
                            letterSpacing: "0.15em", marginBottom: 12,
                        }}>What is Privacy Shield?</p>
                        <h2 style={{ fontSize: "clamp(28px,4vw,42px)", fontWeight: 800, lineHeight: 1.15, marginBottom: 16 }}>
                            Industrial-Grade <span className="gradient-text">Data Anonymization</span>
                        </h2>
                        <p style={{ fontSize: 16, color: "#64748b", maxWidth: 680, margin: "0 auto", lineHeight: 1.7 }}>
                            Privacy Shield uses <strong style={{ color: "#a78bfa" }}>differential privacy</strong>, the same
                            mathematically rigorous framework used by Apple, Google, and the US Census Bureau, to add
                            carefully calibrated noise to your dataset. The result? Your data remains useful for
                            analytics and ML, but individual records become provably unidentifiable.
                        </p>
                    </div>

                    <div className="feature-grid">
                        {FEATURES.map((f, i) => (
                            <div key={i} className="feature-card glass">
                                <div className="feature-icon" style={{ background: `${f.color}15`, color: f.color, border: `1px solid ${f.color}33` }}>
                                    {f.icon}
                                </div>
                                <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 8 }}>{f.title}</h3>
                                <p style={{ fontSize: 14, color: "#64748b", lineHeight: 1.65 }}>{f.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* How to use */}
            <section id="how-it-works" className="section" style={{ padding: "80px 32px", background: "rgba(139,92,246,0.02)" }}>
                <div style={{ maxWidth: 1200, margin: "0 auto" }}>
                    <div style={{ textAlign: "center", marginBottom: 56 }}>
                        <p style={{
                            fontSize: 12, fontWeight: 700, color: "#22d3ee", textTransform: "uppercase",
                            letterSpacing: "0.15em", marginBottom: 12,
                        }}>How to Use</p>
                        <h2 style={{ fontSize: "clamp(28px,4vw,42px)", fontWeight: 800, lineHeight: 1.15, marginBottom: 16 }}>
                            Four Simple <span className="gradient-text">Steps</span>
                        </h2>
                        <p style={{ fontSize: 16, color: "#64748b", maxWidth: 580, margin: "0 auto", lineHeight: 1.7 }}>
                            From raw CSV to provably anonymous dataset in under a minute.
                        </p>
                    </div>

                    <div className="steps-grid">
                        {HOW_TO_STEPS.map((s, i) => (
                            <div key={i} className="step-card">
                                <div className="step-number">{s.step}</div>
                                <div className="step-icon-wrap">
                                    {s.icon}
                                </div>
                                <h3 style={{ fontSize: 17, fontWeight: 700, marginBottom: 8 }}>{s.title}</h3>
                                <p style={{ fontSize: 14, color: "#64748b", lineHeight: 1.65 }}>{s.desc}</p>
                                {i < HOW_TO_STEPS.length - 1 && <div className="step-connector" />}
                            </div>
                        ))}
                    </div>

                    <div style={{ textAlign: "center", marginTop: 48 }}>
                        <button className="btn-brand" onClick={scrollToTool} style={{ fontSize: 16, padding: "16px 36px" }}>
                            <Lock size={16} /> Get Started Now
                            <ArrowRight size={16} />
                        </button>
                    </div>
                </div>
            </section>

            {/* Privacy budget explainer */}
            <section className="section" style={{ padding: "80px 48px" }}>
                <div style={{ display: "flex", gap: 24, alignItems: "flex-start", flexWrap: "wrap" }}>

                    <div style={{ flex: 1, minWidth: 280 }}>
                        <h3 style={{ fontSize: 20, fontWeight: 700, marginBottom: 12 }}>
                            <span style={{ color: "#a78bfa", textAlign: "center", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "clamp(28px,4vw,32px)" }}>Understanding the Privacy Budget (ε)</span>
                        </h3>
                        <p style={{ fontSize: 14, color: "#64748b", lineHeight: 1.7, marginBottom: 16 }}>
                            Epsilon (ε) controls the privacy-utility tradeoff. A <strong style={{ color: "#10b981" }}>lower ε</strong> means
                            more noise is added, offering stronger privacy but potentially reducing data utility. A <strong style={{ color: "#f59e0b" }}>higher ε</strong> preserves
                            more statistical properties but provides weaker privacy guarantees.
                        </p>
                        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12 }}>
                            {[
                                { range: "ε ≤ 0.5", label: "Maximum Privacy", desc: "Best for sharing publicly", color: "#10b981" },
                                { range: "0.5 < ε ≤ 1.5", label: "Balanced", desc: "Good for analytics & ML", color: "#8b5cf6" },
                                { range: "ε > 1.5", label: "Maximum Utility", desc: "Internal QA testing", color: "#f59e0b" },
                            ].map((b) => (
                                <div key={b.range} style={{
                                    padding: "12px 16px", borderRadius: 10,
                                    background: `${b.color}10`, border: `1px solid ${b.color}30`,
                                }}>
                                    <p style={{ fontSize: 13, fontWeight: 700, color: b.color, marginBottom: 2 }}>{b.range}</p>
                                    <p style={{ fontSize: 12, fontWeight: 600, color: "#e2e8f0", marginBottom: 2 }}>{b.label}</p>
                                    <p style={{ fontSize: 11, color: "#64748b" }}>{b.desc}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </section>

            {/* ===== TOOL SECTION ===== */}
            <section ref={toolRef} className="section" style={{ padding: "80px 48px 100px", background: "rgba(0,0,0,0.2)" }}>
                <div style={{ textAlign: "center", marginBottom: 40 }}>
                    <p style={{
                        fontSize: 12, fontWeight: 700, color: "#8b5cf6", textTransform: "uppercase",
                        letterSpacing: "0.15em", marginBottom: 12,
                    }}>Anonymization Tool</p>
                    <h2 style={{ fontSize: "clamp(28px,4vw,42px)", fontWeight: 800, lineHeight: 1.15 }}>
                        Anonymize Your <span className="gradient-text">Data</span>
                    </h2>
                </div>

                <div style={{ padding: 0 }}>
                    {/* Dropzone */}
                    {!processing && (
                        <>
                            <div {...getRootProps()} className={`dropzone ${isDragActive ? "active" : ""}`}>
                                <input {...getInputProps()} />
                                <Upload size={36} color={file ? "#8b5cf6" : "#475569"} style={{ margin: "0 auto 16px" }} />
                                {file ? (
                                    <div>
                                        <p style={{ color: "#a78bfa", fontWeight: 600, fontSize: 15 }}>{file.name}</p>
                                        <p style={{ color: "#475569", fontSize: 13, marginTop: 4 }}>
                                            {(file.size / 1024).toFixed(1)} KB · click or drag to replace
                                        </p>
                                    </div>
                                ) : (
                                    <div>
                                        <p style={{ color: "#94a3b8", fontWeight: 600, fontSize: 15, marginBottom: 4 }}>
                                            {isDragActive ? "Drop it!" : "Drop your CSV file here"}
                                        </p>
                                        <p style={{ color: "#475569", fontSize: 13 }}>or click to browse</p>
                                    </div>
                                )}
                            </div>

                            {/* CSV Preview */}
                            {preview && (
                                <div style={{ marginTop: 24 }}>
                                    <p style={{ fontSize: 12, color: "#64748b", fontWeight: 600, marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                                        Preview (first 5 rows)
                                    </p>
                                    <div style={{ overflowX: "auto", borderRadius: 10, border: "1px solid rgba(255,255,255,0.07)" }}>
                                        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                                            <thead>
                                                <tr style={{ background: "rgba(255,255,255,0.04)" }}>
                                                    {preview.headers.map((h) => (
                                                        <th key={h} style={{ padding: "8px 12px", textAlign: "left", color: "#64748b", fontWeight: 600, borderBottom: "1px solid rgba(255,255,255,0.07)" }}>
                                                            {h}
                                                        </th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {preview.rows.map((row, i) => (
                                                    <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                                                        {preview.headers.map((h) => (
                                                            <td key={h} style={{ padding: "7px 12px", color: "#94a3b8" }}>{row[h]}</td>
                                                        ))}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>

                                    {/* ═══ DATASET AUDIT SUMMARY ═══ */}
                                    {auditSummary && (
                                        <div style={{
                                            marginTop: 20, padding: "16px 20px", borderRadius: 12,
                                            background: "rgba(139,92,246,0.04)", border: "1px solid rgba(139,92,246,0.15)",
                                            display: "flex", gap: 20, flexWrap: "wrap", alignItems: "center"
                                        }}>
                                            <div style={{ textAlign: "center", borderRight: "1px solid rgba(139,92,246,0.1)", paddingRight: 20 }}>
                                                <p style={{ fontSize: 9, color: "#64748b", fontWeight: 800, marginBottom: 2 }}>HEALTH SCORE</p>
                                                <p style={{
                                                    fontSize: 24, fontWeight: 900,
                                                    color: auditSummary.score >= 80 ? "#10b981" : auditSummary.score >= 50 ? "#f59e0b" : "#ef4444"
                                                }}>{auditSummary.score}</p>
                                            </div>
                                            <div style={{ flex: 1, minWidth: 200 }}>
                                                <p style={{ fontSize: 12, color: "#e2e8f0", fontWeight: 600, display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
                                                    <Shield size={12} color="#a78bfa" /> Dataset Pre-Audit Findings
                                                </p>
                                                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                                                    {auditSummary.findings.length === 0 ? (
                                                        <span style={{ fontSize: 11, color: "#10b981", background: "rgba(16,185,129,0.1)", padding: "4px 10px", borderRadius: 6 }}>
                                                            ✅ Healthy: No patterns detected.
                                                        </span>
                                                    ) : (
                                                        <>
                                                            {auditSummary.findings.slice(0, 2).map((f, i) => (
                                                                <span key={i} style={{
                                                                    fontSize: 10, padding: "3px 10px", borderRadius: 6,
                                                                    background: f.severity === "high" ? "rgba(239,68,68,0.1)" : "rgba(245,158,11,0.1)",
                                                                    color: f.severity === "high" ? "#ef4444" : "#f59e0b",
                                                                    fontWeight: 600
                                                                }}>
                                                                    {f.type.toUpperCase()}: {f.message.split(".")[0].split(":")[0]}
                                                                </span>
                                                            ))}
                                                            {auditSummary.findings.length > 2 && (
                                                                <span style={{ fontSize: 10, color: "#64748b", fontWeight: 600 }}>
                                                                    + {auditSummary.findings.length - 2} more
                                                                </span>
                                                            )}
                                                        </>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {/* Exclude columns */}
                                    <div style={{ marginTop: 16 }}>
                                        <p style={{ fontSize: 12, color: "#64748b", fontWeight: 600, marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                                            🎯 Exclude from noise (target columns)
                                        </p>
                                        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                                            {preview.headers.map((h) => (
                                                <button key={h} onClick={() => toggleExclude(h)} style={{
                                                    padding: "4px 12px", borderRadius: 8, border: "1px solid",
                                                    borderColor: excluded.includes(h) ? "rgba(139,92,246,0.6)" : "rgba(255,255,255,0.1)",
                                                    background: excluded.includes(h) ? "rgba(139,92,246,0.15)" : "transparent",
                                                    color: excluded.includes(h) ? "#a78bfa" : "#64748b",
                                                    fontSize: 12, cursor: "pointer", transition: "all 0.15s"
                                                }}>{h}</button>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* ═══ PER-COLUMN CONFIG DASHBOARD ═══ */}
                            {columnAnalysis && columnAnalysis.length > 0 && (
                                <div style={{ marginTop: 24 }}>
                                    <button
                                        onClick={() => setShowColumnConfig(v => !v)}
                                        className="btn-ghost"
                                        style={{ marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}
                                    >
                                        <Settings2 size={14} />
                                        Per-Column Configuration
                                        {showColumnConfig ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                                    </button>

                                    {showColumnConfig && (
                                        <div style={{ display: "grid", gap: 10 }}>
                                            {/* Header row */}
                                            <div style={{
                                                display: "grid",
                                                gridTemplateColumns: "2fr 1.2fr 1.2fr 0.8fr 60px",
                                                gap: 12, padding: "8px 14px",
                                                fontSize: 10, fontWeight: 700, color: "#475569",
                                                textTransform: "uppercase", letterSpacing: "0.1em",
                                            }}>
                                                <span>Column</span>
                                                <span>Type</span>
                                                <span>Mechanism</span>
                                                <span>Epsilon</span>
                                                <span style={{ textAlign: "center" }}>Active</span>
                                            </div>

                                            {columnAnalysis.map((col) => {
                                                const cfg = columnConfigs[col.name];
                                                if (!cfg) return null;
                                                const typeMeta = COL_TYPE_META[cfg.type_override] || COL_TYPE_META.string;
                                                const mechanisms = MECHANISM_OPTIONS[cfg.type_override] || ["mask"];

                                                return (
                                                    <div key={col.name} style={{
                                                        display: "grid",
                                                        gridTemplateColumns: "2fr 1.2fr 1.2fr 0.8fr 60px",
                                                        gap: 12, padding: "10px 14px",
                                                        borderRadius: 10,
                                                        background: cfg.enabled
                                                            ? "rgba(255,255,255,0.03)"
                                                            : "rgba(255,255,255,0.01)",
                                                        border: "1px solid",
                                                        borderColor: cfg.enabled
                                                            ? "rgba(255,255,255,0.07)"
                                                            : "rgba(255,255,255,0.03)",
                                                        opacity: cfg.enabled ? 1 : 0.45,
                                                        transition: "all 0.2s",
                                                        alignItems: "center",
                                                    }}>
                                                        {/* Column name + samples */}
                                                        <div>
                                                            <p style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0", marginBottom: 2 }}>
                                                                {col.name}
                                                            </p>
                                                            <p style={{ fontSize: 10, color: "#475569" }}>
                                                                {col.sample_values.slice(0, 3).join(", ")}
                                                            </p>
                                                        </div>

                                                        {/* Type override */}
                                                        <select
                                                            value={cfg.type_override}
                                                            onChange={(e) => {
                                                                const newType = e.target.value;
                                                                const newMechanisms = MECHANISM_OPTIONS[newType] || ["mask"];
                                                                updateColumnConfig(col.name, {
                                                                    type_override: newType,
                                                                    method: newMechanisms[0],
                                                                });
                                                            }}
                                                            disabled={!cfg.enabled}
                                                            className="input"
                                                            style={{ fontSize: 12, padding: "6px 8px", cursor: "pointer" }}
                                                        >
                                                            {VALID_COL_TYPES.map(t => (
                                                                <option key={t} value={t}>
                                                                    {COL_TYPE_META[t]?.emoji} {COL_TYPE_META[t]?.label || t}
                                                                </option>
                                                            ))}
                                                        </select>

                                                        {/* Mechanism */}
                                                        <select
                                                            value={cfg.method}
                                                            onChange={(e) => updateColumnConfig(col.name, { method: e.target.value })}
                                                            disabled={!cfg.enabled}
                                                            className="input"
                                                            style={{ fontSize: 12, padding: "6px 8px", cursor: "pointer" }}
                                                        >
                                                            {mechanisms.map(m => (
                                                                <option key={m} value={m}>{m.replace(/_/g, " ")}</option>
                                                            ))}
                                                        </select>

                                                        {/* Epsilon */}
                                                        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                                                            <input
                                                                type="number"
                                                                min={0.01}
                                                                max={5.0}
                                                                step={0.05}
                                                                value={cfg.epsilon}
                                                                onChange={(e) => updateColumnConfig(col.name, { epsilon: parseFloat(e.target.value) || 0.1 })}
                                                                disabled={!cfg.enabled}
                                                                className="input"
                                                                style={{ fontSize: 12, padding: "6px 8px", width: "100%" }}
                                                            />
                                                        </div>

                                                        {/* Enabled toggle */}
                                                        <div style={{ textAlign: "center", cursor: "pointer" }}
                                                            onClick={() => updateColumnConfig(col.name, { enabled: !cfg.enabled })}>
                                                            {cfg.enabled
                                                                ? <ToggleRight size={22} color="#8b5cf6" />
                                                                : <ToggleLeft size={22} color="#334155" />}
                                                        </div>
                                                    </div>
                                                );
                                            })}

                                            {/* Summary bar */}
                                            <div style={{
                                                display: "flex", justifyContent: "space-between",
                                                padding: "10px 14px", borderRadius: 8,
                                                background: "rgba(139,92,246,0.08)",
                                                border: "1px solid rgba(139,92,246,0.15)",
                                                fontSize: 12, color: "#94a3b8",
                                            }}>
                                                <span>
                                                    {Object.values(columnConfigs).filter(c => c.enabled).length} columns active
                                                </span>
                                                <span>
                                                    Total column ε: {Object.values(columnConfigs)
                                                        .filter(c => c.enabled)
                                                        .reduce((sum, c) => sum + c.epsilon, 0)
                                                        .toFixed(2)}
                                                </span>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}

                            {analyzing && (
                                <p style={{ fontSize: 12, color: "#64748b", marginTop: 12 }}>
                                    Analyzing columns...
                                </p>
                            )}

                            {/* Divider */}
                            <div style={{ height: 1, background: "rgba(255,255,255,0.06)", margin: "28px 0" }} />

                            {/* Purpose selector */}
                            <div style={{ marginBottom: 24 }}>
                                <label style={{ fontSize: 13, color: "#64748b", fontWeight: 600, display: "block", marginBottom: 10, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                                    Data Purpose
                                </label>
                                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                                    {PURPOSES.map((p) => (
                                        <button key={p.key} onClick={() => { setPurpose(p.key); setEpsilon(p.epsilon); }} style={{
                                            display: "flex", alignItems: "center", gap: 6,
                                            padding: "8px 14px", borderRadius: 9, border: "1px solid",
                                            borderColor: purpose === p.key ? "rgba(139,92,246,0.6)" : "rgba(255,255,255,0.09)",
                                            background: purpose === p.key ? "rgba(139,92,246,0.15)" : "transparent",
                                            color: purpose === p.key ? "#a78bfa" : "#64748b",
                                            fontSize: 13, fontWeight: 500, cursor: "pointer", transition: "all 0.15s"
                                        }}>
                                            {p.icon}{p.label}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Epsilon slider */}
                            <div style={{ marginBottom: 24 }}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                                    <label style={{ fontSize: 13, color: "#64748b", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                                        Privacy Budget (ε)
                                    </label>
                                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                        <span style={{
                                            padding: "4px 14px", borderRadius: 8,
                                            background: "rgba(139,92,246,0.15)", border: "1px solid rgba(139,92,246,0.3)",
                                            color: "#a78bfa", fontWeight: 700, fontSize: 16
                                        }}>ε = {epsilon.toFixed(1)}</span>
                                        <span style={{ fontSize: 12, color: "#475569" }}>
                                            {epsilon < 0.5 ? "Max Privacy" : epsilon < 1.5 ? "Balanced" : "Max Utility"}
                                        </span>
                                    </div>
                                </div>
                                <input type="range" min="0.1" max="5.0" step="0.1"
                                    value={epsilon} onChange={(e) => setEpsilon(parseFloat(e.target.value))}
                                    style={{ width: "100%", accentColor: "#8b5cf6" }}
                                />
                                <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
                                    <span style={{ fontSize: 11, color: "#334155" }}>0.1 · Strong Privacy</span>
                                    <span style={{ fontSize: 11, color: "#334155" }}>5.0 · High Utility</span>
                                </div>
                            </div>

                            {/* Advanced options */}
                            <button onClick={() => setShowAdvanced((v) => !v)} className="btn-ghost" style={{ marginBottom: showAdvanced ? 16 : 0 }}>
                                {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                                Advanced Options
                            </button>
                            {showAdvanced && (
                                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 4 }}>
                                    <div>
                                        <label style={{ fontSize: 12, color: "#64748b", display: "block", marginBottom: 6 }}>Anonymization Limit (Max Rows)</label>
                                        <input
                                            type="number"
                                            min={500}
                                            max={50000}
                                            step={500}
                                            value={maxRows}
                                            onChange={(e) => {
                                                const val = parseInt(e.target.value);
                                                if (!isNaN(val)) setMaxRows(val);
                                            }}
                                            className="input"
                                        />
                                    </div>
                                    <div>
                                        <label style={{ fontSize: 12, color: "#64748b", display: "block", marginBottom: 6 }}>Random seed (optional, 0 = none)</label>
                                        <input type="number" min={0} value={seed ?? 0} onChange={(e) => setSeed(parseInt(e.target.value) || null)} className="input" />
                                    </div>
                                </div>
                            )}

                            {/* Error */}
                            {error && (
                                <div style={{ marginTop: 16, padding: "12px 16px", borderRadius: 10, background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", display: "flex", gap: 8, alignItems: "center" }}>
                                    <AlertTriangle size={14} color="#ef4444" />
                                    <span style={{ color: "#ef4444", fontSize: 13 }}>{error}</span>
                                </div>
                            )}

                            {/* CTA */}
                            <div style={{ marginTop: 28, textAlign: "center" }}>
                                <button className="btn-brand" disabled={!file} onClick={handleAnonymize}
                                    style={{ fontSize: 16, padding: "14px 40px" }}>
                                    <Lock size={16} />
                                    Anonymize Data
                                </button>
                                {!file && (
                                    <p style={{ marginTop: 10, fontSize: 12, color: "#334155" }}>Drop a CSV file above to enable</p>
                                )}
                            </div>
                        </>
                    )}

                    {/* Processing state */}
                    {processing && (
                        <div style={{ textAlign: "center", padding: "48px 0" }}>
                            <div style={{
                                width: 72, height: 72, borderRadius: "50%", margin: "0 auto 24px",
                                background: "linear-gradient(135deg, rgba(139,92,246,0.2), rgba(34,211,238,0.2))",
                                border: "2px solid rgba(139,92,246,0.4)",
                                display: "flex", alignItems: "center", justifyContent: "center",
                                animation: "spin 2s linear infinite",
                            }}>
                                <Shield size={28} color="#8b5cf6" />
                            </div>
                            <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
                            <h2 style={{ fontWeight: 700, fontSize: 20, marginBottom: 8 }}>Applying Differential Privacy…</h2>
                            <p style={{ color: "#64748b", fontSize: 14, marginBottom: 28 }}>{progressMsg}</p>
                            <div style={{ maxWidth: 400, margin: "0 auto" }}>
                                <div className="progress-track">
                                    <div className="progress-fill" style={{ width: `${progress}%` }} />
                                </div>
                                <p style={{ marginTop: 8, fontSize: 13, color: "#475569" }}>{progress}%</p>
                            </div>
                        </div>
                    )}
                </div>
            </section>

            {/* Footer */}
            <footer style={{
                padding: "32px 32px", textAlign: "center",
                borderTop: "1px solid rgba(255,255,255,0.06)",
                color: "#334155", fontSize: 13,
            }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8, marginBottom: 8 }}>
                    <Shield size={14} color="#8b5cf6" />
                    <span style={{ fontWeight: 600, color: "#475569" }}>Privacy Shield</span>
                </div>
                <p>Industrial-grade differential privacy for everyone.</p>
            </footer>
        </div>
    );
}
