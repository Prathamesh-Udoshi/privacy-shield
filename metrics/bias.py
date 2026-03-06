"""
Dataset Integrity and Statistical Diagnostics for Privacy Shield.

Provides objective, dataset-agnostic metrics for data quality and distribution:
  - Data health (duplicates, nulls, cardinality)
  - Representation imbalance (categorical skew)
  - Feature-Target association (potential leakage or strong predictors)
  - General distribution diagnostics
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats as sp_stats


# ── Statistical Helpers ──────────────────────────────────────────────────────

def _association_strength(col_a: pd.Series, col_b: pd.Series, type_a: str, type_b: str) -> float:
    """Computes a normalized association score (0 to 1) based on column types."""
    try:
        # Determine if numeric or categorical
        a_is_num = type_a in ("age", "year", "monetary", "numeric", "count")
        b_is_num = type_b in ("age", "year", "monetary", "numeric", "count")

        if a_is_num and b_is_num:
            # Numeric-Numeric: Pearson
            a = pd.to_numeric(col_a, errors="coerce")
            b = pd.to_numeric(col_b, errors="coerce")
            mask = a.notna() & b.notna()
            if mask.sum() < 5: return 0.0
            r, _ = sp_stats.pearsonr(a[mask], b[mask])
            return abs(float(r))

        elif not a_is_num and not b_is_num:
            # Cat-Cat: Cramér's V
            ct = pd.crosstab(col_a, col_b)
            chi2 = sp_stats.chi2_contingency(ct)[0]
            n = ct.sum().sum()
            min_dim = min(ct.shape[0], ct.shape[1]) - 1
            if min_dim <= 0 or n == 0: return 0.0
            return float(np.sqrt(chi2 / (n * min_dim)))

        else:
            # Cat-Num: Eta-squared approximation
            cat_col = col_a if not a_is_num else col_b
            num_col = col_b if not a_is_num else col_a
            num = pd.to_numeric(num_col, errors="coerce")
            mask = num.notna()
            groups = [num[mask & (cat_col == g)].values for g in cat_col[mask].unique()]
            groups = [g for g in groups if len(g) >= 2]
            if len(groups) < 2: return 0.0
            grand_mean = num[mask].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
            ss_total = ((num[mask] - grand_mean) ** 2).sum()
            return float(ss_between / ss_total) if ss_total > 0 else 0.0
    except Exception:
        return 0.0


# ── Main Analysis ───────────────────────────────────────────────────────────

def analyze_dataset_integrity(
    data: List[Dict[str, Any]],
    column_types: Dict[str, str],
    target_variable: Optional[str] = None
) -> Dict[str, Any]:
    """
    Performs objective statistical diagnostics on the dataset.
    """
    if not data:
        return {"health_score": 0, "findings": [], "summary": "Empty dataset"}

    df = pd.DataFrame(data)
    total_rows = len(df)
    findings = []

    # ── 1. Data Integrity & Health ──────────────────────────────────────────
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = (duplicate_count / total_rows) * 100 if total_rows > 0 else 0.0
    null_counts = {k: int(v) for k, v in df.isnull().sum().to_dict().items()}

    if duplicate_pct > 1:
        findings.append({
            "type": "integrity",
            "severity": "medium" if duplicate_pct > 10 else "low",
            "message": f"Dataset contains {duplicate_count} duplicate rows ({duplicate_pct:.1f}%).",
            "metric": "duplicates"
        })

    for col, count in null_counts.items():
        if count > 0:
            pct = (count / total_rows) * 100
            if pct > 10:
                findings.append({
                    "type": "integrity",
                    "severity": "medium",
                    "message": f"Column '{col}' has high missing values ({pct:.1f}%).",
                    "column": col,
                    "metric": "missing_values"
                })

    # ── 2. Representation & Skew ────────────────────────────────────────────
    imbalances = []
    categorical_cols = [c for c, t in column_types.items() if t in ("string", "boolean", "category", "location")]
    
    for col in categorical_cols:
        if col not in df.columns: continue
        counts = df[col].value_counts(normalize=True)
        if len(counts) > 1:
            ratio = float(counts.max() / (counts.min() if counts.min() > 0 else 1e-6))
            if ratio > 5:
                findings.append({
                    "type": "distribution",
                    "severity": "medium" if ratio > 20 else "low",
                    "message": f"High representation skew in '{col}' (Ratio {ratio:.1f}:1).",
                    "column": col,
                    "metric": "imbalance"
                })
                imbalances.append({"column": col, "ratio": round(ratio, 1)})

    # ── 3. Statistical Associations (Predictive Strength / Leakage) ─────────
    associations = []
    target_impacts = {}
    
    # Check for target correlations if specified
    if target_variable and target_variable in df.columns:
        target_type = column_types.get(target_variable, "numeric")
        for col in df.columns:
            if col == target_variable or col in ("id", "index", "uuid"): continue
            
            score = _association_strength(df[col], df[target_variable], column_types.get(col, "string"), target_type)
            target_impacts[col] = score
            
            if score > 0.95:
                findings.append({
                    "type": "leakage",
                    "severity": "high",
                    "message": f"Extreme correlation ({score:.2f}) between '{col}' and target. Potential data leakage.",
                    "column": col,
                    "metric": "association"
                })
            elif score > 0.7:
                associations.append({"column": col, "score": round(score, 2), "label": "Strong Predictor"})

    # ── 4. Score Calculation ────────────────────────────────────────────────
    # Score starts at 100 and drops for major integrity issues
    score = 100.0
    score -= min(20, duplicate_pct)
    score -= len([f for f in findings if f["severity"] == "high"]) * 15
    score -= len([f for f in findings if f["severity"] == "medium"]) * 5
    score = max(0, min(100, round(score, 1)))

    return {
        "health_score": score,
        "findings": findings,
        "target_impacts": target_impacts,
        "metrics": {
            "total_rows": total_rows,
            "duplicate_count": duplicate_count,
            "duplicate_pct": round(duplicate_pct, 2),
            "null_counts": null_counts,
            "imbalances": imbalances,
            "associations": associations
        },
        "summary": "Dataset Integrity & Statistical Audit Complete"
    }


def generate_diagnostic_report_str(analysis: Dict[str, Any]) -> str:
    """
    Generates a markdown report focusing on statistical diagnostics.
    """
    score = analysis.get("health_score", 0)
    findings = analysis.get("findings", [])
    metrics = analysis.get("metrics", {})
    impacts = analysis.get("target_impacts", {})
    
    status = "EXCELLENT" if score >= 90 else "GOOD" if score >= 75 else "FAIR" if score >= 50 else "POOR"
    
    report = [
        f"DATASET INTEGRITY & STATISTICAL AUDIT",
        f"=====================================",
        f"Health Score: {score}/100 ({status})",
        "",
        "### 📋 Dataset Overview",
        f"- Total Samples: {metrics.get('total_rows')}",
        f"- Unique Records: {metrics.get('total_rows', 0) - metrics.get('duplicate_count', 0)}",
        f"- Duplicate Rate: {metrics.get('duplicate_pct')}%",
        "",
        "### 🔍 Key Observations"
    ]
    
    if not findings:
        report.append("- ✅ No major statistical anomalies or integrity issues detected.")
    else:
        for f in findings:
            emoji = "🔴" if f["severity"] == "high" else "🟡" if f["severity"] == "medium" else "🔵"
            report.append(f"- {emoji} **[{f['type'].upper()}]** {f['message']}")

    if impacts:
        report.append("")
        report.append("### 📊 Statistical Feature Associations")
        report.append("Measures how strongly features relate to the performance target.")
        for col, score_val in sorted(impacts.items(), key=lambda x: -x[1])[:8]:
            bar = "█" * int(score_val * 20)
            report.append(f"- {col}: {score_val:.3f} {bar}")

    return "\n".join(report)
