"""
Analyze router — CSV analysis without anonymization.

POST /api/v1/analyze  — upload CSV, get column analysis (types, samples, stats)
"""

import csv
import io
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException

router = APIRouter(prefix="/api/v1")


@router.post("/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    max_rows: int = Form(5000),
):
    """Upload a CSV and return column analysis without anonymization."""
    fname = file.filename or ""
    if not (fname.endswith(".csv") or fname.endswith(".txt")):
        raise HTTPException(400, "Only CSV files are supported")

    content = await file.read()

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    sample = text[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample)
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","

    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    headers = list(reader.fieldnames or [])
    if not headers:
        raise HTTPException(400, "CSV file has no headers")

    data = []
    for i, row in enumerate(reader):
        if i >= max_rows:
            break
        data.append(dict(row))

    if not data:
        raise HTTPException(400, "CSV file is empty")

    # ── Infer column types ──────────────────────────────────────────────────
    from core.anonymizer import infer_column_types
    column_types, _metadata = infer_column_types(
        headers, data[:min(100, len(data))]
    )

    # ── Build per-column analysis ───────────────────────────────────────────
    columns_analysis = []
    for h in headers:
        col_values = [row.get(h, "") for row in data]
        sample_values = [str(v) for v in col_values[:5] if v != "" and v is not None]
        col_type = column_types.get(h, "string")

        # Compute basic stats for numeric columns
        stats = {}
        if col_type in ["age", "year", "monetary", "numeric", "count"]:
            numeric_vals = []
            for v in col_values:
                try:
                    numeric_vals.append(float(v))
                except (ValueError, TypeError):
                    pass
            if numeric_vals:
                stats = {
                    "min": round(min(numeric_vals), 2),
                    "max": round(max(numeric_vals), 2),
                    "mean": round(sum(numeric_vals) / len(numeric_vals), 2),
                    "count": len(numeric_vals),
                }
        elif col_type in ["boolean"]:
            stats = {"unique": len(set(col_values))}
        elif col_type in ["string", "id"]:
            unique = len(set(v for v in col_values if v))
            stats = {"unique": unique, "total": len(col_values)}

        # Default mechanism for this type
        mechanism_map = {
            "age": "bounded_laplace",
            "year": "bounded_laplace",
            "monetary": "laplace",
            "numeric": "laplace",
            "count": "discrete_laplace",
            "boolean": "randomized_response",
            "id": "hash",
            "string": "mask",
        }

        columns_analysis.append({
            "name": h,
            "detected_type": col_type,
            "mechanism": mechanism_map.get(col_type, "mask"),
            "sample_values": sample_values[:5],
            "stats": stats,
        })

    # ── 5. Dataset Integrity & Statistical Diagnostics ──────────────────────
    health_score = 100.0
    bias_findings = []
    try:
        from metrics.bias import analyze_dataset_integrity
        # Determine a default target variable (last numeric column usually)
        target_var = None
        for col in reversed(headers):
            if column_types.get(col) in ("numeric", "monetary", "count") and col not in ("id", "index", "uuid"):
                target_var = col
                break
        
        integrity_analysis = analyze_dataset_integrity(data, column_types, target_variable=target_var)
        health_score = integrity_analysis.get("health_score", 100.0)
        bias_findings = integrity_analysis.get("findings", [])
    except Exception as e:
        print(f"Dataset diagnostic failed (non-fatal): {e}")

    return {
        "headers": headers,
        "row_count": len(data),
        "columns": columns_analysis,
        "health_score": health_score,
        "bias_findings": bias_findings,
    }
