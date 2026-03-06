"""
Anonymization router — the core of the Privacy Shield API.

POST /api/v1/upload            — upload CSV, start async anonymization job
GET  /api/v1/jobs/{id}/status  — poll job status and progress
GET  /api/v1/jobs/{id}/result  — get full anonymization result
GET  /api/v1/jobs/{id}/download — download anonymized CSV
"""
import asyncio
import csv
import io
import json
import os
import sys
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from backend.job_store import create_job, get_job, update_job
from backend.schemas import JobStatus

# ── ensure the privacy_shield root is importable (dp/, core/, metrics/, etc.) ──
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Upload endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    epsilon: float = Form(1.0),
    purpose: str = Form("general"),
    seed: Optional[int] = Form(None),
    max_rows: int = Form(1000),
    excluded_columns: str = Form(""),
    column_configs: str = Form(""),
    type_overrides: str = Form(""),
):
    """Accept a CSV upload and start an anonymization job."""
    fname = file.filename or ""
    if not (fname.endswith(".csv") or fname.endswith(".txt")):
        raise HTTPException(400, "Only CSV files are supported")

    content = await file.read()
    job_id = create_job()
    excluded = [c.strip() for c in excluded_columns.split(",") if c.strip()]

    # Parse per-column configs from JSON
    col_configs_dict = {}
    if column_configs and column_configs.strip():
        try:
            col_configs_dict = json.loads(column_configs)
        except json.JSONDecodeError:
            pass

    type_overrides_dict = {}
    if type_overrides and type_overrides.strip():
        try:
            type_overrides_dict = json.loads(type_overrides)
        except json.JSONDecodeError:
            pass

    update_job(job_id, status=JobStatus.processing, progress=5, message="File received")

    background_tasks.add_task(
        _run_anonymization,
        job_id, content, epsilon, purpose, seed, max_rows, excluded,
        col_configs_dict, type_overrides_dict,
    )

    return {"job_id": job_id, "status": "processing"}


# ─────────────────────────────────────────────────────────────────────────────
# Background task (async wrapper → sync worker in thread pool)
# ─────────────────────────────────────────────────────────────────────────────

async def _run_anonymization(
    job_id: str,
    content: bytes,
    epsilon: float,
    purpose: str,
    seed: Optional[int],
    max_rows: int,
    excluded: list,
    column_configs: dict = None,
    type_overrides: dict = None,
) -> None:
    try:
        result = await asyncio.to_thread(
            _sync_anonymize, job_id, content, epsilon, purpose, seed, max_rows, excluded,
            column_configs or {}, type_overrides or {},
        )
        update_job(job_id, status=JobStatus.done, progress=100, message="Done!", result=result)
    except Exception as exc:
        update_job(job_id, status=JobStatus.failed, progress=0, message=str(exc), error=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Synchronous anonymization (runs in thread pool, never blocks event loop)
# ─────────────────────────────────────────────────────────────────────────────

def _sync_anonymize(
    job_id: str,
    content: bytes,
    epsilon: float,
    purpose: str,
    seed: Optional[int],
    max_rows: int,
    excluded: list,
    column_configs: dict = None,
    type_overrides: dict = None,
) -> dict:
    from config.loader import ConfigLoader
    from core.anonymizer import apply_anonymization, preprocess_data
    from metrics.utility import get_utility_metrics_data, generate_utility_report
    from metrics.risk import generate_risk_report

    # ── 1. Parse CSV ────────────────────────────────────────────────────────
    update_job(job_id, progress=10, message="Parsing CSV…")
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
        raise ValueError("CSV file has no headers")

    all_data: list = []
    for row in reader:
        all_data.append(dict(row))

    total_dataset_rows = len(all_data)
    processed_rows = min(total_dataset_rows, max_rows)
    data = all_data[:processed_rows]

    if not data:
        raise ValueError("CSV file is empty or contains no valid rows after processing limit")

    # ── 2. Configure ────────────────────────────────────────────────────────
    update_job(job_id, progress=20, message="Loading privacy policy…")
    config_loader = ConfigLoader()
    config_loader.config["global_epsilon"] = epsilon
    config_loader.config["purpose"] = purpose
    if seed is not None:
        config_loader.config["random_seed"] = seed

    # Inject per-column configs from the dashboard
    if column_configs:
        for col_name, col_cfg in column_configs.items():
            config_loader.config["columns"][col_name] = col_cfg

    # ── 3. Anonymize ────────────────────────────────────────────────────────
    update_job(job_id, progress=40, message="Applying differential privacy…")
    anonymized_data, budget, pre_report, pre_data, column_types, ai_active = apply_anonymization(
        data, config_loader, excluded_columns=excluded,
        type_overrides=type_overrides or {},
    )

    # ── 4. Metrics ──────────────────────────────────────────────────────────
    update_job(job_id, progress=80, message="Computing utility & risk metrics…")
    orig_cols = preprocess_data(pre_data)
    anon_cols = preprocess_data(anonymized_data)

    quant_cols = []
    for col, col_type in column_types.items():
        if col_type not in ["age", "year", "monetary", "numeric", "count"]:
            continue
        if col not in orig_cols or col not in anon_cols:
            continue
        try:
            orig_cols[col] = [
                float(v) for v in orig_cols[col]
                if v is not None and str(v).replace(".", "", 1).lstrip("-").isdigit()
            ]
            anon_cols[col] = [
                float(v) for v in anon_cols[col]
                if isinstance(v, (int, float))
            ]
            if len(orig_cols[col]) >= 2 and len(anon_cols[col]) >= 2:
                quant_cols.append(col)
        except Exception:
            pass

    risk_report_str = generate_risk_report(preprocess_data(pre_data), preprocess_data(anonymized_data))
    utility_report_str = (
        generate_utility_report(orig_cols, anon_cols, quant_cols)
        if quant_cols else "No numeric columns to analyze."
    )
    utility_metrics = (
        get_utility_metrics_data(orig_cols, anon_cols, quant_cols)
        if quant_cols else []
    )

    risk_level = "LOW"
    if "Overall Risk Category: CRITICAL" in risk_report_str:
        risk_level = "CRITICAL"
    elif "Overall Risk Category: MODERATE" in risk_report_str:
        risk_level = "MODERATE"

    # ── 6. Metrics and Reporting ───────────────────────────────────────────
    update_job(job_id, progress=90, message="Generating safety reports…")
    
    from metrics.bias import analyze_dataset_integrity, generate_diagnostic_report_str
    
    # Determine a default target variable for diagnostics
    target_var = None
    for col in reversed(headers):
        if column_types.get(col) in ("numeric", "monetary", "count") and col not in ("id", "index", "uuid"):
            target_var = col
            break
            
    integrity_analysis = analyze_dataset_integrity(data, column_types, target_variable=target_var)
    diagnostic_report_str = generate_diagnostic_report_str(integrity_analysis)

    update_job(job_id, progress=95, message="Finalizing…")

    return {
        "job_id": job_id,
        "headers": headers,
        "original_preview": data[:5],
        "anonymized_preview": anonymized_data[:5],
        "anonymized_data": anonymized_data,
        "column_types": column_types,
        "budget_used": round(budget.used_epsilon, 4),
        "budget_total": round(budget.total_epsilon, 4),
        "risk_level": risk_level,
        "risk_report": risk_report_str,
        "utility_report": utility_report_str,
        "utility_metrics": utility_metrics,
        "row_count": len(anonymized_data),
        "total_dataset_rows": total_dataset_rows,
        "max_rows_selected": max_rows,
        "processed_rows": processed_rows,
        "ai_active": ai_active,
        "preprocessing_report": pre_report,
        "bias_report": diagnostic_report_str,
        "bias_analysis": integrity_analysis,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Status / result / download endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "created_at": job["created_at"],
    }


@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] == JobStatus.failed:
        raise HTTPException(400, f"Job failed: {job.get('error', 'Unknown error')}")
    if job["status"] != JobStatus.done:
        raise HTTPException(400, f"Job not done yet (status: {job['status']})")
    return job["result"]


@router.get("/jobs/{job_id}/download")
async def download_result(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != JobStatus.done:
        raise HTTPException(400, "Job is not complete")

    result = job["result"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=result["headers"], extrasaction="ignore")
    writer.writeheader()
    writer.writerows(result["anonymized_data"])

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="anonymized_{job_id[:8]}.csv"'},
    )
