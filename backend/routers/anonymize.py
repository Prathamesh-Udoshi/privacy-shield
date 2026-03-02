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
):
    """Accept a CSV upload and start an anonymization job."""
    fname = file.filename or ""
    if not (fname.endswith(".csv") or fname.endswith(".txt")):
        raise HTTPException(400, "Only CSV files are supported")

    content = await file.read()
    job_id = create_job()
    excluded = [c.strip() for c in excluded_columns.split(",") if c.strip()]

    update_job(job_id, status=JobStatus.processing, progress=5, message="File received")

    background_tasks.add_task(
        _run_anonymization,
        job_id, content, epsilon, purpose, seed, max_rows, excluded,
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
) -> None:
    try:
        result = await asyncio.to_thread(
            _sync_anonymize, job_id, content, epsilon, purpose, seed, max_rows, excluded
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

    data: list = []
    for i, row in enumerate(reader):
        if i >= max_rows:
            break
        data.append(dict(row))

    if not data:
        raise ValueError("CSV file is empty")

    # ── 2. Configure ────────────────────────────────────────────────────────
    update_job(job_id, progress=20, message="Loading privacy policy…")
    config_loader = ConfigLoader()
    config_loader.config["global_epsilon"] = epsilon
    config_loader.config["purpose"] = purpose
    if seed is not None:
        config_loader.config["random_seed"] = seed

    # ── 3. Anonymize ────────────────────────────────────────────────────────
    update_job(job_id, progress=40, message="Applying differential privacy…")
    anonymized_data, budget, pre_report, pre_data, column_types, ai_active = apply_anonymization(
        data, config_loader, excluded_columns=excluded
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
        "ai_active": ai_active,
        "preprocessing_report": pre_report,
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
