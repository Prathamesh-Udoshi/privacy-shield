"""In-memory job store for anonymization jobs."""
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from backend.schemas import JobStatus

# Simple in-memory store — replace with Redis/SQLite for multi-worker deployments
_jobs: Dict[str, Dict[str, Any]] = {}


def create_job() -> str:
    """Create a new job and return its ID."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatus.pending,
        "progress": 0,
        "message": "Job queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result": None,
        "error": None,
    }
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    return _jobs.get(job_id)


def update_job(job_id: str, **kwargs) -> None:
    if job_id in _jobs:
        _jobs[job_id].update(kwargs)


def list_jobs() -> list:
    return list(_jobs.values())
