"""Pydantic schemas for Privacy Shield API."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class PurposeEnum(str, Enum):
    general = "general"
    qa_testing = "qa_testing"
    model_retraining = "model_retraining"
    analytics = "analytics"
    data_sharing = "data_sharing"


class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    failed = "failed"


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int
    message: str
    created_at: str


class UtilityMetric(BaseModel):
    column: str
    utility_score: float
    original_mean: float
    noisy_mean: float
    relative_error: float
    std_change_pct: float
    mae: float


class JobResult(BaseModel):
    job_id: str
    headers: List[str]
    original_preview: List[Dict[str, Any]]
    anonymized_preview: List[Dict[str, Any]]
    anonymized_data: List[Dict[str, Any]]
    column_types: Dict[str, str]
    budget_used: float
    budget_total: float
    risk_level: str
    risk_report: str
    utility_report: str
    utility_metrics: List[Dict[str, Any]]
    row_count: int
    total_dataset_rows: int = 0
    max_rows_selected: int = 0
    processed_rows: int = 0
    ai_active: bool
    preprocessing_report: Dict[str, Any]
    bias_report: Optional[str] = None
    bias_analysis: Optional[Dict[str, Any]] = None


class PolicyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""
    config_yaml: str


class PolicyResponse(BaseModel):
    id: str
    name: str
    description: str
    config_yaml: str
    created_at: str
