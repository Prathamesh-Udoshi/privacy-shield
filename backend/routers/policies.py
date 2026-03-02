"""Policies router — CRUD for named privacy policies."""
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from backend.schemas import PolicyCreate, PolicyResponse

router = APIRouter()

_policies: Dict[str, Dict[str, Any]] = {}


@router.post("/policies", response_model=PolicyResponse, status_code=201)
async def create_policy(policy: PolicyCreate):
    policy_id = str(uuid.uuid4())
    record = {
        "id": policy_id,
        "name": policy.name,
        "description": policy.description,
        "config_yaml": policy.config_yaml,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _policies[policy_id] = record
    return record


@router.get("/policies", response_model=list[PolicyResponse])
async def list_policies():
    return list(_policies.values())


@router.get("/policies/{policy_id}", response_model=PolicyResponse)
async def get_policy(policy_id: str):
    p = _policies.get(policy_id)
    if not p:
        raise HTTPException(404, "Policy not found")
    return p


@router.delete("/policies/{policy_id}")
async def delete_policy(policy_id: str):
    if policy_id not in _policies:
        raise HTTPException(404, "Policy not found")
    del _policies[policy_id]
    return {"deleted": policy_id}
