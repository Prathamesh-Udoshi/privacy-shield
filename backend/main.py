"""Privacy Shield — FastAPI Backend Entry Point."""
import os
import sys

# Make the privacy_shield root importable (dp/, metrics/, config/, etc.)
_ROOT = os.path.dirname(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import anonymize, policies

app = FastAPI(
    title="Privacy Shield API",
    description="Differential Privacy Data Anonymization — REST API",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(anonymize.router, prefix="/api/v1", tags=["Anonymization"])
app.include_router(policies.router, prefix="/api/v1", tags=["Policies"])


@app.get("/api/v1/health", tags=["Health"])
async def health():
    return {"status": "ok", "version": "2.0.0"}
