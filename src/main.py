import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import uuid
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import TicketClassifier

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}'
)
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("API_KEY", "dev-secret-key-change-in-prod")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Depends(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


clf = TicketClassifier()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        clf.load()
        logger.info("Model loaded on startup")
    except Exception as e:
        logger.warning(f"Could not load model on startup: {e}")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Support Ticket Classifier",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        f"path={request.url.path} request_id={request_id} "
        f"status={response.status_code} latency_ms={latency_ms}"
    )
    response.headers["X-Request-ID"] = request_id
    return response


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=2000,
                      description="Support ticket text")


class PredictResponse(BaseModel):
    label: str
    confidence: float
    request_id: str


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": clf.is_trained}


@app.post("/predict", response_model=PredictResponse,
          dependencies=[Depends(verify_api_key)])
async def predict(req: PredictRequest, request: Request):
    if not clf.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    result = clf.predict(req.text)
    logger.info(
        f"event=prediction label={result['label']} "
        f"confidence={result['confidence']}"
    )
    return PredictResponse(**result, request_id=request_id)


@app.get("/model/info", dependencies=[Depends(verify_api_key)])
async def model_info():
    import json
    import pathlib
    metrics = {}
    if pathlib.Path("metrics.json").exists():
        metrics = json.loads(pathlib.Path("metrics.json").read_text())
    return {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "metrics": metrics
    }