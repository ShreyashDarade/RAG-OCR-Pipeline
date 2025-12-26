from __future__ import annotations

import time
import hashlib
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from fastapi import FastAPI, File, Form, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from cachetools import TTLCache

from src.api.schemas import (
    AskContextItem,
    AskRequest,
    AskResponseSchema,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
    RetrievedDocumentSchema,
)
from src.core.config import settings
from src.core.logger import logger
from src.pipelines.ask import AskPipeline
from src.pipelines.ingestion import IngestionPipeline
from src.pipelines.retrieval import RetrievalPipeline
from src.services.reindexer import DataDirectoryWatcher


# === RATE LIMITING ===
limiter = Limiter(key_func=get_remote_address)

# === CACHING ===
# In-memory cache for retrieval results (use Redis in production)
_query_cache: TTLCache = TTLCache(maxsize=1000, ttl=settings.cache_ttl_seconds)

# === THREAD POOL FOR CPU-BOUND TASKS ===
_executor = ThreadPoolExecutor(max_workers=settings.max_workers)

# === PIPELINE SINGLETONS ===
_ingestion: IngestionPipeline | None = None
_retrieval: RetrievalPipeline | None = None
_ask: AskPipeline | None = None
_watcher: DataDirectoryWatcher | None = None


def _get_cache_key(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle management."""
    global _ingestion, _retrieval, _ask, _watcher
    
    logger.info("🚀 Starting RAG-OCR Pipeline (Production Mode)")
    logger.info(f"   Environment: {settings.environment}")
    logger.info(f"   Rate Limit: {settings.rate_limit_per_minute} req/min")
    logger.info(f"   Max Workers: {settings.max_workers}")
    logger.info(f"   OCR Languages: {settings.supported_ocr_languages}")
    
    # Initialize pipelines
    _ingestion = IngestionPipeline()
    _retrieval = RetrievalPipeline()
    _ask = AskPipeline()
    _watcher = DataDirectoryWatcher()
    
    # Start file watcher
    def _reindex_callback(path: Path) -> None:
        if path.suffix.lower() not in settings.allowed_file_extensions:
            return
        logger.info("Watcher triggered reindex for %s", path)
        _ingestion.ingest_path(path, force=True)
    
    _watcher.start(_reindex_callback)
    logger.info(f"📂 Watching directory: {settings.data_dir}")
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("🛑 Shutting down...")
    if _watcher:
        _watcher.stop()
    _executor.shutdown(wait=False)


app = FastAPI(
    title="RAG-OCR Pipeline API",
    version="1.0.0",
    description="Production-ready OCR and RAG pipeline with Hindi/Marathi/English support",
    lifespan=lifespan,
)

# === MIDDLEWARE ===
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === HEALTH CHECK ===
@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and Kubernetes."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "environment": settings.environment,
    }


@app.get("/ready")
async def readiness_check():
    """Readiness probe - checks if all dependencies are ready."""
    checks = {
        "ingestion_pipeline": _ingestion is not None,
        "retrieval_pipeline": _retrieval is not None,
        "ask_pipeline": _ask is not None,
    }
    all_ready = all(checks.values())
    return {
        "ready": all_ready,
        "checks": checks,
    }


# === API ENDPOINTS ===
@app.post("/api/v1/ingest", response_model=IngestResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    force: bool = False,
    image_language: str | None = Form(None),
):
    """
    Ingest a document (PDF or image) into the RAG pipeline.
    
    Supported languages for OCR: en (English), mr (Marathi), hi (Hindi)
    """
    if _ingestion is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Validate language hint
    if image_language and image_language not in settings.supported_ocr_languages:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {image_language}. Supported: {settings.supported_ocr_languages}"
        )
    
    file.file.seek(0)
    try:
        summary = await run_in_threadpool(_ingestion.ingest_upload, file, force, image_language)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    
    return IngestResponse(
        source=str(summary.source),
        text_chunks=summary.text_chunks,
        table_chunks=summary.table_chunks,
        image_chunks=summary.image_chunks,
        skipped_reason=summary.skipped_reason,
        reindexed=summary.reindexed,
    )


@app.post("/api/v1/retrieve", response_model=RetrieveResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def retrieve_documents(request: Request, payload: RetrieveRequest):
    """Retrieve relevant documents for a query using hybrid search."""
    if _retrieval is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Check cache
    cache_key = _get_cache_key(payload.query)
    if cache_key in _query_cache:
        logger.debug(f"Cache hit for query: {payload.query[:50]}...")
        return _query_cache[cache_key]
    
    result = await run_in_threadpool(_retrieval.retrieve, payload.query)
    documents = [
        RetrievedDocumentSchema(
            content=hit.document.page_content,
            score=hit.score,
            source=hit.document.metadata.get("source"),
            page=hit.document.metadata.get("page"),
            type=hit.document.metadata.get("type"),
            keywords=hit.document.metadata.get("keywords"),
        )
        for hit in result.documents
    ]
    response = RetrieveResponse(
        query=payload.query,
        expanded_queries=result.expanded_queries,
        documents=documents,
    )
    
    # Cache result
    _query_cache[cache_key] = response
    return response


@app.post("/api/v1/ask", response_model=AskResponseSchema)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def ask_question(request: Request, payload: AskRequest):
    """Ask a question and get an answer based on retrieved context."""
    if _ask is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    response = await run_in_threadpool(_ask.ask, payload.query)
    context_items = [
        AskContextItem(
            rank=item["rank"],
            score=item["score"],
            source=item["source"],
            page=item["page"],
            type=item["type"],
            keywords=item["keywords"],
            content=item["content"],
        )
        for item in response.context
    ]
    return AskResponseSchema(
        query=response.query,
        expanded_queries=response.expanded_queries,
        answer=response.answer,
        context=context_items,
    )


# === ERROR HANDLERS ===
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__}
    )


__all__ = ["app"]

