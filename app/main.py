"""RG Mining Service — Decentralized LLM training orchestration."""

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

# Optional shared imports for Docker compatibility
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from shared.errors import setup_exception_handlers
    HAS_SHARED_ERRORS = True
except ImportError:
    HAS_SHARED_ERRORS = False
    setup_exception_handlers = None

from .routers import router
from .ws_handler import handle_mining_ws, ws_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("RG Mining Service starting...")
    yield
    logger.info("RG Mining Service stopped")


app = FastAPI(
    title="RG Mining Service",
    description="Decentralized LLM training: genesis seed, task management, gradient aggregation, miner rewards",
    version="0.1.0",
    lifespan=lifespan,
    redirect_slashes=False,
)

# Setup standardized exception handlers
if HAS_SHARED_ERRORS and setup_exception_handlers:
    setup_exception_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {"service": "rg-mining", "version": "0.1.0"}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "rg-mining", "ws_miners": ws_manager.connected_count}


@app.websocket("/ws/mining")
async def mining_websocket(ws: WebSocket):
    """WebSocket endpoint for miner agents — real-time task streaming and gradient submission."""
    await handle_mining_ws(ws)
