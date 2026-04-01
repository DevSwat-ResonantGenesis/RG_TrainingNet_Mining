"""RG Mining Service — Decentralized LLM training orchestration."""

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

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
from .dashboard_api import router as dashboard_router
from .ws_handler import handle_mining_ws, ws_manager
from .chain_bridge import chain_bridge


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("RG Mining Service starting...")
    # Register with Lighthouse (non-blocking, best-effort)
    try:
        await chain_bridge.register_with_lighthouse(service_type="mining")
    except Exception as e:
        logger.warning(f"Lighthouse registration skipped: {e}")
    yield
    await chain_bridge.close()
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
app.include_router(dashboard_router)


@app.get("/")
async def root():
    return {"service": "rg-mining", "version": "0.1.0"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "rg-mining",
        "ws_miners": ws_manager.connected_count,
        "chain_bridge": chain_bridge.get_stats(),
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the mesh dashboard UI."""
    import os
    html_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "index.html")
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws/mining")
async def mining_websocket(ws: WebSocket):
    """WebSocket endpoint for miner agents — real-time task streaming and gradient submission."""
    await handle_mining_ws(ws)
