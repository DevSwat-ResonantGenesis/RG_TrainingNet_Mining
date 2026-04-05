"""RG Mining Service — Decentralized LLM training orchestration."""

import logging
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Environment ──
IS_PRODUCTION = os.getenv("RG_ENV", "development") == "production"

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


# ── Security Headers Middleware ──
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if IS_PRODUCTION:
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("RG Mining Service starting...")

    # Initialize ML database tables
    try:
        from .ml_db import init_ml_tables
        await init_ml_tables()
        logger.info("ML database tables ready")
    except Exception as e:
        logger.warning(f"ML database init skipped (non-fatal): {e}")

    # Load persistent weight shard registry from database
    try:
        from .weight_shard_registry import weight_registry
        await weight_registry.init_persistence()
        logger.info("Weight shard registry persistence initialized")
    except Exception as e:
        logger.warning(f"Weight registry persistence skipped (non-fatal): {e}")

    # Register with Lighthouse (non-blocking, best-effort)
    try:
        await chain_bridge.register_with_lighthouse(service_type="mining")
    except Exception as e:
        logger.warning(f"Lighthouse registration skipped: {e}")
    yield
    await chain_bridge.close()
    logger.info("RG Mining Service stopped")


# ── CORS: env-configurable allowed origins ──
_cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
if _cors_origins:
    ALLOWED_ORIGINS = [o.strip() for o in _cors_origins.split(",") if o.strip()]
elif IS_PRODUCTION:
    ALLOWED_ORIGINS = [
        "https://dev-swat.com",
        "https://www.dev-swat.com",
    ]
else:
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

app = FastAPI(
    title="RG Mining Service",
    description="Decentralized LLM training: genesis seed, task management, gradient aggregation, miner rewards",
    version="0.1.0",
    lifespan=lifespan,
    redirect_slashes=False,
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
)

# Setup standardized exception handlers
if HAS_SHARED_ERRORS and setup_exception_handlers:
    setup_exception_handlers(app)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Internal-Key"],
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
