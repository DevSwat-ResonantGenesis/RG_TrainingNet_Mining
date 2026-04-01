"""
DASHBOARD AGGREGATION API
==========================

Single endpoint that pulls live data from all 3 mesh services
(Mining, Lighthouse, External Blockchain) and returns a unified
JSON payload for the dashboard frontend.
"""

import asyncio
import logging
import os
from typing import Dict, Any

import httpx
from fastapi import APIRouter

from .param_server import param_server
from .training_task import task_manager
from .genesis_seed import genesis_initializer
from .ws_handler import ws_manager
from .chain_bridge import chain_bridge

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

EXTERNAL_BLOCKCHAIN_URL = os.getenv("EXTERNAL_BLOCKCHAIN_URL", "http://localhost:8702")
LIGHTHOUSE_URL = os.getenv("LIGHTHOUSE_URL", "http://localhost:8700")
INTERNAL_SERVICE_KEY = os.getenv("AUTH_INTERNAL_SERVICE_KEY", "")


def _auth_headers() -> Dict[str, str]:
    if INTERNAL_SERVICE_KEY:
        return {"X-Internal-Key": INTERNAL_SERVICE_KEY}
    return {}


async def _fetch(url: str, timeout: float = 5.0) -> Dict:
    """Fetch JSON from a URL, return empty dict on failure."""
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=_auth_headers()) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.debug(f"Dashboard fetch failed: {url} — {e}")
    return {}


@router.get("/data")
async def get_dashboard_data():
    """Aggregate live data from all mesh services."""

    # Fetch from external services in parallel
    chain_status, chain_latest, chain_verify, lh_peers, lh_stats = await asyncio.gather(
        _fetch(f"{EXTERNAL_BLOCKCHAIN_URL}/distributed/status"),
        _fetch(f"{EXTERNAL_BLOCKCHAIN_URL}/distributed/blocks/latest"),
        _fetch(f"{EXTERNAL_BLOCKCHAIN_URL}/distributed/chain/verify"),
        _fetch(f"{LIGHTHOUSE_URL}/lighthouse/peers"),
        _fetch(f"{LIGHTHOUSE_URL}/lighthouse/stats"),
    )

    # Local mining data (no HTTP needed)
    mining = {
        "genesis_initialized": genesis_initializer.state.initialized,
        "genesis_status": genesis_initializer.get_status() if genesis_initializer.state.initialized else None,
        "param_server": param_server.get_stats(),
        "tasks": task_manager.get_stats(),
        "miners": param_server.get_miner_states(),
        "miner_count": len(param_server.miners),
        "ws_connected": ws_manager.connected_count,
        "chain_bridge": chain_bridge.get_stats(),
    }

    return {
        "blockchain": {
            "status": chain_status,
            "latest_block": chain_latest,
            "integrity": chain_verify,
        },
        "lighthouse": {
            "peers": lh_peers.get("peers", []),
            "peer_count": lh_peers.get("count", 0),
            "stats": lh_stats,
        },
        "mining": mining,
    }


@router.get("/blocks")
async def get_recent_blocks(limit: int = 10):
    """Get recent blocks from the external blockchain."""
    chain_status = await _fetch(f"{EXTERNAL_BLOCKCHAIN_URL}/distributed/status")
    height = chain_status.get("chain_height", 0)

    blocks = []
    start = max(0, height - limit)
    fetch_tasks = [
        _fetch(f"{EXTERNAL_BLOCKCHAIN_URL}/distributed/blocks/{i}")
        for i in range(start, height)
    ]
    results = await asyncio.gather(*fetch_tasks)
    for b in results:
        if b:
            blocks.append(b)

    return {"blocks": blocks, "height": height}
