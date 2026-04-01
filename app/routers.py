"""
RG Mining Service API Endpoints
================================

REST API for the decentralized mining network.
Provides endpoints for: genesis initialization, task management,
gradient submission, miner registration, parameter server stats.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from .genesis_seed import genesis_initializer, SeedModelConfig
from .param_server import param_server
from .training_task import task_manager, GradientSubmission
from .gradient_compressor import CompressedGradient, verify_gradient_hash
from .auth_middleware import (
    AuthenticatedUser,
    get_current_user,
    check_rate_limit,
)
from .chain_bridge import chain_bridge

router = APIRouter(prefix="/mining", tags=["mining"])


# ============== Request/Response Models ==============

class GenesisInitRequest(BaseModel):
    model_id: str = "resonant-seed-1b"
    miner_ids: List[str] = []
    ipfs_base_url: str = "ipfs://"


class MinerRegisterRequest(BaseModel):
    miner_id: str
    miner_class: str = "miner"  # validator_miner, core_miner, miner


class GradientSubmitRequest(BaseModel):
    submission_id: str
    task_id: str
    miner_id: str
    model_id: str
    epoch: int
    batch_index: int
    top_k_indices: List[int]
    top_k_values: List[float]
    original_size: int
    compressed_size: int
    compression_ratio: float
    loss_before: float
    loss_after: float
    samples_processed: int
    training_time_seconds: float
    gradient_hash: str
    data_shard_hash: str
    weight_shard_hash: str


class TaskAssignRequest(BaseModel):
    miner_id: str


# ============== Genesis Endpoints ==============

@router.post("/genesis/initialize")
async def initialize_genesis(
    request: GenesisInitRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    req: Request = None,
):
    """Initialize the genesis seed model and create training tasks."""
    check_rate_limit(user.user_id or "anon", "genesis")
    if not user.is_admin() and user.auth_method != "dev":
        raise HTTPException(status_code=403, detail="Admin role required for genesis init")
    config = SeedModelConfig(model_id=request.model_id)
    state = await genesis_initializer.initialize(
        model_config=config,
        miner_ids=request.miner_ids,
        ipfs_base_url=request.ipfs_base_url,
    )
    return {"status": "initialized", "genesis": state.to_dict()}


@router.get("/genesis/status")
async def get_genesis_status(user: AuthenticatedUser = Depends(get_current_user)):
    """Get genesis initialization status."""
    check_rate_limit(user.user_id or "anon", "default")
    return genesis_initializer.get_status()


@router.post("/genesis/assign-tasks")
async def assign_genesis_tasks(user: AuthenticatedUser = Depends(get_current_user)):
    """Assign pending tasks to registered miners."""
    check_rate_limit(user.user_id or "anon", "genesis")
    assignments = await genesis_initializer.assign_tasks_to_miners()
    return {"assignments": assignments, "count": len(assignments)}


# ============== Miner Registration ==============

@router.post("/miners/register")
async def register_miner(
    request: MinerRegisterRequest,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Register a miner with the parameter server."""
    check_rate_limit(user.user_id or "anon", "register")
    state = param_server.register_miner(request.miner_id, request.miner_class)
    return {"status": "registered", "miner": state.to_dict()}


@router.get("/miners")
async def list_miners(user: AuthenticatedUser = Depends(get_current_user)):
    """List all registered miners and their states."""
    check_rate_limit(user.user_id or "anon", "default")
    return {
        "miners": param_server.get_miner_states(),
        "count": len(param_server.miners),
    }


@router.get("/miners/{miner_id}")
async def get_miner(miner_id: str, user: AuthenticatedUser = Depends(get_current_user)):
    """Get a specific miner's state."""
    check_rate_limit(user.user_id or "anon", "default")
    miner = param_server.miners.get(miner_id)
    if not miner:
        raise HTTPException(status_code=404, detail="Miner not found")
    return miner.to_dict()


# ============== Task Management ==============

@router.post("/tasks/assign")
async def assign_task(
    request: TaskAssignRequest,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Assign the next available training task to a miner."""
    check_rate_limit(user.user_id or "anon", "task_assign")
    task = task_manager.assign_task(request.miner_id)
    if not task:
        raise HTTPException(status_code=404, detail="No tasks available")
    return {"status": "assigned", "task": task.to_dict()}


@router.get("/tasks/stats")
async def get_task_stats(user: AuthenticatedUser = Depends(get_current_user)):
    """Get task manager statistics."""
    return task_manager.get_stats()


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, user: AuthenticatedUser = Depends(get_current_user)):
    """Get a specific task."""
    task = task_manager.tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.to_dict()


# ============== Gradient Submission ==============

@router.post("/gradients/submit")
async def submit_gradient(
    request: GradientSubmitRequest,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Submit a compressed gradient from a miner."""
    check_rate_limit(user.user_id or request.miner_id, "gradient_submit")
    submission = GradientSubmission(
        submission_id=request.submission_id,
        task_id=request.task_id,
        miner_id=request.miner_id,
        model_id=request.model_id,
        epoch=request.epoch,
        batch_index=request.batch_index,
        top_k_indices=request.top_k_indices,
        top_k_values=request.top_k_values,
        original_size=request.original_size,
        compressed_size=request.compressed_size,
        compression_ratio=request.compression_ratio,
        loss_before=request.loss_before,
        loss_after=request.loss_after,
        samples_processed=request.samples_processed,
        training_time_seconds=request.training_time_seconds,
        gradient_hash=request.gradient_hash,
        data_shard_hash=request.data_shard_hash,
        weight_shard_hash=request.weight_shard_hash,
    )

    # Build compressed gradients for param server
    compressed = [CompressedGradient(
        indices=request.top_k_indices,
        values=request.top_k_values,
        original_size=request.original_size,
        k=request.compressed_size,
        gradient_hash=request.gradient_hash,
        layer_name=f"layer_{request.batch_index}",
    )]

    # Verify hash integrity via param server FIRST
    ps_accepted = param_server.receive_gradient(submission, compressed)
    if not ps_accepted:
        raise HTTPException(status_code=400, detail="Gradient rejected by parameter server (hash mismatch or unregistered)")

    # Only record in task manager after param server accepts
    accepted = task_manager.submit_result(submission)
    if not accepted:
        raise HTTPException(status_code=400, detail="Gradient submission rejected by task manager")

    # Record on external blockchain (fire-and-forget)
    import asyncio
    asyncio.create_task(chain_bridge.record_gradient_on_chain(
        miner_id=request.miner_id,
        task_id=request.task_id,
        gradient_hash=request.gradient_hash,
        loss_value=request.loss_after,
        samples_processed=request.samples_processed,
        reward_amount=0,  # REST path doesn't calculate reward inline
        submission_id=request.submission_id,
        model_id=request.model_id,
        global_step=param_server.global_step,
    ))

    return {"status": "accepted", "submission_id": request.submission_id}


# ============== Aggregation ==============

@router.post("/aggregate")
async def trigger_aggregation(user: AuthenticatedUser = Depends(get_current_user)):
    """Trigger gradient aggregation on the parameter server."""
    check_rate_limit(user.user_id or "anon", "default")
    merged = param_server.aggregate()
    if merged is None:
        return {"status": "skipped", "reason": "Not enough gradients"}
    return {
        "status": "aggregated",
        "global_step": param_server.global_step,
        "layers_merged": len(merged),
    }


# ============== Parameter Server Stats ==============

@router.get("/param-server/stats")
async def get_param_server_stats(user: AuthenticatedUser = Depends(get_current_user)):
    """Get parameter server statistics."""
    return param_server.get_stats()


@router.get("/param-server/rewards")
async def get_miner_rewards(year: int = 1, user: AuthenticatedUser = Depends(get_current_user)):
    """Get calculated rewards for all miners."""
    rewards = param_server.get_miner_rewards(year)
    return {"year": year, "rewards": rewards}


# ============== Health ==============

@router.get("/health")
async def health():
    """Health check."""
    return {
        "service": "rg-mining",
        "status": "ok",
        "genesis_initialized": genesis_initializer.state.initialized,
        "registered_miners": len(param_server.miners),
        "global_step": param_server.global_step,
    }
