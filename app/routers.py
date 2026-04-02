"""
RG Mining Service API Endpoints
================================

REST API for the decentralized mining network.
Provides endpoints for: genesis initialization, task management,
gradient submission, miner registration, parameter server stats.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import logging
from uuid import uuid4
from datetime import datetime, timezone
from dataclasses import asdict

from .genesis_seed import (
    genesis_initializer, SeedModelConfig,
    MODEL_REGISTRY, TRAINING_DATA_SOURCES,
    list_models, get_model_config, get_best_model_for_network,
)
from .param_server import param_server
from .training_task import task_manager, GradientSubmission
from .gradient_compressor import CompressedGradient, verify_gradient_hash
from .auth_middleware import (
    AuthenticatedUser,
    get_current_user,
    check_rate_limit,
)
from .chain_bridge import chain_bridge
from .p2p_discovery import p2p_discovery
from .slashing import (
    slashing_engine, ViolationType, ViolationSeverity,
)
from .network_dashboard import network_dashboard, DASHBOARD_HTML
from .wallet_service import wallet_service, TokenType, StakeStatus

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


# ============== Model Registry ==============

@router.get("/models")
async def get_models():
    """List all available model tiers and their requirements."""
    return {
        "models": list_models(),
        "current_model": genesis_initializer.state.model_config.model_id if genesis_initializer.state.model_config else None,
        "active_miners": len(param_server.miners),
        "recommended_model": get_best_model_for_network(len(param_server.miners)),
    }


@router.get("/models/{model_id}")
async def get_model_details(model_id: str):
    """Get detailed config for a specific model tier."""
    cfg = get_model_config(model_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}. Use GET /mining/models to list available models.")
    return {"model_id": model_id, **cfg}


@router.get("/training-data")
async def get_training_data_sources():
    """List all training data sources used for model training."""
    return TRAINING_DATA_SOURCES


# ============== Shard Manager ==============

from .shard_manager import shard_manager, MinerCapability
from .sharded_param_server import create_parameter_server


class MinerCapabilityRequest(BaseModel):
    miner_id: str
    gpu_model: str = ""
    gpu_vram_gb: float = 0.0
    system_ram_gb: float = 0.0
    cpu_cores: int = 0
    bandwidth_mbps: float = 0.0
    storage_available_gb: float = 0.0
    location_region: str = "unknown"
    supported_dtypes: List[str] = ["fp32", "fp16"]


@router.post("/shards/register-capability")
async def register_miner_capability(req: MinerCapabilityRequest):
    """Register a miner's hardware capabilities for shard assignment."""
    cap = MinerCapability(
        miner_id=req.miner_id,
        gpu_model=req.gpu_model,
        gpu_vram_gb=req.gpu_vram_gb,
        system_ram_gb=req.system_ram_gb,
        cpu_cores=req.cpu_cores,
        bandwidth_mbps=req.bandwidth_mbps,
        storage_available_gb=req.storage_available_gb,
        location_region=req.location_region,
        supported_dtypes=req.supported_dtypes,
    )
    result = shard_manager.register_miner(cap)
    return {"status": "registered", "miner": result.to_dict()}


@router.get("/shards/assignments")
async def get_shard_assignments():
    """List all current shard assignments."""
    return {
        "assignments": {
            mid: a.to_dict() for mid, a in shard_manager.miner_assignments.items()
        },
        "total": len(shard_manager.miner_assignments),
    }


@router.get("/shards/assignment/{miner_id}")
async def get_miner_assignment(miner_id: str):
    """Get a specific miner's shard assignment."""
    assignment = shard_manager.get_assignment(miner_id)
    if not assignment:
        raise HTTPException(status_code=404, detail=f"No assignment for miner {miner_id}")
    return assignment.to_dict()


@router.get("/shards/pipeline-groups")
async def get_pipeline_groups():
    """List all pipeline groups."""
    return {
        "groups": {
            gid: g.to_dict() for gid, g in shard_manager.pipeline_groups.items()
        },
        "total": len(shard_manager.pipeline_groups),
    }


@router.post("/shards/form-groups")
async def form_pipeline_groups(
    model_id: str = "resonant-seed-1b",
    target_redundancy: int = 2,
):
    """Trigger pipeline group formation from available miners."""
    config = get_model_config(model_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
    groups = shard_manager.form_pipeline_groups(model_id, config, target_redundancy)
    return {
        "formed": len(groups),
        "groups": [g.to_dict() for g in groups],
    }


@router.post("/shards/report-ready/{miner_id}")
async def report_shard_ready(miner_id: str):
    """Miner reports its shard is loaded and ready."""
    ok = shard_manager.report_shard_ready(miner_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"No assignment for miner {miner_id}")
    return {"status": "ready", "miner_id": miner_id}


@router.get("/shards/stats")
async def shard_manager_stats():
    """Shard manager statistics."""
    return shard_manager.get_stats()


@router.get("/shards/needs-sharding/{model_id}")
async def check_needs_sharding(model_id: str):
    """Check if a model needs sharding based on available miners."""
    config = get_model_config(model_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
    needs = shard_manager.needs_sharding(config)
    available = shard_manager.get_available_miners()
    optimal_stages = shard_manager.compute_optimal_stages(config, available) if available else 1
    return {
        "model_id": model_id,
        "needs_sharding": needs,
        "num_available_miners": len(available),
        "optimal_stages": optimal_stages,
        "model_params": config.get("num_parameters", 0),
        "model_layers": config.get("num_layers", 0),
    }


@router.get("/shards/pipeline-peers/{miner_id}")
async def get_pipeline_peers(miner_id: str):
    """Get peers in the same pipeline group (for P2P activation routing)."""
    peers = shard_manager.get_pipeline_peers(miner_id)
    assignment = shard_manager.get_assignment(miner_id)
    return {
        "miner_id": miner_id,
        "peers": peers,
        "upstream": assignment.upstream_miner_id if assignment else None,
        "downstream": assignment.downstream_miner_id if assignment else None,
    }


# ============== Network-Native Model (Weight Registry + Shard Slicer) ==============

from .weight_shard_registry import (
    weight_registry, ShardLocation, ShardState, ReplicaPriority,
)
from .shard_slicer import shard_slicer, WeightTransferRequest, create_transfer_plan
from fastapi.responses import StreamingResponse
import asyncio
import json as _json


class WeightTransferRequestModel(BaseModel):
    miner_id: str
    model_id: str
    layer_start: int
    layer_end: int
    include_embedding: bool = False
    include_lm_head: bool = False
    preferred_source: str = ""


class ShardLoadedReport(BaseModel):
    miner_id: str
    model_id: str
    layer_start: int
    layer_end: int
    weight_hash: str
    size_bytes: int = 0
    num_params: int = 0
    miner_address: str = ""


class ShardIntegrityCheck(BaseModel):
    miner_id: str
    shard_key: str
    reported_hash: str
    global_step: int = 0


@router.get("/weights/model-status/{model_id}")
async def network_model_status(model_id: str):
    """
    Full status of a network-native model — the 'GPS' of weights across the swarm.
    Shows which shards live where, replication health, and on-chain version.
    """
    return weight_registry.get_network_model_status(model_id)


@router.get("/weights/shard-map/{model_id}")
async def get_shard_map(model_id: str):
    """
    Get the complete shard map — which layers live on which miners.
    This is the 'DNS' of the model across the network.
    """
    return {
        "model_id": model_id,
        "shard_map": weight_registry.get_shard_map(model_id),
        "replication": weight_registry.get_replication_status(model_id),
    }


@router.post("/weights/request-transfer")
async def request_weight_transfer(req: WeightTransferRequestModel):
    """
    Request a transfer plan for downloading specific layer weights.
    
    Returns an ordered list of sources (peers + fallback seed slicer)
    that the miner should try to pull weights from.
    """
    transfer_req = WeightTransferRequest(
        requester_miner_id=req.miner_id,
        model_id=req.model_id,
        layer_start=req.layer_start,
        layer_end=req.layer_end,
        include_embedding=req.include_embedding,
        include_lm_head=req.include_lm_head,
        preferred_source=req.preferred_source,
    )
    plan = create_transfer_plan(transfer_req, weight_registry, shard_manager)
    return plan.to_dict()


@router.get("/weights/stream/{model_id}")
async def stream_weight_slice(
    model_id: str,
    layer_start: int = 0,
    layer_end: int = 0,
    include_embedding: bool = False,
    include_lm_head: bool = False,
):
    """
    Stream specific layer weights as chunked response.
    
    This is the seed slicer endpoint — the fallback when no P2P peer
    has the requested shard. Miners should prefer P2P transfers.
    """
    manifest = shard_slicer.create_manifest(
        model_id, layer_start, layer_end,
        include_embedding, include_lm_head,
    )
    if not manifest:
        raise HTTPException(
            status_code=404,
            detail=f"No cached weights for {model_id} layers {layer_start}-{layer_end}"
        )

    return {"manifest": manifest.to_dict()}


@router.post("/weights/report-loaded")
async def report_shard_loaded(req: ShardLoadedReport):
    """
    Miner reports it has finished loading/downloading a weight shard.
    
    This registers the shard in the weight registry, making this miner
    a source for P2P weight transfers to other miners.
    """
    location = ShardLocation(
        model_id=req.model_id,
        miner_id=req.miner_id,
        layer_start=req.layer_start,
        layer_end=req.layer_end,
        weight_hash=req.weight_hash,
        state=ShardState.LOADED,
        priority=ReplicaPriority.PRIMARY,
        size_bytes=req.size_bytes,
        num_params=req.num_params,
        miner_address=req.miner_address,
    )
    result = weight_registry.register_shard(location)

    # Also report shard ready in shard_manager
    shard_manager.report_shard_ready(req.miner_id)

    return {
        "status": "registered",
        "location_id": result.location_id,
        "shard_key": result.shard_key,
        "replicas": len(weight_registry.shard_replicas.get(result.shard_key, set())),
    }


@router.post("/weights/verify-integrity")
async def verify_shard_integrity(req: ShardIntegrityCheck):
    """
    Verify a miner's shard weights match the on-chain expected hash.
    Used to detect weight corruption, tampering, or desync.
    """
    valid = weight_registry.verify_shard_integrity(
        req.miner_id, req.shard_key, req.reported_hash, req.global_step
    )
    return {"valid": valid, "shard_key": req.shard_key, "miner_id": req.miner_id}


@router.post("/weights/snapshot/{model_id}")
async def create_weight_snapshot(
    model_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Create an on-chain version snapshot — Merkle root of all shard hashes.
    
    This anchors the current model state on the blockchain, creating an
    immutable audit trail. Any miner can verify its weights match.
    """
    check_rate_limit(user.user_id or "anon", "default")
    version = weight_registry.create_version_snapshot(model_id, param_server.global_step)

    # Anchor on-chain (fire-and-forget)
    asyncio.create_task(chain_bridge.record_aggregation_on_chain(
        global_step=param_server.global_step,
        layers_merged=version.num_shards,
        miners_contributed=len(weight_registry.miner_shards),
    ))

    return version.to_dict()


@router.get("/weights/registry-stats")
async def weight_registry_stats():
    """Weight shard registry statistics."""
    return weight_registry.get_stats()


@router.get("/weights/miner-shards/{miner_id}")
async def get_miner_weight_shards(miner_id: str):
    """Get all weight shards held by a specific miner."""
    shards = weight_registry.get_miner_shards(miner_id)
    return {
        "miner_id": miner_id,
        "shards": [s.to_dict() for s in shards],
        "count": len(shards),
    }


@router.get("/weights/find-sources/{model_id}")
async def find_weight_sources(
    model_id: str,
    layer_start: int = 0,
    layer_end: int = 0,
):
    """Find all miners that can serve specific layer weights."""
    sources = weight_registry.find_shard_sources(model_id, layer_start, layer_end)
    return {
        "model_id": model_id,
        "layer_range": f"{layer_start}-{layer_end}",
        "sources": [s.to_dict() for s in sources],
        "count": len(sources),
    }


# ============== Bandwidth Reporting ==============

class BandwidthReport(BaseModel):
    miner_id: str
    bandwidth_mbps: float
    peer_miner_id: str = ""
    measurement_method: str = "probe"  # probe, transfer, estimate


@router.post("/miners/report-bandwidth")
async def report_bandwidth(
    request: BandwidthReport,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Miner reports measured P2P bandwidth to its pipeline neighbors.
    
    This feeds into the bandwidth-aware redistribution scorer so that
    liquid redistribution prioritizes miners with the fastest connections.
    Called after a miner completes a P2P probe or weight transfer.
    """
    check_rate_limit(user.user_id or request.miner_id, "heartbeat")
    updated = shard_manager.update_miner_bandwidth(
        request.miner_id, request.bandwidth_mbps
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Miner not registered")
    miner = shard_manager.miners.get(request.miner_id)
    return {
        "status": "updated",
        "miner_id": request.miner_id,
        "bandwidth_mbps": miner.bandwidth_mbps if miner else 0,
        "measurement_method": request.measurement_method,
    }


# ============== Admin / Testing ==============

@router.post("/admin/simulate-disconnect/{miner_id}")
async def simulate_miner_disconnect(
    miner_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Simulate a miner disconnect for fire-drill testing.
    
    Triggers the full disconnect flow:
    1. Orphan the miner's weight shards in the registry
    2. Handle disconnect in the shard manager (mark pipeline DEGRADED)
    3. Return the state changes for observation
    """
    check_rate_limit(user.user_id or "anon", "default")

    # Step 1: Orphan weight shards
    orphaned = weight_registry.orphan_miner_shards(miner_id)

    # Step 2: Shard manager disconnect (marks pipeline DEGRADED)
    affected_group = shard_manager.handle_miner_disconnect(miner_id)

    return {
        "miner_id": miner_id,
        "orphaned_shards": orphaned,
        "affected_pipeline_group": affected_group,
        "registry_stats": weight_registry.get_stats(),
    }


# ============== Liquid Redistribution ==============

@router.post("/shards/auto-heal")
async def auto_heal_pipelines(
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Trigger automatic healing of all degraded pipelines.
    
    Scans for DEGRADED pipelines and reassigns orphaned stages
    to available miners. Like liquid flowing to fill gaps.
    """
    check_rate_limit(user.user_id or "anon", "default")
    healed = shard_manager.auto_heal_degraded_pipelines()
    return {
        "healed_pipelines": len(healed),
        "total_reassignments": sum(len(a) for a in healed.values()),
        "details": {
            gid: [a.to_dict() for a in assignments]
            for gid, assignments in healed.items()
        },
    }


@router.post("/weights/redistribute/{model_id}")
async def redistribute_weights(
    model_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Plan and execute weight redistribution for under-replicated shards.
    
    When miners disconnect, their shards become orphaned. This endpoint
    plans optimal redistribution to maintain the replication policy.
    """
    check_rate_limit(user.user_id or "anon", "default")

    available = [
        {
            "miner_id": m.miner_id,
            "gpu_vram_gb": m.gpu_vram_gb,
            "address": f"{m.location_region}",
            "bandwidth_mbps": m.bandwidth_mbps,
        }
        for m in shard_manager.get_available_miners()
    ]

    plan = weight_registry.plan_redistribution(model_id, available)
    return {
        "model_id": model_id,
        "transfers_planned": len(plan),
        "plan": plan,
    }


# ============== Unified Inference Router ==============

@router.post("/inference/route")
async def route_inference(
    model_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Get the inference routing plan for a model.
    
    Returns the ordered pipeline stages with miner addresses,
    so the caller can send input to stage 0 and collect output
    from the last stage. The model runs across all miners as one.
    """
    check_rate_limit(user.user_id or "anon", "default")

    # Find an active pipeline group for this model
    active_group = None
    for group in shard_manager.pipeline_groups.values():
        if group.model_id == model_id and group.status.value in ("ready", "training"):
            active_group = group
            break

    if not active_group:
        raise HTTPException(
            status_code=404,
            detail=f"No active pipeline for model {model_id}. "
            f"Available groups: {len(shard_manager.pipeline_groups)}"
        )

    stages = []
    for stage_idx in range(active_group.num_stages):
        assignment = active_group.stages.get(stage_idx)
        if not assignment:
            stages.append({"stage": stage_idx, "status": "missing"})
            continue

        miner = shard_manager.miners.get(assignment.miner_id)
        stages.append({
            "stage": stage_idx,
            "miner_id": assignment.miner_id,
            "layer_start": assignment.layer_start,
            "layer_end": assignment.layer_end,
            "has_embedding": assignment.has_embedding,
            "has_lm_head": assignment.has_lm_head,
            "miner_address": miner.location_region if miner else "unknown",
            "status": assignment.status,
        })

    return {
        "model_id": model_id,
        "pipeline_group_id": active_group.group_id,
        "num_stages": active_group.num_stages,
        "status": active_group.status.value,
        "stages": stages,
        "entry_point": stages[0] if stages else None,
        "exit_point": stages[-1] if stages else None,
    }


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
        "shard_manager": {
            "total_miners": len(shard_manager.miners),
            "pipeline_groups": len(shard_manager.pipeline_groups),
            "assigned_miners": len(shard_manager.miner_assignments),
        },
        "weight_registry": {
            "total_locations": len(weight_registry.locations),
            "unique_shards": len(weight_registry.shard_replicas),
            "miners_with_shards": len(weight_registry.miner_shards),
            "latest_version": weight_registry.latest_version,
        },
    }


# ============== P2P Discovery / WebRTC Signaling ==============

@router.websocket("/p2p/signaling/{miner_id}")
async def websocket_signaling(websocket: WebSocket, miner_id: str):
    """
    WebSocket endpoint for WebRTC signaling.
    
    Miners connect here to exchange ICE candidates, offers, and answers
    to establish direct P2P connections with their pipeline peers.
    """
    await websocket.accept()
    
    # Register miner for P2P discovery
    peer_id = p2p_discovery.register_miner(miner_id, websocket)
    
    try:
        logger.info(f"Miner {miner_id} connected for P2P signaling (peer {peer_id})")
        
        while True:
            # Receive signaling message
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                p2p_discovery.handle_signaling_message(miner_id, message)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from miner {miner_id}: {data}")
            except Exception as e:
                logger.error(f"Error handling signaling from {miner_id}: {e}")
                
    except WebSocketDisconnect:
        logger.info(f"Miner {miner_id} disconnected from P2P signaling")
    except Exception as e:
        logger.error(f"P2P signaling error for {miner_id}: {e}")
    finally:
        # Cleanup
        p2p_discovery.unregister_miner(miner_id)


class P2PPipelineAssignment(BaseModel):
    """Request to assign miners to a pipeline for P2P connections."""
    pipeline_group_id: str
    assignments: List[Dict[str, Any]]  # List of {miner_id, stage_index, ...}


@router.post("/p2p/assign-pipeline")
async def assign_pipeline_peers(
    request: P2PPipelineAssignment,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Assign miners to a pipeline and initiate P2P connections.
    
    Called by ShardManager after pipeline formation to establish
    WebRTC connections between adjacent pipeline stages.
    """
    check_rate_limit(user.user_id or "admin", "pipeline-assign")
    
    p2p_discovery.assign_pipeline_peers(
        request.pipeline_group_id,
        request.assignments
    )
    
    return {
        "status": "assigned",
        "pipeline_group_id": request.pipeline_group_id,
        "miners_assigned": len(request.assignments),
    }


@router.get("/p2p/peer-info/{miner_id}")
async def get_peer_info(
    miner_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get P2P connection status for a miner."""
    check_rate_limit(user.user_id or "admin", "peer-info")
    
    info = p2p_discovery.get_peer_info(miner_id)
    if not info:
        raise HTTPException(status_code=404, detail="Miner not registered for P2P")
    
    return info


@router.get("/p2p/pipeline-status/{pipeline_group_id}")
async def get_pipeline_p2p_status(
    pipeline_group_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get P2P connection status for all peers in a pipeline."""
    check_rate_limit(user.user_id or "admin", "pipeline-status")
    
    status = p2p_discovery.get_pipeline_status(pipeline_group_id)
    if not status["peers"]:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    return status


# ============== WebRTC Weight Transfer Endpoints ==============

class WebRTCUpdateRequest(BaseModel):
    """Request to update miner's WebRTC capabilities."""
    webrtc_peer_id: str
    bandwidth_mbps: float = 0.0


@router.post("/webrtc/update-capabilities")
async def update_webrtc_capabilities(
    request: WebRTCUpdateRequest,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Update a miner's WebRTC capabilities for P2P weight transfers.
    
    Called by miners when they establish WebRTC connections with pipeline peers.
    This enables them to serve as P2P weight sources for other miners.
    """
    check_rate_limit(user.user_id or "admin", "webrtc-update")
    
    # Extract miner_id from authenticated user
    miner_id = user.user_id
    if not miner_id:
        raise HTTPException(status_code=400, detail="Miner ID required")
    
    # Update WebRTC info in weight registry
    weight_registry.update_miner_webrtc_info(
        miner_id=miner_id,
        webrtc_peer_id=request.webrtc_peer_id,
        bandwidth_mbps=request.bandwidth_mbps,
    )
    
    return {
        "status": "updated",
        "miner_id": miner_id,
        "webrtc_peer_id": request.webrtc_peer_id,
        "bandwidth_mbps": request.bandwidth_mbps,
    }


@router.get("/webrtc/transfer-sources/{model_id}/{layer_start}/{layer_end}")
async def get_webrtc_transfer_sources(
    model_id: str,
    layer_start: int,
    layer_end: int,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Get WebRTC-enabled sources for a weight shard.
    
    Returns peers that can serve the requested shard via WebRTC DataChannel,
    prioritized by bandwidth and availability. Falls back to HTTP sources
    if no WebRTC peers are available.
    """
    check_rate_limit(user.user_id or "admin", "transfer-sources")
    
    # Find sources (automatically prioritizes WebRTC)
    sources = weight_registry.find_shard_sources(model_id, layer_start, layer_end)
    
    # Format response
    webrtc_sources = []
    http_sources = []
    
    for source in sources:
        source_info = {
            "miner_id": source.miner_id,
            "location_id": source.location_id,
            "version": source.version,
            "weight_hash": source.weight_hash,
            "size_bytes": source.size_bytes,
            "priority": source.priority.value,
        }
        
        if source.can_serve_p2p:
            source_info.update({
                "webrtc_peer_id": source.webrtc_peer_id,
                "bandwidth_mbps": source.webrtc_bandwidth,
                "transfer_method": "webrtc",
            })
            webrtc_sources.append(source_info)
        else:
            source_info.update({
                "miner_address": source.miner_address,
                "transfer_method": "http",
            })
            http_sources.append(source_info)
    
    return {
        "model_id": model_id,
        "layer_range": f"L{layer_start}-{layer_end}",
        "webrtc_sources": webrtc_sources,
        "http_sources": http_sources,
        "total_sources": len(sources),
        "has_webrtc": len(webrtc_sources) > 0,
    }


class WebRTCTransferRequest(BaseModel):
    """Request to initiate WebRTC weight transfer."""
    source_peer_id: str
    target_miner_id: str
    model_id: str
    layer_start: int
    layer_end: int
    transfer_id: str = None


@router.post("/webrtc/initiate-transfer")
async def initiate_webrtc_transfer(
    request: WebRTCTransferRequest,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Initiate a WebRTC-based weight transfer between miners.
    
    This endpoint coordinates the transfer by:
    1. Verifying both peers have WebRTC connections
    2. Sending transfer request via signaling channel
    3. Monitoring transfer progress
    
    The actual data transfer happens directly between miners via WebRTC DataChannel.
    """
    check_rate_limit(user.user_id or "admin", "webrtc-transfer")
    
    # Generate transfer ID if not provided
    if not request.transfer_id:
        request.transfer_id = f"transfer-{uuid4().hex[:12]}"
    
    # Get source peer info
    source_info = p2p_discovery.get_peer_info(request.source_peer_id.split("-")[-1])
    if not source_info or not source_info.get("has_datachannel"):
        raise HTTPException(
            status_code=400,
            detail="Source peer does not have WebRTC DataChannel available"
        )
    
    # Send transfer request via P2P signaling
    transfer_message = {
        "type": "weight-transfer-request",
        "transfer_id": request.transfer_id,
        "target_miner_id": request.target_miner_id,
        "model_id": request.model_id,
        "layer_start": request.layer_start,
        "layer_end": request.layer_end,
    }
    
    # This would be sent via the P2P discovery signaling channel
    # For now, we'll just log it
    logger.info(f"WebRTC transfer initiated: {request.transfer_id}")
    
    return {
        "status": "initiated",
        "transfer_id": request.transfer_id,
        "source_peer_id": request.source_peer_id,
        "target_miner_id": request.target_miner_id,
        "shard": f"{request.model_id}:L{request.layer_start}-{request.layer_end}",
        "transfer_method": "webrtc",
    }


# ============== Slashing and Reputation Endpoints ==============

class WeightVerificationRequest(BaseModel):
    """Request to verify received weights and report violations."""
    source_miner_id: str
    model_id: str
    layer_start: int
    layer_end: int
    weight_data: str  # Base64 encoded weight bytes
    expected_hash: str
    transfer_id: str = None


@router.post("/slashing/verify-weights")
async def verify_received_weights(
    request: WeightVerificationRequest,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Verify weights received from a peer and report violations.
    
    Miners call this when they receive weights via WebRTC DataChannel
    to ensure integrity and report any corrupted weights.
    """
    check_rate_limit(user.user_id or "admin", "verify-weights")
    
    # Decode weight data
    import base64
    try:
        weight_bytes = base64.b64decode(request.weight_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid weight data encoding: {e}")
    
    # Verify weights and report violations if needed
    is_valid, error = slashing_engine.verify_weight_transfer(
        miner_id=request.source_miner_id,
        model_id=request.model_id,
        layer_start=request.layer_start,
        layer_end=request.layer_end,
        weight_data=weight_bytes,
        expected_hash=request.expected_hash,
        requester_id=user.user_id or "anonymous",
    )
    
    return {
        "valid": is_valid,
        "error": error if not is_valid else None,
        "source_miner_id": request.source_miner_id,
        "transfer_id": request.transfer_id,
    }


class ViolationReportRequest(BaseModel):
    """Request to report a violation."""
    violator_miner_id: str
    violation_type: str
    evidence: Dict[str, Any]
    description: str = ""


@router.post("/slashing/report-violation")
async def report_violation(
    request: ViolationReportRequest,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Report a violation by another miner.
    
    Can be used for manual reporting of issues not caught by automatic verification.
    """
    check_rate_limit(user.user_id or "admin", "report-violation")
    
    # Validate violation type
    try:
        violation_type = ViolationType(request.violation_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid violation type: {request.violation_type}"
        )
    
    record_id = slashing_engine.report_violation(
        miner_id=request.violator_miner_id,
        violation_type=violation_type,
        evidence=request.evidence,
        reported_by=user.user_id or "anonymous",
    )
    
    return {
        "record_id": record_id,
        "status": "reported",
        "violation_type": request.violation_type,
        "violator_miner_id": request.violator_miner_id,
    }


@router.get("/slashing/miner-status/{miner_id}")
async def get_miner_slashing_status(
    miner_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get a miner's reputation and slashing status."""
    check_rate_limit(user.user_id or "admin", "miner-status")
    
    status = slashing_engine.get_miner_status(miner_id)
    return status


@router.get("/slashing/network-stats")
async def get_slashing_network_stats(
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get overall network slashing statistics."""
    check_rate_limit(user.user_id or "admin", "network-stats")
    
    stats = slashing_engine.get_network_stats()
    return stats


@router.get("/slashing/violations")
async def get_recent_violations(
    limit: int = 50,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get recent violation records for monitoring."""
    check_rate_limit(user.user_id or "admin", "violations")
    
    violations = sorted(
        slashing_engine.violations,
        key=lambda v: v.reported_at,
        reverse=True
    )[:limit]
    
    return {
        "violations": [
            {
                "record_id": v.record_id,
                "miner_id": v.miner_id,
                "violation_type": v.violation_type.value,
                "severity": v.severity.value,
                "reported_at": v.reported_at,
                "verified": v.verified,
                "slashed": v.slashed,
                "slash_amount": v.slash_amount,
                "suspension_end": v.suspension_end,
            }
            for v in violations
        ],
        "total": len(slashing_engine.violations),
    }


# ============== Network Dashboard Endpoints ==============

@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the network dashboard HTML page."""
    return DASHBOARD_HTML


@router.websocket("/dashboard/ws")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates."""
    client_id = str(uuid.uuid4())
    await network_dashboard.register_client(websocket, client_id)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        await network_dashboard.unregister_client(client_id)
    except Exception as e:
        logger.error(f"Dashboard WebSocket error: {e}")
        await network_dashboard.unregister_client(client_id)


@router.get("/dashboard/data")
async def get_dashboard_data(
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get current dashboard data (REST API endpoint)."""
    check_rate_limit(user.user_id or "admin", "dashboard-data")
    return await network_dashboard.get_dashboard_data()


@router.get("/dashboard/nodes")
async def get_network_nodes(
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get network nodes information."""
    check_rate_limit(user.user_id or "admin", "dashboard-nodes")
    return {"nodes": await network_dashboard._get_network_nodes()}


@router.get("/dashboard/links")
async def get_network_links(
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get network links/connections information."""
    check_rate_limit(user.user_id or "admin", "dashboard-links")
    return {"links": await network_dashboard._get_network_links()}


@router.get("/dashboard/metrics")
async def get_network_metrics(
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get network health metrics."""
    check_rate_limit(user.user_id or "admin", "dashboard-metrics")
    return await network_dashboard._get_network_metrics()


@router.get("/dashboard/wallets")
async def get_wallet_info(
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get wallet and staking information."""
    check_rate_limit(user.user_id or "admin", "dashboard-wallets")
    return {"wallets": await network_dashboard._get_wallet_info()}


@router.get("/dashboard/topology")
async def get_network_topology(
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get network topology for visualization."""
    check_rate_limit(user.user_id or "admin", "dashboard-topology")
    return await network_dashboard._get_network_topology()


@router.get("/dashboard/violations")
async def get_dashboard_violations(
    limit: int = 50,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get recent violations for dashboard display."""
    check_rate_limit(user.user_id or "admin", "dashboard-violations")
    return {"violations": await network_dashboard._get_recent_violations()}


@router.get("/dashboard/health")
async def get_dashboard_health(
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get overall network health score."""
    check_rate_limit(user.user_id or "admin", "dashboard-health")
    metrics = await network_dashboard._get_network_metrics()
    return {
        "health_score": metrics["network_health_score"],
        "status": "healthy" if metrics["network_health_score"] >= 80 else 
                "degraded" if metrics["network_health_score"] >= 50 else "critical",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ============== Wallet Service Endpoints ==============

class CreateWalletRequest(BaseModel):
    """Request to create a new wallet."""
    miner_id: str


class StakeRequest(BaseModel):
    """Request to stake tokens."""
    token_type: str
    amount: float
    lock_period_days: int = 30


class WithdrawStakeRequest(BaseModel):
    """Request to withdraw stake."""
    stake_id: str


@router.post("/wallet/create")
async def create_wallet(
    request: CreateWalletRequest,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Create a new wallet for a miner."""
    check_rate_limit(user.user_id or "admin", "create-wallet")
    
    wallet = await wallet_service.create_wallet(request.miner_id)
    return {
        "wallet_address": wallet.wallet_address,
        "miner_id": wallet.miner_id,
        "created_at": wallet.created_at,
    }


@router.get("/wallet/{miner_id}")
async def get_wallet(
    miner_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get wallet information for a miner."""
    check_rate_limit(user.user_id or "admin", "get-wallet")
    
    wallet = await wallet_service.get_wallet(miner_id)
    if not wallet:
        raise HTTPException(status_code=404, detail="Wallet not found")
    
    stats = await wallet_service.get_wallet_stats(wallet.wallet_address)
    return stats


@router.get("/wallet/{wallet_address}/balance")
async def get_wallet_balance(
    wallet_address: str,
    token_type: Optional[str] = None,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get wallet balance(s)."""
    check_rate_limit(user.user_id or "admin", "get-balance")
    
    if token_type:
        try:
            token_enum = TokenType(token_type)
            balance = await wallet_service.get_balance(wallet_address, token_enum)
            return {"token_type": token_type, "balance": balance}
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid token type: {token_type}")
    else:
        balances = await wallet_service.get_balances(wallet_address)
        return {"balances": [asdict(b) for b in balances]}


@router.post("/wallet/stake")
async def deposit_stake(
    request: StakeRequest,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Deposit tokens as stake."""
    check_rate_limit(user.user_id or "admin", "deposit-stake")
    
    # Get wallet for user
    wallet = await wallet_service.get_wallet(user.user_id or "anonymous")
    if not wallet:
        raise HTTPException(status_code=404, detail="Wallet not found")
    
    try:
        token_enum = TokenType(request.token_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid token type: {request.token_type}")
    
    stake = await wallet_service.deposit_stake(
        wallet_address=wallet.wallet_address,
        token_type=token_enum,
        amount=request.amount,
        lock_period_days=request.lock_period_days,
    )
    
    return {
        "stake_id": stake.stake_id,
        "token_type": stake.token_type.value,
        "amount": stake.amount,
        "status": stake.status.value,
        "locked_until": stake.locked_until,
    }


@router.post("/wallet/stake/withdraw")
async def withdraw_stake(
    request: WithdrawStakeRequest,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Withdraw stake after lock period."""
    check_rate_limit(user.user_id or "admin", "withdraw-stake")
    
    success = await wallet_service.withdraw_stake(request.stake_id)
    return {"success": success, "stake_id": request.stake_id}


@router.get("/wallet/stakes/{wallet_address}")
async def get_wallet_stakes(
    wallet_address: str,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get all stakes for a wallet."""
    check_rate_limit(user.user_id or "admin", "get-stakes")
    
    stakes = [s for s in wallet_service.stakes.values() if s.wallet_address == wallet_address]
    return {"stakes": [asdict(s) for s in stakes]}


@router.get("/wallet/rewards/{wallet_address}")
async def get_wallet_rewards(
    wallet_address: str,
    limit: int = 50,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get reward history for a wallet."""
    check_rate_limit(user.user_id or "admin", "get-rewards")
    
    rewards = [r for r in wallet_service.rewards if r.wallet_address == wallet_address]
    rewards.sort(key=lambda r: r.distributed_at, reverse=True)
    
    return {
        "rewards": [asdict(r) for r in rewards[:limit]],
        "total": len(rewards),
    }


@router.get("/wallet/transactions/{wallet_address}")
async def get_wallet_transactions(
    wallet_address: str,
    limit: int = 50,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get transaction history for a wallet."""
    check_rate_limit(user.user_id or "admin", "get-transactions")
    
    transactions = [t for t in wallet_service.transactions if t.wallet_address == wallet_address]
    transactions.sort(key=lambda t: t.timestamp, reverse=True)
    
    return {
        "transactions": [asdict(t) for t in transactions[:limit]],
        "total": len(transactions),
    }


@router.get("/wallet/network-stats")
async def get_wallet_network_stats(
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get network-wide wallet statistics."""
    check_rate_limit(user.user_id or "admin", "wallet-stats")
    
    return await wallet_service.get_network_stats()


@router.post("/wallet/calculate-rewards")
async def calculate_rewards(
    miner_id: str,
    hours_trained: float = 0.0,
    hours_seeded: float = 0.0,
    bandwidth_mbps: float = 0.0,
    performance_score: float = 1.0,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Calculate rewards for mining activities."""
    check_rate_limit(user.user_id or "admin", "calculate-rewards")
    
    training_rewards = await wallet_service.calculate_training_rewards(
        miner_id, hours_trained, performance_score
    )
    seeding_rewards = await wallet_service.calculate_seeding_rewards(
        miner_id, hours_seeded, bandwidth_mbps
    )
    
    return {
        "miner_id": miner_id,
        "training_rewards": training_rewards,
        "seeding_rewards": seeding_rewards,
        "total_rewards": training_rewards + seeding_rewards,
    }
