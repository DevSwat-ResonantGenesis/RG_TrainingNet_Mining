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
