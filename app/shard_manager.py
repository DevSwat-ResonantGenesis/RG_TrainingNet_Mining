"""
Shard Manager — Decentralized Model Sharding for Unlimited Scale
=================================================================

Assigns model layer ranges to miners, forms pipeline groups, and manages
the shard topology for the entire training network.

Designed for infinite scale:
  - 1 miner training 1B params (Phase 2 — now)
  - 500 miners training 405B MoE (Phase 6)
  - 1 billion miners training 10^21 params (future)

Key concepts:
  - ShardAssignment: which layers a single miner is responsible for
  - PipelineGroup: a set of miners forming one complete forward/backward pipeline
  - SuperGroup: a group of pipeline groups coordinated by a regional aggregator
  - ShardTree: hierarchical tree of super-groups for billion-miner scale

No hard-coded limits. All structures scale dynamically.
"""

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger("rg-mining.shard-manager")


# ══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════

class PipelineStatus(str, Enum):
    FORMING = "forming"          # Waiting for miners to fill all stages
    LOADING = "loading"          # All stages assigned, miners downloading shards
    READY = "ready"              # All miners report shards loaded
    TRAINING = "training"        # Active training in progress
    DEGRADED = "degraded"        # One or more stages lost a miner
    PAUSED = "paused"            # Manually paused
    DISBANDED = "disbanded"      # Group no longer active


class ShardTier(str, Enum):
    """Hierarchy level in the shard tree."""
    MINER = "miner"              # Leaf: single miner holding a layer range
    PIPELINE = "pipeline"        # A complete forward/backward pipeline
    SUPER_GROUP = "super_group"  # Regional cluster of pipelines
    ZONE = "zone"                # Geographic zone (continent-scale)
    ROOT = "root"                # Global root coordinator


@dataclass
class MinerCapability:
    """Hardware capabilities reported by a miner."""
    miner_id: str
    gpu_model: str = ""
    gpu_vram_gb: float = 0.0
    system_ram_gb: float = 0.0
    cpu_cores: int = 0
    bandwidth_mbps: float = 0.0
    storage_available_gb: float = 0.0
    location_region: str = "unknown"
    supported_dtypes: List[str] = field(default_factory=lambda: ["fp32", "fp16"])
    max_layers_capacity: int = 0       # Auto-calculated: how many layers this miner can hold
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_heartbeat: str = ""
    is_available: bool = True

    def estimate_layer_capacity(self, param_per_layer: int, dtype_bytes: int = 2) -> int:
        """Estimate how many transformer layers this miner can hold in VRAM."""
        if self.gpu_vram_gb <= 0:
            return 0
        usable_vram = self.gpu_vram_gb * 0.80 * 1e9  # 80% of VRAM (reserve for activations + optimizer)
        bytes_per_layer = param_per_layer * dtype_bytes * 3  # params + grads + optimizer (Adam: 2x)
        capacity = int(usable_vram / max(bytes_per_layer, 1))
        self.max_layers_capacity = max(capacity, 1)
        return self.max_layers_capacity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "miner_id": self.miner_id,
            "gpu_model": self.gpu_model,
            "gpu_vram_gb": self.gpu_vram_gb,
            "system_ram_gb": self.system_ram_gb,
            "cpu_cores": self.cpu_cores,
            "bandwidth_mbps": self.bandwidth_mbps,
            "storage_available_gb": self.storage_available_gb,
            "location_region": self.location_region,
            "supported_dtypes": self.supported_dtypes,
            "max_layers_capacity": self.max_layers_capacity,
            "is_available": self.is_available,
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
        }


@dataclass
class ShardAssignment:
    """Describes which layers a miner is responsible for."""
    assignment_id: str = field(default_factory=lambda: str(uuid4()))
    miner_id: str = ""
    pipeline_group_id: str = ""
    stage_index: int = 0             # 0, 1, 2, ... N-1 within the pipeline
    num_stages: int = 1              # Total stages in this pipeline
    layer_start: int = 0             # First layer index (inclusive)
    layer_end: int = 0               # Last layer index (exclusive)
    has_embedding: bool = False      # True for stage 0
    has_lm_head: bool = False        # True for last stage
    upstream_miner_id: Optional[str] = None    # Who sends us activations
    downstream_miner_id: Optional[str] = None  # Who we send activations to
    shard_size_bytes: int = 0        # Expected memory footprint
    shard_size_params: int = 0       # Number of parameters in this shard
    status: str = "assigned"         # assigned, loading, ready, training, failed
    assigned_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def num_layers(self) -> int:
        return self.layer_end - self.layer_start

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignment_id": self.assignment_id,
            "miner_id": self.miner_id,
            "pipeline_group_id": self.pipeline_group_id,
            "stage_index": self.stage_index,
            "num_stages": self.num_stages,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "num_layers": self.num_layers,
            "has_embedding": self.has_embedding,
            "has_lm_head": self.has_lm_head,
            "upstream_miner_id": self.upstream_miner_id,
            "downstream_miner_id": self.downstream_miner_id,
            "shard_size_bytes": self.shard_size_bytes,
            "shard_size_params": self.shard_size_params,
            "status": self.status,
            "assigned_at": self.assigned_at,
        }


@dataclass
class PipelineGroup:
    """A group of miners forming one complete forward/backward pipeline."""
    group_id: str = field(default_factory=lambda: f"pg-{uuid4().hex[:12]}")
    model_id: str = ""
    num_stages: int = 1
    stages: Dict[int, ShardAssignment] = field(default_factory=dict)
    status: PipelineStatus = PipelineStatus.FORMING
    super_group_id: Optional[str] = None  # Parent in hierarchy
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    training_step: int = 0
    total_samples: int = 0

    @property
    def is_complete(self) -> bool:
        """All stages have assigned miners."""
        return len(self.stages) == self.num_stages

    @property
    def is_ready(self) -> bool:
        """All stages report shard loaded."""
        return self.is_complete and all(s.status == "ready" for s in self.stages.values())

    @property
    def miner_ids(self) -> List[str]:
        return [s.miner_id for s in self.stages.values()]

    def get_stage_for_miner(self, miner_id: str) -> Optional[ShardAssignment]:
        for s in self.stages.values():
            if s.miner_id == miner_id:
                return s
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "model_id": self.model_id,
            "num_stages": self.num_stages,
            "status": self.status.value,
            "is_complete": self.is_complete,
            "is_ready": self.is_ready,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "miner_ids": self.miner_ids,
            "super_group_id": self.super_group_id,
            "created_at": self.created_at,
            "training_step": self.training_step,
            "total_samples": self.total_samples,
        }


@dataclass
class SuperGroup:
    """
    A cluster of pipeline groups that share a regional aggregator.
    For billion-miner scale, super-groups form a tree (SuperGroup → Zone → Root).
    
    Hierarchy:
      Root
       ├─ Zone (us-west)
       │   ├─ SuperGroup A (100 pipelines)
       │   │   ├─ PipelineGroup 1
       │   │   ├─ PipelineGroup 2
       │   │   └─ ...
       │   └─ SuperGroup B (100 pipelines)
       └─ Zone (eu-central)
           └─ ...
    """
    super_group_id: str = field(default_factory=lambda: f"sg-{uuid4().hex[:12]}")
    tier: ShardTier = ShardTier.SUPER_GROUP
    region: str = "global"
    parent_id: Optional[str] = None       # Parent super-group or zone
    children_ids: List[str] = field(default_factory=list)  # Pipeline group IDs or child super-group IDs
    aggregator_miner_id: Optional[str] = None  # Miner acting as regional aggregator
    max_children: int = 256               # Max pipelines per super-group (scales tree width)

    @property
    def num_children(self) -> int:
        return len(self.children_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "super_group_id": self.super_group_id,
            "tier": self.tier.value,
            "region": self.region,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "num_children": self.num_children,
            "aggregator_miner_id": self.aggregator_miner_id,
            "max_children": self.max_children,
        }


# ══════════════════════════════════════════════════════════════
# SHARD MANAGER
# ══════════════════════════════════════════════════════════════

class ShardManager:
    """
    Manages model sharding across the decentralized miner network.
    
    Scales from 1 miner (no sharding) to 1 billion miners (hierarchical tree).
    No hard-coded limits on any dimension.
    
    Scaling strategy:
      1-10 miners:       No sharding, full model per miner (Phase 2-4)
      10-1000 miners:    Flat pipeline groups (Phase 5)
      1K-1M miners:      Super-groups with regional aggregators (Phase 5-6)
      1M-1B miners:      Hierarchical tree (Zone → SuperGroup → Pipeline) (Future)
    """

    # Minimum layers per pipeline stage (don't split too thin)
    MIN_LAYERS_PER_STAGE = 2

    # Maximum pipeline stages (beyond this, use more pipeline groups instead)
    MAX_STAGES_PER_PIPELINE = 128

    # Max pipelines per super-group before creating a new level
    MAX_PIPELINES_PER_SUPER_GROUP = 256

    def __init__(self):
        self.miners: Dict[str, MinerCapability] = {}
        self.pipeline_groups: Dict[str, PipelineGroup] = {}
        self.super_groups: Dict[str, SuperGroup] = {}
        self.miner_assignments: Dict[str, ShardAssignment] = {}  # miner_id → assignment
        self._model_configs: Dict[str, Dict[str, Any]] = {}

    # ── Miner Registration ──

    def register_miner(self, capability: MinerCapability) -> MinerCapability:
        """Register a miner with its hardware capabilities."""
        capability.last_heartbeat = datetime.now(timezone.utc).isoformat()
        self.miners[capability.miner_id] = capability
        logger.info(
            f"Registered miner {capability.miner_id} "
            f"(GPU: {capability.gpu_model}, VRAM: {capability.gpu_vram_gb:.1f}GB, "
            f"Region: {capability.location_region})"
        )
        return capability

    def heartbeat(self, miner_id: str) -> bool:
        """Update miner heartbeat. Returns False if miner unknown."""
        miner = self.miners.get(miner_id)
        if not miner:
            return False
        miner.last_heartbeat = datetime.now(timezone.utc).isoformat()
        return True

    def get_available_miners(self, min_vram_gb: float = 0) -> List[MinerCapability]:
        """Get all available miners, optionally filtered by VRAM."""
        return [
            m for m in self.miners.values()
            if m.is_available and m.miner_id not in self.miner_assignments
            and m.gpu_vram_gb >= min_vram_gb
        ]

    # ── Core Sharding Logic ──

    def compute_optimal_stages(
        self,
        model_config: Dict[str, Any],
        available_miners: List[MinerCapability],
    ) -> int:
        """
        Determine optimal number of pipeline stages.
        
        Algorithm:
        1. Calculate model size in memory
        2. Find median miner VRAM
        3. Compute minimum stages so each stage fits in VRAM
        4. Clamp to [1, min(MAX_STAGES, num_layers // MIN_LAYERS_PER_STAGE)]
        5. Prefer power-of-2 stages for even layer distribution
        """
        num_layers = model_config["num_layers"]
        num_params = model_config["num_parameters"]
        dtype_bytes = 2  # fp16

        # Model memory: params + gradients + optimizer state (Adam: 2x params)
        model_memory_bytes = num_params * dtype_bytes * 4  # params + grads + 2x optimizer
        
        if not available_miners:
            return 1

        # Median VRAM
        vrams = sorted([m.gpu_vram_gb for m in available_miners])
        median_vram_gb = vrams[len(vrams) // 2]
        usable_vram_bytes = median_vram_gb * 0.75 * 1e9  # 75% usable

        if usable_vram_bytes <= 0:
            return 1

        # Minimum stages to fit model
        min_stages = max(1, math.ceil(model_memory_bytes / usable_vram_bytes))

        # Max stages based on layer count
        max_stages = min(
            self.MAX_STAGES_PER_PIPELINE,
            num_layers // self.MIN_LAYERS_PER_STAGE,
        )

        stages = max(min_stages, 1)
        stages = min(stages, max_stages)

        # Round up to nearest power of 2 for clean layer division
        if stages > 1:
            stages = 2 ** math.ceil(math.log2(stages))
            stages = min(stages, max_stages)

        # Ensure layers divide evenly-ish
        while num_layers % stages != 0 and stages > 1:
            stages -= 1

        return max(stages, 1)

    def compute_layer_assignment(
        self,
        num_layers: int,
        num_stages: int,
    ) -> List[Tuple[int, int]]:
        """
        Compute layer ranges for each stage.
        Returns list of (layer_start, layer_end) tuples.
        
        Handles any number of layers and stages — no upper limit.
        For 10^6 layers with 10^3 stages, each stage gets ~1000 layers.
        """
        if num_stages <= 0:
            return [(0, num_layers)]

        base = num_layers // num_stages
        remainder = num_layers % num_stages

        ranges = []
        current = 0
        for i in range(num_stages):
            # Distribute remainder evenly across first `remainder` stages
            count = base + (1 if i < remainder else 0)
            ranges.append((current, current + count))
            current += count

        return ranges

    def compute_shard_size(
        self,
        model_config: Dict[str, Any],
        layer_start: int,
        layer_end: int,
        has_embedding: bool = False,
        has_lm_head: bool = False,
    ) -> Tuple[int, int]:
        """
        Compute memory footprint and param count for a shard.
        Returns (size_bytes, num_params).
        """
        hidden = model_config["hidden_size"]
        inter = model_config.get("intermediate_size", hidden * 4)
        vocab = model_config["vocab_size"]
        num_heads = model_config["num_heads"]
        num_kv = model_config.get("num_kv_heads", num_heads // 4)
        head_dim = hidden // num_heads

        # Per-layer params: attention (Q, K, V, O) + FFN (gate, up, down) + 2x layernorm
        attn_params = hidden * (num_heads * head_dim + 2 * num_kv * head_dim + hidden)
        ffn_params = hidden * inter * 3  # gate + up + down (SwiGLU)
        norm_params = hidden * 2  # 2 layernorms
        per_layer = attn_params + ffn_params + norm_params

        num_layers = layer_end - layer_start
        total_params = per_layer * num_layers

        if has_embedding:
            total_params += vocab * hidden  # Embedding matrix
        if has_lm_head:
            total_params += vocab * hidden  # LM head (may be tied to embedding)
            total_params += hidden          # Final layernorm

        dtype_bytes = 2  # fp16
        size_bytes = total_params * dtype_bytes

        return size_bytes, total_params

    # ── Pipeline Group Formation ──

    def form_pipeline_groups(
        self,
        model_id: str,
        model_config: Dict[str, Any],
        target_redundancy: int = 2,
    ) -> List[PipelineGroup]:
        """
        Create pipeline groups from available miners.
        
        Algorithm:
        1. Compute optimal stages
        2. Sort available miners by VRAM (largest first)
        3. Fill pipeline groups stage by stage
        4. Create at least `target_redundancy` groups (fault tolerance)
        5. If not enough miners for full redundancy, create partial groups
        
        Scales to any number of miners:
        - 4 miners, 4 stages → 1 pipeline group
        - 40 miners, 4 stages → 10 pipeline groups (10x redundancy)
        - 1M miners, 1000 stages → 1000 pipeline groups, organized in super-groups
        """
        self._model_configs[model_id] = model_config
        available = self.get_available_miners()

        if not available:
            logger.warning("No available miners for pipeline formation")
            return []

        num_stages = self.compute_optimal_stages(model_config, available)
        num_layers = model_config["num_layers"]
        layer_ranges = self.compute_layer_assignment(num_layers, num_stages)

        # Sort by VRAM descending — strongest miners first
        available.sort(key=lambda m: m.gpu_vram_gb, reverse=True)

        # How many complete groups can we form?
        max_groups = len(available) // num_stages
        num_groups = max(1, min(max_groups, max(target_redundancy, len(available) // num_stages)))

        created_groups = []

        for g in range(num_groups):
            group = PipelineGroup(
                model_id=model_id,
                num_stages=num_stages,
            )

            for stage_idx in range(num_stages):
                miner_idx = g * num_stages + stage_idx
                if miner_idx >= len(available):
                    break  # Ran out of miners

                miner = available[miner_idx]
                layer_start, layer_end = layer_ranges[stage_idx]

                size_bytes, num_params = self.compute_shard_size(
                    model_config, layer_start, layer_end,
                    has_embedding=(stage_idx == 0),
                    has_lm_head=(stage_idx == num_stages - 1),
                )

                assignment = ShardAssignment(
                    miner_id=miner.miner_id,
                    pipeline_group_id=group.group_id,
                    stage_index=stage_idx,
                    num_stages=num_stages,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    has_embedding=(stage_idx == 0),
                    has_lm_head=(stage_idx == num_stages - 1),
                    shard_size_bytes=size_bytes,
                    shard_size_params=num_params,
                )

                # Link upstream/downstream
                if stage_idx > 0:
                    prev_stage = group.stages.get(stage_idx - 1)
                    if prev_stage:
                        assignment.upstream_miner_id = prev_stage.miner_id
                        prev_stage.downstream_miner_id = miner.miner_id

                group.stages[stage_idx] = assignment
                self.miner_assignments[miner.miner_id] = assignment
                miner.is_available = False

            if group.is_complete:
                group.status = PipelineStatus.LOADING
            self.pipeline_groups[group.group_id] = group
            created_groups.append(group)

            logger.info(
                f"Pipeline group {group.group_id}: {num_stages} stages, "
                f"miners={group.miner_ids}, status={group.status.value}"
            )

        # Organize into super-groups if we have many pipeline groups
        if len(created_groups) > self.MAX_PIPELINES_PER_SUPER_GROUP:
            self._organize_into_super_groups(created_groups)

        return created_groups

    def _organize_into_super_groups(self, groups: List[PipelineGroup]):
        """
        Organize pipeline groups into a hierarchical tree of super-groups.
        
        For 1M pipelines:
          - 256 pipelines per super-group → 3906 super-groups
          - 256 super-groups per zone → 16 zones
          - 1 root
          
        For 1B pipelines:
          - 256 per SG → 3.9M SGs
          - 256 per zone → 15.3K zones
          - 256 per region → 60 regions
          - 1 root
        
        Tree depth = ceil(log_{MAX_CHILDREN}(num_groups))
        Always O(log N) depth — handles any scale.
        """
        # Group by region for locality
        by_region: Dict[str, List[PipelineGroup]] = {}
        for g in groups:
            # Use first miner's region as group region
            region = "global"
            if g.stages:
                first_miner = self.miners.get(g.stages[0].miner_id)
                if first_miner:
                    region = first_miner.location_region or "global"
            by_region.setdefault(region, []).append(g)

        # Create super-groups per region
        for region, region_groups in by_region.items():
            self._build_tree_level(
                items=[g.group_id for g in region_groups],
                region=region,
                tier=ShardTier.SUPER_GROUP,
                parent_id=None,
            )

    def _build_tree_level(
        self,
        items: List[str],
        region: str,
        tier: ShardTier,
        parent_id: Optional[str],
    ) -> Optional[str]:
        """
        Recursively build tree hierarchy. Returns root ID of this subtree.
        
        If items fit in one node → create single node.
        If too many → split into chunks, create parent, recurse.
        """
        if len(items) <= self.MAX_PIPELINES_PER_SUPER_GROUP:
            sg = SuperGroup(
                tier=tier,
                region=region,
                parent_id=parent_id,
                children_ids=items,
            )
            self.super_groups[sg.super_group_id] = sg

            # Link pipeline groups to their super-group
            for item_id in items:
                pg = self.pipeline_groups.get(item_id)
                if pg:
                    pg.super_group_id = sg.super_group_id

            logger.info(f"SuperGroup {sg.super_group_id} ({tier.value}): {len(items)} children in {region}")
            return sg.super_group_id

        # Too many — split into chunks and recurse
        chunk_size = self.MAX_PIPELINES_PER_SUPER_GROUP
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

        # Determine next tier up
        next_tier = {
            ShardTier.SUPER_GROUP: ShardTier.ZONE,
            ShardTier.ZONE: ShardTier.ROOT,
            ShardTier.ROOT: ShardTier.ROOT,
        }.get(tier, ShardTier.ROOT)

        # Create child nodes first
        child_ids = []
        for chunk in chunks:
            child_id = self._build_tree_level(chunk, region, tier, parent_id=None)
            if child_id:
                child_ids.append(child_id)

        # Create parent node
        if len(child_ids) > self.MAX_PIPELINES_PER_SUPER_GROUP:
            # Need another level
            return self._build_tree_level(child_ids, region, next_tier, parent_id)

        parent = SuperGroup(
            tier=next_tier,
            region=region,
            parent_id=parent_id,
            children_ids=child_ids,
        )
        self.super_groups[parent.super_group_id] = parent

        # Update children's parent
        for cid in child_ids:
            child = self.super_groups.get(cid)
            if child:
                child.parent_id = parent.super_group_id

        logger.info(f"SuperGroup {parent.super_group_id} ({next_tier.value}): {len(child_ids)} child groups in {region}")
        return parent.super_group_id

    # ── Queries ──

    def get_assignment(self, miner_id: str) -> Optional[ShardAssignment]:
        """Get a miner's current shard assignment."""
        return self.miner_assignments.get(miner_id)

    def get_pipeline_group(self, group_id: str) -> Optional[PipelineGroup]:
        """Get a pipeline group by ID."""
        return self.pipeline_groups.get(group_id)

    def get_pipeline_peers(self, miner_id: str) -> List[str]:
        """Get miner IDs of all peers in the same pipeline group."""
        assignment = self.miner_assignments.get(miner_id)
        if not assignment:
            return []
        group = self.pipeline_groups.get(assignment.pipeline_group_id)
        if not group:
            return []
        return [mid for mid in group.miner_ids if mid != miner_id]

    # ── Shard Ready / Status ──

    def report_shard_ready(self, miner_id: str) -> bool:
        """Miner reports that its shard is loaded and ready for training."""
        assignment = self.miner_assignments.get(miner_id)
        if not assignment:
            return False
        assignment.status = "ready"

        group = self.pipeline_groups.get(assignment.pipeline_group_id)
        if group and group.is_ready:
            group.status = PipelineStatus.READY
            logger.info(f"Pipeline group {group.group_id} READY — all {group.num_stages} stages loaded")
        return True

    def start_training(self, group_id: str) -> bool:
        """Transition a pipeline group to training state."""
        group = self.pipeline_groups.get(group_id)
        if not group or not group.is_ready:
            return False
        group.status = PipelineStatus.TRAINING
        for assignment in group.stages.values():
            assignment.status = "training"
        logger.info(f"Pipeline group {group.group_id} → TRAINING")
        return True

    # ── Fault Tolerance ──

    def handle_miner_disconnect(self, miner_id: str) -> Optional[str]:
        """
        Handle a miner leaving the network.
        Returns affected group_id, or None if miner wasn't assigned.
        """
        assignment = self.miner_assignments.pop(miner_id, None)
        if not assignment:
            # Not assigned to a pipeline — just remove from available pool
            miner = self.miners.get(miner_id)
            if miner:
                miner.is_available = False
            return None

        group = self.pipeline_groups.get(assignment.pipeline_group_id)
        if not group:
            return None

        # Mark group as degraded
        group.status = PipelineStatus.DEGRADED
        logger.warning(
            f"Miner {miner_id} disconnected from pipeline {group.group_id} "
            f"(stage {assignment.stage_index}) — group DEGRADED"
        )

        # Remove from group stages
        if assignment.stage_index in group.stages:
            del group.stages[assignment.stage_index]

        # Update upstream/downstream links
        for stage in group.stages.values():
            if stage.downstream_miner_id == miner_id:
                stage.downstream_miner_id = None
            if stage.upstream_miner_id == miner_id:
                stage.upstream_miner_id = None

        return group.group_id

    def reassign_shard(
        self,
        group_id: str,
        stage_index: int,
        new_miner_id: str,
    ) -> Optional[ShardAssignment]:
        """Reassign a shard to a replacement miner (failover)."""
        group = self.pipeline_groups.get(group_id)
        if not group:
            return None

        miner = self.miners.get(new_miner_id)
        if not miner or not miner.is_available:
            return None

        model_config = self._model_configs.get(group.model_id, {})
        layer_ranges = self.compute_layer_assignment(
            model_config.get("num_layers", 0), group.num_stages
        )

        if stage_index >= len(layer_ranges):
            return None

        layer_start, layer_end = layer_ranges[stage_index]
        size_bytes, num_params = self.compute_shard_size(
            model_config, layer_start, layer_end,
            has_embedding=(stage_index == 0),
            has_lm_head=(stage_index == group.num_stages - 1),
        )

        assignment = ShardAssignment(
            miner_id=new_miner_id,
            pipeline_group_id=group_id,
            stage_index=stage_index,
            num_stages=group.num_stages,
            layer_start=layer_start,
            layer_end=layer_end,
            has_embedding=(stage_index == 0),
            has_lm_head=(stage_index == group.num_stages - 1),
            shard_size_bytes=size_bytes,
            shard_size_params=num_params,
            status="loading",
        )

        # Link upstream/downstream
        if stage_index > 0:
            prev = group.stages.get(stage_index - 1)
            if prev:
                assignment.upstream_miner_id = prev.miner_id
                prev.downstream_miner_id = new_miner_id
        if stage_index < group.num_stages - 1:
            nxt = group.stages.get(stage_index + 1)
            if nxt:
                assignment.downstream_miner_id = nxt.miner_id
                nxt.upstream_miner_id = new_miner_id

        group.stages[stage_index] = assignment
        self.miner_assignments[new_miner_id] = assignment
        miner.is_available = False

        logger.info(
            f"Reassigned stage {stage_index} of pipeline {group_id} "
            f"to miner {new_miner_id} (loading shard...)"
        )
        return assignment

    # ── Statistics ──

    def get_stats(self) -> Dict[str, Any]:
        """Get shard manager statistics."""
        active_groups = [g for g in self.pipeline_groups.values() if g.status == PipelineStatus.TRAINING]
        ready_groups = [g for g in self.pipeline_groups.values() if g.status == PipelineStatus.READY]
        degraded_groups = [g for g in self.pipeline_groups.values() if g.status == PipelineStatus.DEGRADED]

        return {
            "total_miners": len(self.miners),
            "available_miners": len(self.get_available_miners()),
            "assigned_miners": len(self.miner_assignments),
            "total_pipeline_groups": len(self.pipeline_groups),
            "active_groups": len(active_groups),
            "ready_groups": len(ready_groups),
            "degraded_groups": len(degraded_groups),
            "forming_groups": sum(1 for g in self.pipeline_groups.values() if g.status == PipelineStatus.FORMING),
            "total_super_groups": len(self.super_groups),
            "hierarchy_depth": self._tree_depth(),
            "scale_tier": self._current_scale_tier(),
        }

    def _tree_depth(self) -> int:
        """Compute current tree hierarchy depth."""
        if not self.super_groups:
            return 0
        # Find root nodes (no parent)
        roots = [sg for sg in self.super_groups.values() if sg.parent_id is None]
        if not roots:
            return 1

        def depth(node_id: str) -> int:
            node = self.super_groups.get(node_id)
            if not node or not node.children_ids:
                return 1
            child_depths = [
                depth(cid) for cid in node.children_ids
                if cid in self.super_groups
            ]
            return 1 + (max(child_depths) if child_depths else 0)

        return max(depth(r.super_group_id) for r in roots)

    def _current_scale_tier(self) -> str:
        """Describe current scale."""
        n = len(self.miners)
        if n == 0:
            return "empty"
        if n <= 10:
            return f"small ({n} miners — no sharding needed)"
        if n <= 1000:
            return f"medium ({n} miners — flat pipeline groups)"
        if n <= 1_000_000:
            return f"large ({n:,} miners — super-groups)"
        return f"planetary ({n:,} miners — hierarchical tree)"

    def needs_sharding(self, model_config: Dict[str, Any]) -> bool:
        """Check if the model needs sharding based on available miners."""
        available = self.get_available_miners()
        if not available:
            return False
        # Check if any single miner can hold the full model
        median_vram = sorted([m.gpu_vram_gb for m in available])[len(available) // 2]
        model_size_gb = model_config["num_parameters"] * 2 / 1e9  # fp16 params only
        full_size_gb = model_size_gb * 4  # params + grads + optimizer
        return full_size_gb > median_vram * 0.75


# Global instance
shard_manager = ShardManager()
