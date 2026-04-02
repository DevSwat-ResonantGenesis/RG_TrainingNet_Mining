"""
WEIGHT SHARD REGISTRY — Network-Native Model Storage
======================================================

The model doesn't live on any single machine. It lives across the swarm.

This registry tracks:
  - Which miner holds which layer weights (the "shard map")
  - Weight version hashes for each shard (integrity)
  - Replication factor per shard (redundancy)
  - On-chain anchoring of shard locations (immutable audit trail)
  - Liquid redistribution when miners join/leave

The model is a living entity distributed across the network:
  - No miner ever downloads the full model
  - Each miner streams only its assigned layers
  - If a miner goes offline, its shard flows to another miner
  - Weights are versioned and hash-verified across the swarm
  - On-chain records ensure accountability and reproducibility

Architecture:
  WeightShardRegistry
    ├── ShardLocation (per-miner, per-layer-range weight tracking)
    ├── WeightVersion (global model version, hash tree of all shards)
    ├── ReplicationPolicy (min replicas, geographic spread)
    └── LiquidRedistributor (automatic shard migration on topology change)

STATUS: PRODUCTION IMPLEMENTATION
UPDATED: 2026-04-02
PURPOSE: Network-native distributed weight storage for decentralized training
"""

import asyncio
import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger("rg-mining.weight-shard-registry")


# ══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════

class ShardState(str, Enum):
    """Lifecycle state of a weight shard on a specific miner."""
    ALLOCATED = "allocated"        # Assigned but not yet downloaded
    STREAMING = "streaming"        # Currently downloading from peers/seed
    LOADED = "loaded"              # In memory, ready for training
    CHECKPOINTED = "checkpointed"  # Saved to local disk
    STALE = "stale"                # Outdated version (needs sync)
    ORPHANED = "orphaned"          # Miner disconnected, shard needs rehoming


class ReplicaPriority(str, Enum):
    """Why this replica exists."""
    PRIMARY = "primary"            # Active training replica
    HOT_SPARE = "hot_spare"        # Ready to take over immediately
    COLD_BACKUP = "cold_backup"    # Stored on disk, not in VRAM
    SEED = "seed"                  # Initial weight source (genesis or checkpoint)


@dataclass
class ShardLocation:
    """
    Tracks a specific weight shard on a specific miner.
    
    One logical shard (e.g., layers 32-40 of model X) may exist
    on multiple miners (replicas). Each replica has its own ShardLocation.
    """
    location_id: str = field(default_factory=lambda: str(uuid4()))
    model_id: str = ""
    miner_id: str = ""
    layer_start: int = 0
    layer_end: int = 0
    version: int = 0                    # Global step when weights were last synced
    weight_hash: str = ""               # SHA-256 of serialized weight tensors
    state: ShardState = ShardState.ALLOCATED
    priority: ReplicaPriority = ReplicaPriority.PRIMARY
    size_bytes: int = 0
    num_params: int = 0
    miner_address: str = ""             # IP:port for P2P download
    download_progress: float = 0.0      # 0.0 → 1.0
    last_sync_at: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def shard_key(self) -> str:
        """Unique key for this logical shard (model + layer range)."""
        return f"{self.model_id}:L{self.layer_start}-{self.layer_end}"

    @property
    def is_available(self) -> bool:
        """Can this location serve weights to other miners?"""
        return self.state in (ShardState.LOADED, ShardState.CHECKPOINTED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location_id": self.location_id,
            "model_id": self.model_id,
            "miner_id": self.miner_id,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "version": self.version,
            "weight_hash": self.weight_hash,
            "state": self.state.value,
            "priority": self.priority.value,
            "size_bytes": self.size_bytes,
            "num_params": self.num_params,
            "miner_address": self.miner_address,
            "download_progress": self.download_progress,
            "last_sync_at": self.last_sync_at,
            "created_at": self.created_at,
        }


@dataclass
class WeightVersion:
    """
    A global model version — the Merkle root of all shard weight hashes.
    
    Each training step produces a new version. The version hash is the
    Merkle root of all shard hashes, anchored on-chain for auditability.
    """
    version_id: str = field(default_factory=lambda: str(uuid4()))
    model_id: str = ""
    global_step: int = 0
    shard_hashes: Dict[str, str] = field(default_factory=dict)  # shard_key → weight_hash
    merkle_root: str = ""
    chain_tx_hash: str = ""             # On-chain anchor transaction
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    num_shards: int = 0
    total_params: int = 0

    def compute_merkle_root(self) -> str:
        """Compute Merkle root from shard hashes."""
        if not self.shard_hashes:
            return hashlib.sha256(b"empty").hexdigest()

        # Sort keys for deterministic ordering
        leaves = [
            hashlib.sha256(f"{k}:{v}".encode()).digest()
            for k, v in sorted(self.shard_hashes.items())
        ]

        # Build Merkle tree
        while len(leaves) > 1:
            if len(leaves) % 2 == 1:
                leaves.append(leaves[-1])  # Duplicate last for odd count
            leaves = [
                hashlib.sha256(leaves[i] + leaves[i + 1]).digest()
                for i in range(0, len(leaves), 2)
            ]

        self.merkle_root = leaves[0].hex()
        return self.merkle_root

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "model_id": self.model_id,
            "global_step": self.global_step,
            "merkle_root": self.merkle_root,
            "chain_tx_hash": self.chain_tx_hash,
            "num_shards": self.num_shards,
            "total_params": self.total_params,
            "created_at": self.created_at,
        }


@dataclass
class ReplicationPolicy:
    """
    Governs how many replicas each shard must maintain.
    
    Like RAID for model weights — no single miner failure loses data.
    """
    min_replicas: int = 2               # At least 2 copies of every shard
    target_replicas: int = 3            # Prefer 3 copies
    max_replicas: int = 10              # Don't over-replicate
    prefer_geographic_spread: bool = True  # Spread replicas across regions
    hot_spare_ratio: float = 0.5        # 50% of replicas should be hot spares
    checkpoint_interval_steps: int = 100  # Save to disk every N steps


# ══════════════════════════════════════════════════════════════
# WEIGHT SHARD REGISTRY
# ══════════════════════════════════════════════════════════════

class WeightShardRegistry:
    """
    The central registry for all weight shard locations across the network.
    
    This is the "DNS of model weights" — when a miner needs layers 32-40,
    it queries this registry to find which peers have those weights,
    then streams them directly via P2P.
    
    The registry itself is replicated across validators for fault tolerance.
    On-chain anchoring provides an immutable audit trail of weight versions.
    
    Key operations:
      register_shard(location)    — Miner reports it has a shard loaded
      find_shard_sources(key)     — Find peers that have specific layers
      orphan_shards(miner_id)     — Mark all of a disconnected miner's shards
      plan_redistribution()       — Compute optimal shard migration plan
      create_version_snapshot()   — Anchor current state on-chain
    """

    def __init__(self, policy: ReplicationPolicy = None):
        self.policy = policy or ReplicationPolicy()

        # Primary index: location_id → ShardLocation
        self.locations: Dict[str, ShardLocation] = {}

        # Index: shard_key → set of location_ids (all replicas of this shard)
        self.shard_replicas: Dict[str, Set[str]] = {}

        # Index: miner_id → set of location_ids (all shards on this miner)
        self.miner_shards: Dict[str, Set[str]] = {}

        # Version history
        self.versions: Dict[int, WeightVersion] = {}  # global_step → version
        self.latest_version: int = 0

        # Statistics
        self._total_registered = 0
        self._total_orphaned = 0
        self._total_redistributed = 0
        self._total_p2p_transfers = 0

    # ── Registration ──

    def register_shard(self, location: ShardLocation) -> ShardLocation:
        """
        Register a weight shard location.
        
        Called when:
          - A miner finishes downloading/initializing its assigned layers
          - A replica is created for redundancy
          - A shard is migrated to a new miner after failover
        """
        self.locations[location.location_id] = location

        # Update shard replica index
        key = location.shard_key
        if key not in self.shard_replicas:
            self.shard_replicas[key] = set()
        self.shard_replicas[key].add(location.location_id)

        # Update miner shard index
        if location.miner_id not in self.miner_shards:
            self.miner_shards[location.miner_id] = set()
        self.miner_shards[location.miner_id].add(location.location_id)

        self._total_registered += 1

        logger.info(
            f"Registered shard {key} on miner {location.miner_id} "
            f"(state={location.state.value}, priority={location.priority.value}, "
            f"replicas={len(self.shard_replicas[key])})"
        )
        return location

    def update_shard_state(
        self,
        location_id: str,
        state: ShardState,
        weight_hash: str = "",
        version: int = 0,
        progress: float = 0.0,
    ) -> Optional[ShardLocation]:
        """Update the state of a specific shard location."""
        loc = self.locations.get(location_id)
        if not loc:
            return None

        loc.state = state
        if weight_hash:
            loc.weight_hash = weight_hash
        if version > 0:
            loc.version = version
        if progress > 0:
            loc.download_progress = progress
        if state in (ShardState.LOADED, ShardState.CHECKPOINTED):
            loc.last_sync_at = datetime.now(timezone.utc).isoformat()

        return loc

    # ── Queries ──

    def find_shard_sources(
        self,
        model_id: str,
        layer_start: int,
        layer_end: int,
        require_loaded: bool = True,
    ) -> List[ShardLocation]:
        """
        Find all miners that have the requested layer range available.
        
        Returns locations sorted by priority (PRIMARY first, then HOT_SPARE).
        This is what miners call to find where to download their weights from.
        """
        key = f"{model_id}:L{layer_start}-{layer_end}"
        location_ids = self.shard_replicas.get(key, set())

        sources = []
        for loc_id in location_ids:
            loc = self.locations.get(loc_id)
            if not loc:
                continue
            if require_loaded and not loc.is_available:
                continue
            sources.append(loc)

        # Sort: PRIMARY > HOT_SPARE > COLD_BACKUP, then by version (newest first)
        priority_order = {
            ReplicaPriority.PRIMARY: 0,
            ReplicaPriority.HOT_SPARE: 1,
            ReplicaPriority.COLD_BACKUP: 2,
            ReplicaPriority.SEED: 3,
        }
        sources.sort(key=lambda s: (priority_order.get(s.priority, 9), -s.version))
        return sources

    def find_overlapping_sources(
        self,
        model_id: str,
        layer_start: int,
        layer_end: int,
    ) -> List[ShardLocation]:
        """
        Find any loaded shard that overlaps with the requested range.
        
        Useful when exact shard boundaries don't match — a miner holding
        layers 0-16 can serve a request for layers 8-16 (partial slice).
        """
        results = []
        for loc in self.locations.values():
            if loc.model_id != model_id or not loc.is_available:
                continue
            # Check overlap
            if loc.layer_start < layer_end and loc.layer_end > layer_start:
                results.append(loc)
        results.sort(key=lambda s: s.layer_start)
        return results

    def get_shard_map(self, model_id: str) -> Dict[str, List[Dict]]:
        """
        Get complete shard map for a model — which layers live where.
        
        Returns: { shard_key: [location_dicts] }
        This is the "GPS" of the model across the network.
        """
        shard_map = {}
        for key, loc_ids in self.shard_replicas.items():
            if not key.startswith(f"{model_id}:"):
                continue
            locations = []
            for loc_id in loc_ids:
                loc = self.locations.get(loc_id)
                if loc:
                    locations.append(loc.to_dict())
            if locations:
                shard_map[key] = locations
        return shard_map

    def get_miner_shards(self, miner_id: str) -> List[ShardLocation]:
        """Get all shards held by a specific miner."""
        loc_ids = self.miner_shards.get(miner_id, set())
        return [self.locations[lid] for lid in loc_ids if lid in self.locations]

    def get_replication_status(self, model_id: str) -> Dict[str, Any]:
        """
        Check replication health for all shards of a model.
        
        Returns which shards are under-replicated, over-replicated, or healthy.
        """
        under_replicated = []
        healthy = []
        over_replicated = []

        for key, loc_ids in self.shard_replicas.items():
            if not key.startswith(f"{model_id}:"):
                continue
            available = [
                lid for lid in loc_ids
                if lid in self.locations and self.locations[lid].is_available
            ]
            count = len(available)

            status = {
                "shard_key": key,
                "replicas": count,
                "total_locations": len(loc_ids),
            }

            if count < self.policy.min_replicas:
                under_replicated.append(status)
            elif count > self.policy.max_replicas:
                over_replicated.append(status)
            else:
                healthy.append(status)

        return {
            "model_id": model_id,
            "total_shards": len(under_replicated) + len(healthy) + len(over_replicated),
            "healthy": len(healthy),
            "under_replicated": under_replicated,
            "over_replicated": over_replicated,
            "min_replicas": self.policy.min_replicas,
            "target_replicas": self.policy.target_replicas,
        }

    # ── Orphaning & Redistribution ──

    def orphan_miner_shards(self, miner_id: str) -> List[ShardLocation]:
        """
        Mark all shards on a disconnected miner as orphaned.
        
        Called when a miner leaves the network. The shards don't disappear —
        they become orphans that need to be rehomed to other miners.
        Like liquid flowing to fill a gap.
        """
        orphaned = []
        loc_ids = self.miner_shards.get(miner_id, set()).copy()

        for loc_id in loc_ids:
            loc = self.locations.get(loc_id)
            if not loc:
                continue
            loc.state = ShardState.ORPHANED
            orphaned.append(loc)
            self._total_orphaned += 1

        logger.warning(
            f"Orphaned {len(orphaned)} shards from miner {miner_id} "
            f"— redistribution needed"
        )
        return orphaned

    def plan_redistribution(
        self,
        model_id: str,
        available_miners: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Compute optimal redistribution plan for under-replicated shards.
        
        This is the "liquid flow" algorithm:
        1. Find all under-replicated shards
        2. Sort by urgency (fewer replicas = more urgent)
        3. For each shard, pick the best available miner:
           - Prefer miners with most free VRAM
           - Prefer miners in different regions (geographic spread)
           - Prefer miners not already holding shards of this model (balance)
        4. Return a list of transfer instructions
        
        The caller executes the plan by notifying miners to pull weights.
        """
        plan = []
        repl_status = self.get_replication_status(model_id)

        # Sort under-replicated by urgency (fewest replicas first)
        urgent = sorted(
            repl_status["under_replicated"],
            key=lambda s: s["replicas"],
        )

        if not urgent:
            return plan

        # Build miner load map (how many shards each miner already holds)
        miner_load = {}
        for m in available_miners:
            mid = m["miner_id"]
            miner_load[mid] = len(self.miner_shards.get(mid, set()))

        for shard_status in urgent:
            key = shard_status["shard_key"]
            current_replicas = shard_status["replicas"]
            needed = self.policy.target_replicas - current_replicas

            if needed <= 0:
                continue

            # Find miners that don't already have this shard
            existing_miners = set()
            for loc_id in self.shard_replicas.get(key, set()):
                loc = self.locations.get(loc_id)
                if loc:
                    existing_miners.add(loc.miner_id)

            candidates = [
                m for m in available_miners
                if m["miner_id"] not in existing_miners
            ]

            # Sort candidates: highest bandwidth first (fastest P2P transfer),
            # then least loaded, then most VRAM
            candidates.sort(
                key=lambda m: (
                    -m.get("bandwidth_mbps", 0),
                    miner_load.get(m["miner_id"], 0),
                    -m.get("gpu_vram_gb", 0),
                )
            )

            # Parse layer range from key
            parts = key.split(":L")
            if len(parts) != 2:
                continue
            layer_range = parts[1].split("-")
            layer_start = int(layer_range[0])
            layer_end = int(layer_range[1])

            # Find a source for this shard (existing loaded replica)
            sources = self.find_shard_sources(model_id, layer_start, layer_end)

            for i in range(min(needed, len(candidates))):
                target = candidates[i]
                source = sources[0] if sources else None

                transfer = {
                    "action": "replicate_shard",
                    "model_id": model_id,
                    "shard_key": key,
                    "layer_start": layer_start,
                    "layer_end": layer_end,
                    "target_miner_id": target["miner_id"],
                    "target_address": target.get("address", ""),
                    "source_miner_id": source.miner_id if source else None,
                    "source_address": source.miner_address if source else None,
                    "priority": "urgent" if current_replicas == 0 else "normal",
                }
                plan.append(transfer)

                # Update load tracking for subsequent assignments
                miner_load[target["miner_id"]] = miner_load.get(target["miner_id"], 0) + 1

        self._total_redistributed += len(plan)
        logger.info(f"Redistribution plan: {len(plan)} transfers for {len(urgent)} under-replicated shards")
        return plan

    # ── Version Snapshots ──

    def create_version_snapshot(
        self,
        model_id: str,
        global_step: int,
    ) -> WeightVersion:
        """
        Create a version snapshot — the Merkle root of all shard hashes.
        
        This gets anchored on-chain, creating an immutable record of the
        model's state at this training step. Any miner can verify its
        weights match the on-chain hash.
        """
        shard_hashes = {}
        total_params = 0

        for key, loc_ids in self.shard_replicas.items():
            if not key.startswith(f"{model_id}:"):
                continue
            # Use the PRIMARY replica's hash (or newest version)
            for loc_id in loc_ids:
                loc = self.locations.get(loc_id)
                if loc and loc.is_available and loc.weight_hash:
                    shard_hashes[key] = loc.weight_hash
                    total_params += loc.num_params
                    break

        version = WeightVersion(
            model_id=model_id,
            global_step=global_step,
            shard_hashes=shard_hashes,
            num_shards=len(shard_hashes),
            total_params=total_params,
        )
        version.compute_merkle_root()

        self.versions[global_step] = version
        self.latest_version = global_step

        logger.info(
            f"Version snapshot: model={model_id} step={global_step} "
            f"shards={len(shard_hashes)} merkle={version.merkle_root[:16]}..."
        )
        return version

    def verify_shard_integrity(
        self,
        miner_id: str,
        shard_key: str,
        reported_hash: str,
        global_step: int = 0,
    ) -> bool:
        """
        Verify a miner's shard weights match the expected hash.
        
        Used to detect weight corruption, tampering, or desync.
        """
        step = global_step or self.latest_version
        version = self.versions.get(step)
        if not version:
            return True  # No version to check against

        expected = version.shard_hashes.get(shard_key)
        if not expected:
            return True  # Shard not tracked in this version

        if reported_hash != expected:
            logger.warning(
                f"Shard integrity MISMATCH: miner={miner_id} shard={shard_key} "
                f"reported={reported_hash[:16]}... expected={expected[:16]}..."
            )
            return False
        return True

    # ── Statistics ──

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        loaded = sum(1 for loc in self.locations.values() if loc.state == ShardState.LOADED)
        streaming = sum(1 for loc in self.locations.values() if loc.state == ShardState.STREAMING)
        orphaned = sum(1 for loc in self.locations.values() if loc.state == ShardState.ORPHANED)

        # Unique models
        models = set(loc.model_id for loc in self.locations.values())

        # Total weight data across network
        total_bytes = sum(loc.size_bytes for loc in self.locations.values() if loc.is_available)

        return {
            "total_locations": len(self.locations),
            "loaded": loaded,
            "streaming": streaming,
            "orphaned": orphaned,
            "unique_shards": len(self.shard_replicas),
            "miners_with_shards": len(self.miner_shards),
            "models_tracked": len(models),
            "model_ids": list(models),
            "total_weight_bytes": total_bytes,
            "total_weight_gb": round(total_bytes / 1e9, 2),
            "versions_tracked": len(self.versions),
            "latest_version": self.latest_version,
            "total_registered": self._total_registered,
            "total_orphaned": self._total_orphaned,
            "total_redistributed": self._total_redistributed,
            "total_p2p_transfers": self._total_p2p_transfers,
            "replication_policy": {
                "min_replicas": self.policy.min_replicas,
                "target_replicas": self.policy.target_replicas,
                "max_replicas": self.policy.max_replicas,
            },
        }

    def get_network_model_status(self, model_id: str) -> Dict[str, Any]:
        """
        Full status of a network-native model.
        
        This is the "health dashboard" for a model living on the network:
        - How many shards exist
        - Which are loaded, streaming, orphaned
        - Replication health
        - Total network capacity being used
        - Latest version anchored on-chain
        """
        shard_map = self.get_shard_map(model_id)
        repl_status = self.get_replication_status(model_id)
        latest = self.versions.get(self.latest_version)

        total_params = 0
        total_bytes = 0
        miners_holding = set()

        for key, locations in shard_map.items():
            for loc_dict in locations:
                total_params = max(total_params, total_params + loc_dict.get("num_params", 0))
                total_bytes += loc_dict.get("size_bytes", 0)
                miners_holding.add(loc_dict["miner_id"])

        return {
            "model_id": model_id,
            "status": "live" if repl_status["healthy"] > 0 else "degraded",
            "total_shards": len(shard_map),
            "miners_holding_shards": len(miners_holding),
            "miner_ids": list(miners_holding),
            "replication": repl_status,
            "total_weight_bytes": total_bytes,
            "total_weight_gb": round(total_bytes / 1e9, 2),
            "latest_version": latest.to_dict() if latest else None,
            "shard_map": {k: len(v) for k, v in shard_map.items()},  # Compact view
        }


class PersistentWeightShardRegistry(WeightShardRegistry):
    """
    Write-through persistent extension of WeightShardRegistry.
    
    Architecture:
      - In-memory dicts for fast reads (inherited from parent)
      - Every mutation (register, update, orphan) writes through to PostgreSQL
      - On startup, load_from_db() rebuilds in-memory state from DB
      - If no database is configured, falls back to pure in-memory (parent behavior)
    
    This ensures the global "shard map" survives Mining service restarts.
    """

    def __init__(self, policy: ReplicationPolicy = None):
        super().__init__(policy)
        self._db_available = False

    async def init_persistence(self):
        """Check if database is available and load existing state."""
        try:
            from .ml_db import ml_engine, MLSessionLocal
            if ml_engine is None or MLSessionLocal is None:
                logger.info("PersistentWeightShardRegistry: no DB configured, running in-memory only")
                return
            self._db_available = True
            await self.load_from_db()
        except Exception as e:
            logger.warning(f"PersistentWeightShardRegistry: DB init failed, running in-memory: {e}")

    async def load_from_db(self):
        """Rebuild in-memory indexes from database rows."""
        if not self._db_available:
            return

        try:
            from .ml_db import MLSessionLocal, WeightShardLocationDB, WeightVersionDB
            from sqlalchemy import select

            async with MLSessionLocal() as session:
                # Load all shard locations
                result = await session.execute(select(WeightShardLocationDB))
                rows = result.scalars().all()

                loaded_count = 0
                for row in rows:
                    loc = ShardLocation(
                        location_id=row.location_id,
                        model_id=row.model_id,
                        miner_id=row.miner_id,
                        layer_start=row.layer_start,
                        layer_end=row.layer_end,
                        version=row.version,
                        weight_hash=row.weight_hash,
                        state=ShardState(row.state),
                        priority=ReplicaPriority(row.priority),
                        size_bytes=row.size_bytes,
                        num_params=row.num_params,
                        miner_address=row.miner_address,
                        download_progress=row.download_progress,
                        last_sync_at=row.last_sync_at or "",
                        created_at=row.created_at.isoformat() if row.created_at else "",
                    )
                    # Insert into in-memory indexes (bypass DB write-through)
                    self.locations[loc.location_id] = loc
                    key = loc.shard_key
                    if key not in self.shard_replicas:
                        self.shard_replicas[key] = set()
                    self.shard_replicas[key].add(loc.location_id)
                    if loc.miner_id not in self.miner_shards:
                        self.miner_shards[loc.miner_id] = set()
                    self.miner_shards[loc.miner_id].add(loc.location_id)
                    loaded_count += 1

                # Load weight versions
                result = await session.execute(
                    select(WeightVersionDB).order_by(WeightVersionDB.global_step)
                )
                version_rows = result.scalars().all()
                for vrow in version_rows:
                    wv = WeightVersion(
                        version_id=vrow.version_id,
                        model_id=vrow.model_id,
                        global_step=vrow.global_step,
                        shard_hashes=vrow.shard_hashes_json or {},
                        merkle_root=vrow.merkle_root,
                        chain_tx_hash=vrow.chain_tx_hash,
                        num_shards=vrow.num_shards,
                        total_params=vrow.total_params,
                        created_at=vrow.created_at.isoformat() if vrow.created_at else "",
                    )
                    self.versions[vrow.global_step] = wv
                    self.latest_version = max(self.latest_version, vrow.global_step)

            logger.info(
                f"PersistentWeightShardRegistry: loaded {loaded_count} shard locations, "
                f"{len(self.versions)} versions from database"
            )
        except Exception as e:
            logger.error(f"PersistentWeightShardRegistry: failed to load from DB: {e}")

    async def _persist_location(self, loc: ShardLocation):
        """Write a single ShardLocation to the database."""
        if not self._db_available:
            return
        try:
            from .ml_db import MLSessionLocal, WeightShardLocationDB
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            async with MLSessionLocal() as session:
                async with session.begin():
                    stmt = pg_insert(WeightShardLocationDB).values(
                        location_id=loc.location_id,
                        model_id=loc.model_id,
                        miner_id=loc.miner_id,
                        layer_start=loc.layer_start,
                        layer_end=loc.layer_end,
                        version=loc.version,
                        weight_hash=loc.weight_hash,
                        state=loc.state.value,
                        priority=loc.priority.value,
                        size_bytes=loc.size_bytes,
                        num_params=loc.num_params,
                        miner_address=loc.miner_address,
                        download_progress=loc.download_progress,
                        last_sync_at=loc.last_sync_at,
                    ).on_conflict_do_update(
                        index_elements=["location_id"],
                        set_={
                            "state": loc.state.value,
                            "version": loc.version,
                            "weight_hash": loc.weight_hash,
                            "download_progress": loc.download_progress,
                            "last_sync_at": loc.last_sync_at,
                            "miner_address": loc.miner_address,
                        },
                    )
                    await session.execute(stmt)
        except Exception as e:
            logger.error(f"Failed to persist shard location {loc.location_id}: {e}")

    async def _persist_version(self, version: WeightVersion):
        """Write a WeightVersion snapshot to the database."""
        if not self._db_available:
            return
        try:
            from .ml_db import MLSessionLocal, WeightVersionDB
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            async with MLSessionLocal() as session:
                async with session.begin():
                    stmt = pg_insert(WeightVersionDB).values(
                        version_id=version.version_id,
                        model_id=version.model_id,
                        global_step=version.global_step,
                        merkle_root=version.merkle_root,
                        chain_tx_hash=version.chain_tx_hash,
                        num_shards=version.num_shards,
                        total_params=version.total_params,
                        shard_hashes_json=version.shard_hashes,
                    ).on_conflict_do_update(
                        index_elements=["version_id"],
                        set_={
                            "merkle_root": version.merkle_root,
                            "chain_tx_hash": version.chain_tx_hash,
                        },
                    )
                    await session.execute(stmt)
        except Exception as e:
            logger.error(f"Failed to persist weight version {version.version_id}: {e}")

    # ── Override mutations to add write-through ──

    def register_shard(self, location: ShardLocation) -> ShardLocation:
        """Register shard in-memory + schedule DB persist."""
        result = super().register_shard(location)
        asyncio.ensure_future(self._persist_location(result))
        return result

    def update_shard_state(
        self,
        location_id: str,
        state: ShardState,
        weight_hash: str = "",
        version: int = 0,
        progress: float = 0.0,
    ) -> Optional[ShardLocation]:
        """Update shard state in-memory + schedule DB persist."""
        result = super().update_shard_state(location_id, state, weight_hash, version, progress)
        if result:
            asyncio.ensure_future(self._persist_location(result))
        return result

    def orphan_miner_shards(self, miner_id: str) -> List[ShardLocation]:
        """Orphan shards in-memory + schedule DB persist for each."""
        orphaned = super().orphan_miner_shards(miner_id)
        for loc in orphaned:
            asyncio.ensure_future(self._persist_location(loc))
        return orphaned

    def create_version_snapshot(
        self,
        model_id: str,
        global_step: int,
    ) -> WeightVersion:
        """Create version snapshot in-memory + schedule DB persist."""
        version = super().create_version_snapshot(model_id, global_step)
        asyncio.ensure_future(self._persist_version(version))
        return version


# Global instance — uses persistent variant so it survives restarts
weight_registry = PersistentWeightShardRegistry()
