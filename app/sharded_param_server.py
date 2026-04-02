"""
Sharded Parameter Server — Hierarchical Gradient Aggregation for Unlimited Scale
==================================================================================

Extends the existing ParameterServer with layer-scoped sharding and
tree-structured aggregation for models that span multiple miners.

Architecture:
  - Each ShardedPS instance handles a SUBSET of model layers
  - Multiple ShardedPS instances form a shard group (one per pipeline stage)
  - Shard groups aggregate within their layer range independently
  - A GlobalAggregator coordinates cross-shard step advancement
  - For billion-miner scale, aggregation forms a tree:
    
    Miners → Regional Aggregators → Zone Aggregators → Global Root
    
    At each level, the same staleness-aware weighted averaging is applied.
    Tree depth = O(log N) — handles any number of miners.

Backward compatible: when num_shards=1, behaves exactly like ParameterServer.
"""

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger("rg-mining.sharded-ps")


# ══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════

class AggregationStatus(str, Enum):
    COLLECTING = "collecting"        # Accepting gradients
    AGGREGATING = "aggregating"      # Running weighted average
    WAITING_CONSENSUS = "waiting_consensus"  # Waiting for cross-shard sync
    COMMITTED = "committed"          # Step committed
    FAILED = "failed"


@dataclass
class ShardGradient:
    """A gradient submission scoped to a specific layer range."""
    submission_id: str = field(default_factory=lambda: str(uuid4()))
    miner_id: str = ""
    pipeline_group_id: str = ""
    stage_index: int = 0
    layer_start: int = 0
    layer_end: int = 0
    compressed_layers: list = field(default_factory=list)  # CompressedGradient objects
    local_step: int = 0
    global_step_at_submission: int = 0
    samples_processed: int = 0
    loss: float = 0.0
    training_time: float = 0.0
    submitted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "miner_id": self.miner_id,
            "pipeline_group_id": self.pipeline_group_id,
            "stage_index": self.stage_index,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "num_layers": self.layer_end - self.layer_start,
            "local_step": self.local_step,
            "samples_processed": self.samples_processed,
            "loss": round(self.loss, 6),
            "training_time": round(self.training_time, 2),
            "submitted_at": self.submitted_at,
        }


@dataclass
class ShardAggregationRound:
    """One round of aggregation within a single shard."""
    round_id: str = field(default_factory=lambda: f"sr-{uuid4().hex[:8]}")
    shard_index: int = 0
    global_step: int = 0
    model_id: str = ""
    layer_start: int = 0
    layer_end: int = 0
    submissions: List[ShardGradient] = field(default_factory=list)
    merged_layers: Dict[str, Any] = field(default_factory=dict)  # layer_name → merged values
    status: AggregationStatus = AggregationStatus.COLLECTING
    total_samples: int = 0
    weighted_loss: float = 0.0
    num_miners: int = 0
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "shard_index": self.shard_index,
            "global_step": self.global_step,
            "model_id": self.model_id,
            "layer_range": f"{self.layer_start}-{self.layer_end}",
            "num_submissions": len(self.submissions),
            "status": self.status.value,
            "total_samples": self.total_samples,
            "weighted_loss": round(self.weighted_loss, 6),
            "num_miners": self.num_miners,
            "num_merged_layers": len(self.merged_layers),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


# ══════════════════════════════════════════════════════════════
# SHARD PARAMETER SERVER
# ══════════════════════════════════════════════════════════════

class ShardParameterServer:
    """
    Parameter server for a specific layer range (shard).
    
    Handles gradient collection and aggregation ONLY for its assigned layers.
    Multiple ShardPS instances run in parallel, one per pipeline stage.
    
    Uses the same staleness-aware weighted averaging as the original PS,
    but scoped to a subset of layers.
    """

    # Same constants as original ParameterServer
    STALENESS_ALPHA = 0.5
    MIN_MINERS_PER_ROUND = 1
    MAX_STALENESS = 50

    def __init__(
        self,
        shard_index: int,
        num_shards: int,
        layer_start: int,
        layer_end: int,
        model_id: str = "default",
    ):
        self.shard_index = shard_index
        self.num_shards = num_shards
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.model_id = model_id

        self.global_step: int = 0
        self.pending_gradients: List[ShardGradient] = []
        self.rounds: List[ShardAggregationRound] = []
        self.miner_steps: Dict[str, int] = {}  # miner_id → local step

        # Aggregated state
        self.merged_layers: Dict[str, Any] = {}

        # Stats
        self.total_gradients_received: int = 0
        self.total_samples_trained: int = 0
        self.total_aggregation_rounds: int = 0

    def _is_our_layer(self, layer_name: str) -> bool:
        """Check if a layer belongs to this shard based on naming convention."""
        # Layer names like "layers.0.attention.q_proj", "layers.15.ffn.gate"
        # Extract layer index from name
        parts = layer_name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                    return self.layer_start <= layer_idx < self.layer_end
                except ValueError:
                    pass

        # Special layers: embedding belongs to shard 0, lm_head to last shard
        if any(k in layer_name for k in ("embed", "token_embedding", "wte")):
            return self.shard_index == 0
        if any(k in layer_name for k in ("lm_head", "output", "final_norm")):
            return self.shard_index == self.num_shards - 1

        # Default: accept (for layers without index, like global norms)
        return True

    def receive_gradient(self, gradient: ShardGradient) -> bool:
        """
        Receive a shard-scoped gradient submission.
        
        Returns True if accepted, False if rejected.
        """
        # Validate layer range
        if gradient.layer_start != self.layer_start or gradient.layer_end != self.layer_end:
            # Filter compressed layers to only our range
            if hasattr(gradient, 'compressed_layers') and gradient.compressed_layers:
                gradient.compressed_layers = [
                    cg for cg in gradient.compressed_layers
                    if self._is_our_layer(getattr(cg, 'layer_name', ''))
                ]
                if not gradient.compressed_layers:
                    return False

        # Check staleness
        miner_step = self.miner_steps.get(gradient.miner_id, 0)
        staleness = abs(miner_step - self.global_step)
        if staleness > self.MAX_STALENESS:
            logger.warning(
                f"Shard {self.shard_index}: Rejecting gradient from {gradient.miner_id} "
                f"(staleness={staleness} > max={self.MAX_STALENESS})"
            )
            return False

        self.pending_gradients.append(gradient)
        self.miner_steps[gradient.miner_id] = gradient.local_step
        self.total_gradients_received += 1
        self.total_samples_trained += gradient.samples_processed

        return True

    def should_aggregate(self) -> bool:
        """Check if we have enough gradients for an aggregation round."""
        return len(self.pending_gradients) >= self.MIN_MINERS_PER_ROUND

    def aggregate(self) -> Optional[ShardAggregationRound]:
        """
        Run staleness-aware weighted averaging on pending gradients.
        
        Same algorithm as original ParameterServer.aggregate(), but:
        1. Only processes layers in our range
        2. Returns a ShardAggregationRound (not full AggregationRound)
        3. Does NOT advance global_step (waits for cross-shard consensus)
        """
        if not self.pending_gradients:
            return None

        round_obj = ShardAggregationRound(
            shard_index=self.shard_index,
            global_step=self.global_step,
            model_id=self.model_id,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            submissions=list(self.pending_gradients),
            status=AggregationStatus.AGGREGATING,
        )

        # Compute staleness-aware weights
        weights = []
        for grad in self.pending_gradients:
            miner_step = self.miner_steps.get(grad.miner_id, 0)
            staleness = abs(miner_step - self.global_step)
            # Same formula: w = 1 / (1 + alpha * staleness)
            w = 1.0 / (1.0 + self.STALENESS_ALPHA * staleness)
            # Weight by samples processed
            w *= max(grad.samples_processed, 1)
            weights.append(w)

        total_weight = sum(weights) or 1.0
        normalized_weights = [w / total_weight for w in weights]

        # Merge compressed gradients layer by layer
        merged = {}
        for grad, w in zip(self.pending_gradients, normalized_weights):
            if not hasattr(grad, 'compressed_layers') or not grad.compressed_layers:
                continue
            for cg in grad.compressed_layers:
                layer_name = getattr(cg, 'layer_name', f'layer_{id(cg)}')
                if not self._is_our_layer(layer_name):
                    continue
                if layer_name not in merged:
                    merged[layer_name] = {
                        "weighted_values": None,
                        "shape": getattr(cg, 'original_shape', None),
                    }
                # Accumulate weighted compressed values
                values = getattr(cg, 'values', None)
                indices = getattr(cg, 'indices', None)
                if values is not None:
                    if merged[layer_name]["weighted_values"] is None:
                        merged[layer_name]["weighted_values"] = {
                            "values": [v * w for v in (values if isinstance(values, list) else [values])],
                            "indices": indices,
                        }
                    else:
                        # Merge weighted values (simplified — real impl uses sparse accumulation)
                        existing = merged[layer_name]["weighted_values"]["values"]
                        new_vals = [v * w for v in (values if isinstance(values, list) else [values])]
                        merged[layer_name]["weighted_values"]["values"] = [
                            e + n for e, n in zip(existing, new_vals)
                        ] if len(existing) == len(new_vals) else existing

        round_obj.merged_layers = merged
        round_obj.num_miners = len(self.pending_gradients)
        round_obj.total_samples = sum(g.samples_processed for g in self.pending_gradients)
        round_obj.weighted_loss = sum(
            g.loss * w for g, w in zip(self.pending_gradients, normalized_weights)
        )
        round_obj.status = AggregationStatus.WAITING_CONSENSUS

        self.merged_layers = merged
        self.rounds.append(round_obj)
        self.total_aggregation_rounds += 1

        # Clear pending
        self.pending_gradients = []

        logger.info(
            f"Shard {self.shard_index} aggregation: "
            f"step={self.global_step}, miners={round_obj.num_miners}, "
            f"layers_merged={len(merged)}, "
            f"samples={round_obj.total_samples}, "
            f"loss={round_obj.weighted_loss:.4f}"
        )

        return round_obj

    def commit_step(self) -> int:
        """
        Advance global step after cross-shard consensus.
        Called by the GlobalAggregator after all shards agree.
        """
        self.global_step += 1
        if self.rounds:
            self.rounds[-1].status = AggregationStatus.COMMITTED
            self.rounds[-1].completed_at = datetime.now(timezone.utc).isoformat()
        return self.global_step

    def get_stats(self) -> Dict[str, Any]:
        return {
            "shard_index": self.shard_index,
            "num_shards": self.num_shards,
            "layer_range": f"{self.layer_start}-{self.layer_end}",
            "num_layers": self.layer_end - self.layer_start,
            "model_id": self.model_id,
            "global_step": self.global_step,
            "pending_gradients": len(self.pending_gradients),
            "total_gradients_received": self.total_gradients_received,
            "total_samples_trained": self.total_samples_trained,
            "total_aggregation_rounds": self.total_aggregation_rounds,
            "registered_miners": len(self.miner_steps),
            "merged_layers": len(self.merged_layers),
        }


# ══════════════════════════════════════════════════════════════
# GLOBAL AGGREGATOR — Cross-Shard Coordination
# ══════════════════════════════════════════════════════════════

class GlobalAggregator:
    """
    Coordinates step advancement across all shard parameter servers.
    
    Ensures all shards complete aggregation before any shard advances
    to the next global step. This is the "consensus" layer.
    
    For small networks: simple barrier sync (all shards report done).
    For large networks: hierarchical consensus via the external blockchain's
    Raft protocol (already built).
    
    Hierarchy for billion-miner scale:
    
      Level 0: ShardPS instances (one per pipeline stage per group)
                  ↓ aggregate within shard
      Level 1: Regional Aggregators (one per super-group)
                  ↓ aggregate across pipeline groups in region
      Level 2: Zone Aggregators
                  ↓ aggregate across regions
      Level 3: Global Root
                  ↓ final consensus, advance global step
    
    Each level uses the same weighted averaging.
    Total aggregation latency = O(log N) where N = number of miners.
    """

    def __init__(self, model_id: str = "default"):
        self.model_id = model_id
        self.shards: Dict[int, ShardParameterServer] = {}
        self.global_step: int = 0
        self.shard_ready: Dict[int, bool] = {}  # shard_index → aggregation complete

        # Hierarchical aggregation
        self.regional_aggregators: Dict[str, "RegionalAggregator"] = {}

        # Stats
        self.total_consensus_rounds: int = 0
        self.total_consensus_time: float = 0.0

    def register_shard(self, shard: ShardParameterServer):
        """Register a shard PS with the global aggregator."""
        self.shards[shard.shard_index] = shard
        self.shard_ready[shard.shard_index] = False
        logger.info(
            f"GlobalAggregator: registered shard {shard.shard_index} "
            f"(layers {shard.layer_start}-{shard.layer_end})"
        )

    def report_shard_aggregated(self, shard_index: int, round_obj: ShardAggregationRound):
        """A shard reports its aggregation is complete."""
        self.shard_ready[shard_index] = True

        # Check if ALL shards are ready
        if all(self.shard_ready.values()) and len(self.shard_ready) == len(self.shards):
            self._commit_global_step()

    def _commit_global_step(self):
        """All shards have aggregated — advance global step."""
        start = time.time()

        # Commit on all shards
        for shard in self.shards.values():
            shard.commit_step()

        self.global_step += 1
        elapsed = time.time() - start
        self.total_consensus_rounds += 1
        self.total_consensus_time += elapsed

        # Reset ready flags
        for k in self.shard_ready:
            self.shard_ready[k] = False

        logger.info(
            f"Global step {self.global_step} committed across {len(self.shards)} shards "
            f"({elapsed * 1000:.1f}ms consensus)"
        )

    def create_shards(
        self,
        model_config: Dict[str, Any],
        layer_ranges: List[Tuple[int, int]],
    ) -> List[ShardParameterServer]:
        """
        Create shard PS instances for a model configuration.
        
        Args:
            model_config: Model configuration dict
            layer_ranges: List of (layer_start, layer_end) per shard
        """
        num_shards = len(layer_ranges)
        created = []

        for i, (start, end) in enumerate(layer_ranges):
            shard = ShardParameterServer(
                shard_index=i,
                num_shards=num_shards,
                layer_start=start,
                layer_end=end,
                model_id=model_config.get("model_id", self.model_id),
            )
            self.register_shard(shard)
            created.append(shard)

        logger.info(
            f"Created {num_shards} shard PS instances for "
            f"{model_config.get('model_id', self.model_id)}"
        )
        return created

    def route_gradient(self, gradient: ShardGradient) -> Optional[ShardParameterServer]:
        """
        Route a gradient to the correct shard PS based on layer range.
        
        Returns the shard PS that accepted the gradient, or None.
        """
        for shard in self.shards.values():
            if shard.layer_start <= gradient.layer_start < shard.layer_end:
                if shard.receive_gradient(gradient):
                    return shard
                break

        # Fallback: try all shards (gradient might span shards)
        for shard in self.shards.values():
            if shard.receive_gradient(gradient):
                return shard

        return None

    def try_aggregate_all(self) -> List[ShardAggregationRound]:
        """Try to aggregate all shards that have enough gradients."""
        rounds = []
        for shard in self.shards.values():
            if shard.should_aggregate():
                round_obj = shard.aggregate()
                if round_obj:
                    rounds.append(round_obj)
                    self.report_shard_aggregated(shard.shard_index, round_obj)
        return rounds

    def get_stats(self) -> Dict[str, Any]:
        shard_stats = [s.get_stats() for s in self.shards.values()]
        return {
            "model_id": self.model_id,
            "global_step": self.global_step,
            "num_shards": len(self.shards),
            "shard_ready": dict(self.shard_ready),
            "total_consensus_rounds": self.total_consensus_rounds,
            "avg_consensus_time_ms": round(
                (self.total_consensus_time / max(self.total_consensus_rounds, 1)) * 1000, 1
            ),
            "shard_stats": shard_stats,
            "total_gradients_all_shards": sum(s.total_gradients_received for s in self.shards.values()),
            "total_samples_all_shards": sum(s.total_samples_trained for s in self.shards.values()),
        }


# ══════════════════════════════════════════════════════════════
# REGIONAL AGGREGATOR — For Hierarchical Scaling
# ══════════════════════════════════════════════════════════════

@dataclass
class RegionalMerge:
    """Result of a regional aggregation (merging multiple pipeline groups)."""
    merge_id: str = field(default_factory=lambda: f"rm-{uuid4().hex[:8]}")
    region: str = "global"
    shard_index: int = 0
    global_step: int = 0
    source_groups: List[str] = field(default_factory=list)  # pipeline group IDs
    num_contributions: int = 0
    total_samples: int = 0
    weighted_loss: float = 0.0
    merged_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RegionalAggregator:
    """
    Aggregates results from multiple pipeline groups within a region.
    
    This is the middle tier of the hierarchical aggregation tree:
    
    ShardPS (per pipeline) → RegionalAggregator → ZoneAggregator → Global
    
    Each regional aggregator handles one shard across multiple pipelines.
    It applies the same staleness-aware weighted averaging to merge
    the already-aggregated results from individual pipeline groups.
    
    For 1M miners with 1000 pipeline groups and 10 regions:
    - Each region has ~100 pipeline groups
    - Each RegionalAggregator merges 100 shard results → 1 regional result
    - 10 regional results → 1 global result
    - Total: 2 levels of aggregation (O(log N))
    """

    STALENESS_ALPHA = 0.3  # Less aggressive at regional level

    def __init__(self, region: str, shard_index: int):
        self.region = region
        self.shard_index = shard_index
        self.pending_merges: List[ShardAggregationRound] = []
        self.completed_merges: List[RegionalMerge] = []
        self.global_step: int = 0

    def receive_shard_result(self, round_obj: ShardAggregationRound):
        """Receive an aggregation result from a pipeline group's shard PS."""
        self.pending_merges.append(round_obj)

    def should_merge(self, min_contributions: int = 2) -> bool:
        """Check if enough pipeline groups have contributed."""
        return len(self.pending_merges) >= min_contributions

    def merge(self) -> Optional[RegionalMerge]:
        """
        Merge results from multiple pipeline groups using weighted averaging.
        
        Weights: staleness * samples * trust_score
        Same principle as shard-level, but applied to pre-aggregated results.
        """
        if not self.pending_merges:
            return None

        weights = []
        for r in self.pending_merges:
            staleness = abs(r.global_step - self.global_step)
            w = 1.0 / (1.0 + self.STALENESS_ALPHA * staleness)
            w *= max(r.total_samples, 1)
            weights.append(w)

        total_w = sum(weights) or 1.0
        norm_weights = [w / total_w for w in weights]

        result = RegionalMerge(
            region=self.region,
            shard_index=self.shard_index,
            global_step=self.global_step,
            source_groups=[r.round_id for r in self.pending_merges],
            num_contributions=len(self.pending_merges),
            total_samples=sum(r.total_samples for r in self.pending_merges),
            weighted_loss=sum(
                r.weighted_loss * w
                for r, w in zip(self.pending_merges, norm_weights)
            ),
        )

        self.completed_merges.append(result)
        self.pending_merges = []

        logger.info(
            f"Regional merge ({self.region}, shard {self.shard_index}): "
            f"{result.num_contributions} pipeline groups, "
            f"{result.total_samples} samples"
        )

        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "region": self.region,
            "shard_index": self.shard_index,
            "global_step": self.global_step,
            "pending_merges": len(self.pending_merges),
            "completed_merges": len(self.completed_merges),
        }


# ══════════════════════════════════════════════════════════════
# FACTORY: Create the right PS configuration for model scale
# ══════════════════════════════════════════════════════════════

def create_parameter_server(
    model_config: Dict[str, Any],
    num_shards: int = 1,
    layer_ranges: Optional[List[Tuple[int, int]]] = None,
) -> GlobalAggregator:
    """
    Factory function to create the appropriate parameter server setup.
    
    For small models (1 shard): creates a single ShardPS — behaves like original.
    For large models (N shards): creates N ShardPS instances + GlobalAggregator.
    
    Args:
        model_config: Model configuration from MODEL_REGISTRY
        num_shards: Number of shards (= number of pipeline stages)
        layer_ranges: Optional explicit layer ranges, auto-computed if None
    """
    model_id = model_config.get("model_id", "default")
    num_layers = model_config.get("num_layers", 24)

    if layer_ranges is None:
        if num_shards <= 1:
            layer_ranges = [(0, num_layers)]
        else:
            base = num_layers // num_shards
            remainder = num_layers % num_shards
            ranges = []
            current = 0
            for i in range(num_shards):
                count = base + (1 if i < remainder else 0)
                ranges.append((current, current + count))
                current += count
            layer_ranges = ranges

    aggregator = GlobalAggregator(model_id=model_id)
    aggregator.create_shards(model_config, layer_ranges)

    logger.info(
        f"Created parameter server: model={model_id}, "
        f"shards={num_shards}, layers={num_layers}"
    )

    return aggregator


# Global instance — lazily initialized when first pipeline group forms
sharded_param_server = GlobalAggregator(model_id="default")
