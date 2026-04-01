"""
PARAMETER SERVER
================

Aggregates compressed gradients from miner agents with staleness-aware weighting.
Runs on T1 Genesis Validator (CLASS_F) Lighthouse nodes.

Algorithm (Federated Averaging with staleness damping):
1. Collect compressed gradient submissions from miners
2. Decompress each gradient to dense form
3. Weight each gradient by: (1 / (1 + alpha * staleness))
   where staleness = current_global_step - miner's_local_step
4. Compute weighted average of all gradients
5. Apply merged gradient to global model weights
6. Increment global step, broadcast new weights

This is standard Local SGD / FedAvg with staleness tolerance,
proven by: McMahan et al. 2017, Stich 2019, Douillard et al. (DiLoCo) 2024.

STATUS: PRODUCTION IMPLEMENTATION
UPDATED: 2026-04-01
PURPOSE: Gradient aggregation for distributed miner training
"""

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .gradient_compressor import (
    CompressedGradient,
    GradientCompressor,
    gradient_compressor,
    verify_gradient_hash,
)
from .training_task import (
    GradientSubmission,
    TaskManager,
    TaskStatus,
    get_miner_reward,
    MINER_REWARD_MULTIPLIERS,
)

logger = logging.getLogger(__name__)

# Try torch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class MinerState:
    """Tracks a miner's training state on the parameter server."""
    miner_id: str
    miner_class: str  # validator_miner, core_miner, miner
    local_step: int = 0
    global_step_at_last_sync: int = 0
    total_gradients_submitted: int = 0
    total_samples_trained: int = 0
    cumulative_loss: float = 0.0
    average_training_time: float = 0.0
    trust_score: float = 1.0
    is_active: bool = True
    last_submission_at: Optional[str] = None

    @property
    def staleness(self) -> int:
        """How many global steps behind this miner is."""
        return max(0, self.local_step - self.global_step_at_last_sync)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "miner_id": self.miner_id,
            "miner_class": self.miner_class,
            "local_step": self.local_step,
            "global_step_at_last_sync": self.global_step_at_last_sync,
            "staleness": self.staleness,
            "total_gradients_submitted": self.total_gradients_submitted,
            "total_samples_trained": self.total_samples_trained,
            "cumulative_loss": round(self.cumulative_loss, 4),
            "average_training_time": round(self.average_training_time, 2),
            "trust_score": round(self.trust_score, 4),
            "is_active": self.is_active,
            "last_submission_at": self.last_submission_at,
        }


@dataclass
class AggregationRound:
    """A single round of gradient aggregation."""
    round_id: str
    global_step: int
    model_id: str
    submissions: List[GradientSubmission] = field(default_factory=list)
    merged_gradient_hash: Optional[str] = None
    total_samples: int = 0
    weighted_loss: float = 0.0
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    num_miners: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "global_step": self.global_step,
            "model_id": self.model_id,
            "num_submissions": len(self.submissions),
            "merged_gradient_hash": self.merged_gradient_hash,
            "total_samples": self.total_samples,
            "weighted_loss": round(self.weighted_loss, 6),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "num_miners": self.num_miners,
        }


class ParameterServer:
    """
    Aggregates gradients from miner agents using staleness-aware weighted averaging.
    
    Designed to run on CLASS_F (Genesis Validator) Lighthouse nodes.
    Integrates with the RG_external_blockchain P2P network via GRADIENT_SUBMIT messages
    and the RAFT consensus for committing merged updates.
    """

    # Staleness damping factor: higher = more aggressive penalty for stale gradients
    STALENESS_ALPHA = 0.5

    # Minimum miners needed for an aggregation round
    MIN_MINERS_PER_ROUND = 1

    # Maximum staleness before a gradient is rejected
    MAX_STALENESS = 50

    def __init__(self, model_id: str = "default"):
        self.model_id = model_id
        self.global_step: int = 0
        self.miners: Dict[str, MinerState] = {}
        self.pending_gradients: List[Tuple[GradientSubmission, List[CompressedGradient]]] = []
        self.rounds: List[AggregationRound] = []
        self.compressor = GradientCompressor(compression_ratio=0.01)

    def register_miner(self, miner_id: str, miner_class: str = "miner") -> MinerState:
        """Register a miner with the parameter server."""
        if miner_id in self.miners:
            logger.info(f"Miner {miner_id} already registered")
            return self.miners[miner_id]

        state = MinerState(
            miner_id=miner_id,
            miner_class=miner_class,
            global_step_at_last_sync=self.global_step,
        )
        self.miners[miner_id] = state
        logger.info(f"Registered miner {miner_id} (class={miner_class}) at global step {self.global_step}")
        return state

    def receive_gradient(
        self,
        submission: GradientSubmission,
        compressed_layers: List[CompressedGradient],
    ) -> bool:
        """
        Receive a compressed gradient from a miner.
        
        Args:
            submission: The gradient submission metadata
            compressed_layers: List of compressed gradients (one per model layer)
            
        Returns:
            True if accepted, False if rejected
        """
        miner_id = submission.miner_id

        # Verify miner is registered
        if miner_id not in self.miners:
            logger.warning(f"Gradient from unregistered miner {miner_id}")
            return False

        miner = self.miners[miner_id]

        # Verify gradient integrity
        for cg in compressed_layers:
            if not verify_gradient_hash(cg):
                logger.warning(f"Gradient hash mismatch from miner {miner_id}")
                miner.trust_score = max(0, miner.trust_score - 0.1)
                return False

        # Check staleness
        staleness = abs(self.global_step - miner.global_step_at_last_sync)
        if staleness > self.MAX_STALENESS:
            logger.warning(f"Gradient from miner {miner_id} too stale ({staleness} steps)")
            return False

        # Accept gradient
        self.pending_gradients.append((submission, compressed_layers))

        # Update miner state
        miner.local_step += 1
        miner.total_gradients_submitted += 1
        miner.total_samples_trained += submission.samples_processed
        miner.cumulative_loss += submission.loss_after
        n = miner.total_gradients_submitted
        miner.average_training_time = (
            (miner.average_training_time * (n - 1) + submission.training_time_seconds) / n
        )
        miner.last_submission_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"Accepted gradient from miner {miner_id} "
            f"(staleness={staleness}, loss={submission.loss_after:.4f}, "
            f"samples={submission.samples_processed})"
        )
        return True

    def aggregate(self) -> Optional[Dict[str, List[float]]]:
        """
        Aggregate all pending gradients into a single merged update.
        
        Uses staleness-aware weighted averaging:
            weight_i = (samples_i / total_samples) * (1 / (1 + alpha * staleness_i))
        
        Returns:
            Dict mapping layer_name → merged dense gradient, or None if not enough gradients
        """
        if len(self.pending_gradients) < self.MIN_MINERS_PER_ROUND:
            logger.debug(f"Not enough gradients to aggregate ({len(self.pending_gradients)} < {self.MIN_MINERS_PER_ROUND})")
            return None

        round_id = str(uuid4())
        round_data = AggregationRound(
            round_id=round_id,
            global_step=self.global_step,
            model_id=self.model_id,
            num_miners=len(self.pending_gradients),
        )

        # Calculate staleness weights for each submission
        weighted_submissions = []
        total_weight = 0.0

        for submission, compressed_layers in self.pending_gradients:
            miner = self.miners.get(submission.miner_id)
            if not miner:
                continue

            staleness = abs(self.global_step - miner.global_step_at_last_sync)
            staleness_weight = 1.0 / (1.0 + self.STALENESS_ALPHA * staleness)
            sample_weight = submission.samples_processed
            combined_weight = sample_weight * staleness_weight * miner.trust_score

            weighted_submissions.append((combined_weight, compressed_layers, submission))
            total_weight += combined_weight

            round_data.submissions.append(submission)
            round_data.total_samples += submission.samples_processed

        if total_weight == 0:
            logger.warning("Total weight is zero — cannot aggregate")
            return None

        # Normalize weights
        for i in range(len(weighted_submissions)):
            w, layers, sub = weighted_submissions[i]
            weighted_submissions[i] = (w / total_weight, layers, sub)

        # Merge gradients per layer
        merged_layers: Dict[str, List[float]] = {}

        # Collect all layer names
        all_layer_names = set()
        for _, compressed_layers, _ in weighted_submissions:
            for cg in compressed_layers:
                all_layer_names.add(cg.layer_name)

        for layer_name in all_layer_names:
            # Decompress and weight each miner's gradient for this layer
            merged = None
            original_size = 0

            for weight, compressed_layers, _ in weighted_submissions:
                # Find this layer's gradient
                layer_grad = None
                for cg in compressed_layers:
                    if cg.layer_name == layer_name:
                        layer_grad = cg
                        break

                if layer_grad is None:
                    continue

                original_size = layer_grad.original_size
                dense = self.compressor.decompress(layer_grad)

                if HAS_TORCH and isinstance(dense, type(None)) is False and hasattr(dense, 'mul_'):
                    weighted_dense = dense * weight
                    if merged is None:
                        merged = weighted_dense
                    else:
                        merged = merged + weighted_dense
                else:
                    # Pure Python path
                    if isinstance(dense, list):
                        weighted_dense = [v * weight for v in dense]
                        if merged is None:
                            merged = weighted_dense
                        else:
                            merged = [merged[j] + weighted_dense[j] for j in range(len(merged))]

            if merged is not None:
                if HAS_TORCH and hasattr(merged, 'tolist'):
                    merged_layers[layer_name] = merged.tolist()
                else:
                    merged_layers[layer_name] = merged

        # Compute weighted average loss
        total_loss_weight = 0.0
        weighted_loss_sum = 0.0
        for weight, _, sub in weighted_submissions:
            weighted_loss_sum += weight * sub.loss_after
            total_loss_weight += weight
        if total_loss_weight > 0:
            round_data.weighted_loss = weighted_loss_sum / total_loss_weight

        # Hash the merged result
        merge_hash_input = json.dumps(
            {k: v[:10] if len(v) > 10 else v for k, v in merged_layers.items()},
            sort_keys=True,
        ).encode()
        round_data.merged_gradient_hash = hashlib.sha256(merge_hash_input).hexdigest()
        round_data.completed_at = datetime.now(timezone.utc).isoformat()

        # Advance global step
        self.global_step += 1

        # Sync miners to new global step
        for submission, _ in self.pending_gradients:
            miner = self.miners.get(submission.miner_id)
            if miner:
                miner.global_step_at_last_sync = self.global_step

        # Clear pending
        self.pending_gradients.clear()

        # Store round
        self.rounds.append(round_data)

        logger.info(
            f"Aggregation round {round_id} complete: "
            f"step={self.global_step}, miners={round_data.num_miners}, "
            f"samples={round_data.total_samples}, loss={round_data.weighted_loss:.6f}"
        )

        return merged_layers

    def get_miner_rewards(self, year: int = 1) -> Dict[str, float]:
        """Calculate rewards for all active miners based on their contributions."""
        rewards = {}
        for miner_id, miner in self.miners.items():
            if not miner.is_active or miner.total_gradients_submitted == 0:
                continue
            reward = get_miner_reward(miner.miner_class, year)
            rewards[miner_id] = reward * miner.total_gradients_submitted
        return rewards

    def get_stats(self) -> Dict[str, Any]:
        """Get parameter server statistics."""
        active_miners = sum(1 for m in self.miners.values() if m.is_active)
        total_samples = sum(m.total_samples_trained for m in self.miners.values())
        total_gradients = sum(m.total_gradients_submitted for m in self.miners.values())

        return {
            "model_id": self.model_id,
            "global_step": self.global_step,
            "registered_miners": len(self.miners),
            "active_miners": active_miners,
            "pending_gradients": len(self.pending_gradients),
            "total_aggregation_rounds": len(self.rounds),
            "total_samples_trained": total_samples,
            "total_gradients_received": total_gradients,
            "staleness_alpha": self.STALENESS_ALPHA,
            "max_staleness": self.MAX_STALENESS,
        }

    def get_miner_states(self) -> List[Dict[str, Any]]:
        """Get all miner states."""
        return [m.to_dict() for m in self.miners.values()]


# Global instance
param_server = ParameterServer()
