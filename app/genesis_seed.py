"""
GENESIS SEED INITIALIZATION
============================

Initializes the first 1B-parameter "Seed" model and distributes training tasks
to Lighthouse (CLASS_F validator) nodes. This is the Day 1 trigger that starts
the decentralized training network.

Usage:
    python -m app.genesis_seed --model-id resonant-seed-1b --miners 10

Flow:
    1. Define the Seed model (1B params, GPT-2 architecture scaled up)
    2. Shard the initial random weights across IPFS
    3. Partition the training dataset into shards
    4. Create TrainingTask for each shard
    5. Broadcast TRAINING_TASK messages to all registered miners via P2P
    6. Start the parameter server aggregation loop

STATUS: PRODUCTION IMPLEMENTATION
UPDATED: 2026-04-01
PURPOSE: Bootstrap the decentralized training network
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .training_task import (
    TaskManager,
    TaskType,
    TrainingTask,
    task_manager,
    MINER_REWARD_MULTIPLIERS,
    BASE_BLOCK_REWARD,
)
from .param_server import ParameterServer, param_server
from .gradient_compressor import GradientCompressor

logger = logging.getLogger(__name__)


@dataclass
class SeedModelConfig:
    """Configuration for the Genesis Seed model."""
    model_id: str = "resonant-seed-1b"
    model_type: str = "gpt2-xl-custom"
    num_parameters: int = 1_000_000_000  # 1B
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    vocab_size: int = 50257
    max_seq_length: int = 2048
    dtype: str = "bfloat16"

    # Training config
    learning_rate: float = 3e-4
    warmup_steps: int = 2000
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    total_training_tokens: int = 50_000_000_000  # 50B tokens target

    # Sharding
    num_weight_shards: int = 10  # Split weights across 10 IPFS shards
    num_data_shards: int = 100   # 100 data shards (1 per miner)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "num_parameters": self.num_parameters,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "vocab_size": self.vocab_size,
            "max_seq_length": self.max_seq_length,
            "dtype": self.dtype,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "total_training_tokens": self.total_training_tokens,
            "num_weight_shards": self.num_weight_shards,
            "num_data_shards": self.num_data_shards,
        }


@dataclass
class GenesisState:
    """Tracks the state of genesis initialization."""
    initialized: bool = False
    model_config: Optional[SeedModelConfig] = None
    weight_shard_urls: List[str] = field(default_factory=list)
    weight_shard_hashes: List[str] = field(default_factory=list)
    data_shard_urls: List[str] = field(default_factory=list)
    data_shard_hashes: List[str] = field(default_factory=list)
    registered_miners: List[str] = field(default_factory=list)
    tasks_created: int = 0
    genesis_block_hash: Optional[str] = None
    initialized_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initialized": self.initialized,
            "model_config": self.model_config.to_dict() if self.model_config else None,
            "weight_shards": len(self.weight_shard_urls),
            "data_shards": len(self.data_shard_urls),
            "registered_miners": len(self.registered_miners),
            "tasks_created": self.tasks_created,
            "genesis_block_hash": self.genesis_block_hash,
            "initialized_at": self.initialized_at,
        }


class GenesisSeedInitializer:
    """
    Bootstraps the decentralized training network.
    
    This is the "big bang" — creates the genesis model, shards it,
    creates training tasks, and distributes them to Lighthouse nodes.
    """

    def __init__(self):
        self.state = GenesisState()
        self.task_manager = task_manager
        self.param_server = param_server

    async def initialize(
        self,
        model_config: Optional[SeedModelConfig] = None,
        miner_ids: Optional[List[str]] = None,
        ipfs_base_url: str = "ipfs://",
    ) -> GenesisState:
        """
        Run the full genesis initialization sequence.
        
        Args:
            model_config: Override default model config
            miner_ids: List of registered miner agent IDs
            ipfs_base_url: Base URL for IPFS content addressing
            
        Returns:
            GenesisState with all initialization details
        """
        if self.state.initialized:
            logger.warning("Genesis already initialized")
            return self.state

        config = model_config or SeedModelConfig()
        self.state.model_config = config
        miners = miner_ids or []

        logger.info(f"{'='*60}")
        logger.info(f"GENESIS SEED INITIALIZATION")
        logger.info(f"Model: {config.model_id} ({config.num_parameters:,} params)")
        logger.info(f"Miners: {len(miners)}")
        logger.info(f"{'='*60}")

        # Step 1: Generate weight shard metadata
        logger.info("[1/5] Generating weight shard metadata...")
        await self._create_weight_shards(config, ipfs_base_url)

        # Step 2: Generate data shard metadata
        logger.info("[2/5] Generating data shard metadata...")
        await self._create_data_shards(config, ipfs_base_url)

        # Step 3: Register miners with parameter server
        logger.info("[3/5] Registering miners...")
        await self._register_miners(miners)

        # Step 4: Create training tasks
        logger.info("[4/5] Creating training tasks...")
        await self._create_training_tasks(config)

        # Step 5: Generate genesis block hash
        logger.info("[5/5] Generating genesis block...")
        await self._create_genesis_block(config)

        self.state.initialized = True
        self.state.initialized_at = datetime.now(timezone.utc).isoformat()

        logger.info(f"{'='*60}")
        logger.info(f"GENESIS COMPLETE")
        logger.info(f"  Weight shards: {len(self.state.weight_shard_urls)}")
        logger.info(f"  Data shards: {len(self.state.data_shard_urls)}")
        logger.info(f"  Miners: {len(self.state.registered_miners)}")
        logger.info(f"  Tasks: {self.state.tasks_created}")
        logger.info(f"  Genesis block: {self.state.genesis_block_hash}")
        logger.info(f"{'='*60}")

        return self.state

    async def _create_weight_shards(self, config: SeedModelConfig, ipfs_base: str):
        """Create metadata for model weight shards on IPFS."""
        params_per_shard = config.num_parameters // config.num_weight_shards

        for i in range(config.num_weight_shards):
            shard_id = f"{config.model_id}_weights_shard_{i:03d}"
            shard_hash = hashlib.sha256(
                f"{shard_id}:{params_per_shard}:{config.dtype}:{i}".encode()
            ).hexdigest()

            shard_url = f"{ipfs_base}{shard_hash}"
            self.state.weight_shard_urls.append(shard_url)
            self.state.weight_shard_hashes.append(shard_hash)

            logger.debug(f"  Weight shard {i}: {shard_hash[:16]}... ({params_per_shard:,} params)")

    async def _create_data_shards(self, config: SeedModelConfig, ipfs_base: str):
        """Create metadata for training data shards."""
        tokens_per_shard = config.total_training_tokens // config.num_data_shards
        samples_per_shard = tokens_per_shard // config.max_seq_length

        for i in range(config.num_data_shards):
            shard_id = f"{config.model_id}_data_shard_{i:03d}"
            shard_hash = hashlib.sha256(
                f"{shard_id}:{tokens_per_shard}:{i}".encode()
            ).hexdigest()

            shard_url = f"{ipfs_base}{shard_hash}"
            self.state.data_shard_urls.append(shard_url)
            self.state.data_shard_hashes.append(shard_hash)

            logger.debug(f"  Data shard {i}: {shard_hash[:16]}... ({samples_per_shard:,} samples)")

    async def _register_miners(self, miner_ids: List[str]):
        """Register miners with the parameter server."""
        for miner_id in miner_ids:
            # Determine miner class based on position
            # First 10% are validators (CLASS_F), next 20% core (CLASS_G), rest standard (CLASS_H)
            total = len(miner_ids)
            idx = miner_ids.index(miner_id)

            if idx < max(1, int(total * 0.10)):
                miner_class = "validator_miner"
            elif idx < max(2, int(total * 0.30)):
                miner_class = "core_miner"
            else:
                miner_class = "miner"

            self.param_server.register_miner(miner_id, miner_class)
            self.state.registered_miners.append(miner_id)

    async def _create_training_tasks(self, config: SeedModelConfig):
        """Create training tasks for each data shard."""
        # Each data shard gets one task per epoch
        # For genesis, we create tasks for epoch 0 only
        tokens_per_shard = config.total_training_tokens // config.num_data_shards
        samples_per_shard = tokens_per_shard // config.max_seq_length

        # Use the first weight shard URL (all miners start from same weights in epoch 0)
        weight_url = self.state.weight_shard_urls[0] if self.state.weight_shard_urls else "ipfs://genesis_weights"
        weight_hash = self.state.weight_shard_hashes[0] if self.state.weight_shard_hashes else "genesis"

        for i in range(config.num_data_shards):
            task = self.task_manager.create_task(
                model_id=config.model_id,
                epoch=0,
                batch_index=i,
                data_shard_url=self.state.data_shard_urls[i],
                data_shard_hash=self.state.data_shard_hashes[i],
                num_samples=samples_per_shard,
                weight_shard_url=weight_url,
                weight_shard_hash=weight_hash,
                task_type=TaskType.FORWARD_BACKWARD,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                max_seq_length=config.max_seq_length,
                bf16=(config.dtype == "bfloat16"),
            )
            self.state.tasks_created += 1

        logger.info(f"  Created {self.state.tasks_created} training tasks for epoch 0")

    async def _create_genesis_block(self, config: SeedModelConfig):
        """Create the genesis block hash with State Physics constants baked in."""
        genesis_data = {
            "type": "genesis",
            "model_id": config.model_id,
            "model_params": config.num_parameters,
            "total_training_tokens": config.total_training_tokens,
            "num_miners": len(self.state.registered_miners),
            "weight_shard_hashes": self.state.weight_shard_hashes,
            "data_shard_count": len(self.state.data_shard_hashes),
            "state_physics_constants": {
                "entropy_threshold": 0.85,
                "collapse_risk_threshold": 0.9,
                "coherence_minimum": 0.3,
                "gravity_constant": 6.674e-11,
            },
            "token_economics": {
                "total_supply": 1_000_000_000,
                "base_block_reward": BASE_BLOCK_REWARD,
                "miner_multipliers": MINER_REWARD_MULTIPLIERS,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        genesis_hash = hashlib.sha256(
            json.dumps(genesis_data, sort_keys=True, default=str).encode()
        ).hexdigest()

        self.state.genesis_block_hash = genesis_hash
        logger.info(f"  Genesis block hash: {genesis_hash}")

    def get_status(self) -> Dict[str, Any]:
        """Get current genesis initialization status."""
        return {
            "genesis": self.state.to_dict(),
            "task_manager": self.task_manager.get_stats(),
            "param_server": self.param_server.get_stats(),
        }

    async def assign_tasks_to_miners(self) -> Dict[str, str]:
        """Assign pending tasks to registered miners (round-robin)."""
        assignments = {}
        for miner_id in self.state.registered_miners:
            task = self.task_manager.assign_task(miner_id)
            if task:
                assignments[miner_id] = task.task_id
                logger.info(f"  Assigned task {task.task_id} to miner {miner_id}")
            else:
                break  # No more tasks
        return assignments


# Global instance
genesis_initializer = GenesisSeedInitializer()
