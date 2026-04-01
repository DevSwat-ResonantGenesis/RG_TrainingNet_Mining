"""
TRAINING TASK DEFINITION
========================

Defines training tasks for distributed LLM training across miner agents.
A training task is assigned to a miner and specifies: which model weights to download,
which data shard to train on, and what hyperparameters to use.

Miners are agents (CLASS_F/G/H). Tasks flow through the existing P2P network
via TRAINING_TASK messages and results submit via GRADIENT_SUBMIT.

STATUS: PRODUCTION IMPLEMENTATION
UPDATED: 2026-04-01
PURPOSE: Distributed LLM training task orchestration
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TaskType(str, Enum):
    FORWARD_BACKWARD = "forward_backward"
    COT_GENERATION = "cot_generation"
    EVALUATION = "evaluation"


@dataclass
class TrainingTask:
    """A single training task assigned to a miner agent."""
    task_id: str
    task_type: TaskType
    model_id: str
    epoch: int
    batch_index: int

    # What to train on
    data_shard_url: str
    data_shard_hash: str
    num_samples: int

    # Model weights to start from
    weight_shard_url: str
    weight_shard_hash: str

    # Hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    bf16: bool = True

    # Assignment
    assigned_miner_id: Optional[str] = None
    assigned_at: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING

    # Results
    gradient_hash: Optional[str] = None
    loss_value: Optional[float] = None
    samples_processed: int = 0
    training_time_seconds: float = 0.0

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    deadline_seconds: int = 600  # 10 min default deadline

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "model_id": self.model_id,
            "epoch": self.epoch,
            "batch_index": self.batch_index,
            "data_shard_url": self.data_shard_url,
            "data_shard_hash": self.data_shard_hash,
            "num_samples": self.num_samples,
            "weight_shard_url": self.weight_shard_url,
            "weight_shard_hash": self.weight_shard_hash,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_seq_length": self.max_seq_length,
            "bf16": self.bf16,
            "assigned_miner_id": self.assigned_miner_id,
            "assigned_at": self.assigned_at,
            "status": self.status.value,
            "gradient_hash": self.gradient_hash,
            "loss_value": self.loss_value,
            "samples_processed": self.samples_processed,
            "training_time_seconds": self.training_time_seconds,
            "created_at": self.created_at,
            "deadline_seconds": self.deadline_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingTask":
        return cls(
            task_id=data["task_id"],
            task_type=TaskType(data["task_type"]),
            model_id=data["model_id"],
            epoch=data["epoch"],
            batch_index=data["batch_index"],
            data_shard_url=data["data_shard_url"],
            data_shard_hash=data["data_shard_hash"],
            num_samples=data["num_samples"],
            weight_shard_url=data["weight_shard_url"],
            weight_shard_hash=data["weight_shard_hash"],
            learning_rate=data.get("learning_rate", 1e-4),
            batch_size=data.get("batch_size", 8),
            gradient_accumulation_steps=data.get("gradient_accumulation_steps", 4),
            max_seq_length=data.get("max_seq_length", 2048),
            bf16=data.get("bf16", True),
            assigned_miner_id=data.get("assigned_miner_id"),
            assigned_at=data.get("assigned_at"),
            status=TaskStatus(data.get("status", "pending")),
            gradient_hash=data.get("gradient_hash"),
            loss_value=data.get("loss_value"),
            samples_processed=data.get("samples_processed", 0),
            training_time_seconds=data.get("training_time_seconds", 0.0),
            created_at=data.get("created_at", ""),
            deadline_seconds=data.get("deadline_seconds", 600),
        )

    @property
    def is_expired(self) -> bool:
        if not self.assigned_at:
            return False
        assigned_time = datetime.fromisoformat(self.assigned_at)
        elapsed = (datetime.now(timezone.utc) - assigned_time).total_seconds()
        return elapsed > self.deadline_seconds


@dataclass
class GradientSubmission:
    """Compressed gradient submitted by a miner after training."""
    submission_id: str
    task_id: str
    miner_id: str
    model_id: str
    epoch: int
    batch_index: int

    # Compressed gradient data
    top_k_indices: List[int]
    top_k_values: List[float]
    original_size: int
    compressed_size: int
    compression_ratio: float

    # Training metrics
    loss_before: float
    loss_after: float
    samples_processed: int
    training_time_seconds: float

    # Integrity
    gradient_hash: str
    data_shard_hash: str
    weight_shard_hash: str

    submitted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "task_id": self.task_id,
            "miner_id": self.miner_id,
            "model_id": self.model_id,
            "epoch": self.epoch,
            "batch_index": self.batch_index,
            "top_k_indices": self.top_k_indices,
            "top_k_values": self.top_k_values,
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "compression_ratio": self.compression_ratio,
            "loss_before": self.loss_before,
            "loss_after": self.loss_after,
            "samples_processed": self.samples_processed,
            "training_time_seconds": self.training_time_seconds,
            "gradient_hash": self.gradient_hash,
            "data_shard_hash": self.data_shard_hash,
            "weight_shard_hash": self.weight_shard_hash,
            "submitted_at": self.submitted_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GradientSubmission":
        return cls(
            submission_id=data["submission_id"],
            task_id=data["task_id"],
            miner_id=data["miner_id"],
            model_id=data["model_id"],
            epoch=data["epoch"],
            batch_index=data["batch_index"],
            top_k_indices=data["top_k_indices"],
            top_k_values=data["top_k_values"],
            original_size=data["original_size"],
            compressed_size=data["compressed_size"],
            compression_ratio=data["compression_ratio"],
            loss_before=data["loss_before"],
            loss_after=data["loss_after"],
            samples_processed=data["samples_processed"],
            training_time_seconds=data["training_time_seconds"],
            gradient_hash=data["gradient_hash"],
            data_shard_hash=data["data_shard_hash"],
            weight_shard_hash=data["weight_shard_hash"],
            submitted_at=data.get("submitted_at", ""),
        )


# ============== MINER REWARD MULTIPLIERS ==============

MINER_REWARD_MULTIPLIERS = {
    "validator_miner": 1.5,   # CLASS_F — Genesis Validator (T1)
    "core_miner": 1.25,       # CLASS_G — Core Contributor (T2)
    "miner": 1.0,             # CLASS_H — Standard Miner (T3/T4)
}

BASE_BLOCK_REWARD = 100  # $RGT per training block (Year 1)

HALVING_SCHEDULE = {
    1: 100,    # Year 1: 100 $RGT/block
    2: 50,     # Year 2: 50 $RGT/block
    3: 25,     # Year 3: 25 $RGT/block
    4: 12.5,   # Year 4: 12.5 $RGT/block
}


def get_block_reward(year: int = 1) -> float:
    """Get block reward for the given year (with halving)."""
    if year in HALVING_SCHEDULE:
        return HALVING_SCHEDULE[year]
    return max(1.0, 100 / (2 ** (year - 1)))


def get_miner_reward(miner_class: str, year: int = 1) -> float:
    """Get miner's actual reward after applying tier multiplier."""
    base = get_block_reward(year)
    multiplier = MINER_REWARD_MULTIPLIERS.get(miner_class, 1.0)
    return base * multiplier


class TaskManager:
    """Manages training task lifecycle — assignment, tracking, expiry."""

    def __init__(self):
        self.tasks: Dict[str, TrainingTask] = {}
        self.submissions: Dict[str, GradientSubmission] = {}
        self._task_queue: List[str] = []

    def create_task(
        self,
        model_id: str,
        epoch: int,
        batch_index: int,
        data_shard_url: str,
        data_shard_hash: str,
        num_samples: int,
        weight_shard_url: str,
        weight_shard_hash: str,
        task_type: TaskType = TaskType.FORWARD_BACKWARD,
        **hyperparams,
    ) -> TrainingTask:
        """Create a new training task."""
        task = TrainingTask(
            task_id=str(uuid4()),
            task_type=task_type,
            model_id=model_id,
            epoch=epoch,
            batch_index=batch_index,
            data_shard_url=data_shard_url,
            data_shard_hash=data_shard_hash,
            num_samples=num_samples,
            weight_shard_url=weight_shard_url,
            weight_shard_hash=weight_shard_hash,
            **hyperparams,
        )
        self.tasks[task.task_id] = task
        self._task_queue.append(task.task_id)
        logger.info(f"Created training task {task.task_id} for model {model_id} epoch {epoch} batch {batch_index}")
        return task

    def assign_task(self, miner_id: str) -> Optional[TrainingTask]:
        """Assign next available task to a miner agent."""
        # Expire old tasks first
        self._expire_stale_tasks()

        while self._task_queue:
            task_id = self._task_queue.pop(0)
            task = self.tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.ASSIGNED
                task.assigned_miner_id = miner_id
                task.assigned_at = datetime.now(timezone.utc).isoformat()
                logger.info(f"Assigned task {task_id} to miner {miner_id}")
                return task

        return None

    def submit_result(self, submission: GradientSubmission) -> bool:
        """Record a gradient submission from a miner."""
        task = self.tasks.get(submission.task_id)
        if not task:
            logger.warning(f"Submission for unknown task {submission.task_id}")
            return False

        if task.assigned_miner_id != submission.miner_id:
            logger.warning(f"Submission from wrong miner: expected {task.assigned_miner_id}, got {submission.miner_id}")
            return False

        if task.status not in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
            logger.warning(f"Task {task.task_id} not in assignable state: {task.status}")
            return False

        task.status = TaskStatus.SUBMITTED
        task.gradient_hash = submission.gradient_hash
        task.loss_value = submission.loss_after
        task.samples_processed = submission.samples_processed
        task.training_time_seconds = submission.training_time_seconds

        self.submissions[submission.submission_id] = submission
        logger.info(f"Recorded gradient submission {submission.submission_id} for task {task.task_id} (loss={submission.loss_after:.4f})")
        return True

    def mark_verified(self, task_id: str) -> bool:
        """Mark a task as verified after spot-check passes."""
        task = self.tasks.get(task_id)
        if task and task.status == TaskStatus.SUBMITTED:
            task.status = TaskStatus.VERIFIED
            logger.info(f"Task {task_id} verified")
            return True
        return False

    def mark_rejected(self, task_id: str, reason: str = "") -> bool:
        """Mark a task as rejected (miner submitted bad gradient)."""
        task = self.tasks.get(task_id)
        if task:
            task.status = TaskStatus.REJECTED
            logger.warning(f"Task {task_id} REJECTED: {reason}")
            return True
        return False

    def _expire_stale_tasks(self):
        """Expire tasks that have been assigned but not completed in time."""
        for task in self.tasks.values():
            if task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS) and task.is_expired:
                task.status = TaskStatus.EXPIRED
                self._task_queue.append(task.task_id)
                task.status = TaskStatus.PENDING
                task.assigned_miner_id = None
                task.assigned_at = None
                logger.info(f"Task {task.task_id} expired, re-queued")

    def get_stats(self) -> Dict[str, Any]:
        """Get task manager statistics."""
        status_counts = {}
        for task in self.tasks.values():
            s = task.status.value
            status_counts[s] = status_counts.get(s, 0) + 1

        return {
            "total_tasks": len(self.tasks),
            "queue_length": len(self._task_queue),
            "total_submissions": len(self.submissions),
            "status_breakdown": status_counts,
        }


# Global instance
task_manager = TaskManager()
