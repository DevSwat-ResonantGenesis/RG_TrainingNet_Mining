"""
ML DATABASE CONNECTION + MODELS
================================

Connects to the DigitalOcean ML database for storing:
- Model checkpoints (metadata + S3 pointers)
- Training state (epochs, global steps, loss curves)
- Weight shards (per-layer chunks for distributed training)
- Data shard assignments (which miner trains which data)
- Gradient history (verified submissions)

Uses the dedicated ML database (ml-registry-db), NOT the main system DB.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column, String, Integer, BigInteger, Float, Boolean, Text,
    DateTime, JSON, LargeBinary, Index, ForeignKey,
    create_engine,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

logger = logging.getLogger(__name__)

# ── Connection ──
ML_DATABASE_URL = os.getenv(
    "ML_DATABASE_URL",
    "postgresql+asyncpg://doadmin:AVNS_Ts6MtEs9WQIYwkw1FoI@ml-registry-db-do-user-18031534-0.g.db.ondigitalocean.com:25060/defaultdb?ssl=require"
)

# S3 config for weight blob storage
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "https://sfo3.digitaloceanspaces.com")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "")
S3_BUCKET = os.getenv("S3_BUCKET", "genesis2026")
S3_WEIGHTS_PREFIX = "model-weights/"

ml_engine = create_async_engine(
    ML_DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

MLSessionLocal = sessionmaker(ml_engine, class_=AsyncSession, expire_on_commit=False)
MLBase = declarative_base()


# ══════════════════════════════════════════════════════════════
# DATABASE MODELS
# ══════════════════════════════════════════════════════════════

class ModelCheckpoint(MLBase):
    """Tracks model training checkpoints."""
    __tablename__ = "model_checkpoints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(128), nullable=False, index=True)       # e.g. "resonant-seed-1b"
    epoch = Column(Integer, nullable=False, default=0)
    global_step = Column(BigInteger, nullable=False, default=0)
    checkpoint_hash = Column(String(64), nullable=False)             # SHA256 of full checkpoint
    s3_path = Column(String(512), nullable=False)                    # s3://genesis2026/model-weights/...
    total_parameters = Column(BigInteger, nullable=False)
    num_weight_shards = Column(Integer, nullable=False)
    loss = Column(Float, nullable=True)                              # training loss at this step
    eval_loss = Column(Float, nullable=True)                         # eval loss if measured
    tokens_trained = Column(BigInteger, nullable=False, default=0)   # total tokens seen
    config_json = Column(JSON, nullable=False)                       # full model config snapshot
    is_best = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class WeightShard(MLBase):
    """Individual weight shard stored on S3. One per layer group."""
    __tablename__ = "weight_shards"

    id = Column(Integer, primary_key=True, autoincrement=True)
    checkpoint_id = Column(Integer, ForeignKey("model_checkpoints.id"), nullable=False)
    model_id = Column(String(128), nullable=False, index=True)
    shard_index = Column(Integer, nullable=False)                    # 0-based shard number
    layer_range = Column(String(64), nullable=False)                 # e.g. "layers.0-5"
    s3_path = Column(String(512), nullable=False)                    # full S3 key
    shard_hash = Column(String(64), nullable=False)                  # SHA256 of shard bytes
    size_bytes = Column(BigInteger, nullable=False)
    num_parameters = Column(BigInteger, nullable=False)
    dtype = Column(String(16), default="bfloat16")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_weight_shard_model_idx", "model_id", "shard_index"),
    )


class DataShardAssignment(MLBase):
    """Tracks which data shard is assigned to which miner."""
    __tablename__ = "data_shard_assignments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(128), nullable=False, index=True)
    epoch = Column(Integer, nullable=False)
    shard_index = Column(Integer, nullable=False)
    dataset_name = Column(String(128), nullable=False)               # e.g. "fineweb-edu"
    shard_path = Column(String(512), nullable=False)                 # S3 or HF path
    num_tokens = Column(BigInteger, nullable=False)
    num_samples = Column(Integer, nullable=False)
    assigned_miner_id = Column(String(128), nullable=True)
    status = Column(String(32), default="pending")                   # pending, training, completed, failed
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class TrainingRun(MLBase):
    """Top-level training run tracking."""
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False, unique=True)
    model_id = Column(String(128), nullable=False)
    status = Column(String(32), default="initializing")              # initializing, training, paused, completed
    current_epoch = Column(Integer, default=0)
    current_global_step = Column(BigInteger, default=0)
    total_tokens_target = Column(BigInteger, nullable=False)
    tokens_completed = Column(BigInteger, default=0)
    best_loss = Column(Float, nullable=True)
    num_active_miners = Column(Integer, default=0)
    config_json = Column(JSON, nullable=False)
    loss_history = Column(JSON, default=list)                        # [{step, loss, timestamp}]
    started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))


class GradientRecord(MLBase):
    """Verified gradient submissions for audit/replay."""
    __tablename__ = "gradient_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False, index=True)
    task_id = Column(String(64), nullable=False)
    miner_id = Column(String(128), nullable=False, index=True)
    model_id = Column(String(128), nullable=False)
    epoch = Column(Integer, nullable=False)
    global_step = Column(BigInteger, nullable=False)
    gradient_hash = Column(String(64), nullable=False)
    loss_before = Column(Float, nullable=False)
    loss_after = Column(Float, nullable=False)
    samples_processed = Column(Integer, nullable=False)
    training_time_seconds = Column(Float, nullable=False)
    compression_ratio = Column(Float, nullable=False)
    verified = Column(Boolean, default=False)
    rgt_reward = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# ── Session helper ──

async def get_ml_session() -> AsyncSession:
    """Get an async session to the ML database."""
    return MLSessionLocal()


async def init_ml_tables():
    """Create all ML tables if they don't exist."""
    try:
        async with ml_engine.begin() as conn:
            await conn.run_sync(MLBase.metadata.create_all)
        logger.info("ML database tables initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ML database tables: {e}")
        raise
