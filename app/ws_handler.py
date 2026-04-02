"""
MINING WEBSOCKET HANDLER
=========================

Real-time WebSocket endpoint for miner agents.
Handles the full mining lifecycle over a persistent connection:
  connect → authenticate → receive tasks → submit gradients → get rewards

Protocol messages (JSON):
  Client → Server:
    {"action": "register", "miner_id": "...", "miner_class": "...", "account_email": "..."}
    {"action": "heartbeat"}
    {"action": "request_task"}
    {"action": "submit_gradient", "task_id": "...", "gradient": {...}}
    
  Server → Client:
    {"event": "welcome", "miner_id": "...", "genesis_status": {...}}
    {"event": "task_assigned", "task": {...}}
    {"event": "gradient_accepted", "submission_id": "...", "reward": ...}
    {"event": "gradient_rejected", "reason": "..."}
    {"event": "aggregation_complete", "global_step": ..., "loss": ...}
    {"event": "error", "message": "..."}
    {"event": "heartbeat_ack", "timestamp": "..."}
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Set
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

from .genesis_seed import genesis_initializer
from .param_server import param_server
from .training_task import task_manager, GradientSubmission
from .gradient_compressor import CompressedGradient, verify_gradient_hash
from .auth_middleware import get_ws_user, check_rate_limit, AuthenticatedUser
from .chain_bridge import chain_bridge
from .shard_manager import shard_manager
from .sharded_param_server import sharded_param_server

logger = logging.getLogger(__name__)


class MiningWSManager:
    """Manages WebSocket connections for miner agents."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}  # miner_id → ws
        self.miner_metadata: Dict[str, Dict] = {}  # miner_id → metadata
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket, miner_id: str, metadata: Dict = None):
        await ws.accept()
        async with self._lock:
            self.active_connections[miner_id] = ws
            self.miner_metadata[miner_id] = metadata or {}
        logger.info(f"WS: Miner {miner_id} connected ({len(self.active_connections)} total)")

    async def disconnect(self, miner_id: str):
        async with self._lock:
            self.active_connections.pop(miner_id, None)
            self.miner_metadata.pop(miner_id, None)
        logger.info(f"WS: Miner {miner_id} disconnected ({len(self.active_connections)} total)")

    async def send_to_miner(self, miner_id: str, data: Dict):
        ws = self.active_connections.get(miner_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception as e:
                logger.warning(f"WS send error to {miner_id}: {e}")
                await self.disconnect(miner_id)

    async def broadcast(self, data: Dict, exclude: str = None):
        for miner_id, ws in list(self.active_connections.items()):
            if miner_id != exclude:
                try:
                    await ws.send_json(data)
                except:
                    await self.disconnect(miner_id)

    @property
    def connected_count(self) -> int:
        return len(self.active_connections)


# Global WebSocket manager
ws_manager = MiningWSManager()


async def handle_mining_ws(ws: WebSocket):
    """Handle a miner WebSocket connection through its full lifecycle."""
    miner_id = None

    try:
        # ── Auth: verify JWT before accepting ──
        auth_user = await get_ws_user(ws)
        if auth_user is None:
            await ws.accept()
            await ws.send_json({"event": "error", "message": "Authentication required — pass ?token=<jwt> or Authorization header"})
            await ws.close(code=4001)
            return

        # Wait for first message (must be register)
        await ws.accept()
        raw = await asyncio.wait_for(ws.receive_json(), timeout=30)

        if raw.get("action") != "register":
            await ws.send_json({"event": "error", "message": "First message must be 'register'"})
            await ws.close()
            return

        # Register miner — email comes from verified JWT, not self-reported
        miner_id = raw.get("miner_id", f"miner-{uuid4().hex[:8]}")
        miner_class = raw.get("miner_class", "miner")
        account_email = auth_user.email or raw.get("account_email", "")

        # Rate limit WS connections per user
        check_rate_limit(auth_user.user_id or miner_id, "register")

        # Register with param server
        miner_state = param_server.register_miner(miner_id, miner_class)

        # Track WS connection
        async with ws_manager._lock:
            ws_manager.active_connections[miner_id] = ws
            ws_manager.miner_metadata[miner_id] = {
                "account_email": account_email,
                "user_id": auth_user.user_id,
                "org_id": auth_user.org_id,
                "role": auth_user.role,
                "miner_class": miner_class,
                "connected_at": datetime.now(timezone.utc).isoformat(),
            }

        logger.info(f"WS: Miner {miner_id} registered (class={miner_class}, user={auth_user.user_id}, email={account_email})")

        # Check shard assignment for this miner
        shard_assignment = shard_manager.miner_assignments.get(miner_id)
        shard_info = None
        if shard_assignment:
            shard_info = {
                "pipeline_group_id": shard_assignment.pipeline_group_id,
                "stage_index": shard_assignment.stage_index,
                "num_stages": shard_assignment.num_stages,
                "layer_start": shard_assignment.layer_start,
                "layer_end": shard_assignment.layer_end,
                "has_embedding": shard_assignment.has_embedding,
                "has_lm_head": shard_assignment.has_lm_head,
                "upstream_miner_id": shard_assignment.upstream_miner_id,
                "downstream_miner_id": shard_assignment.downstream_miner_id,
            }
            logger.info(
                f"WS: Miner {miner_id} is pipeline stage "
                f"{shard_assignment.stage_index}/{shard_assignment.num_stages} "
                f"in group {shard_assignment.pipeline_group_id[:12]}..."
            )

        # Send welcome
        await ws.send_json({
            "event": "welcome",
            "miner_id": miner_id,
            "miner_class": miner_class,
            "account_email": account_email,
            "genesis_initialized": genesis_initializer.state.initialized,
            "genesis_status": genesis_initializer.get_status() if genesis_initializer.state.initialized else None,
            "param_server": param_server.get_stats(),
            "connected_miners": ws_manager.connected_count,
            "shard_assignment": shard_info,
        })

        # Main message loop
        while True:
            data = await ws.receive_json()
            action = data.get("action", "")

            if action == "heartbeat":
                hb_response = {
                    "event": "heartbeat_ack",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "global_step": param_server.global_step,
                }
                # Include shard health if this miner is in a pipeline
                if shard_assignment:
                    group = shard_manager.get_pipeline_group(shard_assignment.pipeline_group_id)
                    hb_response["shard_health"] = {
                        "pipeline_status": group.status.value if group else "unknown",
                        "stage_index": shard_assignment.stage_index,
                        "is_ready": shard_assignment.miner_id in shard_manager._ready_miners if hasattr(shard_manager, '_ready_miners') else False,
                    }
                await ws.send_json(hb_response)

            elif action == "request_task":
                task = task_manager.assign_task(miner_id)
                if task:
                    await ws.send_json({
                        "event": "task_assigned",
                        "task": task.to_dict(),
                    })
                    logger.info(f"WS: Assigned task {task.task_id} to {miner_id}")
                else:
                    await ws.send_json({
                        "event": "no_tasks",
                        "message": "No tasks available in queue",
                        "queue_stats": task_manager.get_stats(),
                    })

            elif action == "submit_gradient":
                result = await _handle_gradient_submit(miner_id, data, ws)

            elif action == "get_status":
                miner = param_server.miners.get(miner_id)
                await ws.send_json({
                    "event": "status",
                    "miner": miner.to_dict() if miner else None,
                    "param_server": param_server.get_stats(),
                    "tasks": task_manager.get_stats(),
                })

            elif action == "get_rewards":
                rewards = param_server.get_miner_rewards(year=data.get("year", 1))
                await ws.send_json({
                    "event": "rewards",
                    "miner_reward": rewards.get(miner_id, 0),
                    "all_rewards": rewards,
                })

            else:
                await ws.send_json({
                    "event": "error",
                    "message": f"Unknown action: {action}",
                })

    except WebSocketDisconnect:
        logger.info(f"WS: Miner {miner_id or 'unknown'} disconnected")
    except asyncio.TimeoutError:
        logger.warning(f"WS: Timeout waiting for register from new connection")
    except Exception as e:
        logger.error(f"WS: Error for miner {miner_id or 'unknown'}: {e}")
        try:
            await ws.send_json({"event": "error", "message": str(e)})
        except:
            pass
    finally:
        if miner_id:
            await ws_manager.disconnect(miner_id)
            # Notify ShardManager — may mark pipeline group as DEGRADED
            affected_group = shard_manager.handle_miner_disconnect(miner_id)
            if affected_group:
                logger.warning(
                    f"WS: Miner {miner_id} disconnect degraded pipeline group {affected_group}"
                )
                # Notify remaining miners in the group
                group = shard_manager.get_pipeline_group(affected_group)
                if group:
                    for stage in group.stages.values():
                        if stage.miner_id != miner_id:
                            await ws_manager.send_to_miner(stage.miner_id, {
                                "event": "pipeline_degraded",
                                "pipeline_group_id": affected_group,
                                "disconnected_miner": miner_id,
                                "disconnected_stage": stage.stage_index,
                                "group_status": group.status.value,
                            })


async def _handle_gradient_submit(miner_id: str, data: Dict, ws: WebSocket):
    """Process a gradient submission from the WebSocket."""
    try:
        grad = data.get("gradient", {})

        # Build submission
        submission = GradientSubmission(
            submission_id=grad.get("submission_id", str(uuid4())),
            task_id=grad["task_id"],
            miner_id=miner_id,
            model_id=grad.get("model_id", "resonant-seed-1b"),
            epoch=grad.get("epoch", 0),
            batch_index=grad.get("batch_index", 0),
            top_k_indices=grad["top_k_indices"],
            top_k_values=grad["top_k_values"],
            original_size=grad["original_size"],
            compressed_size=grad["compressed_size"],
            compression_ratio=grad.get("compression_ratio", 0),
            loss_before=grad["loss_before"],
            loss_after=grad["loss_after"],
            samples_processed=grad["samples_processed"],
            training_time_seconds=grad.get("training_time_seconds", 0),
            gradient_hash=grad["gradient_hash"],
            data_shard_hash=grad.get("data_shard_hash", ""),
            weight_shard_hash=grad.get("weight_shard_hash", ""),
        )

        # Build compressed gradient
        compressed = [CompressedGradient(
            indices=grad["top_k_indices"],
            values=grad["top_k_values"],
            original_size=grad["original_size"],
            k=grad["compressed_size"],
            gradient_hash=grad["gradient_hash"],
            layer_name=f"layer_{grad.get('batch_index', 0)}",
        )]

        # Verify via param server first
        ps_accepted = param_server.receive_gradient(submission, compressed)
        if not ps_accepted:
            await ws.send_json({
                "event": "gradient_rejected",
                "reason": "Parameter server rejected (hash mismatch, staleness, or unregistered)",
            })
            return

        # Record in task manager
        task_manager.submit_result(submission)

        # Calculate reward
        miner = param_server.miners.get(miner_id)
        from .training_task import get_miner_reward
        reward = get_miner_reward(miner.miner_class if miner else "miner", year=1)

        await ws.send_json({
            "event": "gradient_accepted",
            "submission_id": submission.submission_id,
            "task_id": submission.task_id,
            "loss_after": submission.loss_after,
            "samples_processed": submission.samples_processed,
            "reward": reward,
            "global_step": param_server.global_step,
            "miner_total_gradients": miner.total_gradients_submitted if miner else 0,
        })

        logger.info(f"WS: Gradient accepted from {miner_id} (loss={submission.loss_after:.4f}, reward={reward})")

        # Credit miner's wallet via Crypto service (fire-and-forget)
        meta = ws_manager.miner_metadata.get(miner_id, {})
        asyncio.create_task(chain_bridge.credit_miner_wallet(
            user_id=meta.get("user_id"),
            email=meta.get("account_email"),
            rgt_amount=reward,
            samples_processed=submission.samples_processed,
            trust_score=miner.trust_score if miner else 1.0,
            tier=miner.miner_class if miner else "miner",
            gradient_hash=submission.gradient_hash,
            task_id=submission.task_id,
            global_step=param_server.global_step,
        ))

        # Record on external blockchain (fire-and-forget)
        asyncio.create_task(chain_bridge.record_gradient_on_chain(
            miner_id=miner_id,
            task_id=submission.task_id,
            gradient_hash=submission.gradient_hash,
            loss_value=submission.loss_after,
            samples_processed=submission.samples_processed,
            reward_amount=reward,
            submission_id=submission.submission_id,
            model_id=submission.model_id,
            global_step=param_server.global_step,
        ))

        # Auto-aggregate if enough gradients pending
        if len(param_server.pending_gradients) >= param_server.MIN_MINERS_PER_ROUND:
            merged = param_server.aggregate()
            if merged:
                # Broadcast aggregation to all connected miners
                await ws_manager.broadcast({
                    "event": "aggregation_complete",
                    "global_step": param_server.global_step,
                    "layers_merged": len(merged),
                })
                # Record aggregation on-chain
                asyncio.create_task(chain_bridge.record_aggregation_on_chain(
                    global_step=param_server.global_step,
                    layers_merged=len(merged),
                    miners_contributed=len(param_server.miners),
                ))

    except KeyError as e:
        await ws.send_json({"event": "error", "message": f"Missing field: {e}"})
    except Exception as e:
        await ws.send_json({"event": "error", "message": f"Gradient submit error: {e}"})
