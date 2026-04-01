#!/usr/bin/env python3
"""
RG MINER CLIENT AGENT
======================

Standalone miner agent that connects to the RG Mining service via WebSocket
and runs the full mining lifecycle:

  1. Connect to Lighthouse → discover peers
  2. Connect to Mining service via WebSocket
  3. Register as miner under account email
  4. Request training tasks
  5. Train model (REAL forward/backward pass on GPU/CPU)
  6. Submit compressed gradient with correct SHA256 hash
  7. Receive reward confirmation
  8. Loop

Usage:
    # REAL TRAINING (default — requires torch):
    python3 miner_client.py --email nemesh.liubov@gmail.com --cycles 5

    # Simulated training (for testing without GPU):
    python3 miner_client.py --email nemesh.liubov@gmail.com --cycles 5 --simulate

    # With JWT auth (production):
    python3 miner_client.py --token <jwt> --cycles 5

    # Or via env var:
    AUTH_TOKEN=<jwt> python3 miner_client.py --cycles 5
"""

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

import httpx

try:
    import websockets
except ImportError:
    print("ERROR: pip3 install websockets")
    sys.exit(1)

# Real training imports (optional — falls back to simulation)
REAL_TRAINING_AVAILABLE = False
try:
    import torch
    REAL_TRAINING_AVAILABLE = True
    _DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
except ImportError:
    _DEVICE = "cpu"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("miner-agent")


# ────────────────────────────────────────────────────────────────
# TRAINING — Real or Simulated
# ────────────────────────────────────────────────────────────────

# Persistent model & data across cycles (loaded once, reused)
_model = None
_data = None
_tokenizer = None


def _init_real_training(task: dict):
    """Initialize model + data for real training. Called once."""
    global _model, _data, _tokenizer
    if _model is not None:
        return  # Already initialized

    from app.model_architecture import create_model, ResonantModelConfig
    from app.real_trainer import get_tokenizer, DataShardLoader, DEVICE

    model_id = task.get("model_id", "resonant-seed-1b")
    logger.info(f"  [REAL] Initializing model: {model_id}")
    _model, _ = create_model(model_id)
    _model = _model.to(DEVICE)
    logger.info(f"  [REAL] Model loaded on {DEVICE} ({sum(p.numel() for p in _model.parameters()):,} params)")

    # Load training data
    _tokenizer = get_tokenizer()
    loader = DataShardLoader(_tokenizer, max_seq_length=task.get("max_seq_length", 4096))
    try:
        logger.info("  [REAL] Loading training data from HuggingFace (FineWeb-Edu)...")
        _data = loader.load_from_huggingface(
            dataset_name="HuggingFaceFW/fineweb-edu",
            num_samples=max(500, task.get("batch_size", 8) * 20),
        )
    except Exception as e:
        logger.warning(f"  [REAL] HF dataset unavailable ({e}), generating synthetic data")
        _data = loader._generate_synthetic_data(
            num_samples=max(500, task.get("batch_size", 8) * 20)
        )
    logger.info(f"  [REAL] Data loaded: {_data.shape}")


def real_training(task: dict, cycle: int) -> dict:
    """
    REAL training step: forward/backward pass on actual PyTorch model.
    Uses GPU (CUDA/MPS) when available, falls back to CPU.
    """
    from app.real_trainer import RealTrainer, compress_gradients, DEVICE

    # Initialize model + data on first call
    _init_real_training(task)

    batch_size = task.get("batch_size", 8)
    learning_rate = task.get("learning_rate", 3e-4)

    # Select batch for this cycle
    start_idx = (cycle * batch_size) % len(_data)
    batch = _data[start_idx:start_idx + batch_size]
    if len(batch) < batch_size:
        batch = _data[:batch_size]

    input_ids = batch[:, :-1].to(DEVICE)
    labels = batch[:, 1:].to(DEVICE)

    # Loss BEFORE training
    _model.eval()
    with torch.no_grad():
        pre = _model(input_ids=input_ids, labels=labels)
        loss_before = pre["loss"].item()

    # Real forward + backward
    trainer = RealTrainer(_model, task, DEVICE)
    result = trainer.train_step(input_ids, labels, learning_rate)
    loss_after = result["loss"]

    # Top-K gradient compression
    compressed = compress_gradients(result["gradients"], top_k_ratio=0.01)

    return {
        "top_k_indices": compressed["top_k_indices"],
        "top_k_values": compressed["top_k_values"],
        "original_size": compressed["original_size"],
        "compressed_size": compressed["compressed_size"],
        "compression_ratio": compressed["compression_ratio"],
        "gradient_hash": compressed["gradient_hash"],
        "loss_before": round(loss_before, 6),
        "loss_after": round(loss_after, 6),
        "samples_processed": result["samples_in_batch"] * result["seq_length"],
        "training_time_seconds": round(result["training_time_seconds"], 2),
        "device": result["device"],
        "grad_norm": result["grad_norm"],
        "real_training": True,
    }


def simulate_training(task: dict, cycle: int) -> dict:
    """
    LEGACY: Simulate a training step with fake gradients.
    Only used with --simulate flag for testing without PyTorch.
    """
    num_params = 100_000
    k = 10

    indices = sorted(random.sample(range(num_params), k))
    values = [round(random.gauss(0, 0.05), 6) for _ in range(k)]

    hash_input = json.dumps({"indices": indices, "values": values}, sort_keys=True).encode()
    gradient_hash = hashlib.sha256(hash_input).hexdigest()

    base_loss = 11.0 - (cycle * 0.3)
    loss_before = base_loss + random.uniform(-0.1, 0.1)
    loss_after = loss_before - random.uniform(0.5, 1.5)
    training_time = random.uniform(20.0, 60.0)

    return {
        "top_k_indices": indices,
        "top_k_values": values,
        "original_size": num_params,
        "compressed_size": k,
        "compression_ratio": num_params / k,
        "gradient_hash": gradient_hash,
        "loss_before": round(loss_before, 4),
        "loss_after": round(max(0.5, loss_after), 4),
        "samples_processed": task.get("num_samples", 244140),
        "training_time_seconds": round(training_time, 2),
        "real_training": False,
    }


# ────────────────────────────────────────────────────────────────
# LIGHTHOUSE REGISTRATION (HTTP)
# ────────────────────────────────────────────────────────────────

def _auth_headers(token: str = None) -> dict:
    """Build auth headers from JWT token."""
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


async def register_with_lighthouse(
    lighthouse_url: str,
    miner_id: str,
    miner_class: str,
    token: str = None,
) -> dict:
    """Register this miner with the Lighthouse for peer discovery."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{lighthouse_url}/lighthouse/register",
            headers=_auth_headers(token),
            json={
                "peer_id": miner_id,
                "peer_type": "validator" if "validator" in miner_class else "miner",
                "address": "127.0.0.1",
                "p2p_port": 8600,
                "api_port": 8701,
                "node_version": "0.1.0",
                "capabilities": ["training", "gradient_submit"],
                "miner_class": miner_class,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data


async def send_lighthouse_heartbeat(lighthouse_url: str, miner_id: str, token: str = None):
    """Send heartbeat to Lighthouse."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{lighthouse_url}/lighthouse/heartbeat",
                headers=_auth_headers(token),
                json={"peer_id": miner_id},
            )
    except Exception:
        pass


# ────────────────────────────────────────────────────────────────
# GENESIS INITIALIZATION (HTTP)
# ────────────────────────────────────────────────────────────────

async def ensure_genesis_initialized(mining_url: str, miner_id: str, token: str = None) -> dict:
    """Check if genesis is initialized; if not, initialize it."""
    async with httpx.AsyncClient(timeout=10, headers=_auth_headers(token)) as client:
        # Check current status
        resp = await client.get(f"{mining_url}/mining/genesis/status")
        status = resp.json()

        if status.get("genesis", {}).get("initialized"):
            logger.info("Genesis already initialized")
            return status

        # Initialize genesis with this miner
        logger.info("Initializing Genesis Seed (resonant-seed-1b, 1B params)...")
        resp = await client.post(
            f"{mining_url}/mining/genesis/initialize",
            json={
                "model_id": "resonant-seed-1b",
                "miner_ids": [miner_id],
                "ipfs_base_url": "ipfs://",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        genesis = data.get("genesis", {})
        logger.info(
            f"  Genesis block: {genesis.get('genesis_block_hash', '?')[:16]}..."
        )
        logger.info(
            f"  Tasks created: {genesis.get('tasks_created', 0)} | "
            f"Weight shards: {genesis.get('weight_shards', 0)} | "
            f"Data shards: {genesis.get('data_shards', 0)}"
        )
        return data


# ────────────────────────────────────────────────────────────────
# WEBSOCKET MINING LOOP
# ────────────────────────────────────────────────────────────────

async def run_mining_loop(
    ws_url: str,
    miner_id: str,
    miner_class: str,
    account_email: str,
    num_cycles: int = 5,
    token: str = None,
    use_real_training: bool = True,
):
    """Connect via WebSocket and run the mining loop."""
    # Append JWT token as query param for WS auth
    ws_connect_url = ws_url
    if token:
        sep = "&" if "?" in ws_url else "?"
        ws_connect_url = f"{ws_url}{sep}token={token}"
    logger.info(f"Connecting to Mining WS: {ws_url}")

    async with websockets.connect(ws_connect_url) as ws:
        # Step 1: Register
        await ws.send(json.dumps({
            "action": "register",
            "miner_id": miner_id,
            "miner_class": miner_class,
            "account_email": account_email,
        }))

        welcome = json.loads(await ws.recv())
        if welcome.get("event") != "welcome":
            logger.error(f"Expected welcome, got: {welcome}")
            return False

        logger.info(f"  Connected as: {welcome['miner_id']} ({welcome['miner_class']})")
        logger.info(f"  Account: {welcome.get('account_email')}")
        logger.info(f"  Genesis initialized: {welcome.get('genesis_initialized')}")
        logger.info(f"  Global step: {welcome.get('param_server', {}).get('global_step', 0)}")

        results = []

        for cycle in range(1, num_cycles + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"  MINING CYCLE {cycle}/{num_cycles}")
            logger.info(f"{'='*60}")

            # Step 2: Request task
            await ws.send(json.dumps({"action": "request_task"}))
            task_msg = json.loads(await ws.recv())

            if task_msg.get("event") == "no_tasks":
                logger.warning(f"  No tasks available — queue exhausted")
                break

            if task_msg.get("event") != "task_assigned":
                logger.error(f"  Unexpected: {task_msg}")
                break

            task = task_msg["task"]
            logger.info(f"  Task assigned: {task['task_id'][:12]}...")
            logger.info(
                f"    Model: {task['model_id']} | Epoch: {task['epoch']} | "
                f"Batch: {task['batch_index']} | Samples: {task['num_samples']:,}"
            )
            logger.info(
                f"    LR: {task['learning_rate']} | BS: {task['batch_size']} | "
                f"bf16: {task['bf16']} | Deadline: {task['deadline_seconds']}s"
            )

            # Step 3: Train (real or simulated)
            use_real = use_real_training and REAL_TRAINING_AVAILABLE
            mode_str = f"REAL on {_DEVICE}" if use_real else "SIMULATED"
            logger.info(f"  Training... ({mode_str})")
            train_start = time.time()

            if use_real:
                grad_data = real_training(task, cycle)
            else:
                grad_data = simulate_training(task, cycle)
                await asyncio.sleep(0.5)

            train_elapsed = time.time() - train_start

            logger.info(
                f"  Training complete in {train_elapsed:.1f}s"
            )
            logger.info(
                f"    Loss: {grad_data['loss_before']:.4f} → {grad_data['loss_after']:.4f} "
                f"(Δ={grad_data['loss_before'] - grad_data['loss_after']:.4f})"
            )
            logger.info(
                f"    Compression: {grad_data['original_size']:,} → {grad_data['compressed_size']} "
                f"({grad_data['compression_ratio']:.0f}x)"
            )
            logger.info(f"    Hash: {grad_data['gradient_hash'][:16]}...")

            # Step 4: Submit gradient
            submission_id = f"sub-{uuid4().hex[:8]}"
            await ws.send(json.dumps({
                "action": "submit_gradient",
                "gradient": {
                    "submission_id": submission_id,
                    "task_id": task["task_id"],
                    "model_id": task["model_id"],
                    "epoch": task["epoch"],
                    "batch_index": task["batch_index"],
                    "data_shard_hash": task.get("data_shard_hash", ""),
                    "weight_shard_hash": task.get("weight_shard_hash", ""),
                    **grad_data,
                },
            }))

            # Step 5: Get response (might be gradient_accepted then aggregation_complete)
            submit_msg = json.loads(await ws.recv())

            if submit_msg.get("event") == "gradient_accepted":
                reward = submit_msg.get("reward", 0)
                total_grads = submit_msg.get("miner_total_gradients", 0)
                logger.info(f"  ✓ Gradient ACCEPTED")
                logger.info(f"    Reward: {reward} $RGT | Total gradients: {total_grads}")

                results.append({
                    "cycle": cycle,
                    "task_id": task["task_id"][:12],
                    "loss_before": grad_data["loss_before"],
                    "loss_after": grad_data["loss_after"],
                    "reward": reward,
                    "global_step": submit_msg.get("global_step", 0),
                    "status": "accepted",
                })

                # Check for aggregation broadcast
                try:
                    agg_msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    agg = json.loads(agg_msg)
                    if agg.get("event") == "aggregation_complete":
                        logger.info(
                            f"  ⚡ Aggregation complete — global step: {agg['global_step']} "
                            f"({agg['layers_merged']} layers merged)"
                        )
                except asyncio.TimeoutError:
                    pass

            elif submit_msg.get("event") == "gradient_rejected":
                logger.error(f"  ✗ Gradient REJECTED: {submit_msg.get('reason')}")
                results.append({
                    "cycle": cycle,
                    "task_id": task["task_id"][:12],
                    "status": "rejected",
                    "reason": submit_msg.get("reason"),
                })
            else:
                logger.error(f"  Unexpected response: {submit_msg}")
                results.append({"cycle": cycle, "status": "error", "msg": str(submit_msg)})

            # Heartbeat
            await ws.send(json.dumps({"action": "heartbeat"}))
            hb = json.loads(await ws.recv())

        # Final status check
        await ws.send(json.dumps({"action": "get_status"}))
        final_status = json.loads(await ws.recv())

        await ws.send(json.dumps({"action": "get_rewards"}))
        final_rewards = json.loads(await ws.recv())

    return results, final_status, final_rewards


# ────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="RG Miner Agent")
    parser.add_argument("--email", default="nemesh.liubov@gmail.com", help="Account email")
    parser.add_argument("--miner-id", default=None, help="Miner ID (auto-generated if not set)")
    parser.add_argument("--miner-class", default="validator_miner", help="Miner class")
    parser.add_argument("--cycles", type=int, default=5, help="Number of mining cycles")
    parser.add_argument("--lighthouse-url", default="http://localhost:8700", help="Lighthouse URL")
    parser.add_argument("--mining-url", default="http://localhost:8701", help="Mining service URL")
    parser.add_argument("--mining-ws", default="ws://localhost:8701/ws/mining", help="Mining WS URL")
    parser.add_argument("--token", default=None, help="JWT auth token (or set AUTH_TOKEN env var)")
    parser.add_argument("--simulate", action="store_true", help="Use simulated training (no GPU needed)")
    parser.add_argument("--model", default="resonant-seed-1b", help="Model tier from registry")
    args = parser.parse_args()

    # Token from CLI arg or env var
    token = args.token or os.getenv("AUTH_TOKEN", "")
    auth_mode = "JWT" if token else "dev (no auth)"
    use_real = not args.simulate and REAL_TRAINING_AVAILABLE
    training_mode = f"REAL ({_DEVICE.upper()})" if use_real else "SIMULATED"

    miner_id = args.miner_id or f"miner-louie-{uuid4().hex[:6]}"

    print()
    print("=" * 60)
    print("  RG MINER AGENT — ResonantGenesis Network")
    print("=" * 60)
    print(f"  Miner ID:    {miner_id}")
    print(f"  Class:       {args.miner_class}")
    print(f"  Account:     {args.email}")
    print(f"  Cycles:      {args.cycles}")
    print(f"  Lighthouse:  {args.lighthouse_url}")
    print(f"  Mining API:  {args.mining_url}")
    print(f"  Mining WS:   {args.mining_ws}")
    print(f"  Auth:        {auth_mode}")
    print(f"  Training:    {training_mode}")
    print(f"  Model:       {args.model}")
    print(f"  PyTorch:     {'✓ ' + torch.__version__ if REAL_TRAINING_AVAILABLE else '✗ not installed (simulating)'}")
    if REAL_TRAINING_AVAILABLE and _DEVICE != 'cpu':
        if _DEVICE == 'cuda':
            print(f"  GPU:         {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB)")
        elif _DEVICE == 'mps':
            print(f"  GPU:         Apple Silicon (MPS)")
    print("=" * 60)
    print()

    # Phase 1: Lighthouse registration
    logger.info("[PHASE 1] Registering with Lighthouse...")
    try:
        lh_result = await register_with_lighthouse(
            args.lighthouse_url, miner_id, args.miner_class, token=token,
        )
        bootstrap_peers = lh_result.get("bootstrap_peers", [])
        logger.info(f"  Registered with Lighthouse — {len(bootstrap_peers)} bootstrap peers discovered")
        for p in bootstrap_peers[:5]:
            logger.info(f"    → {p['peer_id']} ({p['peer_type']}) at {p['address']}:{p['p2p_port']}")
    except Exception as e:
        logger.error(f"  Lighthouse registration failed: {e}")
        logger.info("  Continuing without Lighthouse...")

    # Phase 2: Ensure genesis is initialized
    logger.info("\n[PHASE 2] Checking Genesis status...")
    try:
        await ensure_genesis_initialized(args.mining_url, miner_id, token=token)
    except Exception as e:
        logger.error(f"  Genesis init failed: {e}")
        return

    # Phase 3: WebSocket mining loop
    logger.info(f"\n[PHASE 3] Starting WebSocket mining loop ({args.cycles} cycles)...")
    try:
        results, final_status, final_rewards = await run_mining_loop(
            ws_url=args.mining_ws,
            miner_id=miner_id,
            miner_class=args.miner_class,
            account_email=args.email,
            num_cycles=args.cycles,
            token=token,
            use_real_training=use_real,
        )
    except Exception as e:
        logger.error(f"  Mining loop failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Phase 4: Lighthouse heartbeat
    logger.info("\n[PHASE 4] Sending final Lighthouse heartbeat...")
    await send_lighthouse_heartbeat(args.lighthouse_url, miner_id, token=token)

    # ─── FINAL REPORT ───
    print()
    print("=" * 60)
    print("  MINING SESSION REPORT")
    print("=" * 60)
    print(f"  Miner:       {miner_id}")
    print(f"  Account:     {args.email}")
    print()

    accepted = [r for r in results if r.get("status") == "accepted"]
    rejected = [r for r in results if r.get("status") == "rejected"]
    total_reward = sum(r.get("reward", 0) for r in accepted)

    print(f"  Cycles Run:     {len(results)}")
    print(f"  Accepted:       {len(accepted)}")
    print(f"  Rejected:       {len(rejected)}")
    print(f"  Total Reward:   {total_reward} $RGT")
    print()

    print("  ┌────────┬──────────────────┬───────────┬──────────┬─────────┐")
    print("  │ Cycle  │ Task             │ Loss      │ Reward   │ Status  │")
    print("  ├────────┼──────────────────┼───────────┼──────────┼─────────┤")
    for r in results:
        cycle = r.get("cycle", "?")
        task = r.get("task_id", "?")
        loss = f"{r.get('loss_after', 0):.4f}" if "loss_after" in r else "—"
        reward = f"{r.get('reward', 0):.1f}" if "reward" in r else "—"
        status = r.get("status", "?")
        icon = "✓" if status == "accepted" else "✗"
        print(f"  │ {cycle:>5}  │ {task:<16} │ {loss:>9} │ {reward:>7}  │ {icon} {status:<5}  │")
    print("  └────────┴──────────────────┴───────────┴──────────┴─────────┘")
    print()

    # Final miner state
    miner_state = final_status.get("miner", {}) if final_status.get("event") == "status" else {}
    if miner_state:
        print(f"  Miner State:")
        print(f"    Class:             {miner_state.get('miner_class')}")
        print(f"    Trust Score:       {miner_state.get('trust_score')}")
        print(f"    Total Gradients:   {miner_state.get('total_gradients_submitted')}")
        print(f"    Total Samples:     {miner_state.get('total_samples_trained'):,}")
        print(f"    Avg Loss:          {miner_state.get('cumulative_loss', 0) / max(1, miner_state.get('total_gradients_submitted', 1)):.4f}")
        print(f"    Avg Training Time: {miner_state.get('average_training_time', 0):.1f}s")
        print(f"    Staleness:         {miner_state.get('staleness')}")
        print()

    # Rewards
    rwd = final_rewards.get("miner_reward", 0) if final_rewards.get("event") == "rewards" else 0
    print(f"  Total $RGT Earned: {rwd}")
    print()

    all_ok = len(accepted) == len(results) and len(results) == args.cycles
    if all_ok:
        print(f"  ✅ ALL {args.cycles} CYCLES COMPLETED SUCCESSFULLY")
    else:
        print(f"  ⚠️  {len(accepted)}/{args.cycles} cycles completed successfully")
    print("=" * 60)
    print()

    return all_ok


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
