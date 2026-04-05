# RG Mining Service

Decentralized LLM training orchestration for the ResonantGenesis network.

## Architecture

This module handles the **mining layer** — training task distribution, gradient compression, parameter aggregation, and miner rewards. It is separate from:

- **RG_Blockchain** (internal) — DSID logging, governance, Hash Sphere memory
- **RG_external_blockchain** — Own distributed chain: Raft consensus, P2P network, block production

## Components

| File | Purpose |
|------|---------|
| `training_task.py` | Task definitions, assignment, lifecycle, reward multipliers |
| `gradient_compressor.py` | Top-K sparsification with error feedback (100-1000x compression) |
| `param_server.py` | Staleness-aware gradient aggregation (FedAvg) |
| `genesis_seed.py` | Genesis model init, weight/data sharding, miner registration |
| `routers.py` | REST API endpoints for all mining operations |
| `main.py` | FastAPI application entry point |
| `config.py` | Service configuration |

## Miner Tiers

| Tier | Class | Role | Reward Multiplier |
|------|-------|------|-------------------|
| T1 | CLASS_F | Genesis Validator (Lighthouse) | 1.5x |
| T2 | CLASS_G | Core Contributor | 1.25x |
| T3/T4 | CLASS_H | Standard Miner | 1.0x |

## API Endpoints

```
POST /mining/genesis/initialize    — Bootstrap the training network
GET  /mining/genesis/status        — Genesis state
POST /mining/miners/register       — Register a miner
POST /mining/tasks/assign          — Get next training task
POST /mining/gradients/submit      — Submit compressed gradient
POST /mining/aggregate             — Trigger gradient aggregation
GET  /mining/param-server/stats    — Parameter server stats
GET  /mining/param-server/rewards  — Miner reward calculations
GET  /mining/health                — Health check
```

## Running

```bash
docker build -t rg-mining .
docker run -p 8000:8000 rg-mining
```

## Token Economics (Halving Schedule)

| Year | Block Reward ($RGT) |
|------|---------------------|
| 1 | 100 |
| 2 | 50 |
| 3 | 25 |
| 4 | 12.5 |

## Reward Security (Proof-of-Training)

The `/crypto/miner/credit` endpoint uses 5-layer verification to prevent unauthorized token minting:

| Layer | Check |
|-------|-------|
| Internal Key Auth | `X-Internal-Key` header must match `AUTH_INTERNAL_SERVICE_KEY` |
| Gradient Hash Required | `gradient_hash` mandatory — must actually process a gradient |
| HMAC-SHA256 Signature | Mining service signs `{gradient_hash}:{user_id}:{amount}:{timestamp}` — crypto service verifies |
| Replay Protection | Each `gradient_hash` recorded in `mining_credits` table (UNIQUE constraint) — credited exactly once |
| Reward Cap | Max **500 $RGT per credit call** — prevents inflation from any single request |

**Per-call vs per-session:** The 500 $RGT cap applies to each individual API call, not to the total session. Each mining cycle triggers one credit call (~100-150 $RGT). A 5-cycle session = 5 separate calls × ~150 = ~750 $RGT total — legitimate and within bounds. The cap blocks single-call abuse (e.g., a compromised service trying to mint 10,000 $RGT at once).

### Chain Bridge (`app/chain_bridge.py`)

The `ChainBridge.credit_miner_wallet()` method:
1. Requires a valid `gradient_hash` (no hash = skip)
2. Generates HMAC-SHA256 signature with current timestamp
3. POSTs to `CRYPTO_SERVICE_URL/crypto/miner/credit` with `X-Internal-Key` header
4. Fire-and-forget via `asyncio.create_task()` — never blocks training
