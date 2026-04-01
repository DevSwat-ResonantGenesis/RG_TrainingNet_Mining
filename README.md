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
