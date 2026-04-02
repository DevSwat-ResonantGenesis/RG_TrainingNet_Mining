# Decentralized Sharded Training Architecture
## Scaling ResonantGenesis from 1B to 405B Parameters

**Version:** 1.0  
**Date:** 2026-04-01  
**Status:** Design — Pre-Implementation  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture (Phase 2)](#2-current-architecture-phase-2)
3. [The Scaling Problem](#3-the-scaling-problem)
4. [Model Sharding Design](#4-model-sharding-design)
5. [Pipeline Parallelism](#5-pipeline-parallelism)
6. [Shard Manager](#6-shard-manager)
7. [Activation Router](#7-activation-router)
8. [Sharded Parameter Server](#8-sharded-parameter-server)
9. [Mixture of Experts (MoE) Architecture](#9-mixture-of-experts-moe-architecture)
10. [Lighthouse Enhancements](#10-lighthouse-enhancements)
11. [Miner App Enhancements](#11-miner-app-enhancements)
12. [Weight Storage & Checkpointing](#12-weight-storage--checkpointing)
13. [Fault Tolerance](#13-fault-tolerance)
14. [Security Considerations](#14-security-considerations)
15. [Phased Rollout Plan](#15-phased-rollout-plan)
16. [New Files & Modules](#16-new-files--modules)
17. [API Changes](#17-api-changes)
18. [Open Questions](#18-open-questions)

---

## 1. Executive Summary

ResonantGenesis trains LLMs via a decentralized miner network. Currently (Phase 2), each miner holds a **full copy** of the 1B-parameter seed model in local memory. This architecture cannot scale beyond ~13B parameters because no single consumer machine can hold the full model.

To train 70B–405B models, we need **model sharding** — splitting the model across multiple miners who coordinate via **pipeline parallelism**. This document specifies the complete architecture for decentralized sharded training.

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Parallelism strategy | Pipeline parallelism | Minimizes inter-node communication (only activations between stages, not full model state) |
| Shard granularity | Layer groups | Transformers have natural layer boundaries; avoids splitting tensor operations across nodes |
| Redundancy model | Multiple pipeline groups | Same layers replicated across groups for fault tolerance |
| Large model strategy | Mixture of Experts (MoE) | 405B total params but only ~50B active per token — 8x compute reduction |
| Activation transport | P2P WebSocket | Already used for gradient transport; low-latency, bidirectional |
| Coordination | Lighthouse | Already tracks peers; extended to manage pipeline topology |

---

## 2. Current Architecture (Phase 2)

```
┌─────────────────────────────────────────────────────────────────┐
│                     CURRENT FLOW (1B model)                     │
│                                                                 │
│  Miner A ──┐                                                    │
│  (full 1B) │   compressed gradients    ┌────────────────────┐   │
│            ├──────────────────────────▶│  Parameter Server  │   │
│  Miner B ──┤   (0.01% of full size)   │  (single instance) │   │
│  (full 1B) │                           │  merges all grads  │   │
│            ├──────────────────────────▶│  advances step     │   │
│  Miner C ──┤                           └────────────────────┘   │
│  (full 1B) │                                                    │
│            │                                                    │
│  Each miner:                                                    │
│    1. Downloads full model (~2GB)                               │
│    2. Receives training task from WS                            │
│    3. Trains on local GPU (full forward + backward pass)        │
│    4. Compresses gradients (top-k 0.01%)                        │
│    5. Sends compressed gradients via WS                         │
│    6. Param server aggregates, miner syncs                      │
└─────────────────────────────────────────────────────────────────┘
```

### What Works

- Full model fits in 8GB+ VRAM (1B params ≈ 2GB in fp16)
- Simple architecture: each miner is independent
- Gradient compression (10000x) keeps bandwidth manageable
- Staleness-aware aggregation handles async miners

### What Doesn't Scale

| Model | Size (fp16) | Fits in 24GB GPU? | Fits in 80GB A100? |
|---|---|---|---|
| resonant-seed-1b | 2 GB | Yes | Yes |
| resonant-v1-7b | 14 GB | Yes | Yes |
| resonant-v1-13b | 26 GB | **No** | Yes |
| resonant-v2-70b | 140 GB | **No** | **No** |
| resonant-frontier-405b | 810 GB | **No** | **No** |

---

## 3. The Scaling Problem

### Memory: Can't Fit the Model

A 405B parameter model requires ~810GB in fp16 just for parameters. Even with quantization (int8 ≈ 405GB, int4 ≈ 200GB), no single consumer GPU can hold it. The model MUST be split.

### Compute: Single GPU Too Slow

Even if a single GPU could hold 405B params, a forward+backward pass would take hours per batch. Training requires parallelism across many GPUs.

### Bandwidth: Can't Send Full Gradients

Full gradient for 405B = 810GB. Even compressed 10000x = 81MB per submission. With 500 miners, that's ~40GB per aggregation round hitting the parameter server. Must shard the parameter server too.

### The Solution: Shard Everything

```
Model → sharded across miners (pipeline parallelism)
Param Server → sharded across validator nodes
Gradients → each miner only sends their shard's gradients
Activations → forwarded between pipeline stages via P2P
```

---

## 4. Model Sharding Design

### Layer-Group Sharding

Transformer models are stacks of identical layers. We shard at layer boundaries:

```
resonant-frontier-405b: 126 layers

4-stage pipeline (32 layers per stage):
  Stage 0: Embedding + Layers 0-31    (~100GB)
  Stage 1: Layers 32-63               (~100GB)
  Stage 2: Layers 64-95               (~100GB)
  Stage 3: Layers 96-125 + LM Head    (~100GB)

8-stage pipeline (16 layers per stage):
  Stage 0: Embedding + Layers 0-15    (~50GB)
  Stage 1: Layers 16-31               (~50GB)
  ...
  Stage 7: Layers 112-125 + LM Head   (~50GB)
```

### Shard Assignment Rules

1. **Embedding layer** always assigned to Stage 0 (first stage)
2. **LM Head** always assigned to last stage (tied to embedding weights)
3. **Layers divided evenly** across stages, respecting layer boundaries
4. **No splitting within a layer** — attention + FFN + normalization stay together
5. **MoE expert blocks** are NOT split — full expert stays on one stage

### Data Structures

```python
@dataclass
class ShardAssignment:
    """Describes which layers a miner is responsible for."""
    miner_id: str
    pipeline_group_id: str       # Which redundancy group
    stage_index: int             # 0, 1, 2, ... N-1
    num_stages: int              # Total stages in pipeline
    layer_start: int             # First layer index (inclusive)
    layer_end: int               # Last layer index (exclusive)
    has_embedding: bool          # True for stage 0
    has_lm_head: bool            # True for last stage
    upstream_miner_id: Optional[str]    # Who sends us activations
    downstream_miner_id: Optional[str]  # Who we send activations to
    shard_size_bytes: int        # Expected memory footprint

@dataclass
class PipelineGroup:
    """A group of miners forming one complete pipeline."""
    group_id: str
    model_id: str
    num_stages: int
    stages: Dict[int, ShardAssignment]  # stage_index → assignment
    status: str  # "forming", "ready", "training", "degraded"
```

### How Shards Are Created

```python
def compute_shard_assignments(
    model_id: str,
    available_miners: List[MinerCapability],
    target_stages: int = 4,
) -> List[PipelineGroup]:
    """
    Given a model and available miners, create pipeline groups.
    
    Algorithm:
    1. Get model config (num_layers, param count)
    2. Calculate layers_per_stage = ceil(num_layers / target_stages)
    3. Sort miners by VRAM (largest first)
    4. Greedily assign miners to stages, forming groups
    5. Each group must have exactly target_stages miners
    6. Remaining miners form additional groups (redundancy)
    """
```

---

## 5. Pipeline Parallelism

### Forward Pass

```
Batch of tokens
    │
    ▼
┌─────────┐   hidden_states    ┌─────────┐   hidden_states    ┌─────────┐
│ Stage 0  │──────────────────▶│ Stage 1  │──────────────────▶│ Stage 2  │ ...
│ Embed +  │   [batch, seq,    │ Layers   │   [batch, seq,    │ Layers   │
│ L0-L31   │    hidden_size]   │ L32-L63  │    hidden_size]   │ L64-L95  │
└─────────┘                    └─────────┘                    └─────────┘
```

**Activation size per transfer:**
- `batch_size × seq_length × hidden_size × dtype_bytes`
- For 405B: `8 × 32768 × 16384 × 2 = 8.6 GB` per forward transfer
- With microbatching (batch=1): `1 × 32768 × 16384 × 2 = 1.07 GB`

### Backward Pass (Reverse Direction)

```
                                                                   Loss
                                                                    │
┌─────────┐   grad_hidden      ┌─────────┐   grad_hidden      ┌─────────┐
│ Stage 0  │◀──────────────────│ Stage 1  │◀──────────────────│ Stage 2  │ ...
│          │                    │          │                    │          │
│ compute  │                    │ compute  │                    │ compute  │
│ local    │                    │ local    │                    │ local    │
│ grads    │                    │ grads    │                    │ grads    │
└────┬─────┘                    └────┬─────┘                    └────┬─────┘
     │                               │                               │
     ▼                               ▼                               ▼
  Shard PS 0                    Shard PS 1                    Shard PS 2
```

### Microbatch Scheduling (GPipe-style)

To avoid pipeline bubbles (stages sitting idle), we split each batch into microbatches:

```
Time →
Stage 0: [F0][F1][F2][F3]          [B3][B2][B1][B0]
Stage 1:     [F0][F1][F2][F3]      [B3][B2][B1][B0]
Stage 2:         [F0][F1][F2][F3]  [B3][B2][B1][B0]
Stage 3:             [F0][F1][F2][F3][B3][B2][B1][B0]

F = forward microbatch, B = backward microbatch
Pipeline bubble = empty slots at start/end
Efficiency = (num_microbatches) / (num_microbatches + num_stages - 1)
```

With 4 microbatches and 4 stages: 4/7 = 57% efficiency  
With 16 microbatches and 4 stages: 16/19 = 84% efficiency  
With 32 microbatches and 4 stages: 32/35 = 91% efficiency  

### Alternative: 1F1B Schedule (More Memory Efficient)

```
Time →
Stage 0: [F0][F1][F2][F3][B0][F4][B1][F5][B2][F6][B3]...
Stage 1:     [F0][F1][F2][B0][F3][B1][F4][B2][F5][B3]...
```

Interleaves forward and backward to reduce peak activation memory. Recommended for large models.

---

## 6. Shard Manager

**Location:** `RG_Mining/app/shard_manager.py`

### Responsibilities

1. **Query Lighthouse** for available miners and their capabilities
2. **Compute pipeline groups** — assign miners to stages
3. **Track pipeline topology** — who connects to whom
4. **Handle group formation** — wait for all stages to be filled before starting
5. **Handle failures** — reassign shards when a miner drops
6. **Expose API** for miners to query their shard assignment

### Core Class

```python
class ShardManager:
    """Manages model sharding across the decentralized miner network."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model_config = get_model_config(model_id)
        self.pipeline_groups: Dict[str, PipelineGroup] = {}
        self.miner_assignments: Dict[str, ShardAssignment] = {}
    
    def compute_optimal_stages(self, num_miners: int, avg_vram_gb: float) -> int:
        """Determine how many pipeline stages based on model + available compute."""
        model_size_gb = self.model_config["num_parameters"] * 2 / 1e9  # fp16
        min_stages = math.ceil(model_size_gb / avg_vram_gb)
        max_stages = self.model_config["num_layers"] // 4  # min 4 layers per stage
        # Target: enough groups for redundancy (at least 2)
        miners_per_group = min_stages
        num_groups = num_miners // miners_per_group
        if num_groups < 2:
            min_stages = num_miners // 2  # sacrifice stages for redundancy
        return max(min_stages, 2)
    
    def form_pipeline_groups(self, miners: List[MinerCapability]) -> List[PipelineGroup]:
        """Create pipeline groups from available miners."""
        ...
    
    def get_assignment(self, miner_id: str) -> Optional[ShardAssignment]:
        """Get a miner's shard assignment."""
        return self.miner_assignments.get(miner_id)
    
    def handle_miner_disconnect(self, miner_id: str) -> Optional[str]:
        """Handle a miner leaving. Returns affected group_id or None."""
        ...
    
    def reassign_shard(self, group_id: str, stage_index: int, new_miner_id: str):
        """Reassign a shard to a different miner (failover)."""
        ...
```

### API Endpoints (added to `routers.py`)

```
GET  /mining/shards/assignments          → list all current assignments
GET  /mining/shards/my-assignment        → miner queries their own shard
GET  /mining/shards/pipeline-groups      → list all pipeline groups
POST /mining/shards/form-groups          → trigger group formation
POST /mining/shards/report-ready         → miner reports shard loaded
```

---

## 7. Activation Router

**Location:** `RG_Mining/app/activation_router.py`

### Responsibilities

1. **Route activations** between pipeline stages during forward pass
2. **Route gradients** between pipeline stages during backward pass
3. **Manage microbatch scheduling** (GPipe or 1F1B)
4. **Handle compression** — activations can be large, may need fp16→int8 or selective transmission
5. **Handle failures** — detect dead upstream/downstream, trigger reassignment

### Transport Protocol

Activations are transferred via P2P WebSocket (same infrastructure as gradient submission):

```python
# Activation message format (forward pass)
{
    "type": "activation_forward",
    "pipeline_group_id": "pg-abc123",
    "microbatch_index": 0,
    "stage_from": 0,
    "stage_to": 1,
    "shape": [1, 32768, 16384],      # [batch, seq, hidden]
    "dtype": "float16",
    "compressed": true,                # int8 quantized for transfer
    "data": "<base64-encoded tensor>",
    "forward_pass_id": "fp-xyz789",
}

# Gradient message format (backward pass)
{
    "type": "activation_backward",
    "pipeline_group_id": "pg-abc123",
    "microbatch_index": 0,
    "stage_from": 1,
    "stage_to": 0,
    "shape": [1, 32768, 16384],
    "dtype": "float16",
    "data": "<base64-encoded gradient tensor>",
    "forward_pass_id": "fp-xyz789",
}
```

### Activation Compression

Raw activation size for 405B: `batch × seq × hidden = 1 × 32768 × 16384 × 2 bytes = 1.07 GB`

Compression strategies:
1. **Microbatching** — batch_size=1 per microbatch (1.07 GB vs 8.6 GB)
2. **INT8 quantization** — reduces to ~537 MB per transfer
3. **Activation checkpointing** — recompute activations during backward (saves memory, costs compute)
4. **Sequence parallelism** — split sequence dimension (advanced, Phase 6+)

---

## 8. Sharded Parameter Server

**Location:** Updates to `RG_Mining/app/param_server.py`

### Current: Single Parameter Server

```python
class ParameterServer:
    def __init__(self):
        self.pending_gradients: List[Tuple[GradientSubmission, List[CompressedGradient]]] = []
        # All gradients for ALL layers land here
```

### Future: Sharded Parameter Server

```python
class ShardedParameterServer:
    """Each instance handles a subset of model layers."""
    
    def __init__(self, shard_index: int, num_shards: int, layer_start: int, layer_end: int):
        self.shard_index = shard_index
        self.num_shards = num_shards
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.pending_gradients = []  # Only for our layer range
        self.global_step = 0
    
    def receive_gradient(self, submission, compressed_layers):
        """Accept gradients only for layers in our range."""
        relevant_layers = [
            cg for cg in compressed_layers
            if self._is_our_layer(cg.layer_name)
        ]
        if not relevant_layers:
            return False  # Not our shard
        # ... aggregate only our layers
    
    def aggregate(self):
        """Aggregate pending gradients for our layer range only."""
        # Same staleness-aware weighted averaging, but scoped to our layers
        ...
    
    def sync_global_step(self, other_shards: List["ShardedParameterServer"]):
        """All shards must agree on global step before advancing."""
        # Requires consensus across shards (use Raft from external blockchain)
        ...
```

### Backward Compatibility

For Phase 2-4 (small models), the single `ParameterServer` continues to work. The sharded version activates automatically when:
- `model_config.num_weight_shards > 1`
- `num_registered_miners >= model_config.min_miners`

---

## 9. Mixture of Experts (MoE) Architecture

**Location:** `RG_Mining/app/model_architecture.py` (new classes)

### Why MoE for 405B?

| Property | Dense 405B | MoE 405B (8/64 experts) |
|---|---|---|
| Total parameters | 405B | 405B |
| Active params per token | 405B | **~50B** |
| FLOPs per token | 810 TFLOPs | **~100 TFLOPs** |
| Memory for weights | 810 GB | 810 GB (same) |
| Memory for activations | Very large | **8x smaller** |
| Inter-stage activation size | 1.07 GB | **Same** (hidden_size unchanged) |

MoE doesn't reduce the total model size (still need 810GB for weights), but it massively reduces compute and activation memory per token.

### MoE Layer Architecture

```
Standard Transformer Layer:
  Input → LayerNorm → Attention → Residual → LayerNorm → FFN → Residual

MoE Transformer Layer:
  Input → LayerNorm → Attention → Residual → LayerNorm → MoE(FFN) → Residual
                                                           │
                                                    ┌──────┴──────┐
                                                    │  Router     │
                                                    │  (top-k=2)  │
                                                    └──────┬──────┘
                                              ┌─────┬──────┼──────┬─────┐
                                              │E0   │E1    │E2    │...  │E63
                                              │FFN  │FFN   │FFN   │     │FFN
                                              └─────┴──────┴──────┴─────┘
```

### Router Design

```python
class MoERouter(nn.Module):
    """Top-k router for Mixture of Experts."""
    
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts
    
    def forward(self, x):
        # x: [batch, seq, hidden]
        logits = self.gate(x)                          # [batch, seq, num_experts]
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        return weights, indices  # Which experts to use + their weights
```

### Load Balancing Loss

To prevent all tokens routing to the same experts:

```python
def load_balancing_loss(router_logits, top_k_indices, num_experts):
    """Auxiliary loss to encourage even expert utilization."""
    # Count how many tokens each expert receives
    expert_counts = torch.zeros(num_experts, device=router_logits.device)
    for k in range(top_k_indices.shape[-1]):
        expert_counts.scatter_add_(0, top_k_indices[..., k].flatten(), 
                                    torch.ones_like(top_k_indices[..., k].flatten(), dtype=torch.float))
    # Ideal: each expert gets equal share
    ideal = router_logits.shape[0] * router_logits.shape[1] * top_k / num_experts
    return ((expert_counts / ideal - 1.0) ** 2).mean()
```

### MoE Sharding Implication

With MoE, experts can be distributed across miners differently:

**Option A: Expert-parallel (within a stage)**
```
Stage 1 has 64 experts for layers 32-63.
  Miner A: experts 0-15   (16 experts)
  Miner B: experts 16-31  (16 experts)
  Miner C: experts 32-47  (16 experts)
  Miner D: experts 48-63  (16 experts)
```
Pro: Reduces memory per miner. Con: Requires all-to-all communication within stage.

**Option B: Pipeline-parallel only (recommended initially)**
```
Stage 1 (Miner B): ALL 64 experts for layers 32-63
```
Pro: Simple, no intra-stage communication. Con: Each stage miner needs to hold all experts.

**Recommendation:** Start with Option B (simpler). Move to Option A for Phase 6+ if memory is the bottleneck.

---

## 10. Lighthouse Enhancements

**Repo:** `RG_lighthouse`

### New Responsibilities

1. **Miner capability reporting** — miners report GPU VRAM, bandwidth, CPU cores
2. **Pipeline group formation** — Lighthouse suggests optimal groupings
3. **Topology tracking** — knows the full pipeline graph
4. **Health monitoring** — detects dead miners, triggers reassignment

### New Data in Peer Registry

```python
@dataclass
class MinerCapability:
    peer_id: str
    gpu_model: str              # "NVIDIA RTX 4090"
    gpu_vram_gb: float          # 24.0
    system_ram_gb: float        # 64.0
    cpu_cores: int              # 16
    bandwidth_mbps: float       # 1000.0  (measured)
    storage_available_gb: float # 500.0
    location_region: str        # "us-west" (for latency optimization)
    supported_dtypes: List[str] # ["fp16", "bf16", "int8"]
```

### New API Endpoints

```
POST /lighthouse/capabilities    → miner reports its hardware capabilities
GET  /lighthouse/pipeline-groups → get current pipeline topology
GET  /lighthouse/shard-peers     → get peers in same pipeline group (for P2P activation routing)
```

---

## 11. Miner App Enhancements

**Repo:** `RG_miner_app`

### Current: Download Full Model

```python
# server.py — _mining_loop
model, _ = create_model(model_id)  # Creates FULL model
model = model.to(device)           # Loads FULL model to GPU
```

### Future: Download Only My Shard

```python
# server.py — _mining_loop (sharded mode)
assignment = await get_my_shard_assignment(miner_id)
# assignment = ShardAssignment(layer_start=32, layer_end=63, ...)

shard_model = create_model_shard(
    model_id=model_id,
    layer_start=assignment.layer_start,
    layer_end=assignment.layer_end,
    has_embedding=assignment.has_embedding,
    has_lm_head=assignment.has_lm_head,
)
shard_model = shard_model.to(device)

# Training loop now:
# 1. Receive activations from upstream (or create from input tokens if stage 0)
# 2. Forward through our layers only
# 3. Send activations to downstream
# 4. Receive gradients from downstream (or compute loss if last stage)
# 5. Backward through our layers only
# 6. Send gradients to upstream
# 7. Submit our shard's compressed gradients to our shard's param server
```

### Capability Reporting

On startup, the miner app detects and reports hardware:

```python
def detect_capabilities() -> MinerCapability:
    import torch
    import psutil
    
    gpu_model = ""
    gpu_vram_gb = 0
    if torch.cuda.is_available():
        gpu_model = torch.cuda.get_device_name(0)
        gpu_vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_model = "Apple Silicon (MPS)"
        gpu_vram_gb = psutil.virtual_memory().total / 1e9 * 0.75  # Unified memory
    
    return MinerCapability(
        gpu_model=gpu_model,
        gpu_vram_gb=gpu_vram_gb,
        system_ram_gb=psutil.virtual_memory().total / 1e9,
        cpu_cores=psutil.cpu_count(logical=False),
        bandwidth_mbps=0,  # Measured during peer discovery
        storage_available_gb=psutil.disk_usage('/').free / 1e9,
    )
```

---

## 12. Weight Storage & Checkpointing

### Current: WeightStorage in real_trainer.py

Already has `save_weight_shard()` and `save_checkpoint()` with S3 support. This is the foundation.

### Sharded Checkpoint Format

```
s3://genesis2026/model-weights/resonant-frontier-405b/step_1000/
  ├── metadata.json            # Model config, shard map, global step
  ├── shard_0000.pt            # Embedding + layers 0-31
  ├── shard_0001.pt            # Layers 32-63
  ├── shard_0002.pt            # Layers 64-95
  ├── shard_0003.pt            # Layers 96-125 + LM head
  └── shard_checksums.json     # SHA256 for each shard file
```

### Checkpoint Flow

1. After aggregation round completes on all shards
2. Each shard PS saves its layers to S3 as `shard_NNNN.pt`
3. A metadata file records the mapping: `shard_index → layer_range`
4. New miners joining can download only their shard
5. Full model reconstruction: download all shards + concatenate

### Local Miner Checkpoints

Each miner saves only their shard locally:

```
~/.rg_miner/checkpoints/
  resonant-frontier-405b_shard2_step1000.pt   # Only layers 64-95
```

---

## 13. Fault Tolerance

### Miner Disconnects Mid-Pipeline

**Problem:** If Miner B (Stage 1) disconnects, the entire pipeline stalls.

**Solution: Pipeline Group Redundancy**

```
Pipeline Group A:  M1 → M2 → M3 → M4    (primary)
Pipeline Group B:  M5 → M6 → M7 → M8    (redundant)
Pipeline Group C:  M9 → M10 → M11 → M12  (redundant)
```

- If M2 disconnects, Group A is degraded
- Lighthouse reassigns M2's shard to a waiting miner (M13)
- M13 downloads shard from S3, loads it, reports ready
- Group A resumes training
- Groups B and C continue training uninterrupted

**Recovery time:** Dominated by shard download time  
- 100GB shard on 1 Gbps connection ≈ 800 seconds (13 min)  
- With S3 acceleration + CDN: ~3-5 minutes

### Stale Activations

If a pipeline stage is slow, upstream activations may accumulate. Use a **bounded queue** with timeout:

```python
ACTIVATION_QUEUE_MAX = 16    # Max microbatches buffered
ACTIVATION_TIMEOUT_SEC = 300  # 5 min timeout per microbatch
```

### Gradient Staleness Across Shards

Each shard PS tracks its own global step. All shards must agree before advancing. Use the external blockchain's Raft consensus for cross-shard step synchronization.

---

## 14. Security Considerations

### Activation Poisoning

A malicious miner could send corrupted activations to downstream stages, poisoning the training.

**Mitigation:**
- Redundant pipeline groups can cross-validate activations (sample random microbatches)
- Activation checksums recorded on external blockchain
- Trust scoring already in parameter server — extend to activation integrity

### Shard Withholding

A miner could refuse to share their shard's weights during checkpointing.

**Mitigation:**
- Multiple miners per shard (via redundant groups)
- Checkpoint duty rotates — if one miner fails to checkpoint, another in the same group does
- Reward penalties for checkpoint failures

### Free-Riding

A miner claims to train but sends random gradients.

**Mitigation (already partially built):**
- Gradient hash verification in param server
- Loss value monitoring — random gradients produce high loss
- Spot-check verification (task manager has `mark_verified` / `mark_rejected`)
- Trust score decay for rejected submissions

---

## 15. Phased Rollout Plan

### Phase 2 — NOW ✅
- **Model:** resonant-seed-1b (1B params)
- **Architecture:** Full model per miner, single param server
- **Miners:** 1+
- **Status:** Working. Enterprise auth, metrics, checkpointing done.

### Phase 4 — Next
- **Model:** resonant-v1-7b / resonant-v1-13b
- **Architecture:** Still full model per miner (needs 16-24GB VRAM)
- **Miners:** 10-25
- **New infrastructure needed:**
  - Better gradient compression (adaptive top-k)
  - Multi-node param server (not yet sharded, just replicated)
  - Weight sync via S3 checkpoints
  - Miner capability reporting to Lighthouse

### Phase 5 — Model Sharding
- **Model:** resonant-v2-70b
- **Architecture:** 4-stage pipeline parallelism
- **Miners:** 100+ (25 pipeline groups of 4)
- **New infrastructure needed:**
  - Shard Manager
  - Pipeline Coordinator + Activation Router
  - Sharded Parameter Server
  - Pipeline-aware Mining WebSocket protocol
  - Shard-aware miner app (download shard only)

### Phase 6 — Frontier
- **Model:** resonant-frontier-405b (MoE)
- **Architecture:** 8-stage pipeline + MoE
- **Miners:** 500+ (62+ pipeline groups of 8)
- **New infrastructure needed:**
  - MoE model architecture (router, expert blocks)
  - Expert-parallel within stages (optional)
  - Cross-shard Raft consensus for step sync
  - Advanced microbatch scheduling (1F1B)
  - Activation compression (INT8)

---

## 16. New Files & Modules

### RG_Mining (Mining Service)

| File | Purpose | Phase |
|---|---|---|
| `app/shard_manager.py` | Layer assignment, pipeline group formation | 5 |
| `app/pipeline.py` | Pipeline coordinator, microbatch scheduling | 5 |
| `app/activation_router.py` | P2P activation forwarding, compression | 5 |
| `app/sharded_param_server.py` | Layer-scoped parameter server | 5 |
| `app/moe_architecture.py` | MoE layer, router, expert blocks | 6 |

### RG_lighthouse

| File | Purpose | Phase |
|---|---|---|
| `app/capability_registry.py` | Track miner hardware capabilities | 4 |
| `app/pipeline_topology.py` | Pipeline group formation logic | 5 |

### RG_miner_app

| File | Purpose | Phase |
|---|---|---|
| `shard_loader.py` | Download and load model shards | 5 |
| `activation_client.py` | Send/receive activations to/from peers | 5 |

---

## 17. API Changes

### Mining Service — New Endpoints

```
# Shard Management
GET  /mining/shards/assignments                → All shard assignments
GET  /mining/shards/assignment/{miner_id}      → Single miner's assignment
GET  /mining/shards/pipeline-groups             → All pipeline groups
POST /mining/shards/form-groups                 → Trigger group formation
POST /mining/shards/report-ready               → Miner reports shard loaded

# Pipeline
POST /mining/pipeline/forward                  → Receive activation from upstream
POST /mining/pipeline/backward                 → Receive gradient from downstream
GET  /mining/pipeline/status/{group_id}        → Pipeline group training status
```

### Mining WebSocket — Extended Protocol

```json
// New message types for sharded training:

// Miner → Server: register with capabilities
{"action": "register", "miner_id": "...", "capabilities": {"gpu_vram_gb": 24, ...}}

// Server → Miner: shard assignment
{"event": "shard_assigned", "assignment": {"layer_start": 32, "layer_end": 63, ...}}

// Miner → Miner (P2P): forward activation
{"type": "activation_forward", "microbatch": 0, "data": "..."}

// Miner → Miner (P2P): backward gradient
{"type": "activation_backward", "microbatch": 0, "data": "..."}

// Miner → Server: submit shard gradient (same as today, but layer-scoped)
{"action": "submit_gradient", "gradient": {"shard_index": 1, ...}}
```

### Lighthouse — New Endpoints

```
POST /lighthouse/capabilities    → Report hardware capabilities
GET  /lighthouse/pipeline-groups → Current pipeline topology
GET  /lighthouse/shard-peers/{group_id} → Peers in my pipeline group
```

---

## 18. Open Questions

1. **Activation transport protocol**: Should we use WebSocket (existing) or gRPC (faster for large tensors)? WebSocket is simpler and already deployed. gRPC would be better for multi-GB activation transfers.

2. **Cross-region latency**: Pipeline parallelism is latency-sensitive (each stage waits for the previous). Should we enforce that pipeline groups are geographically co-located? Lighthouse could factor in region during group formation.

3. **Mixed hardware**: What if some miners have 8GB VRAM and others have 80GB? Uneven stage sizes? Or restrict pipeline groups to similar-capability miners?

4. **Optimizer state sharding**: Adam optimizer state is 2x model size. Each shard miner holds optimizer state for their layers only — this works naturally with pipeline parallelism.

5. **Gradient accumulation across microbatches**: Do we accumulate on the miner before sending, or send each microbatch's gradient separately? Accumulating is more bandwidth-efficient; separate gives the PS more data for staleness-aware averaging.

6. **Token economics**: Should pipeline stage miners get different rewards? Stage 0 (embedding) has more parameters. Last stage (LM head + loss) has extra compute. Currently all miners get the same reward per gradient.

7. **MoE expert placement**: When should we move from Option B (all experts on one miner) to Option A (expert-parallel across miners)? Depends on VRAM per miner vs expert count.

---

*This document is the starting point for implementation. Each section should be expanded into detailed design docs as we enter Phase 5 development.*
