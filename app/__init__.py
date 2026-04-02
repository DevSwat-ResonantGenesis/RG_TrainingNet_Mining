"""RG Mining Service — Decentralized LLM training orchestration.

Modules:
  - genesis_seed: Model registry, scaling tiers, genesis initialization
  - model_architecture: Dense Transformer (GQA + RoPE + SwiGLU)
  - moe_architecture: Mixture of Experts extension + ModelShard for pipeline parallelism
  - param_server: Original single-instance parameter server
  - sharded_param_server: Layer-scoped sharded PS + hierarchical aggregation
  - shard_manager: Pipeline group formation, layer assignment, hierarchical tree
  - pipeline: Microbatch scheduling (GPipe, 1F1B)
  - activation_router: P2P tensor forwarding with compression
  - gradient_compressor: Top-k gradient compression
  - training_task: Task lifecycle management
  - real_trainer: Training engine + tokenizer + weight storage
  - chain_bridge: External blockchain integration
  - ws_handler: WebSocket handler for miner connections
"""
