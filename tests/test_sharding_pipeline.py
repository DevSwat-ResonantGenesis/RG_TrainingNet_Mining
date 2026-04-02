"""
Full production pipeline test for sharded training infrastructure.
Tests all 5 new modules end-to-end.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json
import time
import traceback

results = {}

def test(name):
    def decorator(fn):
        def wrapper():
            try:
                fn()
                results[name] = ("PASS", "")
                print(f"  ✓ {name}")
            except Exception as e:
                results[name] = ("FAIL", str(e))
                print(f"  ✗ {name}: {e}")
                traceback.print_exc()
        return wrapper
    return decorator

# ═══════════════════════════════════════════
# TEST 2: SHARD MANAGER
# ═══════════════════════════════════════════
print("=" * 60)
print("TEST 2: Shard Manager")
print("=" * 60)

@test("sm_register_miners")
def t():
    from app.shard_manager import ShardManager, MinerCapability
    sm = ShardManager()
    for i in range(8):
        sm.register_miner(MinerCapability(miner_id=f'm-{i}', gpu_vram_gb=24.0, bandwidth_mbps=1000, location_region='us-west'))
    assert len(sm.miners) == 8
    assert len(sm.get_available_miners()) == 8
t()

@test("sm_needs_sharding_70b")
def t():
    from app.shard_manager import ShardManager, MinerCapability
    sm = ShardManager()
    for i in range(8):
        sm.register_miner(MinerCapability(miner_id=f'm-{i}', gpu_vram_gb=24.0))
    cfg = {'num_parameters': 70_000_000_000, 'num_layers': 80, 'hidden_size': 8192, 'num_heads': 64, 'num_kv_heads': 8, 'intermediate_size': 28672, 'vocab_size': 128_256}
    assert sm.needs_sharding(cfg) == True
t()

@test("sm_no_sharding_1b")
def t():
    from app.shard_manager import ShardManager, MinerCapability
    sm = ShardManager()
    for i in range(4):
        sm.register_miner(MinerCapability(miner_id=f'm-{i}', gpu_vram_gb=24.0))
    cfg = {'num_parameters': 1_000_000_000, 'num_layers': 24, 'hidden_size': 2048, 'num_heads': 16, 'num_kv_heads': 4, 'intermediate_size': 5504, 'vocab_size': 128_256}
    assert sm.needs_sharding(cfg) == False
t()

@test("sm_optimal_stages")
def t():
    from app.shard_manager import ShardManager, MinerCapability
    sm = ShardManager()
    for i in range(8):
        sm.register_miner(MinerCapability(miner_id=f'm-{i}', gpu_vram_gb=24.0))
    cfg = {'num_parameters': 70_000_000_000, 'num_layers': 80, 'hidden_size': 8192, 'num_heads': 64, 'num_kv_heads': 8, 'intermediate_size': 28672, 'vocab_size': 128_256}
    stages = sm.compute_optimal_stages(cfg, sm.get_available_miners())
    assert stages >= 2, f"Expected >=2 stages, got {stages}"
    assert stages <= 80, f"Stages should not exceed layers"
t()

@test("sm_layer_assignment")
def t():
    from app.shard_manager import ShardManager
    sm = ShardManager()
    ranges = sm.compute_layer_assignment(80, 4)
    assert len(ranges) == 4
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 80
    total_layers = sum(e - s for s, e in ranges)
    assert total_layers == 80
t()

@test("sm_form_pipeline_groups")
def t():
    from app.shard_manager import ShardManager, MinerCapability
    sm = ShardManager()
    # Use 32 miners so we have enough for multiple groups of any stage count
    for i in range(32):
        sm.register_miner(MinerCapability(miner_id=f'm-{i}', gpu_vram_gb=24.0, location_region='us-west'))
    cfg = {'num_parameters': 70_000_000_000, 'num_layers': 80, 'hidden_size': 8192, 'num_heads': 64, 'num_kv_heads': 8, 'intermediate_size': 28672, 'vocab_size': 128_256}
    groups = sm.form_pipeline_groups('70b', cfg, target_redundancy=2)
    assert len(groups) >= 1
    # At least one complete group
    complete = [g for g in groups if g.is_complete]
    assert len(complete) >= 1, f"No complete groups from {len(groups)} formed"
    for g in complete:
        assert len(g.stages) == g.num_stages
        for idx, s in g.stages.items():
            if idx == 0:
                assert s.has_embedding
                assert s.upstream_miner_id is None
            if idx == g.num_stages - 1:
                assert s.has_lm_head
                assert s.downstream_miner_id is None
t()

@test("sm_ready_and_training")
def t():
    from app.shard_manager import ShardManager, MinerCapability, PipelineStatus
    sm = ShardManager()
    # 1B model on 24GB GPUs = 1 stage per group, easy to fill
    for i in range(4):
        sm.register_miner(MinerCapability(miner_id=f'm-{i}', gpu_vram_gb=24.0))
    cfg = {'num_parameters': 1_000_000_000, 'num_layers': 24, 'hidden_size': 2048, 'num_heads': 16, 'num_kv_heads': 4, 'intermediate_size': 5504, 'vocab_size': 128_256}
    groups = sm.form_pipeline_groups('1b', cfg)
    g = groups[0]
    assert g.is_complete
    for s in g.stages.values():
        sm.report_shard_ready(s.miner_id)
    assert g.is_ready
    sm.start_training(g.group_id)
    assert g.status == PipelineStatus.TRAINING
t()

@test("sm_disconnect_failover")
def t():
    from app.shard_manager import ShardManager, MinerCapability, PipelineStatus
    sm = ShardManager()
    for i in range(8):
        sm.register_miner(MinerCapability(miner_id=f'm-{i}', gpu_vram_gb=24.0))
    cfg = {'num_parameters': 70_000_000_000, 'num_layers': 80, 'hidden_size': 8192, 'num_heads': 64, 'num_kv_heads': 8, 'intermediate_size': 28672, 'vocab_size': 128_256}
    groups = sm.form_pipeline_groups('70b', cfg)
    victim = groups[0].stages[0].miner_id
    affected = sm.handle_miner_disconnect(victim)
    assert affected is not None
    assert sm.get_pipeline_group(affected).status == PipelineStatus.DEGRADED
t()

@test("sm_stats")
def t():
    from app.shard_manager import ShardManager, MinerCapability
    sm = ShardManager()
    for i in range(4):
        sm.register_miner(MinerCapability(miner_id=f'm-{i}', gpu_vram_gb=24.0))
    stats = sm.get_stats()
    assert stats["total_miners"] == 4
    assert "scale_tier" in stats
t()

# ═══════════════════════════════════════════
# TEST 3: PIPELINE COORDINATOR
# ═══════════════════════════════════════════
print()
print("=" * 60)
print("TEST 3: Pipeline Coordinator")
print("=" * 60)

@test("pipe_gpipe_schedule")
def t():
    from app.pipeline import generate_gpipe_schedule
    sched = generate_gpipe_schedule(4, 8)
    assert sched.num_stages == 4
    assert sched.num_microbatches == 8
    assert len(sched.stage_steps) == 4
    assert sched.total_ticks > 0
    # Each stage should have forward + backward + submit
    for stage_idx in range(4):
        steps = sched.get_steps_for_stage(stage_idx)
        fwd = [s for s in steps if s.action.value == "forward"]
        bwd = [s for s in steps if s.action.value == "backward"]
        assert len(fwd) == 8, f"Stage {stage_idx}: expected 8 forward, got {len(fwd)}"
        assert len(bwd) == 8, f"Stage {stage_idx}: expected 8 backward, got {len(bwd)}"
t()

@test("pipe_1f1b_schedule")
def t():
    from app.pipeline import generate_1f1b_schedule
    sched = generate_1f1b_schedule(4, 8)
    assert sched.num_stages == 4
    assert sched.num_microbatches == 8
    assert sched.strategy.value == "1f1b"
    for stage_idx in range(4):
        steps = sched.get_steps_for_stage(stage_idx)
        fwd = [s for s in steps if s.action.value == "forward"]
        bwd = [s for s in steps if s.action.value == "backward"]
        assert len(fwd) == 8
        assert len(bwd) == 8
t()

@test("pipe_choose_schedule")
def t():
    from app.pipeline import choose_schedule
    # 1 stage = trivial
    s1 = choose_schedule(1, 4)
    assert s1.num_stages == 1
    # 8 stages = should prefer 1f1b
    s8 = choose_schedule(8, 16, prefer_memory_efficient=True)
    assert s8.strategy.value == "1f1b"
t()

@test("pipe_optimal_microbatches")
def t():
    from app.pipeline import compute_optimal_microbatches
    mb, mb_size = compute_optimal_microbatches(4, 128, target_efficiency=0.85)
    assert mb >= 4
    assert mb_size >= 1
    assert mb * mb_size >= 128 or mb_size == 1
t()

@test("pipe_throughput_estimate")
def t():
    from app.pipeline import estimate_pipeline_throughput, ScheduleStrategy
    r = estimate_pipeline_throughput(4, 16, 0.1, ScheduleStrategy.ONE_F_ONE_B)
    assert r["efficiency"] > 0.5
    assert r["total_step_time_sec"] > 0
t()

@test("pipe_coordinator_create")
def t():
    from app.pipeline import PipelineCoordinator
    pc = PipelineCoordinator("pg-test", num_stages=4)
    sched = pc.create_schedule(num_microbatches=8)
    assert sched.num_stages == 4
    stats = pc.get_stats()
    assert stats["num_stages"] == 4
t()

# ═══════════════════════════════════════════
# TEST 4: ACTIVATION ROUTER
# ═══════════════════════════════════════════
print()
print("=" * 60)
print("TEST 4: Activation Router")
print("=" * 60)

@test("act_serialize_none")
def t():
    import torch
    from app.activation_router import TensorSerializer, CompressionMethod
    tensor = torch.randn(2, 128, 512)
    data, meta = TensorSerializer.serialize(tensor, CompressionMethod.NONE)
    assert len(data) == 2 * 128 * 512 * 4  # float32
    assert meta.compression == CompressionMethod.NONE
    restored = TensorSerializer.deserialize(data, meta)
    assert restored.shape == (2, 128, 512)
    assert torch.allclose(tensor.float(), restored.float(), atol=1e-5)
t()

@test("act_serialize_int8")
def t():
    import torch
    from app.activation_router import TensorSerializer, CompressionMethod
    tensor = torch.randn(2, 128, 512)
    data, meta = TensorSerializer.serialize(tensor, CompressionMethod.INT8)
    assert meta.compression == CompressionMethod.INT8
    assert meta.compressed_size_bytes < meta.original_size_bytes
    restored = TensorSerializer.deserialize(data, meta)
    assert restored.shape == (2, 128, 512)
t()

@test("act_serialize_zlib")
def t():
    import torch
    from app.activation_router import TensorSerializer, CompressionMethod
    tensor = torch.randn(2, 128, 512)
    data, meta = TensorSerializer.serialize(tensor, CompressionMethod.ZLIB)
    assert meta.compression == CompressionMethod.ZLIB
    restored = TensorSerializer.deserialize(data, meta)
    assert restored.shape == (2, 128, 512)
    assert torch.allclose(tensor.float(), restored.float(), atol=1e-5)
t()

@test("act_serialize_int8_zlib")
def t():
    import torch
    from app.activation_router import TensorSerializer, CompressionMethod
    tensor = torch.randn(2, 128, 512)
    data, meta = TensorSerializer.serialize(tensor, CompressionMethod.INT8_ZLIB)
    assert meta.compression == CompressionMethod.INT8_ZLIB
    assert meta.compression_ratio > 1.0
    restored = TensorSerializer.deserialize(data, meta)
    assert restored.shape == (2, 128, 512)
t()

@test("act_serialize_topk")
def t():
    import torch
    from app.activation_router import TensorSerializer, CompressionMethod
    tensor = torch.randn(2, 128, 512)
    data, meta = TensorSerializer.serialize(tensor, CompressionMethod.TOP_K)
    assert meta.compression == CompressionMethod.TOP_K
    restored = TensorSerializer.deserialize(data, meta)
    assert restored.shape == (2, 128, 512)
t()

@test("act_auto_compression")
def t():
    from app.activation_router import TensorSerializer, TransferDirection, CompressionMethod
    # Small tensor: no compression
    c = TensorSerializer.choose_compression(500_000, TransferDirection.FORWARD)
    assert c == CompressionMethod.NONE
    # Large tensor: INT8
    c = TensorSerializer.choose_compression(50_000_000, TransferDirection.FORWARD)
    assert c == CompressionMethod.INT8
    # Very large backward: INT8_ZLIB
    c = TensorSerializer.choose_compression(500_000_000, TransferDirection.BACKWARD)
    assert c == CompressionMethod.INT8_ZLIB
t()

@test("act_estimate_size")
def t():
    from app.activation_router import estimate_activation_size
    r = estimate_activation_size(2, 4096, 8192, dtype_bytes=2)
    assert r["raw_size_mb"] > 0
    assert r["int8_size_mb"] < r["raw_size_mb"]
t()

@test("act_chunk_roundtrip")
def t():
    from app.activation_router import ActivationChunk
    import base64
    chunk = ActivationChunk(transfer_id="t1", chunk_index=0, total_chunks=3, data=b"hello world", is_last=False)
    msg = chunk.to_message()
    assert msg["type"] == "activation_chunk"
    restored = ActivationChunk.from_message(msg)
    assert restored.data == b"hello world"
    assert restored.chunk_index == 0
t()

@test("act_router_stats")
def t():
    from app.activation_router import ActivationRouter
    router = ActivationRouter("pg-test")
    stats = router.get_stats()
    assert stats["total_transfers"] == 0
    assert stats["pipeline_group_id"] == "pg-test"
t()

# ═══════════════════════════════════════════
# TEST 5: SHARDED PARAM SERVER
# ═══════════════════════════════════════════
print()
print("=" * 60)
print("TEST 5: Sharded Parameter Server")
print("=" * 60)

@test("sps_create_shards")
def t():
    from app.sharded_param_server import create_parameter_server
    cfg = {'model_id': '70b', 'num_layers': 80}
    agg = create_parameter_server(cfg, num_shards=4)
    assert len(agg.shards) == 4
    # Check layer ranges cover all 80 layers
    all_layers = set()
    for s in agg.shards.values():
        for l in range(s.layer_start, s.layer_end):
            all_layers.add(l)
    assert len(all_layers) == 80
t()

@test("sps_receive_gradient")
def t():
    from app.sharded_param_server import create_parameter_server, ShardGradient
    cfg = {'model_id': '70b', 'num_layers': 80}
    agg = create_parameter_server(cfg, num_shards=4)
    grad = ShardGradient(miner_id='m1', layer_start=0, layer_end=20, samples_processed=100, loss=2.5)
    shard = agg.shards[0]
    ok = shard.receive_gradient(grad)
    assert ok == True
    assert shard.total_gradients_received == 1
t()

@test("sps_aggregate")
def t():
    from app.sharded_param_server import create_parameter_server, ShardGradient
    cfg = {'model_id': '70b', 'num_layers': 80}
    agg = create_parameter_server(cfg, num_shards=2)
    # Submit gradients to shard 0
    for i in range(3):
        grad = ShardGradient(miner_id=f'm{i}', layer_start=0, layer_end=40, samples_processed=100, loss=2.0-i*0.1)
        agg.shards[0].receive_gradient(grad)
    assert agg.shards[0].should_aggregate()
    rnd = agg.shards[0].aggregate()
    assert rnd is not None
    assert rnd.num_miners == 3
    assert rnd.total_samples == 300
t()

@test("sps_global_consensus")
def t():
    from app.sharded_param_server import create_parameter_server, ShardGradient
    cfg = {'model_id': '70b', 'num_layers': 80}
    agg = create_parameter_server(cfg, num_shards=2)
    # Submit to both shards
    for s_idx in range(2):
        shard = agg.shards[s_idx]
        grad = ShardGradient(miner_id=f'm-s{s_idx}', layer_start=shard.layer_start, layer_end=shard.layer_end, samples_processed=50, loss=1.5)
        shard.receive_gradient(grad)
    # Aggregate all
    rounds = agg.try_aggregate_all()
    assert len(rounds) == 2
    # Both shards aggregated => global step should advance
    assert agg.global_step == 1
t()

@test("sps_staleness_reject")
def t():
    from app.sharded_param_server import ShardParameterServer, ShardGradient
    sps = ShardParameterServer(shard_index=0, num_shards=1, layer_start=0, layer_end=24, model_id='test')
    sps.global_step = 100
    # Gradient from step 0 — staleness = 100 > MAX_STALENESS(50)
    grad = ShardGradient(miner_id='stale-miner', layer_start=0, layer_end=24, local_step=0, samples_processed=50, loss=3.0)
    sps.miner_steps['stale-miner'] = 0
    ok = sps.receive_gradient(grad)
    assert ok == False
t()

@test("sps_stats")
def t():
    from app.sharded_param_server import create_parameter_server
    cfg = {'model_id': 'test', 'num_layers': 24}
    agg = create_parameter_server(cfg, num_shards=2)
    stats = agg.get_stats()
    assert stats["num_shards"] == 2
    assert stats["global_step"] == 0
t()

@test("sps_regional_aggregator")
def t():
    from app.sharded_param_server import RegionalAggregator, ShardAggregationRound
    ra = RegionalAggregator(region='us-west', shard_index=0)
    r1 = ShardAggregationRound(shard_index=0, global_step=0, total_samples=100, weighted_loss=2.0)
    r2 = ShardAggregationRound(shard_index=0, global_step=0, total_samples=200, weighted_loss=1.5)
    ra.receive_shard_result(r1)
    ra.receive_shard_result(r2)
    assert ra.should_merge()
    merge = ra.merge()
    assert merge is not None
    assert merge.num_contributions == 2
    assert merge.total_samples == 300
t()

# ═══════════════════════════════════════════
# TEST 6: MoE ARCHITECTURE
# ═══════════════════════════════════════════
print()
print("=" * 60)
print("TEST 6: MoE Architecture")
print("=" * 60)

@test("moe_config")
def t():
    from app.moe_architecture import MoEConfig
    cfg = MoEConfig(hidden_size=256, num_layers=4, num_heads=4, num_kv_heads=2, intermediate_size=512, vocab_size=1000, max_seq_length=128, num_experts=8, num_experts_per_token=2)
    assert cfg.sparsity_ratio == 0.25
    assert cfg.total_expert_params > 0
    d = cfg.to_dict()
    assert d["num_experts"] == 8
t()

@test("moe_expert_router")
def t():
    import torch
    from app.moe_architecture import MoEConfig, ExpertRouter
    cfg = MoEConfig(hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=128, vocab_size=100, max_seq_length=32, num_experts=8, num_experts_per_token=2)
    router = ExpertRouter(cfg)
    x = torch.randn(2, 16, 64)
    weights, selected, loss = router(x)
    assert weights.shape == (2, 16, 2)
    assert selected.shape == (2, 16, 2)
    assert loss.item() >= 0
t()

@test("moe_layer_forward")
def t():
    import torch
    from app.moe_architecture import MoEConfig, MoELayer
    cfg = MoEConfig(hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=128, vocab_size=100, max_seq_length=32, num_experts=4, num_experts_per_token=2)
    layer = MoELayer(cfg)
    x = torch.randn(2, 16, 64)
    out, loss = layer(x)
    assert out.shape == (2, 16, 64)
    assert loss.item() >= 0
t()

@test("moe_transformer_block")
def t():
    import torch
    from app.moe_architecture import MoEConfig, MoETransformerBlock
    from app.model_architecture import precompute_rope_freqs
    cfg = MoEConfig(hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=128, vocab_size=100, max_seq_length=32, num_experts=4, num_experts_per_token=2)
    block = MoETransformerBlock(cfg, layer_idx=0, use_moe=True)
    freqs = precompute_rope_freqs(cfg.head_dim, 32)
    x = torch.randn(2, 16, 64)
    out, loss = block(x, freqs)
    assert out.shape == (2, 16, 64)
t()

@test("moe_full_model")
def t():
    import torch
    from app.moe_architecture import MoEConfig, ResonantMoEModel
    cfg = MoEConfig(hidden_size=64, num_layers=4, num_heads=4, num_kv_heads=2, intermediate_size=128, vocab_size=1000, max_seq_length=32, num_experts=4, num_experts_per_token=2, moe_layer_frequency=2)
    model = ResonantMoEModel(cfg)
    ids = torch.randint(0, 1000, (2, 16))
    labels = torch.randint(0, 1000, (2, 16))
    result = model(ids, labels)
    assert "logits" in result
    assert "loss" in result
    assert "router_loss" in result
    assert "total_loss" in result
    assert result["logits"].shape == (2, 16, 1000)
    params = model.get_num_params()
    assert params["total"] > 0
    assert params["active_per_token"] <= params["total"]
t()

@test("moe_model_shard_stage0")
def t():
    import torch
    from app.moe_architecture import MoEConfig, ModelShard
    cfg = MoEConfig(hidden_size=64, num_layers=4, num_heads=4, num_kv_heads=2, intermediate_size=128, vocab_size=1000, max_seq_length=32, num_experts=4, num_experts_per_token=2, moe_layer_frequency=2)
    shard = ModelShard(cfg, layer_start=0, layer_end=2, has_embedding=True, has_lm_head=False, is_moe=True)
    ids = torch.randint(0, 1000, (2, 16))
    result = shard(hidden_states=None, input_ids=ids)
    assert "hidden_states" in result
    assert result["hidden_states"].shape == (2, 16, 64)
t()

@test("moe_model_shard_last_stage")
def t():
    import torch
    from app.moe_architecture import MoEConfig, ModelShard
    cfg = MoEConfig(hidden_size=64, num_layers=4, num_heads=4, num_kv_heads=2, intermediate_size=128, vocab_size=1000, max_seq_length=32, num_experts=4, num_experts_per_token=2, moe_layer_frequency=2, tie_word_embeddings=False)
    shard = ModelShard(cfg, layer_start=2, layer_end=4, has_embedding=False, has_lm_head=True, is_moe=True)
    hidden = torch.randn(2, 16, 64)
    labels = torch.randint(0, 100, (2, 16))  # Keep labels well within vocab range
    result = shard(hidden_states=hidden, labels=labels)
    assert "logits" in result
    assert "loss" in result
    assert result["logits"].shape == (2, 16, 1000)
t()

@test("moe_dense_shard")
def t():
    import torch
    from app.model_architecture import ResonantModelConfig
    from app.moe_architecture import ModelShard
    cfg = ResonantModelConfig(hidden_size=64, num_layers=4, num_heads=4, num_kv_heads=2, intermediate_size=128, vocab_size=1000, max_seq_length=32)
    shard = ModelShard(cfg, layer_start=0, layer_end=2, has_embedding=True, has_lm_head=False, is_moe=False)
    ids = torch.randint(0, 1000, (2, 16))
    result = shard(hidden_states=None, input_ids=ids)
    assert result["hidden_states"].shape == (2, 16, 64)
t()

# ═══════════════════════════════════════════
# TEST 7: EXISTING MODULES STILL WORK
# ═══════════════════════════════════════════
print()
print("=" * 60)
print("TEST 7: Existing Modules Backward Compatibility")
print("=" * 60)

@test("existing_model_registry")
def t():
    from app.genesis_seed import MODEL_REGISTRY
    assert "resonant-seed-1b" in MODEL_REGISTRY
    assert "resonant-frontier-405b" in MODEL_REGISTRY
    assert MODEL_REGISTRY["resonant-frontier-405b"]["model_type"] == "transformer-gqa-moe"
t()

@test("existing_create_model_1b")
def t():
    import torch
    from app.model_architecture import create_model
    model, cfg = create_model("resonant-seed-1b")
    ids = torch.randint(0, cfg.vocab_size, (1, 32))
    result = model(ids)
    assert "logits" in result
t()

@test("existing_param_server")
def t():
    from app.param_server import ParameterServer
    ps = ParameterServer("test")
    state = ps.register_miner("m1", "miner")
    assert state.miner_id == "m1"
    stats = ps.get_stats()
    assert stats["global_step"] == 0
t()

@test("existing_gradient_compressor")
def t():
    import torch
    from app.gradient_compressor import GradientCompressor
    gc = GradientCompressor(compression_ratio=0.01)
    grad = torch.randn(1000)
    compressed = gc.compress(grad, "test_layer")
    assert compressed.layer_name == "test_layer"
    assert compressed.compressed_size > 0
t()

# ═══════════════════════════════════════════
# TEST 8: CROSS-MODULE INTEGRATION
# ═══════════════════════════════════════════
print()
print("=" * 60)
print("TEST 8: Cross-Module Integration")
print("=" * 60)

@test("integration_shard_to_pipeline")
def t():
    from app.shard_manager import ShardManager, MinerCapability
    from app.pipeline import PipelineCoordinator, choose_schedule
    sm = ShardManager()
    for i in range(4):
        sm.register_miner(MinerCapability(miner_id=f'm-{i}', gpu_vram_gb=24.0))
    cfg = {'num_parameters': 70_000_000_000, 'num_layers': 80, 'hidden_size': 8192, 'num_heads': 64, 'num_kv_heads': 8, 'intermediate_size': 28672, 'vocab_size': 128_256}
    groups = sm.form_pipeline_groups('70b', cfg)
    g = groups[0]
    # Create coordinator from group
    pc = PipelineCoordinator(g.group_id, g.num_stages)
    sched = pc.create_schedule(num_microbatches=8)
    assert sched.num_stages == g.num_stages
    assert sched.num_microbatches >= g.num_stages
t()

@test("integration_shard_to_ps")
def t():
    from app.shard_manager import ShardManager, MinerCapability
    from app.sharded_param_server import create_parameter_server, ShardGradient
    sm = ShardManager()
    for i in range(4):
        sm.register_miner(MinerCapability(miner_id=f'm-{i}', gpu_vram_gb=24.0))
    # Use 1B model so 1 stage per group, guaranteed complete
    cfg = {'num_parameters': 1_000_000_000, 'num_layers': 24, 'hidden_size': 2048, 'num_heads': 16, 'num_kv_heads': 4, 'intermediate_size': 5504, 'vocab_size': 128_256, 'model_id': '1b'}
    groups = sm.form_pipeline_groups('1b', cfg)
    g = groups[0]
    assert g.is_complete
    # Create sharded PS matching pipeline stages
    layer_ranges = [(s.layer_start, s.layer_end) for s in g.stages.values()]
    agg = create_parameter_server(cfg, num_shards=g.num_stages, layer_ranges=layer_ranges)
    assert len(agg.shards) == g.num_stages
    # Submit gradient from each stage
    for idx, stage in g.stages.items():
        grad = ShardGradient(miner_id=stage.miner_id, layer_start=stage.layer_start, layer_end=stage.layer_end, samples_processed=64, loss=2.0)
        shard = agg.shards[idx]
        ok = shard.receive_gradient(grad)
        assert ok, f"Shard {idx} rejected gradient"
    # Aggregate
    rounds = agg.try_aggregate_all()
    assert len(rounds) == g.num_stages
    assert agg.global_step == 1
t()

@test("integration_model_shard_pipeline")
def t():
    import torch
    from app.moe_architecture import MoEConfig, ModelShard
    # Simulate 2-stage pipeline for small model
    cfg = MoEConfig(hidden_size=64, num_layers=4, num_heads=4, num_kv_heads=2, intermediate_size=128, vocab_size=1000, max_seq_length=32, num_experts=4, num_experts_per_token=2, moe_layer_frequency=2, tie_word_embeddings=False)
    stage0 = ModelShard(cfg, layer_start=0, layer_end=2, has_embedding=True, has_lm_head=False, is_moe=True)
    stage1 = ModelShard(cfg, layer_start=2, layer_end=4, has_embedding=False, has_lm_head=True, is_moe=True)
    ids = torch.randint(0, 100, (2, 16))
    labels = torch.randint(0, 100, (2, 16))
    # Stage 0: embed + first 2 layers
    r0 = stage0(hidden_states=None, input_ids=ids)
    hidden = r0["hidden_states"]
    # Stage 1: last 2 layers + LM head + loss
    r1 = stage1(hidden_states=hidden, labels=labels)
    assert "logits" in r1
    assert "loss" in r1
    assert r1["logits"].shape == (2, 16, 1000)
t()

@test("integration_activation_roundtrip")
def t():
    import torch
    from app.activation_router import TensorSerializer, ActivationMetadata, CompressionMethod
    from app.moe_architecture import MoEConfig, ModelShard
    cfg = MoEConfig(hidden_size=64, num_layers=4, num_heads=4, num_kv_heads=2, intermediate_size=128, vocab_size=1000, max_seq_length=32, num_experts=4, num_experts_per_token=2, moe_layer_frequency=2)
    stage0 = ModelShard(cfg, layer_start=0, layer_end=2, has_embedding=True, has_lm_head=False, is_moe=True)
    ids = torch.randint(0, 1000, (2, 16))
    r0 = stage0(hidden_states=None, input_ids=ids)
    hidden = r0["hidden_states"]
    # Serialize activation for network transfer
    data, meta = TensorSerializer.serialize(hidden, CompressionMethod.INT8)
    assert meta.compressed_size_bytes < meta.original_size_bytes
    # Deserialize on receiving miner
    restored = TensorSerializer.deserialize(data, meta)
    assert restored.shape == hidden.shape
    # Should be close (INT8 is lossy)
    diff = (hidden.float() - restored.float()).abs().mean().item()
    assert diff < 0.5, f"INT8 roundtrip error too large: {diff}"
t()

# ═══════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════
print()
print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
passed = sum(1 for v in results.values() if v[0] == "PASS")
failed = sum(1 for v in results.values() if v[0] == "FAIL")
total = len(results)
print(f"\n  PASSED: {passed}/{total}")
print(f"  FAILED: {failed}/{total}")
if failed:
    print("\n  FAILURES:")
    for name, (status, err) in results.items():
        if status == "FAIL":
            print(f"    ✗ {name}: {err}")
print()
sys.exit(0 if failed == 0 else 1)
