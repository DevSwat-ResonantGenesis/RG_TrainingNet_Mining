"""
Tests for Network-Native Model Infrastructure
===============================================

Tests for:
  - WeightShardRegistry: shard registration, queries, orphaning, redistribution
  - ShardSlicer: layer extraction, manifest creation, streaming
  - Liquid Redistribution: auto-heal degraded pipelines
  - Merkle root versioning and integrity verification
"""

import asyncio
import hashlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from app.weight_shard_registry import (
    WeightShardRegistry, ShardLocation, ShardState, ReplicaPriority,
    WeightVersion, ReplicationPolicy,
)
from app.shard_slicer import (
    ShardSlicer, LayerWeightChunk, SliceManifest,
    WeightTransferRequest, WeightTransferPlan, create_transfer_plan,
)
from app.shard_manager import (
    ShardManager, MinerCapability, PipelineStatus,
)


# ══════════════════════════════════════════════════════════════
# Test helpers
# ══════════════════════════════════════════════════════════════

passed = 0
failed = 0
errors = []


def run_test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✓ {name}")
        passed += 1
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        failed += 1
        errors.append((name, str(e)))


def make_registry(n_shards=4, n_miners=4, model_id="test-model"):
    """Create a registry with pre-registered shards."""
    reg = WeightShardRegistry()
    for i in range(n_shards):
        miner_id = f"miner-{i % n_miners}"
        loc = ShardLocation(
            model_id=model_id,
            miner_id=miner_id,
            layer_start=i * 8,
            layer_end=(i + 1) * 8,
            version=1,
            weight_hash=hashlib.sha256(f"shard-{i}".encode()).hexdigest(),
            state=ShardState.LOADED,
            priority=ReplicaPriority.PRIMARY,
            size_bytes=1_000_000 * (i + 1),
            num_params=500_000 * (i + 1),
            miner_address=f"10.0.0.{i}:3000",
        )
        reg.register_shard(loc)
    return reg


# ══════════════════════════════════════════════════════════════
# TEST 1: Weight Shard Registry — Registration & Queries
# ══════════════════════════════════════════════════════════════

def test_registry_register():
    reg = WeightShardRegistry()
    loc = ShardLocation(
        model_id="m1", miner_id="miner-0",
        layer_start=0, layer_end=8,
        state=ShardState.LOADED,
        weight_hash="abc123",
    )
    result = reg.register_shard(loc)
    assert result.location_id == loc.location_id
    assert loc.shard_key == "m1:L0-8"
    assert "m1:L0-8" in reg.shard_replicas
    assert loc.location_id in reg.shard_replicas["m1:L0-8"]
    assert "miner-0" in reg.miner_shards
    assert reg._total_registered == 1


def test_registry_multiple_replicas():
    reg = WeightShardRegistry()
    for i in range(3):
        loc = ShardLocation(
            model_id="m1", miner_id=f"miner-{i}",
            layer_start=0, layer_end=8,
            state=ShardState.LOADED,
            priority=ReplicaPriority.PRIMARY if i == 0 else ReplicaPriority.HOT_SPARE,
        )
        reg.register_shard(loc)
    assert len(reg.shard_replicas["m1:L0-8"]) == 3
    assert len(reg.miner_shards) == 3


def test_registry_find_sources():
    reg = make_registry(4, 4)
    sources = reg.find_shard_sources("test-model", 0, 8)
    assert len(sources) == 1
    assert sources[0].miner_id == "miner-0"
    assert sources[0].layer_start == 0
    assert sources[0].layer_end == 8


def test_registry_find_sources_sorted_by_priority():
    reg = WeightShardRegistry()
    # Add cold backup first
    reg.register_shard(ShardLocation(
        model_id="m1", miner_id="backup",
        layer_start=0, layer_end=8,
        state=ShardState.LOADED,
        priority=ReplicaPriority.COLD_BACKUP,
    ))
    # Add primary
    reg.register_shard(ShardLocation(
        model_id="m1", miner_id="primary",
        layer_start=0, layer_end=8,
        state=ShardState.LOADED,
        priority=ReplicaPriority.PRIMARY,
    ))
    sources = reg.find_shard_sources("m1", 0, 8)
    assert len(sources) == 2
    assert sources[0].miner_id == "primary"  # PRIMARY first
    assert sources[1].miner_id == "backup"


def test_registry_find_overlapping():
    reg = WeightShardRegistry()
    reg.register_shard(ShardLocation(
        model_id="m1", miner_id="miner-a",
        layer_start=0, layer_end=16,
        state=ShardState.LOADED,
    ))
    reg.register_shard(ShardLocation(
        model_id="m1", miner_id="miner-b",
        layer_start=16, layer_end=32,
        state=ShardState.LOADED,
    ))
    # Request layers 8-24 — overlaps both shards
    results = reg.find_overlapping_sources("m1", 8, 24)
    assert len(results) == 2


def test_registry_shard_map():
    reg = make_registry(4, 4)
    smap = reg.get_shard_map("test-model")
    assert len(smap) == 4
    assert "test-model:L0-8" in smap
    assert "test-model:L24-32" in smap


def test_registry_miner_shards():
    reg = make_registry(4, 2)  # 4 shards across 2 miners
    shards = reg.get_miner_shards("miner-0")
    assert len(shards) == 2  # miner-0 gets shards 0 and 2
    shards1 = reg.get_miner_shards("miner-1")
    assert len(shards1) == 2  # miner-1 gets shards 1 and 3


def test_registry_update_state():
    reg = WeightShardRegistry()
    loc = ShardLocation(
        model_id="m1", miner_id="miner-0",
        layer_start=0, layer_end=8,
        state=ShardState.STREAMING,
    )
    reg.register_shard(loc)
    assert reg.locations[loc.location_id].state == ShardState.STREAMING

    updated = reg.update_shard_state(
        loc.location_id, ShardState.LOADED,
        weight_hash="newhash", version=5,
    )
    assert updated.state == ShardState.LOADED
    assert updated.weight_hash == "newhash"
    assert updated.version == 5
    assert updated.last_sync_at != ""


# ══════════════════════════════════════════════════════════════
# TEST 2: Weight Shard Registry — Orphaning & Replication
# ══════════════════════════════════════════════════════════════

def test_registry_orphan_shards():
    reg = make_registry(4, 4)
    orphaned = reg.orphan_miner_shards("miner-0")
    assert len(orphaned) == 1
    assert orphaned[0].state == ShardState.ORPHANED
    assert reg._total_orphaned == 1


def test_registry_replication_status():
    reg = make_registry(4, 4)
    status = reg.get_replication_status("test-model")
    assert status["total_shards"] == 4
    # All shards have only 1 replica, min_replicas=2, so all under-replicated
    assert len(status["under_replicated"]) == 4
    assert status["healthy"] == 0


def test_registry_replication_healthy():
    reg = WeightShardRegistry(policy=ReplicationPolicy(min_replicas=1))
    loc = ShardLocation(
        model_id="m1", miner_id="miner-0",
        layer_start=0, layer_end=8,
        state=ShardState.LOADED,
    )
    reg.register_shard(loc)
    status = reg.get_replication_status("m1")
    assert status["healthy"] == 1
    assert len(status["under_replicated"]) == 0


def test_registry_redistribution_plan():
    reg = make_registry(4, 4)
    # Orphan miner-0's shard
    reg.orphan_miner_shards("miner-0")

    available = [
        {"miner_id": "miner-new-1", "gpu_vram_gb": 24.0, "address": "us-west"},
        {"miner_id": "miner-new-2", "gpu_vram_gb": 16.0, "address": "us-east"},
    ]
    plan = reg.plan_redistribution("test-model", available)
    assert len(plan) > 0
    # The orphaned shard (0-8) should be in the plan
    shard_keys = [p["shard_key"] for p in plan]
    assert "test-model:L0-8" in shard_keys


# ══════════════════════════════════════════════════════════════
# TEST 3: Merkle Versioning & Integrity
# ══════════════════════════════════════════════════════════════

def test_version_merkle_root():
    reg = make_registry(4, 4)
    version = reg.create_version_snapshot("test-model", global_step=100)
    assert version.merkle_root != ""
    assert version.global_step == 100
    assert version.num_shards == 4
    assert version.total_params > 0
    assert reg.latest_version == 100


def test_version_deterministic():
    """Same shard hashes should produce same Merkle root."""
    v1 = WeightVersion(shard_hashes={"a": "hash1", "b": "hash2"})
    v2 = WeightVersion(shard_hashes={"b": "hash2", "a": "hash1"})
    assert v1.compute_merkle_root() == v2.compute_merkle_root()


def test_integrity_verification():
    reg = make_registry(4, 4)
    reg.create_version_snapshot("test-model", global_step=1)

    # Correct hash should pass
    shard_key = "test-model:L0-8"
    correct_hash = hashlib.sha256(b"shard-0").hexdigest()
    assert reg.verify_shard_integrity("miner-0", shard_key, correct_hash, 1) == True

    # Wrong hash should fail
    assert reg.verify_shard_integrity("miner-0", shard_key, "wrong_hash", 1) == False


def test_network_model_status():
    reg = make_registry(4, 4)
    reg.create_version_snapshot("test-model", global_step=50)
    status = reg.get_network_model_status("test-model")
    assert status["model_id"] == "test-model"
    assert status["total_shards"] == 4
    assert status["miners_holding_shards"] == 4
    assert status["latest_version"]["global_step"] == 50


def test_registry_stats():
    reg = make_registry(4, 4)
    stats = reg.get_stats()
    assert stats["total_locations"] == 4
    assert stats["loaded"] == 4
    assert stats["unique_shards"] == 4
    assert stats["miners_with_shards"] == 4
    assert stats["total_registered"] == 4


# ══════════════════════════════════════════════════════════════
# TEST 4: Shard Slicer
# ══════════════════════════════════════════════════════════════

def test_slicer_extract_layers():
    if not HAS_TORCH:
        return
    slicer = ShardSlicer()
    # Create a fake state dict with layer-named params
    state = {}
    for i in range(16):
        state[f"layers.{i}.attention.q_proj.weight"] = torch.randn(64, 64)
        state[f"layers.{i}.ffn.gate.weight"] = torch.randn(128, 64)
    state["embed_tokens.weight"] = torch.randn(1000, 64)
    state["lm_head.weight"] = torch.randn(1000, 64)
    state["norm.weight"] = torch.randn(64)

    slicer.load_model_weights("test", state)

    # Extract layers 4-8
    sliced = slicer.extract_layer_params("test", 4, 8)
    assert len(sliced) == 8  # 4 layers × 2 params each
    # Keys should be remapped: layers.4 → layers.0
    assert "layers.0.attention.q_proj.weight" in sliced
    assert "layers.3.ffn.gate.weight" in sliced
    assert "layers.4.attention.q_proj.weight" not in sliced

    # With embedding
    sliced_emb = slicer.extract_layer_params("test", 0, 4, include_embedding=True)
    assert "embed_tokens.weight" in sliced_emb
    assert len(sliced_emb) == 9  # 4 layers × 2 + embed

    # With lm_head
    sliced_head = slicer.extract_layer_params("test", 12, 16, include_lm_head=True)
    assert "lm_head.weight" in sliced_head
    assert "norm.weight" in sliced_head


def test_slicer_manifest():
    if not HAS_TORCH:
        return
    slicer = ShardSlicer()
    state = {}
    for i in range(8):
        state[f"layers.{i}.w"] = torch.randn(32, 32)
    slicer.load_model_weights("test", state)

    manifest = slicer.create_manifest("test", 2, 6, version=10)
    assert manifest is not None
    assert manifest.layer_start == 2
    assert manifest.layer_end == 6
    assert manifest.num_chunks > 0
    assert manifest.total_params > 0
    assert manifest.slice_hash != ""
    assert manifest.version == 10


def test_slicer_stream():
    if not HAS_TORCH:
        return
    slicer = ShardSlicer()
    state = {}
    for i in range(4):
        state[f"layers.{i}.w"] = torch.randn(16, 16)
    slicer.load_model_weights("test", state)

    chunks = []
    async def collect():
        async for header, data in slicer.stream_slice("test", 1, 3):
            chunks.append((header, data))

    asyncio.get_event_loop().run_until_complete(collect())
    assert len(chunks) == 2  # 2 layers, 1 param each
    assert all(ch[0]["hash"] != "" for ch in chunks)
    assert slicer.get_stats()["slices_served"] == 1


def test_slicer_weight_hash():
    if not HAS_TORCH:
        return
    slicer = ShardSlicer()
    state = {f"layers.{i}.w": torch.randn(8, 8) for i in range(4)}
    slicer.load_model_weights("test", state)

    h1 = slicer.compute_weight_hash("test", 0, 2)
    h2 = slicer.compute_weight_hash("test", 0, 2)
    h3 = slicer.compute_weight_hash("test", 2, 4)
    assert h1 == h2  # Same layers → same hash
    assert h1 != h3  # Different layers → different hash


def test_slicer_serialize_deserialize():
    if not HAS_TORCH:
        return
    slicer = ShardSlicer()
    t = torch.randn(4, 4)
    data = slicer._serialize_tensor(t)
    t2 = slicer.deserialize_tensor(data)
    assert torch.allclose(t, t2)


# ══════════════════════════════════════════════════════════════
# TEST 5: Transfer Plan
# ══════════════════════════════════════════════════════════════

def test_transfer_plan_with_peers():
    reg = make_registry(4, 4)
    req = WeightTransferRequest(
        requester_miner_id="new-miner",
        model_id="test-model",
        layer_start=0, layer_end=8,
    )
    plan = create_transfer_plan(req, reg)
    assert len(plan.sources) >= 2  # At least 1 peer + 1 fallback
    assert plan.sources[0]["type"] == "peer"
    assert plan.sources[-1]["type"] == "seed_slicer"


def test_transfer_plan_no_peers():
    reg = WeightShardRegistry()
    req = WeightTransferRequest(
        requester_miner_id="new-miner",
        model_id="test-model",
        layer_start=0, layer_end=8,
    )
    plan = create_transfer_plan(req, reg)
    assert len(plan.sources) == 1  # Only fallback
    assert plan.sources[0]["type"] == "seed_slicer"


# ══════════════════════════════════════════════════════════════
# TEST 6: Liquid Redistribution (ShardManager)
# ══════════════════════════════════════════════════════════════

def test_liquid_redistribute():
    sm = ShardManager()
    # 7B model on 24GB miners → 4 stages. Register 5 miners → 1 spare.
    model_config = {
        "num_parameters": 7_000_000_000,
        "num_layers": 32,
        "hidden_size": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "intermediate_size": 11008,
        "vocab_size": 32000,
    }

    for i in range(5):
        sm.register_miner(MinerCapability(
            miner_id=f"m-{i}",
            gpu_model="A100",
            gpu_vram_gb=24.0,
            location_region="us-west",
        ))

    groups = sm.form_pipeline_groups("test-model", model_config, target_redundancy=1)
    assert len(groups) >= 1
    group = groups[0]
    assert group.num_stages == 4
    assert group.status == PipelineStatus.LOADING

    assigned_miner = group.miner_ids[0]

    # Simulate disconnect
    affected = sm.handle_miner_disconnect(assigned_miner)
    assert affected == group.group_id
    assert group.status == PipelineStatus.DEGRADED

    # Spare miner should fill the gap
    available = sm.get_available_miners()
    assert len(available) >= 1, "No spare miner available for redistribution"

    new_assignments = sm.liquid_redistribute(group.group_id)
    assert len(new_assignments) == 1
    assert group.is_complete


def test_auto_heal():
    sm = ShardManager()
    # 7B on 24GB → 4 stages
    model_config = {
        "num_parameters": 7_000_000_000,
        "num_layers": 32,
        "hidden_size": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "intermediate_size": 11008,
        "vocab_size": 32000,
    }

    # Register exactly 4 miners (fills one pipeline, no spares)
    for i in range(4):
        sm.register_miner(MinerCapability(
            miner_id=f"m-{i}", gpu_model="A100", gpu_vram_gb=24.0,
        ))

    groups = sm.form_pipeline_groups("test-model", model_config, target_redundancy=1)
    assert len(groups) >= 1
    group = groups[0]
    assigned_miner = group.miner_ids[0]

    # Disconnect one miner
    sm.handle_miner_disconnect(assigned_miner)
    assert group.status == PipelineStatus.DEGRADED

    # Register a new spare
    sm.register_miner(MinerCapability(
        miner_id="m-spare", gpu_model="A100", gpu_vram_gb=24.0,
    ))

    # Auto-heal
    healed = sm.auto_heal_degraded_pipelines()
    assert group.group_id in healed
    assert group.is_complete


# ══════════════════════════════════════════════════════════════
# TEST 7: Layer Weight Chunk
# ══════════════════════════════════════════════════════════════

def test_chunk_hash():
    chunk = LayerWeightChunk(
        model_id="m1",
        layer_index=5,
        param_name="layers.5.w",
        data=b"fake_tensor_data_here",
        shape=(64, 64),
    )
    h = chunk.compute_hash()
    assert h == hashlib.sha256(b"fake_tensor_data_here").hexdigest()
    assert chunk.size_bytes == len(b"fake_tensor_data_here")


def test_chunk_header():
    chunk = LayerWeightChunk(
        model_id="m1", layer_index=3, param_name="layers.3.w",
        data=b"data", shape=(4, 4), dtype="float16",
    )
    chunk.compute_hash()
    header = chunk.to_header()
    assert header["model_id"] == "m1"
    assert header["layer_index"] == 3
    assert header["shape"] == [4, 4]
    assert header["hash"] != ""


def test_slice_manifest_hash():
    manifest = SliceManifest(
        model_id="m1", layer_start=0, layer_end=8,
        chunk_headers=[
            {"layer_index": 0, "param_name": "a", "hash": "h1"},
            {"layer_index": 1, "param_name": "b", "hash": "h2"},
        ],
    )
    h = manifest.compute_slice_hash()
    assert h != ""
    # Deterministic
    assert manifest.compute_slice_hash() == h


# ══════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sections = [
        ("TEST 1: Weight Shard Registry — Registration & Queries", [
            ("reg_register", test_registry_register),
            ("reg_multiple_replicas", test_registry_multiple_replicas),
            ("reg_find_sources", test_registry_find_sources),
            ("reg_find_sources_sorted", test_registry_find_sources_sorted_by_priority),
            ("reg_find_overlapping", test_registry_find_overlapping),
            ("reg_shard_map", test_registry_shard_map),
            ("reg_miner_shards", test_registry_miner_shards),
            ("reg_update_state", test_registry_update_state),
        ]),
        ("TEST 2: Weight Shard Registry — Orphaning & Replication", [
            ("reg_orphan_shards", test_registry_orphan_shards),
            ("reg_replication_status", test_registry_replication_status),
            ("reg_replication_healthy", test_registry_replication_healthy),
            ("reg_redistribution_plan", test_registry_redistribution_plan),
        ]),
        ("TEST 3: Merkle Versioning & Integrity", [
            ("ver_merkle_root", test_version_merkle_root),
            ("ver_deterministic", test_version_deterministic),
            ("ver_integrity", test_integrity_verification),
            ("ver_model_status", test_network_model_status),
            ("ver_registry_stats", test_registry_stats),
        ]),
        ("TEST 4: Shard Slicer", [
            ("slicer_extract_layers", test_slicer_extract_layers),
            ("slicer_manifest", test_slicer_manifest),
            ("slicer_stream", test_slicer_stream),
            ("slicer_weight_hash", test_slicer_weight_hash),
            ("slicer_serialize_deserialize", test_slicer_serialize_deserialize),
        ]),
        ("TEST 5: Transfer Plan", [
            ("plan_with_peers", test_transfer_plan_with_peers),
            ("plan_no_peers", test_transfer_plan_no_peers),
        ]),
        ("TEST 6: Liquid Redistribution", [
            ("liquid_redistribute", test_liquid_redistribute),
            ("auto_heal", test_auto_heal),
        ]),
        ("TEST 7: Layer Weight Chunk & Manifest", [
            ("chunk_hash", test_chunk_hash),
            ("chunk_header", test_chunk_header),
            ("manifest_hash", test_slice_manifest_hash),
        ]),
    ]

    for section_name, tests in sections:
        print(f"\n{'=' * 60}")
        print(f"{section_name}")
        print(f"{'=' * 60}")
        for name, fn in tests:
            run_test(name, fn)

    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n  PASSED: {passed}/{passed + failed}")
    print(f"  FAILED: {failed}/{passed + failed}")

    if errors:
        print(f"\n  ERRORS:")
        for name, err in errors:
            print(f"    {name}: {err}")

    sys.exit(1 if failed else 0)
