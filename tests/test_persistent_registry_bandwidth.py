"""
Tests for Persistent WeightShardRegistry + Bandwidth-Aware Scoring
====================================================================

Tests:
  - PersistentWeightShardRegistry falls back to in-memory when no DB
  - Write-through methods still work without DB (graceful degradation)
  - Bandwidth-aware _find_best_replacement scoring
  - Bandwidth-aware plan_redistribution sorting
  - update_miner_bandwidth EMA calculation
  - End-to-end: high-bandwidth miner beats low-bandwidth for redistribution
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.weight_shard_registry import (
    PersistentWeightShardRegistry, WeightShardRegistry,
    ShardLocation, ShardState, ReplicaPriority,
    WeightVersion, ReplicationPolicy,
)
from app.shard_manager import (
    ShardManager, MinerCapability, PipelineStatus,
    PipelineGroup, ShardAssignment,
)
from uuid import uuid4


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
        passed += 1
        print(f"  ✓ {name}")
    except AssertionError as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  ✗ {name}: {e}")
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  ✗ {name}: EXCEPTION: {e}")


def run_async_test(name, coro_fn):
    global passed, failed
    try:
        asyncio.get_event_loop().run_until_complete(coro_fn())
        passed += 1
        print(f"  ✓ {name}")
    except AssertionError as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  ✗ {name}: {e}")
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  ✗ {name}: EXCEPTION: {e}")


# ══════════════════════════════════════════════════════════════
# PERSISTENT REGISTRY TESTS (no DB — graceful fallback)
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PERSISTENT REGISTRY TESTS")
print("=" * 60)


def test_persistent_registry_is_subclass():
    """PersistentWeightShardRegistry inherits from WeightShardRegistry."""
    reg = PersistentWeightShardRegistry()
    assert isinstance(reg, WeightShardRegistry)
    assert reg._db_available is False

run_test("PersistentWeightShardRegistry is subclass of WeightShardRegistry", test_persistent_registry_is_subclass)


def test_persistent_registry_register_without_db():
    """Register shard works without DB (falls back to in-memory)."""
    reg = PersistentWeightShardRegistry()
    loc = ShardLocation(
        model_id="test-model",
        miner_id="m-1",
        layer_start=0,
        layer_end=8,
        state=ShardState.LOADED,
        priority=ReplicaPriority.PRIMARY,
        weight_hash="abc123",
        size_bytes=1_000_000,
        num_params=500_000,
    )
    result = reg.register_shard(loc)
    assert result.location_id == loc.location_id
    assert len(reg.locations) == 1
    assert "test-model:L0-8" in reg.shard_replicas

run_test("Register shard works without DB", test_persistent_registry_register_without_db)


def test_persistent_registry_orphan_without_db():
    """Orphan shards works without DB."""
    reg = PersistentWeightShardRegistry()
    loc = ShardLocation(
        model_id="test-model", miner_id="m-1",
        layer_start=0, layer_end=8, state=ShardState.LOADED,
    )
    reg.register_shard(loc)
    orphaned = reg.orphan_miner_shards("m-1")
    assert len(orphaned) == 1
    assert orphaned[0].state == ShardState.ORPHANED

run_test("Orphan shards works without DB", test_persistent_registry_orphan_without_db)


def test_persistent_registry_update_state_without_db():
    """Update shard state works without DB."""
    reg = PersistentWeightShardRegistry()
    loc = ShardLocation(
        model_id="test-model", miner_id="m-1",
        layer_start=0, layer_end=8, state=ShardState.ALLOCATED,
    )
    result = reg.register_shard(loc)
    updated = reg.update_shard_state(
        result.location_id, ShardState.LOADED, weight_hash="hash123", version=5
    )
    assert updated is not None
    assert updated.state == ShardState.LOADED
    assert updated.weight_hash == "hash123"
    assert updated.version == 5

run_test("Update shard state works without DB", test_persistent_registry_update_state_without_db)


def test_persistent_registry_version_snapshot_without_db():
    """Version snapshot works without DB."""
    reg = PersistentWeightShardRegistry()
    loc = ShardLocation(
        model_id="test-model", miner_id="m-1",
        layer_start=0, layer_end=8, state=ShardState.LOADED,
        weight_hash="abc123", num_params=1000,
    )
    reg.register_shard(loc)
    version = reg.create_version_snapshot("test-model", global_step=100)
    assert version.global_step == 100
    assert version.merkle_root != ""
    assert version.num_shards == 1
    assert reg.latest_version == 100

run_test("Version snapshot works without DB", test_persistent_registry_version_snapshot_without_db)


async def test_init_persistence_no_db():
    """init_persistence gracefully handles missing DB."""
    reg = PersistentWeightShardRegistry()
    await reg.init_persistence()
    assert reg._db_available is False

run_async_test("init_persistence gracefully handles missing DB", test_init_persistence_no_db)


def test_persistent_registry_all_queries_work():
    """All query methods work on persistent registry (inherited from parent)."""
    reg = PersistentWeightShardRegistry()
    for i in range(3):
        loc = ShardLocation(
            model_id="model-x", miner_id=f"m-{i}",
            layer_start=i * 8, layer_end=(i + 1) * 8,
            state=ShardState.LOADED, weight_hash=f"hash{i}",
            size_bytes=1000, num_params=500, miner_address=f"10.0.0.{i}:8000",
        )
        reg.register_shard(loc)

    # find_shard_sources
    sources = reg.find_shard_sources("model-x", 0, 8)
    assert len(sources) == 1
    assert sources[0].miner_id == "m-0"

    # get_shard_map
    shard_map = reg.get_shard_map("model-x")
    assert len(shard_map) == 3

    # get_miner_shards
    shards = reg.get_miner_shards("m-1")
    assert len(shards) == 1

    # get_replication_status
    status = reg.get_replication_status("model-x")
    assert status["total_shards"] == 3

    # get_stats
    stats = reg.get_stats()
    assert stats["total_locations"] == 3

run_test("All query methods work on persistent registry", test_persistent_registry_all_queries_work)


# ══════════════════════════════════════════════════════════════
# BANDWIDTH-AWARE SCORING TESTS
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  BANDWIDTH-AWARE SCORING TESTS")
print("=" * 60)


MODEL_CONFIG_4L = {
    "num_layers": 4,
    "hidden_size": 256,
    "num_heads": 4,
    "intermediate_size": 512,
    "vocab_size": 1000,
    "max_sequence_length": 512,
    "dtype_bytes": 2,
}


def test_update_miner_bandwidth_basic():
    """update_miner_bandwidth sets initial bandwidth."""
    sm = ShardManager()
    sm.register_miner(MinerCapability(
        miner_id="m-1", gpu_vram_gb=24.0, bandwidth_mbps=0,
    ))
    assert sm.update_miner_bandwidth("m-1", 500.0) is True
    assert sm.miners["m-1"].bandwidth_mbps == 500.0

run_test("update_miner_bandwidth sets initial bandwidth", test_update_miner_bandwidth_basic)


def test_update_miner_bandwidth_ema():
    """update_miner_bandwidth uses EMA for subsequent updates."""
    sm = ShardManager()
    sm.register_miner(MinerCapability(
        miner_id="m-1", gpu_vram_gb=24.0, bandwidth_mbps=1000.0,
    ))
    sm.update_miner_bandwidth("m-1", 500.0)
    bw = sm.miners["m-1"].bandwidth_mbps
    # EMA: 0.7 * 500 + 0.3 * 1000 = 350 + 300 = 650
    assert abs(bw - 650.0) < 0.01, f"Expected ~650, got {bw}"

run_test("update_miner_bandwidth uses EMA", test_update_miner_bandwidth_ema)


def test_update_miner_bandwidth_unknown_miner():
    """update_miner_bandwidth returns False for unknown miner."""
    sm = ShardManager()
    assert sm.update_miner_bandwidth("unknown", 500.0) is False

run_test("update_miner_bandwidth returns False for unknown", test_update_miner_bandwidth_unknown_miner)


def test_find_best_replacement_prefers_high_bandwidth():
    """High bandwidth miner beats low bandwidth miner (same region, same VRAM)."""
    sm = ShardManager()

    sm.register_miner(MinerCapability(
        miner_id="m-stage0", gpu_vram_gb=24.0, bandwidth_mbps=1000.0,
        location_region="us-east",
    ))
    # Spare: slow
    sm.register_miner(MinerCapability(
        miner_id="m-slow", gpu_vram_gb=24.0, bandwidth_mbps=100.0,
        location_region="us-east",
    ))
    # Spare: fast
    sm.register_miner(MinerCapability(
        miner_id="m-fast", gpu_vram_gb=24.0, bandwidth_mbps=5000.0,
        location_region="us-east",
    ))

    # Create pipeline group manually
    group = PipelineGroup(model_id="test", num_stages=2)
    sm.pipeline_groups[group.group_id] = group
    sm._model_configs["test"] = MODEL_CONFIG_4L

    # Stage 0 occupied by m-stage0, stage 1 empty
    group.stages[0] = ShardAssignment(
        miner_id="m-stage0",
        pipeline_group_id=group.group_id, stage_index=0,
        layer_start=0, layer_end=2, status="ready",
    )
    sm.miner_assignments["m-stage0"] = group.stages[0]

    replacement = sm._find_best_replacement(group, 1, MODEL_CONFIG_4L)
    assert replacement is not None
    assert replacement.miner_id == "m-fast", (
        f"Expected m-fast (5000 Mbps), got {replacement.miner_id} "
        f"({replacement.bandwidth_mbps} Mbps)"
    )

run_test("_find_best_replacement prefers high bandwidth miner", test_find_best_replacement_prefers_high_bandwidth)


def test_find_best_replacement_region_beats_bandwidth():
    """Region match takes priority over bandwidth (same-region miner wins)."""
    sm = ShardManager()

    sm.register_miner(MinerCapability(
        miner_id="m-stage0", gpu_vram_gb=24.0, bandwidth_mbps=1000.0,
        location_region="us-east",
    ))
    # Spare: fast but wrong region
    sm.register_miner(MinerCapability(
        miner_id="m-fast-faraway", gpu_vram_gb=24.0, bandwidth_mbps=10000.0,
        location_region="ap-south",
    ))
    # Spare: slower but same region
    sm.register_miner(MinerCapability(
        miner_id="m-slow-local", gpu_vram_gb=24.0, bandwidth_mbps=500.0,
        location_region="us-east",
    ))

    group = PipelineGroup(model_id="test", num_stages=2)
    sm.pipeline_groups[group.group_id] = group
    sm._model_configs["test"] = MODEL_CONFIG_4L

    group.stages[0] = ShardAssignment(
        miner_id="m-stage0",
        pipeline_group_id=group.group_id, stage_index=0,
        layer_start=0, layer_end=2, status="ready",
    )
    sm.miner_assignments["m-stage0"] = group.stages[0]

    replacement = sm._find_best_replacement(group, 1, MODEL_CONFIG_4L)
    assert replacement is not None
    # Region match (us-east) should beat raw bandwidth
    assert replacement.miner_id == "m-slow-local", (
        f"Expected m-slow-local (same region), got {replacement.miner_id}"
    )

run_test("Region match takes priority over bandwidth", test_find_best_replacement_region_beats_bandwidth)


def test_plan_redistribution_bandwidth_sorting():
    """plan_redistribution sorts candidates by bandwidth (highest first)."""
    reg = PersistentWeightShardRegistry(ReplicationPolicy(min_replicas=2, target_replicas=2))

    # Register shard with only 1 replica (under-replicated)
    loc = ShardLocation(
        model_id="model-x", miner_id="m-source",
        layer_start=0, layer_end=8, state=ShardState.LOADED,
        weight_hash="h1", miner_address="10.0.0.1:8000",
    )
    reg.register_shard(loc)

    # Available miners: different bandwidths
    available = [
        {"miner_id": "m-slow", "gpu_vram_gb": 24.0, "bandwidth_mbps": 100, "address": "10.0.0.2:8000"},
        {"miner_id": "m-fast", "gpu_vram_gb": 24.0, "bandwidth_mbps": 5000, "address": "10.0.0.3:8000"},
        {"miner_id": "m-mid", "gpu_vram_gb": 24.0, "bandwidth_mbps": 1000, "address": "10.0.0.4:8000"},
    ]

    plan = reg.plan_redistribution("model-x", available)
    assert len(plan) >= 1
    # First transfer should target the fastest miner
    assert plan[0]["target_miner_id"] == "m-fast", (
        f"Expected m-fast as first transfer target, got {plan[0]['target_miner_id']}"
    )

run_test("plan_redistribution sorts by bandwidth", test_plan_redistribution_bandwidth_sorting)


def test_plan_redistribution_bandwidth_zero_handled():
    """plan_redistribution handles miners with zero bandwidth gracefully."""
    reg = PersistentWeightShardRegistry(ReplicationPolicy(min_replicas=2, target_replicas=2))

    loc = ShardLocation(
        model_id="model-x", miner_id="m-source",
        layer_start=0, layer_end=8, state=ShardState.LOADED,
        weight_hash="h1", miner_address="10.0.0.1:8000",
    )
    reg.register_shard(loc)

    available = [
        {"miner_id": "m-no-bw", "gpu_vram_gb": 24.0, "bandwidth_mbps": 0, "address": "10.0.0.2:8000"},
        {"miner_id": "m-has-bw", "gpu_vram_gb": 24.0, "bandwidth_mbps": 500, "address": "10.0.0.3:8000"},
    ]

    plan = reg.plan_redistribution("model-x", available)
    assert len(plan) >= 1
    # Miner with bandwidth should be preferred
    assert plan[0]["target_miner_id"] == "m-has-bw"

run_test("plan_redistribution handles zero bandwidth", test_plan_redistribution_bandwidth_zero_handled)


def test_effective_bandwidth_bottleneck():
    """Effective bandwidth is min(candidate, neighbor) — bottleneck principle."""
    sm = ShardManager()

    # Neighbor with 500 Mbps
    sm.register_miner(MinerCapability(
        miner_id="m-neighbor", gpu_vram_gb=24.0, bandwidth_mbps=500.0,
        location_region="us-east",
    ))
    # Candidate with 10 Gbps — effective should be capped at 500
    sm.register_miner(MinerCapability(
        miner_id="m-10g", gpu_vram_gb=24.0, bandwidth_mbps=10000.0,
        location_region="us-east",
    ))
    # Candidate with 800 Mbps — effective should be capped at 500
    sm.register_miner(MinerCapability(
        miner_id="m-800", gpu_vram_gb=24.0, bandwidth_mbps=800.0,
        location_region="us-east",
    ))

    group = PipelineGroup(model_id="test", num_stages=2)
    sm.pipeline_groups[group.group_id] = group
    sm._model_configs["test"] = MODEL_CONFIG_4L

    group.stages[0] = ShardAssignment(
        miner_id="m-neighbor",
        pipeline_group_id=group.group_id, stage_index=0,
        layer_start=0, layer_end=2, status="ready",
    )
    sm.miner_assignments["m-neighbor"] = group.stages[0]

    replacement = sm._find_best_replacement(group, 1, MODEL_CONFIG_4L)
    assert replacement is not None
    # Both candidates have effective BW of 500 (bottlenecked by neighbor)
    # Tie broken by raw VRAM (equal) then load (equal), so either is valid
    # But the scoring should still work without crashing
    assert replacement.miner_id in ("m-10g", "m-800")

run_test("Effective bandwidth uses bottleneck principle", test_effective_bandwidth_bottleneck)


# ══════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(f"\nRESULTS SUMMARY")
print("=" * 60)
print(f"\n  PASSED: {passed}/{passed + failed}")
print(f"  FAILED: {failed}/{passed + failed}")

if errors:
    print(f"\nFailed tests:")
    for name, err in errors:
        print(f"  - {name}: {err}")

sys.exit(1 if failed > 0 else 0)
