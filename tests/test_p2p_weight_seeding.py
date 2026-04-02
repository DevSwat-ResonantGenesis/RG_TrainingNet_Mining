"""
Tests for P2P Weight Seeding via WebRTC DataChannels
======================================================

Tests:
  - WebRTC capability tracking in WeightShardRegistry
  - Transfer plan prioritizes WebRTC peers over HTTP
  - WebRTC transfer source selection and routing
  - Weight serving via DataChannel
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.weight_shard_registry import (
    WeightShardRegistry, ShardLocation, ShardState, ReplicaPriority,
    ReplicationPolicy,
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
# P2P WEIGHT SEEDING TESTS
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  P2P WEIGHT SEEDING TESTS")
print("=" * 60)


def test_webrtc_capability_tracking():
    """Test tracking of WebRTC capabilities in ShardLocation."""
    registry = WeightShardRegistry()
    
    # Create a shard location
    location = ShardLocation(
        model_id="test-model",
        miner_id="miner-1",
        layer_start=0,
        layer_end=8,
        state=ShardState.LOADED,
        priority=ReplicaPriority.PRIMARY,
        size_bytes=1000000,
        num_params=500000,
    )
    
    # Initially no WebRTC
    assert not location.can_serve_p2p
    assert location.transfer_priority == 50  # HTTP priority
    
    # Enable WebRTC
    location.webrtc_peer_id = "peer-abc123"
    location.has_webrtc = True
    location.webrtc_bandwidth = 1000.0
    
    # Now should have P2P capability
    assert location.can_serve_p2p
    assert location.transfer_priority == 1100  # 100 + bandwidth
    
    # Check to_dict includes WebRTC fields
    loc_dict = location.to_dict()
    assert "webrtc_peer_id" in loc_dict
    assert loc_dict["webrtc_peer_id"] == "peer-abc123"
    assert loc_dict["has_webrtc"] == True
    assert loc_dict["webrtc_bandwidth"] == 1000.0

run_test("WebRTC capability tracking", test_webrtc_capability_tracking)


def test_transfer_plan_prioritizes_webrtc():
    """Test that transfer plans prioritize WebRTC peers over HTTP."""
    registry = WeightShardRegistry()
    registry.policy = ReplicationPolicy(min_replicas=2, target_replicas=3)
    
    # Create WebRTC-enabled source
    webrtc_source = ShardLocation(
        model_id="test-model",
        miner_id="miner-webrtc",
        layer_start=0,
        layer_end=8,
        state=ShardState.LOADED,
        priority=ReplicaPriority.PRIMARY,
        size_bytes=1000000,
        num_params=500000,
        webrtc_peer_id="peer-webrtc",
        has_webrtc=True,
        webrtc_bandwidth=1000.0,
    )
    
    # Create HTTP-only source
    http_source = ShardLocation(
        model_id="test-model",
        miner_id="miner-http",
        layer_start=0,
        layer_end=8,
        state=ShardState.LOADED,
        priority=ReplicaPriority.PRIMARY,
        size_bytes=1000000,
        num_params=500000,
        miner_address="http://miner-http:8701",
    )
    
    # Register both sources
    registry.register_shard(webrtc_source)
    registry.register_shard(http_source)
    
    # Find sources - WebRTC should be first
    sources = registry.find_shard_sources("test-model", 0, 8)
    assert len(sources) == 2
    assert sources[0].can_serve_p2p  # WebRTC source first
    assert not sources[1].can_serve_p2p  # HTTP source second
    
    # Create transfer plan - need under-replicated shards
    registry.policy = ReplicationPolicy(min_replicas=3, target_replicas=4)  # Need more replicas
    available_miners = [
        {"miner_id": "miner-target", "address": "http://target:8701", "bandwidth_mbps": 500},
    ]
    
    plan = registry.plan_redistribution("test-model", available_miners)
    assert len(plan) > 0
    
    transfer = plan[0]
    assert transfer["transfer_method"] == "webrtc"
    assert transfer["source_webrtc_peer_id"] == "peer-webrtc"
    assert transfer["source_has_webrtc"] == True

run_test("Transfer plan prioritizes WebRTC", test_transfer_plan_prioritizes_webrtc)


def test_update_miner_webrtc_info():
    """Test updating WebRTC capabilities for all miner shards."""
    registry = WeightShardRegistry()
    
    # Register multiple shards for the same miner
    for i in range(3):
        location = ShardLocation(
            model_id="test-model",
            miner_id="miner-1",
            layer_start=i*8,
            layer_end=(i+1)*8,
            state=ShardState.LOADED,
            priority=ReplicaPriority.PRIMARY,
            size_bytes=1000000,
            num_params=500000,
        )
        registry.register_shard(location)
    
    # Initially no WebRTC
    for loc in registry.locations.values():
        if loc.miner_id == "miner-1":
            assert not loc.can_serve_p2p
    
    # Update WebRTC info
    registry.update_miner_webrtc_info(
        miner_id="miner-1",
        webrtc_peer_id="peer-abc123",
        bandwidth_mbps=800.0,
    )
    
    # All miner-1 shards should now have WebRTC
    updated_count = 0
    for loc in registry.locations.values():
        if loc.miner_id == "miner-1":
            assert loc.can_serve_p2p
            assert loc.webrtc_peer_id == "peer-abc123"
            assert loc.webrtc_bandwidth == 800.0
            updated_count += 1
    
    assert updated_count == 3

run_test("Update miner WebRTC info", test_update_miner_webrtc_info)


def test_webrtc_source_selection():
    """Test WebRTC source selection with bandwidth prioritization."""
    registry = WeightShardRegistry()
    
    # Create multiple WebRTC sources with different bandwidths
    sources = []
    for i, bandwidth in enumerate([200, 1000, 500]):
        source = ShardLocation(
            model_id="test-model",
            miner_id=f"miner-{i}",
            layer_start=0,
            layer_end=8,
            state=ShardState.LOADED,
            priority=ReplicaPriority.PRIMARY,
            size_bytes=1000000,
            num_params=500000,
            webrtc_peer_id=f"peer-{i}",
            has_webrtc=True,
            webrtc_bandwidth=bandwidth,
        )
        sources.append(source)
        registry.register_shard(source)
    
    # Find sources - should be sorted by bandwidth (highest first)
    found_sources = registry.find_shard_sources("test-model", 0, 8)
    assert len(found_sources) == 3
    
    # Check bandwidth ordering
    assert found_sources[0].webrtc_bandwidth == 1000.0  # Highest bandwidth first
    assert found_sources[1].webrtc_bandwidth == 500.0
    assert found_sources[2].webrtc_bandwidth == 200.0   # Lowest bandwidth last

run_test("WebRTC source selection with bandwidth", test_webrtc_source_selection)


def test_mixed_webrtc_http_sources():
    """Test transfer plan with mixed WebRTC and HTTP sources."""
    registry = WeightShardRegistry()
    registry.policy = ReplicationPolicy(min_replicas=1, target_replicas=2)
    
    # Create WebRTC source
    webrtc_source = ShardLocation(
        model_id="test-model",
        miner_id="miner-webrtc",
        layer_start=0,
        layer_end=8,
        state=ShardState.LOADED,
        priority=ReplicaPriority.HOT_SPARE,  # Lower priority
        size_bytes=1000000,
        num_params=500000,
        webrtc_peer_id="peer-webrtc",
        has_webrtc=True,
        webrtc_bandwidth=1000.0,
    )
    
    # Create HTTP source with higher replica priority
    http_source = ShardLocation(
        model_id="test-model",
        miner_id="miner-http",
        layer_start=0,
        layer_end=8,
        state=ShardState.LOADED,
        priority=ReplicaPriority.PRIMARY,  # Higher priority
        size_bytes=1000000,
        num_params=500000,
        miner_address="http://miner-http:8701",
    )
    
    registry.register_shard(webrtc_source)
    registry.register_shard(http_source)
    
    # WebRTC should still be first despite lower replica priority
    sources = registry.find_shard_sources("test-model", 0, 8)
    assert len(sources) == 2
    assert sources[0].can_serve_p2p  # WebRTC first due to transfer priority
    assert sources[1].priority == ReplicaPriority.PRIMARY  # HTTP second

run_test("Mixed WebRTC/HTTP source prioritization", test_mixed_webrtc_http_sources)


def test_transfer_plan_with_no_webrtc():
    """Test transfer plan falls back to HTTP when no WebRTC available."""
    registry = WeightShardRegistry()
    registry.policy = ReplicationPolicy(min_replicas=1, target_replicas=2)
    
    # Create only HTTP source
    http_source = ShardLocation(
        model_id="test-model",
        miner_id="miner-http",
        layer_start=0,
        layer_end=8,
        state=ShardState.LOADED,
        priority=ReplicaPriority.PRIMARY,
        size_bytes=1000000,
        num_params=500000,
        miner_address="http://miner-http:8701",
    )
    
    registry.register_shard(http_source)
    
    # Create transfer plan - need under-replicated shards
    registry.policy = ReplicationPolicy(min_replicas=2, target_replicas=3)  # Need more replicas
    available_miners = [
        {"miner_id": "miner-target", "address": "http://target:8701", "bandwidth_mbps": 500},
    ]
    
    plan = registry.plan_redistribution("test-model", available_miners)
    assert len(plan) > 0
    
    transfer = plan[0]
    assert transfer["transfer_method"] == "http"
    assert transfer["source_webrtc_peer_id"] is None
    assert transfer["source_has_webrtc"] == False
    assert transfer["source_address"] == "http://miner-http:8701"

run_test("Transfer plan fallback to HTTP", test_transfer_plan_with_no_webrtc)


def test_registry_stats_includes_webrtc():
    """Test registry stats include WebRTC information."""
    registry = WeightShardRegistry()
    
    # Add some locations
    for i in range(5):
        location = ShardLocation(
            model_id="test-model",
            miner_id=f"miner-{i}",
            layer_start=0,
            layer_end=8,
            state=ShardState.LOADED,
            priority=ReplicaPriority.PRIMARY,
            size_bytes=1000000,
            num_params=500000,
        )
        if i < 3:  # First 3 have WebRTC
            location.webrtc_peer_id = f"peer-{i}"
            location.has_webrtc = True
            location.webrtc_bandwidth = 1000.0
        registry.register_shard(location)
    
    stats = registry.get_registry_stats()
    assert "webrtc_enabled_locations" in stats
    assert stats["webrtc_enabled_locations"] == 3
    assert stats["total_locations"] == 5

run_test("Registry stats include WebRTC", test_registry_stats_includes_webrtc)


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
