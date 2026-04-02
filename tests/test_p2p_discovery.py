"""
Tests for P2P Discovery Service — WebRTC NAT Traversal
========================================================

Tests:
  - Peer registration and cleanup
  - Pipeline peer assignment and connection initiation
  - Signaling message routing (offer, answer, ICE candidates)
  - Pipeline status tracking
  - Graceful handling of missing WebRTC dependencies
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.p2p_discovery import P2PDiscovery, PeerInfo, SignalingMessage


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
        import traceback
        traceback.print_exc()
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  ✗ {name}: EXCEPTION: {e}")
        import traceback
        traceback.print_exc()


# Mock WebSocket for testing
class MockWebSocket:
    def __init__(self):
        self.sent_messages = []
        self.closed = False
    
    def send_text(self, message: str):
        """Synchronous version for testing."""
        self.sent_messages.append(json.loads(message))
    
    async def close(self):
        self.closed = True


# ══════════════════════════════════════════════════════════════
# P2P DISCOVERY TESTS
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  P2P DISCOVERY TESTS")
print("=" * 60)


def test_peer_registration():
    """Register and unregister miners."""
    discovery = P2PDiscovery()
    ws = MockWebSocket()
    
    # Register miner
    peer_id = discovery.register_miner("miner-1", ws)
    assert peer_id.startswith("peer-")
    assert len(peer_id) == 17  # peer- + 12 hex chars
    
    # Check registration
    assert "miner-1" in discovery.miner_to_peer
    assert peer_id in discovery.peers
    assert discovery.peers[peer_id].miner_id == "miner-1"
    assert discovery.peers[peer_id].websocket == ws
    
    # Unregister
    removed_peer_id = discovery.unregister_miner("miner-1")
    assert removed_peer_id == peer_id
    assert "miner-1" not in discovery.miner_to_peer
    assert peer_id not in discovery.peers

run_test("Peer registration and cleanup", test_peer_registration)


def test_pipeline_assignment():
    """Assign miners to pipeline and establish peer links."""
    discovery = P2PDiscovery()
    
    # Register 3 miners for a 3-stage pipeline
    miners = {}
    for i in range(3):
        ws = MockWebSocket()
        peer_id = discovery.register_miner(f"miner-{i}", ws)
        miners[f"miner-{i}"] = peer_id
    
    # Create pipeline assignment
    assignments = [
        {"miner_id": "miner-0", "stage_index": 0},
        {"miner_id": "miner-1", "stage_index": 1},
        {"miner_id": "miner-2", "stage_index": 2},
    ]
    
    discovery.assign_pipeline_peers("pipeline-123", assignments)
    
    # Check pipeline group created
    assert "pipeline-123" in discovery.pipeline_peers
    assert len(discovery.pipeline_peers["pipeline-123"]) == 3
    
    # Check peer links (upstream/downstream)
    peer_0 = discovery.peers[miners["miner-0"]]
    peer_1 = discovery.peers[miners["miner-1"]]
    peer_2 = discovery.peers[miners["miner-2"]]
    
    # miner-0 (stage 0) should have downstream only
    assert peer_0.upstream_peer_id is None
    assert peer_0.downstream_peer_id == miners["miner-1"]
    
    # miner-1 (stage 1) should have both
    assert peer_1.upstream_peer_id == miners["miner-0"]
    assert peer_1.downstream_peer_id == miners["miner-2"]
    
    # miner-2 (stage 2) should have upstream only
    assert peer_2.upstream_peer_id == miners["miner-1"]
    assert peer_2.downstream_peer_id is None

run_test("Pipeline assignment with peer links", test_pipeline_assignment)


def test_webrtc_connection_initiation():
    """Test WebRTC connection initiation between peers."""
    discovery = P2PDiscovery()
    
    # Register 2 miners
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()
    peer1_id = discovery.register_miner("miner-1", ws1)
    peer2_id = discovery.register_miner("miner-2", ws2)
    
    # Assign to pipeline
    assignments = [
        {"miner_id": "miner-1", "stage_index": 0},
        {"miner_id": "miner-2", "stage_index": 1},
    ]
    discovery.assign_pipeline_peers("pipeline-123", assignments)
    
    # Check that offer creation message was sent to initiator
    assert len(ws1.sent_messages) == 1
    offer_msg = ws1.sent_messages[0]
    assert offer_msg["type"] == "create-offer"
    assert offer_msg["data"]["target_peer_id"] == peer2_id
    assert offer_msg["data"]["target_miner_id"] == "miner-2"

run_test("WebRTC connection initiation", test_webrtc_connection_initiation)


def test_signaling_offer_routing():
    """Test WebRTC offer message routing."""
    discovery = P2PDiscovery()
    
    # Register 2 miners
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()
    peer1_id = discovery.register_miner("miner-1", ws1)
    peer2_id = discovery.register_miner("miner-2", ws2)
    
    # Simulate offer from peer1 to peer2
    offer_message = {
        "type": "offer",
        "target_peer_id": peer2_id,
        "offer": "mock-offer-data",
    }
    
    discovery.handle_signaling_message("miner-1", offer_message)
    
    # Check offer was forwarded to peer2
    assert len(ws2.sent_messages) == 1
    forwarded = ws2.sent_messages[0]
    assert forwarded["type"] == "offer"
    assert forwarded["from_peer_id"] == peer1_id
    assert forwarded["to_peer_id"] == peer2_id
    assert forwarded["data"]["offer"] == "mock-offer-data"
    
    # Check offer stored in peer1
    peer1 = discovery.peers[peer1_id]
    assert peer1.offer["offer"] == "mock-offer-data"

run_test("Signaling offer routing", test_signaling_offer_routing)


def test_signaling_answer_routing():
    """Test WebRTC answer message routing."""
    discovery = P2PDiscovery()
    
    # Register 2 miners
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()
    peer1_id = discovery.register_miner("miner-1", ws1)
    peer2_id = discovery.register_miner("miner-2", ws2)
    
    # Simulate answer from peer2 to peer1
    answer_message = {
        "type": "answer",
        "target_peer_id": peer1_id,
        "answer": "mock-answer-data",
    }
    
    discovery.handle_signaling_message("miner-2", answer_message)
    
    # Check answer was forwarded to peer1
    assert len(ws1.sent_messages) == 1
    forwarded = ws1.sent_messages[0]
    assert forwarded["type"] == "answer"
    assert forwarded["from_peer_id"] == peer2_id
    assert forwarded["to_peer_id"] == peer1_id
    assert forwarded["data"]["answer"] == "mock-answer-data"
    
    # Check answer stored in peer2
    peer2 = discovery.peers[peer2_id]
    assert peer2.answer["answer"] == "mock-answer-data"

run_test("Signaling answer routing", test_signaling_answer_routing)


def test_ice_candidate_routing():
    """Test ICE candidate message routing."""
    discovery = P2PDiscovery()
    
    # Register 2 miners
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()
    peer1_id = discovery.register_miner("miner-1", ws1)
    peer2_id = discovery.register_miner("miner-2", ws2)
    
    # Simulate ICE candidate from peer1 to peer2
    candidate_message = {
        "type": "ice-candidate",
        "target_peer_id": peer2_id,
        "candidate": "mock-candidate-data",
    }
    
    discovery.handle_signaling_message("miner-1", candidate_message)
    
    # Check candidate was forwarded to peer2
    assert len(ws2.sent_messages) == 1
    forwarded = ws2.sent_messages[0]
    assert forwarded["type"] == "ice-candidate"
    assert forwarded["from_peer_id"] == peer1_id
    assert forwarded["to_peer_id"] == peer2_id
    assert forwarded["data"]["candidate"] == "mock-candidate-data"
    
    # Check candidate stored in peer1
    peer1 = discovery.peers[peer1_id]
    assert len(peer1.ice_candidates) == 1
    assert peer1.ice_candidates[0]["candidate"] == "mock-candidate-data"

run_test("ICE candidate routing", test_ice_candidate_routing)


def test_datachannel_open_notification():
    """Test DataChannel open notification handling."""
    discovery = P2PDiscovery()
    
    # Register miner
    ws = MockWebSocket()
    peer_id = discovery.register_miner("miner-1", ws)
    
    # Simulate DataChannel open
    dc_message = {
        "type": "datachannel-open",
        "connected_at": 1234567890.0,
    }
    
    discovery.handle_signaling_message("miner-1", dc_message)
    
    # Check connected_at was set
    peer = discovery.peers[peer_id]
    assert peer.connected_at == 1234567890.0

run_test("DataChannel open notification", test_datachannel_open_notification)


def test_get_peer_info():
    """Test getting peer information."""
    discovery = P2PDiscovery()
    ws = MockWebSocket()
    peer_id = discovery.register_miner("miner-1", ws)
    
    # Assign to pipeline
    assignments = [{"miner_id": "miner-1", "stage_index": 0}]
    discovery.assign_pipeline_peers("pipeline-123", assignments)
    
    # Get peer info
    info = discovery.get_peer_info("miner-1")
    assert info is not None
    assert info["peer_id"] == peer_id
    assert info["miner_id"] == "miner-1"
    assert info["pipeline_group_id"] == "pipeline-123"
    assert info["stage_index"] == 0
    assert info["connected_at"] is None  # Not connected yet
    
    # Get info for unknown miner
    info = discovery.get_peer_info("unknown")
    assert info is None

run_test("Get peer information", test_get_peer_info)


def test_get_pipeline_status():
    """Test getting pipeline status."""
    discovery = P2PDiscovery()
    
    # Register 3 miners
    miners = {}
    for i in range(3):
        ws = MockWebSocket()
        peer_id = discovery.register_miner(f"miner-{i}", ws)
        miners[f"miner-{i}"] = peer_id
    
    # Assign to pipeline
    assignments = [
        {"miner_id": "miner-0", "stage_index": 0},
        {"miner_id": "miner-1", "stage_index": 1},
        {"miner_id": "miner-2", "stage_index": 2},
    ]
    discovery.assign_pipeline_peers("pipeline-123", assignments)
    
    # Get pipeline status
    status = discovery.get_pipeline_status("pipeline-123")
    assert status["pipeline_group_id"] == "pipeline-123"
    assert status["total_peers"] == 3
    assert status["connected_peers"] == 0  # None connected yet
    assert len(status["peers"]) == 3
    
    # Check peer details
    peer_statuses = {p["miner_id"]: p for p in status["peers"]}
    assert peer_statuses["miner-0"]["stage_index"] == 0
    assert peer_statuses["miner-1"]["stage_index"] == 1
    assert peer_statuses["miner-2"]["stage_index"] == 2
    assert all(not p["connected"] for p in status["peers"])
    
    # Get status for unknown pipeline
    status = discovery.get_pipeline_status("unknown")
    assert status["total_peers"] == 0
    assert len(status["peers"]) == 0

run_test("Get pipeline status", test_get_pipeline_status)


def test_unknown_message_type():
    """Test graceful handling of unknown message types."""
    discovery = P2PDiscovery()
    ws = MockWebSocket()
    discovery.register_miner("miner-1", ws)
    
    # Send unknown message type
    unknown_message = {"type": "unknown-type", "data": {}}
    # Should not raise exception
    discovery.handle_signaling_message("miner-1", unknown_message)
    
    # No messages should be sent
    assert len(ws.sent_messages) == 0

run_test("Unknown message type handling", test_unknown_message_type)


def test_unregister_cleanup():
    """Test cleanup when unregistering miners."""
    discovery = P2PDiscovery()
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()
    peer1_id = discovery.register_miner("miner-1", ws1)
    peer2_id = discovery.register_miner("miner-2", ws2)
    
    # Assign to pipeline
    assignments = [
        {"miner_id": "miner-1", "stage_index": 0},
        {"miner_id": "miner-2", "stage_index": 1},
    ]
    discovery.assign_pipeline_peers("pipeline-123", assignments)
    
    # Verify pipeline exists
    assert "pipeline-123" in discovery.pipeline_peers
    assert peer1_id in discovery.pipeline_peers["pipeline-123"]
    
    # Unregister one miner
    discovery.unregister_miner("miner-1")
    
    # Check cleanup
    assert "miner-1" not in discovery.miner_to_peer
    assert peer1_id not in discovery.peers
    assert peer1_id not in discovery.pipeline_peers["pipeline-123"]
    assert peer2_id in discovery.pipeline_peers["pipeline-123"]  # Other peer remains

run_test("Unregister cleanup", test_unregister_cleanup)


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
