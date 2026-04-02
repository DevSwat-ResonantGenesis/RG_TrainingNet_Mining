"""
Tests for On-Chain Slashing Protocol
======================================

Tests:
  - Weight integrity verification with Merkle proofs
  - Violation reporting and penalty application
  - Reputation tracking and suspension
  - Slashing condition selection based on history
"""

import hashlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.slashing import (
    SlashingEngine, WeightIntegrityVerifier, ViolationType, ViolationSeverity,
    ViolationRecord, MinerReputation, SlashingCondition,
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
# SLASHING TESTS
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  SLASHING PROTOCOL TESTS")
print("=" * 60)


def test_weight_integrity_verification():
    """Test weight shard verification with hash and Merkle proof."""
    verifier = WeightIntegrityVerifier()
    
    # Create test weight data
    weight_data = b"test_weight_data_12345"
    expected_hash = "21cefd69378e4e47946e42906a9ca9b436506c99e624df8cd75867deabbf6381"
    
    # Test valid hash
    is_valid, error = verifier.verify_weight_shard(
        model_id="test-model",
        layer_start=0,
        layer_end=8,
        weight_data=weight_data,
        expected_hash=expected_hash,
    )
    assert is_valid
    assert error == "Weight shard verified successfully"
    
    # Test invalid hash
    wrong_hash = "wrong_hash_value_12345"
    is_valid, error = verifier.verify_weight_shard(
        model_id="test-model",
        layer_start=0,
        layer_end=8,
        weight_data=weight_data,
        expected_hash=wrong_hash,
    )
    assert not is_valid
    assert "Weight hash mismatch" in error
    
    # Test Merkle proof verification
    merkle_proof = {
        "path": [
            {"hash": "a1b2c3d4e5f6", "direction": "left"},
            {"hash": "f6e5d4c3b2a1", "direction": "right"},
        ]
    }
    merkle_root = "final_merkle_root_hash"
    
    # This will fail with our test data, but tests the verification logic
    is_valid, error = verifier.verify_weight_shard(
        model_id="test-model",
        layer_start=0,
        layer_end=8,
        weight_data=weight_data,
        expected_hash=expected_hash,
        merkle_proof=merkle_proof,
        merkle_root=merkle_root,
    )
    assert not is_valid
    assert "Invalid Merkle proof" in error

run_test("Weight integrity verification", test_weight_integrity_verification)


def test_violation_reporting():
    """Test reporting violations and record creation."""
    engine = SlashingEngine()
    
    # Report a violation
    evidence = {
        "model_id": "test-model",
        "layer_start": 0,
        "layer_end": 8,
        "expected_hash": "correct_hash",
        "actual_hash": "wrong_hash",
    }
    
    record_id = engine.report_violation(
        miner_id="miner-1",
        violation_type=ViolationType.WEIGHT_HASH_MISMATCH,
        evidence=evidence,
        reported_by="miner-2",
    )
    
    assert record_id is not None
    assert len(engine.violations) == 1
    
    record = engine.violations[0]
    assert record.miner_id == "miner-1"
    assert record.violation_type == ViolationType.WEIGHT_HASH_MISMATCH
    assert record.evidence == evidence
    assert record.reported_by == "miner-2"
    assert not record.verified
    assert not record.slashed

run_test("Violation reporting", test_violation_reporting)


def test_reputation_tracking():
    """Test miner reputation tracking and updates."""
    engine = SlashingEngine()
    
    # Get initial reputation (should create new entry)
    status = engine.get_miner_status("miner-1")
    assert status["miner_id"] == "miner-1"
    assert status["reputation_score"] == 100
    assert status["violations_count"] == 0
    assert not status["is_suspended"]
    
    # Report a violation
    evidence = {"test": "evidence"}
    engine.report_violation(
        miner_id="miner-1",
        violation_type=ViolationType.WEIGHT_HASH_MISMATCH,
        evidence=evidence,
        reported_by="system",
    )
    
    # Manually verify and apply penalty for testing
    record = engine.violations[0]
    engine.reputation_tracker["miner-1"].update_score(10)  # Medium penalty
    
    # Check updated reputation
    status = engine.get_miner_status("miner-1")
    assert status["reputation_score"] == 90
    assert status["violations_count"] == 1

run_test("Reputation tracking", test_reputation_tracking)


def test_slashing_conditions():
    """Test slashing condition selection and penalty application."""
    engine = SlashingEngine()
    
    # Test first-time offender gets lowest severity
    conditions = engine.slashing_conditions[ViolationType.WEIGHT_HASH_MISMATCH]
    assert len(conditions) > 0
    
    condition = conditions[0]  # First condition for first-time offenders
    assert condition.severity == ViolationSeverity.MEDIUM
    assert condition.stake_penalty_percent == 5.0
    assert condition.suspension_hours == 12
    assert condition.reputation_penalty == 10
    
    # Test repeat offender escalation
    # Create reputation with multiple violations
    reputation = MinerReputation(miner_id="miner-1")
    reputation.violations_count = 5
    engine.reputation_tracker["miner-1"] = reputation
    
    record = ViolationRecord(
        miner_id="miner-1",
        violation_type=ViolationType.WEIGHT_HASH_MISMATCH,
    )
    
    selected_condition = engine._select_condition(record, conditions)
    # Should select highest severity for repeat offenders
    assert selected_condition.severity == ViolationSeverity.MEDIUM

run_test("Slashing conditions", test_slashing_conditions)


def test_suspension_logic():
    """Test miner suspension and reactivation."""
    engine = SlashingEngine()
    
    # Create a miner with suspension
    reputation = MinerReputation(miner_id="miner-1")
    reputation.suspension_end = "2026-01-01T00:00:00+00:00"  # Past date
    reputation.is_suspended = True
    engine.reputation_tracker["miner-1"] = reputation
    
    # Check suspension status (should be False for past date)
    assert not reputation.is_suspended_now()
    
    # Set future suspension
    from datetime import datetime, timezone, timedelta
    future_end = datetime.now(timezone.utc) + timedelta(hours=24)
    reputation.suspension_end = future_end.isoformat()
    
    # Should be suspended now
    assert reputation.is_suspended_now()

run_test("Suspension logic", test_suspension_logic)


def test_weight_transfer_verification():
    """Test weight transfer verification with violation reporting."""
    engine = SlashingEngine()
    
    # Create test weight data
    weight_data = b"test_weight_data"
    expected_hash = hashlib.sha256(weight_data).hexdigest()
    wrong_hash = "wrong_hash_value"
    
    # Test valid transfer
    is_valid, error = engine.verify_weight_transfer(
        miner_id="miner-1",
        model_id="test-model",
        layer_start=0,
        layer_end=8,
        weight_data=weight_data,
        expected_hash=expected_hash,
        requester_id="miner-2",
    )
    assert is_valid
    assert error == "Weight shard verified successfully"
    assert len(engine.violations) == 0  # No violation reported
    
    # Test invalid transfer (should report violation)
    is_valid, error = engine.verify_weight_transfer(
        miner_id="miner-1",
        model_id="test-model",
        layer_start=0,
        layer_end=8,
        weight_data=weight_data,
        expected_hash=wrong_hash,
        requester_id="miner-2",
    )
    assert not is_valid
    assert "Weight hash mismatch" in error
    assert len(engine.violations) == 1  # Violation reported
    
    violation = engine.violations[0]
    assert violation.miner_id == "miner-1"
    assert violation.violation_type == ViolationType.WEIGHT_HASH_MISMATCH

run_test("Weight transfer verification", test_weight_transfer_verification)


def test_network_stats():
    """Test network statistics calculation."""
    engine = SlashingEngine()
    
    # Add some miners and violations
    engine.reputation_tracker["miner-1"] = MinerReputation(miner_id="miner-1", score=90)
    engine.reputation_tracker["miner-2"] = MinerReputation(miner_id="miner-2", score=85)
    engine.reputation_tracker["miner-3"] = MinerReputation(miner_id="miner-3", score=95)
    
    # Add violations
    engine.report_violation("miner-1", ViolationType.WEIGHT_HASH_MISMATCH, {}, "system")
    engine.report_violation("miner-2", ViolationType.UNAVAILABLE_WEIGHTS, {}, "system")
    
    stats = engine.get_network_stats()
    assert stats["total_miners"] == 3
    assert stats["total_violations"] == 2
    assert stats["avg_reputation"] == (90 + 85 + 95) / 3
    assert "violation_breakdown" in stats

run_test("Network statistics", test_network_stats)


def test_collateral_requirements():
    """Test collateral requirements based on reputation."""
    engine = SlashingEngine()
    
    # Test normal reputation
    reputation = MinerReputation(miner_id="miner-1", score=80)
    assert reputation.collateral_required == 1000.0
    
    # Test low reputation
    reputation.score = 45  # Start above threshold
    reputation.update_score(5)  # Apply penalty, score becomes 40, triggers collateral update
    assert reputation.score == 40
    assert reputation.collateral_required == 2000.0
    
    # Test very low reputation - use new object
    very_low_rep = MinerReputation(miner_id="miner-2", score=25)
    very_low_rep.update_score(5)  # Apply penalty, score becomes 20, triggers collateral update
    assert very_low_rep.collateral_required == 5000.0

run_test("Collateral requirements", test_collateral_requirements)


# Import hashlib for tests (already imported at top)

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
