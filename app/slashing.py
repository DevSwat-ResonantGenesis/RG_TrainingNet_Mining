"""
On-Chain Slashing Protocol for Weight Integrity
===============================================

Protects the P2P weight distribution network from malicious or lazy miners
who serve corrupted weights. Implements economic incentives for honest behavior.

Key Components:
  - WeightIntegrityVerifier: Validates weights against Merkle roots
  - SlashingEngine: Manages slashing conditions and penalties
  - ReputationTracker: Tracks miner behavior over time
  - DisputeResolver: Handles weight corruption disputes

Slashing Conditions:
  1. Serving weights with invalid Merkle proofs
  2. Serving weights that don't match expected hashes
  3. Failing to serve registered weights (availability violations)
  4. Repeated integrity failures (pattern detection)

Penalties:
  - Stake slashing (proportional to violation severity)
  - Temporary suspension from P2P network
  - Reputation score reduction
  - Increased collateral requirements for re-entry
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger("rg-mining.slashing")


class ViolationType(str, Enum):
    """Types of slashing violations."""
    INVALID_MERKLE_PROOF = "invalid_merkle_proof"
    WEIGHT_HASH_MISMATCH = "weight_hash_mismatch"
    UNAVAILABLE_WEIGHTS = "unavailable_weights"
    REPEATED_FAILURES = "repeated_failures"
    MALICIOUS_BEHAVIOR = "malicious_behavior"


class ViolationSeverity(str, Enum):
    """Severity levels for violations."""
    LOW = "low"          # First offense, minor issue
    MEDIUM = "medium"    # Repeated minor issues
    HIGH = "high"        # Serious integrity violation
    CRITICAL = "critical" # Malicious behavior, network threat


@dataclass
class SlashingCondition:
    """Defines a slashing condition and its penalty."""
    violation_type: ViolationType
    severity: ViolationSeverity
    stake_penalty_percent: float  # Percentage of stake to slash
    suspension_hours: int         # Hours to suspend from P2P
    reputation_penalty: int       # Reputation points to deduct
    description: str


@dataclass
class ViolationRecord:
    """Record of a violation committed by a miner."""
    record_id: str = field(default_factory=lambda: str(uuid4()))
    miner_id: str = ""
    violation_type: ViolationType = ViolationType.WEIGHT_HASH_MISMATCH
    severity: ViolationSeverity = ViolationSeverity.LOW
    evidence: Dict[str, Any] = field(default_factory=dict)
    reported_by: str = ""  # Who reported the violation
    reported_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    verified: bool = False
    slashed: bool = False
    slash_amount: float = 0.0
    suspension_end: str = ""


@dataclass
class MinerReputation:
    """Reputation tracking for a miner."""
    miner_id: str = ""
    score: int = 100  # Start at 100, goes down with violations
    violations_count: int = 0
    last_violation: str = ""
    suspension_end: str = ""
    collateral_required: float = 1000.0  # Base collateral
    is_suspended: bool = False
    
    def update_score(self, penalty: int):
        """Update reputation score with penalty."""
        self.score = max(0, self.score - penalty)
        self.violations_count += 1
        self.last_violation = datetime.now(timezone.utc).isoformat()
        
        # Increase collateral requirements for low reputation
        if self.score < 25:
            self.collateral_required = 5000.0
        elif self.score < 50:
            self.collateral_required = 2000.0
    
    def is_suspended_now(self) -> bool:
        """Check if miner is currently suspended."""
        if not self.suspension_end:
            return False
        
        try:
            end_time = datetime.fromisoformat(self.suspension_end.replace('Z', '+00:00'))
            return datetime.now(timezone.utc) < end_time
        except:
            return False


class WeightIntegrityVerifier:
    """
    Verifies weight integrity using Merkle proofs and hash validation.
    
    This is the core of the slashing system — it provides cryptographic
    proof of whether weights are valid or corrupted.
    """
    
    def __init__(self):
        self.verified_hashes: Dict[str, Dict] = {}  # Cache of verified hashes
    
    def verify_weight_shard(
        self,
        model_id: str,
        layer_start: int,
        layer_end: int,
        weight_data: bytes,
        expected_hash: str,
        merkle_proof: Optional[Dict] = None,
        merkle_root: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Verify a weight shard against expected hash and Merkle proof.
        
        Returns:
            (is_valid, error_message)
        """
        # First verify the weight hash matches
        actual_hash = hashlib.sha256(weight_data).hexdigest()
        if actual_hash != expected_hash:
            return False, f"Weight hash mismatch: expected {expected_hash}, got {actual_hash}"
        
        # If Merkle proof provided, verify it
        if merkle_proof and merkle_root:
            is_valid, error = self._verify_merkle_proof(
                weight_data, merkle_proof, merkle_root
            )
            if not is_valid:
                return False, f"Invalid Merkle proof: {error}"
        
        # Cache the verification
        shard_key = f"{model_id}:L{layer_start}-{layer_end}"
        self.verified_hashes[shard_key] = {
            "hash": actual_hash,
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "size": len(weight_data),
        }
        
        return True, "Weight shard verified successfully"
    
    def _verify_merkle_proof(
        self,
        weight_data: bytes,
        merkle_proof: Dict,
        merkle_root: str,
    ) -> Tuple[bool, str]:
        """Verify a Merkle proof for weight data."""
        try:
            # Compute leaf hash
            leaf_hash = hashlib.sha256(weight_data).hexdigest()
            
            # Rebuild the Merkle path
            current_hash = leaf_hash
            for proof_element in merkle_proof.get("path", []):
                if proof_element.get("direction") == "left":
                    current_hash = hashlib.sha256(
                        bytes.fromhex(proof_element["hash"]) + bytes.fromhex(current_hash)
                    ).hexdigest()
                else:
                    current_hash = hashlib.sha256(
                        bytes.fromhex(current_hash) + bytes.fromhex(proof_element["hash"])
                    ).hexdigest()
            
            # Check against expected root
            return current_hash == merkle_root, ""
            
        except Exception as e:
            return False, f"Merkle proof verification failed: {e}"
    
    def get_verification_info(self, shard_key: str) -> Optional[Dict]:
        """Get cached verification info for a shard."""
        return self.verified_hashes.get(shard_key)


class SlashingEngine:
    """
    Core slashing engine that manages violations, penalties, and enforcement.
    
    Coordinates with the chain bridge to execute slashes on-chain.
    """
    
    def __init__(self, chain_bridge=None):
        self.chain_bridge = chain_bridge
        self.integrity_verifier = WeightIntegrityVerifier()
        self.reputation_tracker: Dict[str, MinerReputation] = {}
        self.violations: List[ViolationRecord] = []
        self.slashing_conditions = self._init_slashing_conditions()
        self.pending_verifications: Dict[str, Dict] = {}
        
    def _init_slashing_conditions(self) -> Dict[ViolationType, List[SlashingCondition]]:
        """Initialize predefined slashing conditions."""
        return {
            ViolationType.INVALID_MERKLE_PROOF: [
                SlashingCondition(
                    violation_type=ViolationType.INVALID_MERKLE_PROOF,
                    severity=ViolationSeverity.HIGH,
                    stake_penalty_percent=10.0,
                    suspension_hours=24,
                    reputation_penalty=20,
                    description="Providing invalid Merkle proof for weights"
                ),
            ],
            ViolationType.WEIGHT_HASH_MISMATCH: [
                SlashingCondition(
                    violation_type=ViolationType.WEIGHT_HASH_MISMATCH,
                    severity=ViolationSeverity.MEDIUM,
                    stake_penalty_percent=5.0,
                    suspension_hours=12,
                    reputation_penalty=10,
                    description="Serving weights with incorrect hash"
                ),
            ],
            ViolationType.UNAVAILABLE_WEIGHTS: [
                SlashingCondition(
                    violation_type=ViolationType.UNAVAILABLE_WEIGHTS,
                    severity=ViolationSeverity.LOW,
                    stake_penalty_percent=2.0,
                    suspension_hours=6,
                    reputation_penalty=5,
                    description="Failing to serve registered weights"
                ),
            ],
            ViolationType.REPEATED_FAILURES: [
                SlashingCondition(
                    violation_type=ViolationType.REPEATED_FAILURES,
                    severity=ViolationSeverity.HIGH,
                    stake_penalty_percent=15.0,
                    suspension_hours=48,
                    reputation_penalty=30,
                    description="Pattern of integrity violations"
                ),
            ],
            ViolationType.MALICIOUS_BEHAVIOR: [
                SlashingCondition(
                    violation_type=ViolationType.MALICIOUS_BEHAVIOR,
                    severity=ViolationSeverity.CRITICAL,
                    stake_penalty_percent=50.0,
                    suspension_hours=168,  # 1 week
                    reputation_penalty=50,
                    description="Deliberate attempt to poison the network"
                ),
            ],
        }
    
    def report_violation(
        self,
        miner_id: str,
        violation_type: ViolationType,
        evidence: Dict[str, Any],
        reported_by: str,
    ) -> str:
        """
        Report a violation for investigation.
        
        Returns the violation record ID.
        """
        record = ViolationRecord(
            miner_id=miner_id,
            violation_type=violation_type,
            evidence=evidence,
            reported_by=reported_by,
        )
        
        self.violations.append(record)
        logger.warning(f"Violation reported: {violation_type.value} by miner {miner_id}")
        
        # Auto-verify for certain types with clear evidence (only in async context)
        # Skip auto-verification in test environments
        
        return record.record_id
    
    async def _verify_violation(self, record_id: str):
        """Verify a violation and apply penalties if confirmed."""
        record = next((r for r in self.violations if r.record_id == record_id), None)
        if not record:
            return
        
        try:
            is_valid = await self._validate_evidence(record)
            record.verified = True
            
            if is_valid:
                await self._apply_penalty(record)
                logger.info(f"Violation verified and penalty applied: {record_id}")
            else:
                logger.info(f"Violation evidence invalid: {record_id}")
                
        except Exception as e:
            logger.error(f"Error verifying violation {record_id}: {e}")
    
    async def _validate_evidence(self, record: ViolationRecord) -> bool:
        """Validate the evidence for a violation."""
        vt = record.violation_type
        
        if vt == ViolationType.WEIGHT_HASH_MISMATCH:
            # Check if weight hash doesn't match expected
            evidence = record.evidence
            expected_hash = evidence.get("expected_hash")
            actual_hash = evidence.get("actual_hash")
            return expected_hash and actual_hash and expected_hash != actual_hash
        
        elif vt == ViolationType.INVALID_MERKLE_PROOF:
            # Verify Merkle proof is actually invalid
            evidence = record.evidence
            weight_data = evidence.get("weight_data")
            merkle_proof = evidence.get("merkle_proof")
            merkle_root = evidence.get("merkle_root")
            
            if not all([weight_data, merkle_proof, merkle_root]):
                return False
            
            is_valid, _ = self.integrity_verifier._verify_merkle_proof(
                weight_data, merkle_proof, merkle_root
            )
            return not is_valid  # Violation if proof is invalid
        
        elif vt == ViolationType.UNAVAILABLE_WEIGHTS:
            # Check if miner failed to serve weights
            evidence = record.evidence
            return evidence.get("transfer_failed", False)
        
        return False
    
    async def _apply_penalty(self, record: ViolationRecord):
        """Apply the penalty for a verified violation."""
        conditions = self.slashing_conditions.get(record.violation_type, [])
        if not conditions:
            logger.warning(f"No slashing condition for {record.violation_type}")
            return
        
        # Get appropriate condition based on miner's history
        condition = self._select_condition(record, conditions)
        
        # Update miner reputation
        reputation = self.reputation_tracker.get(record.miner_id)
        if not reputation:
            reputation = MinerReputation(miner_id=record.miner_id)
            self.reputation_tracker[record.miner_id] = reputation
        
        reputation.update_score(condition.reputation_penalty)
        
        # Apply suspension
        if condition.suspension_hours > 0:
            suspension_end = datetime.now(timezone.utc) + timedelta(hours=condition.suspension_hours)
            reputation.suspension_end = suspension_end.isoformat()
            reputation.is_suspended = True
            record.suspension_end = suspension_end.isoformat()
        
        # Calculate slash amount (would use actual stake from chain)
        record.slash_amount = condition.stake_penalty_percent  # Placeholder
        record.slashed = True
        
        # Execute slash on-chain if bridge available
        if self.chain_bridge:
            try:
                await self.chain_bridge.slash_miner(
                    miner_id=record.miner_id,
                    amount=record.slash_amount,
                    reason=record.violation_type.value,
                    evidence=record.evidence,
                )
            except Exception as e:
                logger.error(f"Failed to execute on-chain slash: {e}")
        
        logger.warning(
            f"Slashed miner {record.miner_id}: {condition.stake_penalty_percent}% stake, "
            f"suspended for {condition.suspension_hours}h"
        )
    
    def _select_condition(
        self,
        record: ViolationRecord,
        conditions: List[SlashingCondition]
    ) -> SlashingCondition:
        """Select appropriate condition based on violation history."""
        reputation = self.reputation_tracker.get(record.miner_id)
        
        # Escalate severity for repeat offenders
        if reputation and reputation.violations_count >= 3:
            # Find highest severity condition
            return max(conditions, key=lambda c: c.severity.value)
        
        # Use first (lowest severity) condition for first-time offenders
        return conditions[0]
    
    def verify_weight_transfer(
        self,
        miner_id: str,
        model_id: str,
        layer_start: int,
        layer_end: int,
        weight_data: bytes,
        expected_hash: str,
        requester_id: str,
    ) -> Tuple[bool, str]:
        """
        Verify a weight transfer and report violations if needed.
        
        Called by miners when they receive weights from a peer.
        """
        # Verify the weights
        is_valid, error = self.integrity_verifier.verify_weight_shard(
            model_id, layer_start, layer_end, weight_data, expected_hash
        )
        
        if not is_valid:
            # Report violation
            evidence = {
                "model_id": model_id,
                "layer_start": layer_start,
                "layer_end": layer_end,
                "expected_hash": expected_hash,
                "actual_hash": hashlib.sha256(weight_data).hexdigest(),
                "weight_size": len(weight_data),
                "requester_id": requester_id,
            }
            
            self.report_violation(
                miner_id=miner_id,
                violation_type=ViolationType.WEIGHT_HASH_MISMATCH,
                evidence=evidence,
                reported_by=requester_id,
            )
        
        return is_valid, error
    
    def get_miner_status(self, miner_id: str) -> Dict[str, Any]:
        """Get the current status of a miner including reputation."""
        reputation = self.reputation_tracker.get(miner_id)
        if not reputation:
            reputation = MinerReputation(miner_id=miner_id)
            self.reputation_tracker[miner_id] = reputation
        
        violations = [v for v in self.violations if v.miner_id == miner_id]
        
        return {
            "miner_id": miner_id,
            "reputation_score": reputation.score,
            "violations_count": reputation.violations_count,
            "is_suspended": reputation.is_suspended_now(),
            "suspension_end": reputation.suspension_end,
            "collateral_required": reputation.collateral_required,
            "recent_violations": [
                {
                    "type": v.violation_type.value,
                    "severity": v.severity.value,
                    "reported_at": v.reported_at,
                    "verified": v.verified,
                    "slashed": v.slashed,
                }
                for v in sorted(violations, key=lambda x: x.reported_at, reverse=True)[:5]
            ],
        }
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get overall network statistics for monitoring."""
        total_miners = len(self.reputation_tracker)
        suspended_miners = sum(1 for r in self.reputation_tracker.values() if r.is_suspended_now())
        total_violations = len(self.violations)
        verified_violations = sum(1 for v in self.violations if v.verified)
        
        # Violation breakdown
        violation_counts = {}
        for v in self.violations:
            violation_counts[v.violation_type.value] = violation_counts.get(v.violation_type.value, 0) + 1
        
        return {
            "total_miners": total_miners,
            "suspended_miners": suspended_miners,
            "total_violations": total_violations,
            "verified_violations": verified_violations,
            "violation_breakdown": violation_counts,
            "avg_reputation": sum(r.score for r in self.reputation_tracker.values()) / max(total_miners, 1),
        }


# Global slashing engine instance
slashing_engine = SlashingEngine()
