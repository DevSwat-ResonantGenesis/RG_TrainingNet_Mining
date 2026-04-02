"""
Tests for Wallet Service
========================

Tests:
  - Wallet creation and management
  - Token balance tracking
  - Staking operations
  - Reward calculation and distribution
  - Slashing penalty execution
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.wallet_service import (
    WalletService, Wallet, TokenBalance, Stake, Reward,
    TokenType, StakeStatus, TransactionStatus,
)

# Mock slashing_engine for tests
class MockSlashingEngine:
    def __init__(self):
        self.reputation_tracker = {}
    
    def get_reputation_tracker(self):
        return self.reputation_tracker

# Create global mock
slashing_engine = MockSlashingEngine()


# ══════════════════════════════════════════════════════════════
# Test helpers
# ══════════════════════════════════════════════════════════════

passed = 0
failed = 0
errors = []


def run_test(name, fn):
    global passed, failed
    try:
        if asyncio.iscoroutinefunction(fn):
            asyncio.run(fn())
        else:
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
# WALLET SERVICE TESTS
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  WALLET SERVICE TESTS")
print("=" * 60)


async def test_wallet_creation():
    """Test wallet creation for miners."""
    service = WalletService()
    
    # Create wallet for miner
    wallet = await service.create_wallet("miner-1")
    
    assert wallet.miner_id == "miner-1"
    assert wallet.wallet_address.startswith("0x")
    assert len(wallet.wallet_address) == 34  # 0x + 32 hex chars
    assert wallet.is_active == True
    
    # Check balances initialized
    balances = service.balances[wallet.wallet_address]
    assert len(balances) == 3  # RG_TOKEN, ETH, USDC
    assert all(b.balance == 0.0 for b in balances)
    
    # Check wallet retrieval
    retrieved_wallet = await service.get_wallet("miner-1")
    assert retrieved_wallet.wallet_address == wallet.wallet_address

run_test("Wallet creation", test_wallet_creation)


async def test_token_balances():
    """Test token balance tracking."""
    service = WalletService()
    
    wallet = await service.create_wallet("miner-1")
    
    # Get initial balances
    balances = service.get_balances(wallet.wallet_address)
    assert len(balances) == 3
    
    # Get specific token balance
    rg_balance = service.get_balance(wallet.wallet_address, TokenType.RG_TOKEN)
    assert rg_balance == 0.0
    
    # Update balance
    service._update_balance(wallet.wallet_address, TokenType.RG_TOKEN, 1000.0)
    rg_balance = service.get_balance(wallet.wallet_address, TokenType.RG_TOKEN)
    assert rg_balance == 1000.0
    
    # Check USD value
    balances = service.get_balances(wallet.wallet_address)
    rg_balance_info = next(b for b in balances if b.token_type == TokenType.RG_TOKEN)
    assert rg_balance_info.balance == 1000.0
    assert rg_balance_info.usd_value == 1000.0

run_test("Token balances", test_token_balances)


async def test_staking_operations():
    """Test staking deposit and withdrawal."""
    service = WalletService()
    
    wallet = await service.create_wallet("miner-1")
    
    # Add some balance first
    service._update_balance(wallet.wallet_address, TokenType.RG_TOKEN, 2000.0)
    
    # Deposit stake
    stake = await service.deposit_stake(
        wallet_address=wallet.wallet_address,
        token_type=TokenType.RG_TOKEN,
        amount=1000.0,
        lock_period_days=30,
    )
    
    assert stake.token_type == TokenType.RG_TOKEN
    assert stake.amount == 1000.0
    assert stake.status == StakeStatus.ACTIVE
    assert stake.wallet_address == wallet.wallet_address
    
    # Check balance decreased
    balance = service.get_balance(wallet.wallet_address, TokenType.RG_TOKEN)
    assert balance == 1000.0  # 2000 - 1000 staked
    
    # Try to withdraw before lock period (should fail)
    try:
        await service.withdraw_stake(stake.stake_id)
        assert False, "Should not be able to withdraw before lock period"
    except ValueError as e:
        assert "locked until" in str(e)
    
    # Manually set lock period to past for testing
    from datetime import datetime, timezone, timedelta
    stake.locked_until = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    
    # Withdraw stake
    success = await service.withdraw_stake(stake.stake_id)
    assert success == True
    assert stake.status == StakeStatus.WITHDRAWN
    
    # Check balance restored
    balance = service.get_balance(wallet.wallet_address, TokenType.RG_TOKEN)
    assert balance == 2000.0  # 1000 + 1000 returned

run_test("Staking operations", test_staking_operations)


async def test_slashing_penalties():
    """Test stake slashing for violations."""
    service = WalletService()
    
    wallet = await service.create_wallet("miner-1")
    service._update_balance(wallet.wallet_address, TokenType.RG_TOKEN, 2000.0)
    
    # Create stake
    stake = await service.deposit_stake(
        wallet_address=wallet.wallet_address,
        token_type=TokenType.RG_TOKEN,
        amount=1000.0,
        lock_period_days=30,
    )
    
    # Slash stake
    slash_amount = service.slash_stake(
        stake_id=stake.stake_id,
        slash_percent=10.0,  # 10% slash
        reason="Weight hash mismatch",
        violation_id="violation-123",
    )
    
    assert slash_amount == 100.0  # 10% of 1000
    assert stake.amount == 900.0  # 1000 - 100
    assert stake.status == StakeStatus.SLASHED
    assert len(stake.slash_history) == 1
    assert stake.slash_history[0]["amount"] == 100.0
    assert stake.slash_history[0]["reason"] == "Weight hash mismatch"

run_test("Slashing penalties", test_slashing_penalties)


async def test_reward_distribution():
    """Test reward calculation and distribution."""
    service = WalletService()
    
    wallet = await service.create_wallet("miner-1")
    
    # Distribute reward
    reward = await service.distribute_reward(
        wallet_address=wallet.wallet_address,
        token_type=TokenType.RG_TOKEN,
        amount=50.0,
        reason="training",
        epoch=12345,
    )
    
    assert reward.wallet_address == wallet.wallet_address
    assert reward.amount == 50.0
    assert reward.reason == "training"
    assert reward.epoch == 12345
    
    # Check balance updated
    balance = service.get_balance(wallet.wallet_address, TokenType.RG_TOKEN)
    assert balance == 50.0
    
    # Check reward in history
    rewards = [r for r in service.rewards if r.wallet_address == wallet.wallet_address]
    assert len(rewards) == 1
    assert rewards[0].reward_id == reward.reward_id

run_test("Reward distribution", test_reward_distribution)


async def test_training_rewards_calculation():
    """Test training reward calculation."""
    service = WalletService()
    
    # Create wallet for miner
    await service.create_wallet("miner-1")
    
    # Calculate rewards
    rewards = await service.calculate_training_rewards(
        miner_id="miner-1",
        hours_trained=10.0,
        performance_score=1.0,
    )
    
    assert rewards == 100.0  # 10 hours * 10 RG tokens/hour * 1.0 performance
    
    # Test with performance bonus
    rewards = await service.calculate_training_rewards(
        miner_id="miner-1",
        hours_trained=10.0,
        performance_score=1.5,
    )
    
    assert rewards == 150.0  # 10 hours * 10 RG tokens/hour * 1.5 performance

run_test("Training rewards calculation", test_training_rewards_calculation)


async def test_seeding_rewards_calculation():
    """Test P2P seeding reward calculation."""
    service = WalletService()
    
    # Create wallet for miner
    await service.create_wallet("miner-1")
    
    # Calculate rewards
    rewards = await service.calculate_seeding_rewards(
        miner_id="miner-1",
        hours_seeded=8.0,
        bandwidth_mbps=100.0,
    )
    
    assert rewards == 40.0  # 8 hours * 5 RG tokens/hour * 1.0 bandwidth bonus
    
    # Test with high bandwidth bonus
    rewards = await service.calculate_seeding_rewards(
        miner_id="miner-1",
        hours_seeded=8.0,
        bandwidth_mbps=200.0,  # 2x bandwidth bonus
    )
    
    assert rewards == 80.0  # 8 hours * 5 RG tokens/hour * 2.0 bandwidth bonus

run_test("Seeding rewards calculation", test_seeding_rewards_calculation)


async def test_wallet_statistics():
    """Test wallet statistics calculation."""
    service = WalletService()
    
    wallet = await service.create_wallet("miner-1")
    
    # Add balance and stakes
    service._update_balance(wallet.wallet_address, TokenType.RG_TOKEN, 1000.0)
    service._update_balance(wallet.wallet_address, TokenType.ETH, 1.0)
    
    # Add more balance for staking
    service._update_balance(wallet.wallet_address, TokenType.RG_TOKEN, 1000.0)
    
    stake = await service.deposit_stake(
        wallet_address=wallet.wallet_address,
        token_type=TokenType.RG_TOKEN,
        amount=1000.0,
        lock_period_days=30,
    )
    
    # Distribute some rewards
    await service.distribute_reward(
        wallet_address=wallet.wallet_address,
        token_type=TokenType.RG_TOKEN,
        amount=100.0,
        reason="training",
    )
    
    # Get stats
    stats = await service.get_wallet_stats(wallet.wallet_address)
    
    assert stats["wallet_address"] == wallet.wallet_address
    assert stats["total_balance_usd"] == 3100.0  # 1100 RG (1000 + 100 reward) + 2000 ETH
    assert stats["total_staked"] == 1000.0
    assert stats["rewards_24h"] == 100.0
    assert stats["total_rewards"] == 1

run_test("Wallet statistics", test_wallet_statistics)


async def test_network_statistics():
    """Test network-wide statistics."""
    service = WalletService()
    
    # Create multiple wallets
    wallet1 = await service.create_wallet("miner-1")
    wallet2 = await service.create_wallet("miner-2")
    
    # Add balances
    service._update_balance(wallet1.wallet_address, TokenType.RG_TOKEN, 1000.0)
    service._update_balance(wallet2.wallet_address, TokenType.RG_TOKEN, 2000.0)
    
    # Add more balance for staking
    service._update_balance(wallet1.wallet_address, TokenType.RG_TOKEN, 1000.0)
    service._update_balance(wallet2.wallet_address, TokenType.RG_TOKEN, 1000.0)
    
    # Create stakes
    await service.deposit_stake(wallet1.wallet_address, TokenType.RG_TOKEN, 1000.0, 30)
    await service.deposit_stake(wallet2.wallet_address, TokenType.RG_TOKEN, 1000.0, 30)
    
    # Get network stats
    stats = await service.get_network_stats()
    
    assert stats["total_wallets"] == 2
    assert stats["active_wallets"] == 2
    assert stats["total_balances"]["rg_token"] == 3000.0  # 1000 + 2000
    assert stats["total_staked"] == 2000.0  # 1000 + 1000

run_test("Network statistics", test_network_statistics)


async def test_minimum_stake_requirements():
    """Test minimum stake requirements."""
    service = WalletService()
    
    wallet = await service.create_wallet("miner-1")
    service._update_balance(wallet.wallet_address, TokenType.RG_TOKEN, 500.0)
    
    # Try to stake below minimum (should fail)
    try:
        await service.deposit_stake(
            wallet_address=wallet.wallet_address,
            token_type=TokenType.RG_TOKEN,
            amount=500.0,  # Below minimum of 1000
        )
        assert False, "Should not allow staking below minimum"
    except ValueError as e:
        assert "Minimum stake" in str(e)
    
    # Stake exactly minimum
    service._update_balance(wallet.wallet_address, TokenType.RG_TOKEN, 1000.0)
    stake = await service.deposit_stake(
        wallet_address=wallet.wallet_address,
        token_type=TokenType.RG_TOKEN,
        amount=1000.0,
    )
    assert stake.amount == 1000.0

run_test("Minimum stake requirements", test_minimum_stake_requirements)


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
