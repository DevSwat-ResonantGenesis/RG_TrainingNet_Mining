"""
Wallet Service for Resonant Genesis Mining
==========================================

Handles blockchain wallet operations, staking, and reward distribution.

Features:
- Wallet creation and management
- Stake deposit and withdrawal
- Reward calculation and distribution
- Slashing penalty execution
- Balance tracking and history
- Multi-token support (RG Token, ETH, etc.)

Integrates with external blockchain service for on-chain operations.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

logger = logging.getLogger("rg-mining.wallet")


class TokenType(str, Enum):
    """Supported token types."""
    RG_TOKEN = "rg_token"      # Native governance token
    ETH = "eth"               # Ethereum for gas fees
    USDC = "usdc"             # Stablecoin for rewards


class TransactionStatus(str, Enum):
    """Transaction status."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    REVERTED = "reverted"


class StakeStatus(str, Enum):
    """Stake status."""
    ACTIVE = "active"
    LOCKED = "locked"
    SLASHED = "slashed"
    WITHDRAWN = "withdrawn"


@dataclass
class Wallet:
    """Miner wallet information."""
    wallet_address: str
    miner_id: str
    created_at: str
    last_active: str
    is_active: bool = True


@dataclass
class TokenBalance:
    """Token balance for a wallet."""
    wallet_address: str
    token_type: TokenType
    balance: float
    usd_value: float
    last_updated: str


@dataclass
class Stake:
    """Staking information."""
    stake_id: str
    wallet_address: str
    token_type: TokenType
    amount: float
    status: StakeStatus
    locked_until: str  # ISO timestamp
    created_at: str
    slash_history: List[Dict] = None


@dataclass
class Transaction:
    """Blockchain transaction."""
    tx_hash: str
    wallet_address: str
    token_type: TokenType
    amount: float
    gas_fee: float
    status: TransactionStatus
    block_number: Optional[int]
    timestamp: str
    metadata: Dict[str, Any] = None


@dataclass
class Reward:
    """Reward distribution record."""
    reward_id: str
    wallet_address: str
    token_type: TokenType
    amount: float
    reason: str  # training, seeding, etc.
    epoch: int
    distributed_at: str
    tx_hash: Optional[str] = None


class WalletService:
    """
    Wallet and staking service for miners.
    
    Manages wallet creation, balance tracking, staking, rewards,
    and slashing penalties through integration with blockchain service.
    """
    
    def __init__(self, blockchain_service=None):
        self.blockchain_service = blockchain_service
        self.wallets: Dict[str, Wallet] = {}
        self.balances: Dict[str, List[TokenBalance]] = {}
        self.stakes: Dict[str, Stake] = {}
        self.transactions: List[Transaction] = []
        self.rewards: List[Reward] = []
        
        # Reward rates (per hour)
        self.reward_rates = {
            "training": 10.0,      # RG tokens per hour for training
            "seeding": 5.0,        # RG tokens per hour for P2P seeding
            "verification": 2.0,   # RG tokens per hour for verification
        }
        
        # Minimum stake requirements
        self.min_stakes = {
            TokenType.RG_TOKEN: 1000.0,
            TokenType.ETH: 0.1,
            TokenType.USDC: 1000.0,
        }
    
    async def create_wallet(self, miner_id: str) -> Wallet:
        """
        Create a new wallet for a miner.
        
        Returns the wallet information with address.
        """
        # Generate wallet address (in production, this would come from blockchain service)
        wallet_address = f"0x{uuid.uuid4().hex[:32]}"  # 32 hex chars + 0x = 34 chars total
        
        wallet = Wallet(
            wallet_address=wallet_address,
            miner_id=miner_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            last_active=datetime.now(timezone.utc).isoformat(),
        )
        
        self.wallets[wallet_address] = wallet
        
        # Initialize balances
        self.balances[wallet_address] = [
            TokenBalance(
                wallet_address=wallet_address,
                token_type=TokenType.RG_TOKEN,
                balance=0.0,
                usd_value=0.0,
                last_updated=datetime.now(timezone.utc).isoformat(),
            ),
            TokenBalance(
                wallet_address=wallet_address,
                token_type=TokenType.ETH,
                balance=0.0,
                usd_value=0.0,
                last_updated=datetime.now(timezone.utc).isoformat(),
            ),
            TokenBalance(
                wallet_address=wallet_address,
                token_type=TokenType.USDC,
                balance=0.0,
                usd_value=0.0,
                last_updated=datetime.now(timezone.utc).isoformat(),
            ),
        ]
        
        logger.info(f"Created wallet {wallet_address} for miner {miner_id}")
        return wallet
    
    async def get_wallet(self, miner_id: str) -> Optional[Wallet]:
        """Get wallet for a miner."""
        for wallet in self.wallets.values():
            if wallet.miner_id == miner_id:
                return wallet
        return None
    
    def get_balances(self, wallet_address: str) -> List[TokenBalance]:
        """Get all token balances for a wallet."""
        return self.balances.get(wallet_address, [])
    
    def get_balance(self, wallet_address: str, token_type: TokenType) -> float:
        """Get specific token balance."""
        balances = self.balances.get(wallet_address, [])
        for balance in balances:
            if balance.token_type == token_type:
                return balance.balance
        return 0.0
    
    async def deposit_stake(
        self,
        wallet_address: str,
        token_type: TokenType,
        amount: float,
        lock_period_days: int = 30,
    ) -> Stake:
        """
        Deposit tokens as stake.
        
        Creates a stake that locks tokens for a specified period.
        """
        # Check minimum stake
        min_stake = self.min_stakes.get(token_type, 0)
        if amount < min_stake:
            raise ValueError(f"Minimum stake for {token_type} is {min_stake}")
        
        # Check balance
        current_balance = self.get_balance(wallet_address, token_type)
        if current_balance < amount:
            raise ValueError(f"Insufficient balance: have {current_balance}, need {amount}")
        
        # Create stake
        stake_id = f"stake_{uuid.uuid4().hex[:12]}"
        locked_until = datetime.now(timezone.utc) + timedelta(days=lock_period_days)
        
        stake = Stake(
            stake_id=stake_id,
            wallet_address=wallet_address,
            token_type=token_type,
            amount=amount,
            status=StakeStatus.ACTIVE,
            locked_until=locked_until.isoformat(),
            created_at=datetime.now(timezone.utc).isoformat(),
            slash_history=[],
        )
        
        self.stakes[stake_id] = stake
        
        # Update balance (subtract staked amount)
        self._update_balance(wallet_address, token_type, -amount)
        
        logger.info(f"Created stake {stake_id}: {amount} {token_type} until {locked_until}")
        return stake
    
    async def withdraw_stake(self, stake_id: str) -> bool:
        """
        Withdraw stake if lock period has expired.
        
        Returns True if withdrawal was successful.
        """
        stake = self.stakes.get(stake_id)
        if not stake:
            raise ValueError(f"Stake {stake_id} not found")
        
        if stake.status != StakeStatus.ACTIVE:
            raise ValueError(f"Stake {stake_id} is not active")
        
        # Check lock period
        locked_until = datetime.fromisoformat(stake.locked_until.replace('Z', '+00:00'))
        if datetime.now(timezone.utc) < locked_until:
            raise ValueError(f"Stake locked until {locked_until}")
        
        # Return staked amount
        self._update_balance(stake.wallet_address, stake.token_type, stake.amount)
        stake.status = StakeStatus.WITHDRAWN
        
        logger.info(f"Withdrew stake {stake_id}: {stake.amount} {stake.token_type}")
        return True
    
    def slash_stake(
        self,
        stake_id: str,
        slash_percent: float,
        reason: str,
        violation_id: str = None,
    ) -> float:
        """
        Slash a portion of stake as penalty.
        
        Returns the amount slashed.
        """
        stake = self.stakes.get(stake_id)
        if not stake:
            raise ValueError(f"Stake {stake_id} not found")
        
        if stake.status != StakeStatus.ACTIVE:
            raise ValueError(f"Stake {stake_id} is not active")
        
        # Calculate slash amount
        slash_amount = stake.amount * (slash_percent / 100)
        
        # Update stake
        stake.amount -= slash_amount
        stake.status = StakeStatus.SLASHED
        
        # Record slash history
        slash_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "amount": slash_amount,
            "percent": slash_percent,
            "reason": reason,
            "violation_id": violation_id,
        }
        
        if stake.slash_history is None:
            stake.slash_history = []
        stake.slash_history.append(slash_record)
        
        # Transfer slashed amount to treasury (in production)
        self._transfer_to_treasury(stake.wallet_address, stake.token_type, slash_amount)
        
        logger.warning(f"Slashed stake {stake_id}: {slash_amount} {stake.token_type} ({slash_percent}%)")
        return slash_amount
    
    async def distribute_reward(
        self,
        wallet_address: str,
        token_type: TokenType,
        amount: float,
        reason: str,
        epoch: int = None,
    ) -> Reward:
        """
        Distribute rewards to a miner.
        
        Creates reward record and updates balance.
        """
        reward_id = f"reward_{uuid.uuid4().hex[:12]}"
        
        reward = Reward(
            reward_id=reward_id,
            wallet_address=wallet_address,
            token_type=token_type,
            amount=amount,
            reason=reason,
            epoch=epoch or int(time.time() // 3600),  # Current hour as epoch
            distributed_at=datetime.now(timezone.utc).isoformat(),
        )
        
        self.rewards.append(reward)
        
        # Update balance
        self._update_balance(wallet_address, token_type, amount)
        
        logger.info(f"Distributed reward {reward_id}: {amount} {token_type} to {wallet_address}")
        return reward
    
    async def calculate_training_rewards(
        self,
        miner_id: str,
        hours_trained: float,
        performance_score: float = 1.0,
    ) -> float:
        """
        Calculate training rewards based on hours and performance.
        
        Returns reward amount in RG tokens.
        """
        wallet = await self.get_wallet(miner_id)
        if not wallet:
            return 0.0
        
        base_rate = self.reward_rates["training"]
        reward = base_rate * hours_trained * performance_score
        
        # Apply reputation bonus
        if slashing_engine and hasattr(slashing_engine, 'reputation_tracker'):
            reputation = slashing_engine.reputation_tracker.get(miner_id)
            if reputation:
                reputation_bonus = reputation.score / 100  # 0.5 to 1.5 multiplier
                reward *= reputation_bonus
        
        return reward
    
    async def calculate_seeding_rewards(
        self,
        miner_id: str,
        hours_seeded: float,
        bandwidth_mbps: float,
    ) -> float:
        """
        Calculate P2P seeding rewards.
        
        Returns reward amount in RG tokens.
        """
        wallet = await self.get_wallet(miner_id)
        if not wallet:
            return 0.0
        
        base_rate = self.reward_rates["seeding"]
        bandwidth_bonus = min(bandwidth_mbps / 100, 2.0)  # Up to 2x bonus for high bandwidth
        reward = base_rate * hours_seeded * bandwidth_bonus
        
        return reward
    
    async def get_wallet_stats(self, wallet_address: str) -> Dict[str, Any]:
        """Get comprehensive wallet statistics."""
        balances = self.get_balances(wallet_address)
        stakes = [s for s in self.stakes.values() if s.wallet_address == wallet_address]
        rewards = [r for r in self.rewards if r.wallet_address == wallet_address]
        transactions = [t for t in self.transactions if t.wallet_address == wallet_address]
        
        # Calculate totals
        total_balance_usd = sum(b.usd_value for b in balances)
        total_staked = sum(s.amount for s in stakes if s.status == StakeStatus.ACTIVE)
        total_rewards_24h = sum(
            r.amount for r in rewards 
            if datetime.fromisoformat(r.distributed_at.replace('Z', '+00:00')) > 
            datetime.now(timezone.utc) - timedelta(hours=24)
        )
        
        return {
            "wallet_address": wallet_address,
            "balances": [asdict(b) for b in balances],
            "total_balance_usd": total_balance_usd,
            "stakes": [asdict(s) for s in stakes],
            "total_staked": total_staked,
            "rewards_24h": total_rewards_24h,
            "total_rewards": len(rewards),
            "transactions_count": len(transactions),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
    
    async def get_network_stats(self) -> Dict[str, Any]:
        """Get network-wide wallet statistics."""
        total_wallets = len(self.wallets)
        active_wallets = sum(1 for w in self.wallets.values() if w.is_active)
        
        # Calculate totals
        total_balances = {}
        for token_type in TokenType:
            total = 0.0
            for balances in self.balances.values():
                for balance in balances:
                    if balance.token_type == token_type:
                        total += balance.balance
            total_balances[token_type.value] = total
        
        total_staked = sum(s.amount for s in self.stakes.values() if s.status == StakeStatus.ACTIVE)
        total_slashed = sum(
            sum(record["amount"] for record in s.slash_history or [])
            for s in self.stakes.values()
        )
        
        rewards_24h = sum(
            r.amount for r in self.rewards
            if datetime.fromisoformat(r.distributed_at.replace('Z', '+00:00')) > 
            datetime.now(timezone.utc) - timedelta(hours=24)
        )
        
        return {
            "total_wallets": total_wallets,
            "active_wallets": active_wallets,
            "total_balances": total_balances,
            "total_staked": total_staked,
            "total_slashed": total_slashed,
            "rewards_24h": rewards_24h,
            "total_rewards": len(self.rewards),
            "total_transactions": len(self.transactions),
        }
    
    def _update_balance(self, wallet_address: str, token_type: TokenType, delta: float):
        """Update token balance for a wallet."""
        balances = self.balances.get(wallet_address, [])
        for balance in balances:
            if balance.token_type == token_type:
                balance.balance += delta
                balance.last_updated = datetime.now(timezone.utc).isoformat()
                
                # Update USD value (mock calculation)
                if token_type == TokenType.RG_TOKEN:
                    balance.usd_value = balance.balance * 1.0  # $1 per RG token
                elif token_type == TokenType.ETH:
                    balance.usd_value = balance.balance * 2000.0  # $2000 per ETH
                elif token_type == TokenType.USDC:
                    balance.usd_value = balance.balance  # $1 per USDC
                break
    
    def _transfer_to_treasury(self, wallet_address: str, token_type: TokenType, amount: float):
        """Transfer slashed amount to treasury (mock implementation)."""
        # In production, this would execute an on-chain transfer
        logger.info(f"Transferred {amount} {token_type} from {wallet_address} to treasury")
    
    async def create_transaction(
        self,
        wallet_address: str,
        token_type: TokenType,
        amount: float,
        gas_fee: float,
        metadata: Dict[str, Any] = None,
    ) -> Transaction:
        """Create a new transaction record."""
        tx_hash = f"0x{uuid.uuid4().hex[:64]}"
        
        transaction = Transaction(
            tx_hash=tx_hash,
            wallet_address=wallet_address,
            token_type=token_type,
            amount=amount,
            gas_fee=gas_fee,
            status=TransactionStatus.PENDING,
            block_number=None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )
        
        self.transactions.append(transaction)
        return transaction


# Global wallet service instance
wallet_service = WalletService()

# Import slashing engine for reputation calculations (will be injected)
slashing_engine = None
