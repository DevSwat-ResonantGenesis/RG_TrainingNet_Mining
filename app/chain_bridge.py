"""
CHAIN BRIDGE — Mining → External Blockchain
=============================================

Posts accepted gradient submissions to the external blockchain as
immutable on-chain records. Also handles service registration with
Lighthouse on startup.

The bridge is fire-and-forget: if the chain or lighthouse is down,
mining continues and logs a warning. No training is ever blocked by
a chain write failure.
"""

import asyncio
import hashlib
import hmac
import logging
import os
import time
from typing import Dict, Any, Optional

import httpx

logger = logging.getLogger(__name__)

# ── Configuration ──
EXTERNAL_BLOCKCHAIN_URL = os.getenv("EXTERNAL_BLOCKCHAIN_URL", "http://localhost:8702")
LIGHTHOUSE_URL = os.getenv("LIGHTHOUSE_URL", "http://localhost:8700")
CRYPTO_SERVICE_URL = os.getenv("CRYPTO_SERVICE_URL", "http://localhost:8010")
NODE_ID = os.getenv("NODE_ID", "mining-node-0")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
INTERNAL_SERVICE_KEY = os.getenv("AUTH_INTERNAL_SERVICE_KEY", "")


def _headers() -> Dict[str, str]:
    """Build auth headers for inter-service calls."""
    if INTERNAL_SERVICE_KEY:
        return {"X-Internal-Key": INTERNAL_SERVICE_KEY}
    if AUTH_TOKEN:
        return {"Authorization": f"Bearer {AUTH_TOKEN}"}
    return {}


class ChainBridge:
    """Async bridge from Mining service to External Blockchain."""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._chain_url = EXTERNAL_BLOCKCHAIN_URL
        self._lighthouse_url = LIGHTHOUSE_URL
        self._crypto_url = CRYPTO_SERVICE_URL
        self._enabled = True
        self._tx_count = 0
        self._tx_errors = 0
        self._credit_count = 0
        self._credit_errors = 0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0, headers=_headers())
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── On-chain gradient recording ──

    async def record_gradient_on_chain(
        self,
        miner_id: str,
        task_id: str,
        gradient_hash: str,
        loss_value: float,
        samples_processed: int,
        reward_amount: float,
        submission_id: str = "",
        model_id: str = "",
        global_step: int = 0,
    ):
        """
        Submit a training_gradient transaction to the external blockchain.
        Fire-and-forget — never blocks training.
        """
        if not self._enabled:
            return

        tx = {
            "tx_type": "training_gradient",
            "payload": {
                "miner_id": miner_id,
                "task_id": task_id,
                "gradient_hash": gradient_hash,
                "loss_value": loss_value,
                "samples_processed": samples_processed,
                "reward_amount": reward_amount,
                "submission_id": submission_id,
                "model_id": model_id,
                "global_step": global_step,
            },
        }

        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self._chain_url}/distributed/transactions",
                json=tx,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._tx_count += 1
                logger.info(
                    f"⛓ On-chain: gradient {gradient_hash[:12]}… "
                    f"→ tx_hash={data.get('tx_hash', '?')[:16]}… "
                    f"(miner={miner_id}, reward={reward_amount})"
                )
            else:
                self._tx_errors += 1
                logger.warning(f"Chain bridge: tx rejected ({resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            self._tx_errors += 1
            logger.warning(f"Chain bridge: failed to post gradient tx: {e}")

    async def record_aggregation_on_chain(
        self,
        global_step: int,
        layers_merged: int,
        miners_contributed: int,
    ):
        """Record an aggregation event on-chain."""
        if not self._enabled:
            return

        tx = {
            "tx_type": "set",
            "payload": {
                "key": f"aggregation:step:{global_step}",
                "value": {
                    "global_step": global_step,
                    "layers_merged": layers_merged,
                    "miners_contributed": miners_contributed,
                },
            },
        }

        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self._chain_url}/distributed/transactions",
                json=tx,
            )
            if resp.status_code == 200:
                self._tx_count += 1
                logger.info(f"⛓ On-chain: aggregation step {global_step} recorded")
        except Exception as e:
            logger.warning(f"Chain bridge: failed to post aggregation tx: {e}")

    # ── Crypto wallet credit ──

    @staticmethod
    def _sign_credit(gradient_hash: str, user_id: str, rgt_amount: float, timestamp: int) -> str:
        """
        HMAC-SHA256 proof-of-training signature.
        Only the mining service (which holds INTERNAL_SERVICE_KEY) can produce this.
        The crypto service verifies it before crediting — like Bitcoin's coinbase tx
        being tied to the block's proof-of-work.
        """
        msg = f"{gradient_hash}:{user_id}:{rgt_amount}:{timestamp}"
        return hmac.new(
            INTERNAL_SERVICE_KEY.encode(),
            msg.encode(),
            hashlib.sha256,
        ).hexdigest()

    async def credit_miner_wallet(
        self,
        user_id: Optional[str],
        email: Optional[str],
        rgt_amount: float,
        samples_processed: int,
        trust_score: float = 1.0,
        tier: str = "miner",
        gradient_hash: str = "",
        task_id: str = "",
        global_step: int = 0,
    ):
        """
        Credit mined $RGT to the user's wallet via Crypto service.
        Fire-and-forget — never blocks training.

        Security: HMAC-signed proof-of-training prevents anyone from
        calling the credit endpoint without having actually processed
        a gradient through the mining service.
        """
        if not self._enabled:
            return
        if not user_id and not email:
            return
        if not gradient_hash:
            logger.warning("Wallet credit skipped: no gradient_hash (no proof-of-training)")
            return

        timestamp = int(time.time())
        signature = self._sign_credit(gradient_hash, user_id or "", rgt_amount, timestamp)

        payload = {
            "user_id": user_id,
            "email": email,
            "rgt_amount": rgt_amount,
            "tasks_delta": 1,
            "samples_delta": samples_processed,
            "trust_score": trust_score,
            "tier": tier,
            "gradient_hash": gradient_hash,
            "task_id": task_id,
            "global_step": global_step,
            "timestamp": timestamp,
            "signature": signature,
        }

        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self._crypto_url}/crypto/miner/credit",
                json=payload,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._credit_count += 1
                logger.info(
                    f"💰 Wallet credit: {rgt_amount} RGT → "
                    f"user={user_id or email} "
                    f"(total={data.get('rgt_earned', '?')})"
                )
            else:
                self._credit_errors += 1
                logger.warning(f"Wallet credit failed ({resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            self._credit_errors += 1
            logger.warning(f"Wallet credit failed: {e}")

    # ── Lighthouse registration ──

    async def register_with_lighthouse(self, service_type: str = "mining"):
        """Register this service as a peer with Lighthouse."""
        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self._lighthouse_url}/lighthouse/register",
                json={
                    "peer_id": NODE_ID,
                    "peer_type": "chain" if service_type == "blockchain" else "validator",
                    "address": "127.0.0.1",
                    "p2p_port": 8602 if service_type == "blockchain" else 8601,
                    "api_port": 8702 if service_type == "blockchain" else 8701,
                    "node_version": "0.1.0",
                    "capabilities": ["training", "gradient_submit", "block_production"],
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                peers = data.get("bootstrap_peers", [])
                logger.info(
                    f"Registered with Lighthouse as {service_type} "
                    f"({len(peers)} peers discovered)"
                )
                return data
            else:
                logger.warning(f"Lighthouse registration failed: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Lighthouse registration failed: {e}")
        return None

    async def send_lighthouse_heartbeat(self):
        """Send heartbeat to Lighthouse."""
        try:
            client = await self._get_client()
            await client.post(
                f"{self._lighthouse_url}/lighthouse/heartbeat",
                json={"peer_id": NODE_ID},
            )
        except Exception:
            pass

    # ── Stats ──

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "chain_url": self._chain_url,
            "lighthouse_url": self._lighthouse_url,
            "crypto_url": self._crypto_url,
            "tx_count": self._tx_count,
            "tx_errors": self._tx_errors,
            "credit_count": self._credit_count,
            "credit_errors": self._credit_errors,
        }


# Global singleton
chain_bridge = ChainBridge()
