"""
CHAIN BRIDGE — Mining → External Blockchain
=============================================

Posts accepted gradient submissions to the external blockchain as
immutable on-chain records. Also handles service registration with
Lighthouse on startup.

Wallet credits use a persistent queue (PendingCredit table) so that
tokens are never lost even if the service restarts mid-flight.
"""

import asyncio
import hashlib
import hmac
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import httpx
from sqlalchemy import select, update

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

    # ── Persistent credit queue helpers ──

    async def _save_pending_credit(self, **kwargs):
        """Save a credit to the persistent queue BEFORE sending."""
        try:
            from .ml_db import MLSessionLocal, PendingCredit
            if not MLSessionLocal:
                return
            async with MLSessionLocal() as session:
                pc = PendingCredit(**kwargs)
                session.add(pc)
                await session.commit()
                logger.debug(f"Saved pending credit: {kwargs.get('gradient_hash', '')[:16]}")
        except Exception as e:
            logger.warning(f"Failed to save pending credit to DB: {e}")

    async def _mark_credit_sent(self, gradient_hash: str):
        """Mark a pending credit as successfully sent."""
        try:
            from .ml_db import MLSessionLocal, PendingCredit
            if not MLSessionLocal:
                return
            async with MLSessionLocal() as session:
                await session.execute(
                    update(PendingCredit)
                    .where(PendingCredit.gradient_hash == gradient_hash)
                    .values(status="sent", sent_at=datetime.now(timezone.utc))
                )
                await session.commit()
        except Exception as e:
            logger.warning(f"Failed to mark credit sent: {e}")

    async def _mark_credit_failed(self, gradient_hash: str, error: str, attempts: int):
        """Mark a pending credit as failed with error details."""
        try:
            from .ml_db import MLSessionLocal, PendingCredit
            if not MLSessionLocal:
                return
            async with MLSessionLocal() as session:
                await session.execute(
                    update(PendingCredit)
                    .where(PendingCredit.gradient_hash == gradient_hash)
                    .values(
                        status="failed" if attempts >= 5 else "pending",
                        attempts=attempts,
                        last_error=error[:500],
                    )
                )
                await session.commit()
        except Exception as e:
            logger.warning(f"Failed to mark credit failed: {e}")

    async def retry_pending_credits(self):
        """
        Called on startup — retries any credits stuck in 'pending' status.
        This is the key mechanism that prevents token loss on restart.
        """
        try:
            from .ml_db import MLSessionLocal, PendingCredit
            if not MLSessionLocal:
                return
            async with MLSessionLocal() as session:
                result = await session.execute(
                    select(PendingCredit)
                    .where(PendingCredit.status == "pending")
                    .order_by(PendingCredit.created_at)
                )
                pending = result.scalars().all()

            if not pending:
                return

            logger.info(f"🔄 Retrying {len(pending)} pending wallet credits...")
            for pc in pending:
                try:
                    await self._send_credit_to_crypto(
                        user_id=pc.user_id,
                        email=pc.email,
                        rgt_amount=pc.rgt_amount,
                        samples_processed=pc.samples_processed,
                        trust_score=pc.trust_score,
                        tier=pc.tier,
                        gradient_hash=pc.gradient_hash,
                        task_id=pc.task_id,
                        global_step=pc.global_step,
                    )
                    await self._mark_credit_sent(pc.gradient_hash)
                    logger.info(f"✅ Retry succeeded: {pc.rgt_amount} RGT → {pc.user_id or pc.email}")
                except Exception as e:
                    await self._mark_credit_failed(pc.gradient_hash, str(e), (pc.attempts or 0) + 1)
                    logger.warning(f"Retry failed for {pc.gradient_hash[:16]}: {e}")
        except Exception as e:
            logger.warning(f"Pending credit retry scan failed: {e}")

    async def _send_credit_to_crypto(
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
        """Send the actual HTTP credit request to crypto service. Raises on failure."""
        if not INTERNAL_SERVICE_KEY:
            logger.error(
                "AUTH_INTERNAL_SERVICE_KEY is NOT set — wallet credits will be rejected by crypto service. "
                "Set this env var to the same value used by crypto_service."
            )

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
            elif resp.status_code == 403:
                self._credit_errors += 1
                detail = resp.text[:200]
                logger.error(
                    f"Wallet credit REJECTED (403): {detail}. "
                    f"Check AUTH_INTERNAL_SERVICE_KEY matches between mining and crypto services."
                )
                raise RuntimeError(f"Crypto service rejected credit (403): {detail}")
            elif resp.status_code == 409:
                # Already credited — treat as success
                logger.info(f"Wallet credit: gradient already credited (replay blocked)")
            else:
                self._credit_errors += 1
                detail = resp.text[:200]
                logger.warning(f"Wallet credit failed ({resp.status_code}): {detail}")
                raise RuntimeError(f"Crypto service returned {resp.status_code}: {detail}")
        except (RuntimeError, ValueError):
            raise
        except Exception as e:
            self._credit_errors += 1
            logger.warning(f"Wallet credit failed (network/connection): {e}")
            raise RuntimeError(f"Wallet credit network error: {e}") from e

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

        Flow: save to DB → send HTTP → mark sent.
        If the service dies between save and send, retry_pending_credits()
        picks it up on next startup. Tokens are NEVER lost.

        Raises on failure so the caller can notify the miner.
        """
        if not self._enabled:
            logger.warning("Wallet credit skipped: chain bridge disabled")
            return
        if not user_id and not email:
            raise ValueError("Wallet credit failed: no user_id or email provided")
        if not gradient_hash:
            raise ValueError("Wallet credit failed: no gradient_hash (no proof-of-training)")

        # Step 1: Persist to DB BEFORE sending (crash-safe)
        await self._save_pending_credit(
            gradient_hash=gradient_hash,
            user_id=user_id,
            email=email,
            rgt_amount=rgt_amount,
            samples_processed=samples_processed,
            trust_score=trust_score,
            tier=tier,
            task_id=task_id,
            global_step=global_step,
        )

        # Step 2: Send to crypto service
        await self._send_credit_to_crypto(
            user_id=user_id,
            email=email,
            rgt_amount=rgt_amount,
            samples_processed=samples_processed,
            trust_score=trust_score,
            tier=tier,
            gradient_hash=gradient_hash,
            task_id=task_id,
            global_step=global_step,
        )

        # Step 3: Mark as sent in DB
        await self._mark_credit_sent(gradient_hash)

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
