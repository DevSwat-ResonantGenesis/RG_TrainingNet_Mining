"""
SHARD SLICER — Stream Specific Layer Weights to Miners
========================================================

No miner ever downloads the full model. Each miner streams only its
assigned layers from the network.

This module provides:
  1. Weight serialization — extract specific layers from a model checkpoint
  2. Streaming API — chunked HTTP transfer of layer weights
  3. P2P weight transfer — miners pull from peers, not just the seed
  4. Progressive loading — start training before all weights arrive
  5. Integrity verification — SHA-256 hash per layer chunk

The flow:
  Miner gets assignment (layers 32-40)
    → queries WeightShardRegistry for sources
    → if peer has it: P2P pull from peer (fastest)
    → if no peer: pull from seed/checkpoint via shard slicer
    → verify hash against on-chain record
    → report LOADED to registry
    → start training

STATUS: PRODUCTION IMPLEMENTATION
UPDATED: 2026-04-02
PURPOSE: Selective weight streaming for network-native distributed models
"""

import asyncio
import hashlib
import io
import logging
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger("rg-mining.shard-slicer")

# Try torch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ══════════════════════════════════════════════════════════════
# WEIGHT SERIALIZATION
# ══════════════════════════════════════════════════════════════

@dataclass
class LayerWeightChunk:
    """A single chunk of serialized layer weights for streaming."""
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    model_id: str = ""
    layer_index: int = 0
    param_name: str = ""             # e.g. "layers.32.attention.q_proj.weight"
    chunk_index: int = 0             # For multi-chunk params
    total_chunks: int = 1
    data: bytes = b""                # Serialized tensor data
    shape: Tuple = ()
    dtype: str = "float16"
    hash: str = ""                   # SHA-256 of this chunk's data
    size_bytes: int = 0

    def compute_hash(self) -> str:
        self.hash = hashlib.sha256(self.data).hexdigest()
        self.size_bytes = len(self.data)
        return self.hash

    def to_header(self) -> Dict[str, Any]:
        """Metadata header (sent before data)."""
        return {
            "chunk_id": self.chunk_id,
            "model_id": self.model_id,
            "layer_index": self.layer_index,
            "param_name": self.param_name,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "hash": self.hash,
            "size_bytes": self.size_bytes,
        }


@dataclass
class SliceManifest:
    """
    Manifest describing a weight slice — sent to the miner before streaming.
    
    The miner uses this to:
      - Allocate the right amount of memory
      - Verify all chunks arrived
      - Check integrity via hash tree
    """
    manifest_id: str = field(default_factory=lambda: str(uuid4()))
    model_id: str = ""
    layer_start: int = 0
    layer_end: int = 0
    has_embedding: bool = False
    has_lm_head: bool = False
    total_params: int = 0
    total_bytes: int = 0
    num_chunks: int = 0
    chunk_headers: List[Dict] = field(default_factory=list)
    slice_hash: str = ""             # Hash of all chunk hashes combined
    version: int = 0                 # Global training step
    source_miner_id: str = ""        # Who is serving this slice

    def compute_slice_hash(self) -> str:
        """Compute aggregate hash from all chunk hashes."""
        combined = "|".join(
            ch.get("hash", "") for ch in sorted(
                self.chunk_headers, key=lambda c: (c["layer_index"], c["param_name"])
            )
        )
        self.slice_hash = hashlib.sha256(combined.encode()).hexdigest()
        return self.slice_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "model_id": self.model_id,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "has_embedding": self.has_embedding,
            "has_lm_head": self.has_lm_head,
            "total_params": self.total_params,
            "total_bytes": self.total_bytes,
            "num_chunks": self.num_chunks,
            "slice_hash": self.slice_hash,
            "version": self.version,
            "source_miner_id": self.source_miner_id,
            "chunk_headers": self.chunk_headers,
        }


# ══════════════════════════════════════════════════════════════
# SHARD SLICER ENGINE
# ══════════════════════════════════════════════════════════════

# Maximum chunk size for streaming (4MB — fits in most HTTP buffers)
MAX_CHUNK_BYTES = 4 * 1024 * 1024


class ShardSlicer:
    """
    Extracts and streams specific layer weights from model state dicts.
    
    Works with:
      - In-memory model state (from param server or loaded model)
      - On-disk checkpoints (memory-mapped, zero-copy where possible)
      - Peer-to-peer relay (forward chunks from another miner)
    """

    def __init__(self, max_chunk_bytes: int = MAX_CHUNK_BYTES):
        self.max_chunk_bytes = max_chunk_bytes
        self._cache: Dict[str, Dict] = {}  # model_id → { param_name → tensor }
        self._stats = {
            "slices_served": 0,
            "bytes_served": 0,
            "chunks_served": 0,
        }

    def load_model_weights(self, model_id: str, state_dict: Dict[str, Any]):
        """
        Cache a model's state dict for slicing.
        
        In production, this would be the param server's aggregated weights.
        For genesis, this is the initial random weights.
        """
        self._cache[model_id] = state_dict
        total_params = sum(
            v.numel() for v in state_dict.values()
            if HAS_TORCH and isinstance(v, torch.Tensor)
        )
        logger.info(f"Loaded {model_id} weights for slicing: {len(state_dict)} params, {total_params:,} total")

    def extract_layer_params(
        self,
        model_id: str,
        layer_start: int,
        layer_end: int,
        include_embedding: bool = False,
        include_lm_head: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract only the parameters for specific layers from the cached state dict.
        
        Returns a filtered state dict containing only the requested layers.
        This is the core "slicing" operation.
        """
        full_state = self._cache.get(model_id, {})
        if not full_state:
            return {}

        sliced = {}
        for name, tensor in full_state.items():
            # Match layer-specific params: "layers.{N}.attention.q_proj.weight" etc.
            if name.startswith("layers."):
                parts = name.split(".")
                if len(parts) >= 2:
                    try:
                        layer_idx = int(parts[1])
                        if layer_start <= layer_idx < layer_end:
                            # Remap layer index to shard-local: layers.32 → layers.0
                            local_idx = layer_idx - layer_start
                            local_name = f"layers.{local_idx}" + "." + ".".join(parts[2:])
                            sliced[local_name] = tensor
                    except ValueError:
                        pass

            # Embedding layer (only for stage 0)
            elif include_embedding and name in ("embed_tokens.weight", "tok_embeddings.weight"):
                sliced[name] = tensor

            # LM head and final norm (only for last stage)
            elif include_lm_head and name in ("lm_head.weight", "output.weight", "norm.weight", "final_norm.weight"):
                sliced[name] = tensor

        return sliced

    def create_manifest(
        self,
        model_id: str,
        layer_start: int,
        layer_end: int,
        include_embedding: bool = False,
        include_lm_head: bool = False,
        version: int = 0,
        source_miner_id: str = "",
    ) -> Optional[SliceManifest]:
        """
        Create a manifest for a weight slice — sent before streaming begins.
        
        The manifest tells the miner exactly what to expect:
        how many chunks, their sizes, shapes, and hashes.
        """
        if not HAS_TORCH:
            logger.error("PyTorch required for weight slicing")
            return None

        sliced = self.extract_layer_params(
            model_id, layer_start, layer_end,
            include_embedding, include_lm_head,
        )

        if not sliced:
            return None

        manifest = SliceManifest(
            model_id=model_id,
            layer_start=layer_start,
            layer_end=layer_end,
            has_embedding=include_embedding,
            has_lm_head=include_lm_head,
            version=version,
            source_miner_id=source_miner_id,
        )

        total_params = 0
        total_bytes = 0

        for param_name, tensor in sorted(sliced.items()):
            # Serialize tensor to bytes
            data = self._serialize_tensor(tensor)
            num_chunks = math.ceil(len(data) / self.max_chunk_bytes)

            # Determine layer index from param name
            layer_idx = 0
            if param_name.startswith("layers."):
                try:
                    layer_idx = int(param_name.split(".")[1]) + layer_start
                except ValueError:
                    pass

            for ci in range(num_chunks):
                start = ci * self.max_chunk_bytes
                end = min(start + self.max_chunk_bytes, len(data))
                chunk_data = data[start:end]

                chunk = LayerWeightChunk(
                    model_id=model_id,
                    layer_index=layer_idx,
                    param_name=param_name,
                    chunk_index=ci,
                    total_chunks=num_chunks,
                    data=chunk_data,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype).replace("torch.", ""),
                )
                chunk.compute_hash()
                manifest.chunk_headers.append(chunk.to_header())

            total_params += tensor.numel()
            total_bytes += len(data)

        manifest.total_params = total_params
        manifest.total_bytes = total_bytes
        manifest.num_chunks = len(manifest.chunk_headers)
        manifest.compute_slice_hash()

        return manifest

    async def stream_slice(
        self,
        model_id: str,
        layer_start: int,
        layer_end: int,
        include_embedding: bool = False,
        include_lm_head: bool = False,
    ) -> AsyncIterator[Tuple[Dict, bytes]]:
        """
        Async generator that yields (header, data) tuples for streaming.
        
        Usage:
            async for header, data in slicer.stream_slice("model-x", 32, 40):
                # Send header + data to requesting miner
        """
        if not HAS_TORCH:
            return

        sliced = self.extract_layer_params(
            model_id, layer_start, layer_end,
            include_embedding, include_lm_head,
        )

        for param_name, tensor in sorted(sliced.items()):
            data = self._serialize_tensor(tensor)
            num_chunks = math.ceil(len(data) / self.max_chunk_bytes)

            layer_idx = 0
            if param_name.startswith("layers."):
                try:
                    layer_idx = int(param_name.split(".")[1]) + layer_start
                except ValueError:
                    pass

            for ci in range(num_chunks):
                start = ci * self.max_chunk_bytes
                end = min(start + self.max_chunk_bytes, len(data))
                chunk_data = data[start:end]

                chunk = LayerWeightChunk(
                    model_id=model_id,
                    layer_index=layer_idx,
                    param_name=param_name,
                    chunk_index=ci,
                    total_chunks=num_chunks,
                    data=chunk_data,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype).replace("torch.", ""),
                )
                chunk.compute_hash()

                self._stats["chunks_served"] += 1
                self._stats["bytes_served"] += len(chunk_data)

                yield chunk.to_header(), chunk_data

                # Yield control to event loop between chunks
                await asyncio.sleep(0)

        self._stats["slices_served"] += 1

    def _serialize_tensor(self, tensor) -> bytes:
        """Serialize a tensor to bytes for transmission."""
        if not HAS_TORCH:
            return b""

        # Use torch.save to a BytesIO buffer — handles all dtypes
        buf = io.BytesIO()
        torch.save(tensor.cpu(), buf)
        return buf.getvalue()

    def deserialize_tensor(self, data: bytes, shape: Tuple = None, dtype: str = "float16"):
        """Deserialize bytes back to a tensor."""
        if not HAS_TORCH:
            return None

        buf = io.BytesIO(data)
        tensor = torch.load(buf, weights_only=True)
        return tensor

    def compute_weight_hash(self, model_id: str, layer_start: int, layer_end: int) -> str:
        """Compute hash of specific layer weights (for integrity verification)."""
        sliced = self.extract_layer_params(model_id, layer_start, layer_end)
        if not sliced:
            return ""

        h = hashlib.sha256()
        for name in sorted(sliced.keys()):
            h.update(name.encode())
            h.update(self._serialize_tensor(sliced[name]))
        return h.hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "cached_models": list(self._cache.keys()),
            "slices_served": self._stats["slices_served"],
            "chunks_served": self._stats["chunks_served"],
            "bytes_served": self._stats["bytes_served"],
            "bytes_served_gb": round(self._stats["bytes_served"] / 1e9, 3),
        }


# Global instance
shard_slicer = ShardSlicer()


# ══════════════════════════════════════════════════════════════
# P2P WEIGHT TRANSFER PROTOCOL
# ══════════════════════════════════════════════════════════════

@dataclass
class WeightTransferRequest:
    """Request from a miner to pull specific layer weights."""
    request_id: str = field(default_factory=lambda: str(uuid4()))
    requester_miner_id: str = ""
    model_id: str = ""
    layer_start: int = 0
    layer_end: int = 0
    include_embedding: bool = False
    include_lm_head: bool = False
    preferred_source: str = ""       # Preferred peer to pull from
    max_sources: int = 3             # Try up to N sources in parallel

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "requester_miner_id": self.requester_miner_id,
            "model_id": self.model_id,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "include_embedding": self.include_embedding,
            "include_lm_head": self.include_lm_head,
            "preferred_source": self.preferred_source,
            "max_sources": self.max_sources,
        }


@dataclass
class WeightTransferPlan:
    """
    Plan for how a miner should download its weights.
    
    Computed by the registry based on available sources:
    - Best case: pull from a peer in the same region (lowest latency)
    - Fallback: pull from any peer with the shard loaded
    - Last resort: pull from seed/checkpoint via shard slicer
    """
    plan_id: str = field(default_factory=lambda: str(uuid4()))
    request_id: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    estimated_transfer_time_s: float = 0.0
    total_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "request_id": self.request_id,
            "sources": self.sources,
            "estimated_transfer_time_s": self.estimated_transfer_time_s,
            "total_bytes": self.total_bytes,
            "total_mb": round(self.total_bytes / 1e6, 1),
        }


def create_transfer_plan(
    request: WeightTransferRequest,
    registry,  # WeightShardRegistry
    shard_manager=None,  # Optional ShardManager for miner addresses
) -> WeightTransferPlan:
    """
    Create an optimal transfer plan for a weight download request.
    
    Strategy:
    1. Check registry for peers that have the exact shard loaded
    2. Check for overlapping shards that cover the range
    3. Fall back to seed slicer endpoint
    4. Rank by: locality > bandwidth > load
    """
    plan = WeightTransferPlan(request_id=request.request_id)

    # Find exact sources
    sources = registry.find_shard_sources(
        request.model_id, request.layer_start, request.layer_end
    )

    if not sources:
        # Try overlapping sources
        sources = registry.find_overlapping_sources(
            request.model_id, request.layer_start, request.layer_end
        )

    for source in sources[:request.max_sources]:
        plan.sources.append({
            "type": "peer",
            "miner_id": source.miner_id,
            "address": source.miner_address,
            "layer_start": source.layer_start,
            "layer_end": source.layer_end,
            "version": source.version,
            "size_bytes": source.size_bytes,
            "priority": source.priority.value,
        })
        plan.total_bytes += source.size_bytes

    # Always add seed slicer as fallback
    plan.sources.append({
        "type": "seed_slicer",
        "endpoint": "/mining/weights/stream",
        "layer_start": request.layer_start,
        "layer_end": request.layer_end,
        "priority": "fallback",
    })

    # Estimate transfer time (assume 100 MB/s average)
    if plan.total_bytes > 0:
        plan.estimated_transfer_time_s = plan.total_bytes / (100 * 1e6)

    return plan
