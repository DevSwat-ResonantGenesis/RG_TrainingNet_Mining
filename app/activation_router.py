"""
Activation Router — P2P Tensor Forwarding Between Pipeline Stages
==================================================================

Routes activations (forward pass) and gradients (backward pass) between
miners in a pipeline group. Handles serialization, compression, and
transfer verification.

Transport options:
  1. WebSocket (default) — already used for gradient submission
  2. Direct TCP — lower latency for large tensors (future)
  3. Shared storage — S3/NFS for very large activations (future)

Compression strategies:
  1. None — raw fp16 tensors (small models)
  2. INT8 quantization — 2x compression, minimal quality loss
  3. Top-K sparsification — send only largest values (extreme compression)
  4. Mixed — INT8 for forward, full precision for backward

Scales to any activation size — from 1KB (tiny model) to 100GB+ (future).
Uses chunked transfer for activations that don't fit in one message.
"""

import asyncio
import base64
import hashlib
import logging
import struct
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger("rg-mining.activation-router")


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

class CompressionMethod(str, Enum):
    NONE = "none"
    INT8 = "int8"           # Quantize fp16 → int8 (2x compression)
    ZLIB = "zlib"           # zlib compression on raw bytes
    INT8_ZLIB = "int8_zlib" # Quantize then compress (best ratio)
    TOP_K = "top_k"         # Only send top-k values (sparse, lossy)


class TransferDirection(str, Enum):
    FORWARD = "forward"     # Activations: stage N → stage N+1
    BACKWARD = "backward"   # Gradients: stage N+1 → stage N


# Max single message size (chunk larger transfers)
MAX_MESSAGE_BYTES = 64 * 1024 * 1024  # 64 MB per chunk

# Default compression threshold — compress if activation > this size
COMPRESSION_THRESHOLD_BYTES = 1 * 1024 * 1024  # 1 MB


# ══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════

@dataclass
class ActivationMetadata:
    """Metadata for an activation transfer between pipeline stages."""
    transfer_id: str = field(default_factory=lambda: f"act-{uuid4().hex[:10]}")
    execution_id: str = ""           # Pipeline execution this belongs to
    pipeline_group_id: str = ""
    direction: TransferDirection = TransferDirection.FORWARD
    microbatch_index: int = 0
    source_stage: int = 0
    target_stage: int = 0
    source_miner_id: str = ""
    target_miner_id: str = ""
    # Tensor metadata
    shape: List[int] = field(default_factory=list)  # e.g. [batch, seq, hidden]
    dtype: str = "float16"
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    compression: CompressionMethod = CompressionMethod.NONE
    num_chunks: int = 1              # How many chunks this transfer is split into
    checksum: str = ""               # SHA256 of original tensor bytes
    # Timing
    created_at: float = field(default_factory=time.time)
    transfer_start: Optional[float] = None
    transfer_end: Optional[float] = None

    @property
    def compression_ratio(self) -> float:
        if self.compressed_size_bytes <= 0:
            return 1.0
        return self.original_size_bytes / self.compressed_size_bytes

    @property
    def transfer_time(self) -> float:
        if self.transfer_start and self.transfer_end:
            return self.transfer_end - self.transfer_start
        return 0.0

    @property
    def bandwidth_mbps(self) -> float:
        t = self.transfer_time
        if t <= 0:
            return 0.0
        return (self.compressed_size_bytes * 8 / 1e6) / t

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transfer_id": self.transfer_id,
            "execution_id": self.execution_id,
            "pipeline_group_id": self.pipeline_group_id,
            "direction": self.direction.value,
            "microbatch_index": self.microbatch_index,
            "source_stage": self.source_stage,
            "target_stage": self.target_stage,
            "source_miner_id": self.source_miner_id,
            "target_miner_id": self.target_miner_id,
            "shape": self.shape,
            "dtype": self.dtype,
            "original_size_bytes": self.original_size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "compression": self.compression.value,
            "compression_ratio": round(self.compression_ratio, 2),
            "num_chunks": self.num_chunks,
            "checksum": self.checksum,
            "transfer_time_ms": round(self.transfer_time * 1000, 1),
            "bandwidth_mbps": round(self.bandwidth_mbps, 1),
        }


@dataclass
class ActivationChunk:
    """One chunk of a potentially multi-chunk activation transfer."""
    transfer_id: str
    chunk_index: int
    total_chunks: int
    data: bytes = b""
    is_last: bool = False

    def to_message(self) -> Dict[str, Any]:
        return {
            "type": "activation_chunk",
            "transfer_id": self.transfer_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "data": base64.b64encode(self.data).decode("ascii"),
            "is_last": self.is_last,
        }

    @classmethod
    def from_message(cls, msg: Dict[str, Any]) -> "ActivationChunk":
        return cls(
            transfer_id=msg["transfer_id"],
            chunk_index=msg["chunk_index"],
            total_chunks=msg["total_chunks"],
            data=base64.b64decode(msg["data"]),
            is_last=msg.get("is_last", False),
        )


# ══════════════════════════════════════════════════════════════
# TENSOR SERIALIZATION & COMPRESSION
# ══════════════════════════════════════════════════════════════

class TensorSerializer:
    """
    Serialize and compress tensors for network transfer.
    
    Supports any tensor size — uses chunked transfer for large tensors.
    Compression is adaptive: no compression for small tensors, INT8 for
    medium, INT8+zlib for large.
    """

    @staticmethod
    def serialize(tensor, compression: CompressionMethod = CompressionMethod.NONE) -> Tuple[bytes, ActivationMetadata]:
        """
        Serialize a PyTorch tensor to bytes with optional compression.
        
        Returns (compressed_bytes, metadata).
        Works with any tensor shape and dtype.
        """
        import torch
        import numpy as np

        # Get raw bytes
        if tensor.is_cuda or (hasattr(tensor, 'device') and str(tensor.device) != 'cpu'):
            tensor = tensor.detach().cpu()
        else:
            tensor = tensor.detach()

        # Convert to contiguous numpy array
        np_arr = tensor.float().numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
        raw_bytes = np_arr.tobytes()

        metadata = ActivationMetadata(
            shape=list(tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
            original_size_bytes=len(raw_bytes),
        )

        # Compute checksum
        metadata.checksum = hashlib.sha256(raw_bytes).hexdigest()[:16]

        # Apply compression
        if compression == CompressionMethod.NONE:
            metadata.compression = CompressionMethod.NONE
            metadata.compressed_size_bytes = len(raw_bytes)
            return raw_bytes, metadata

        elif compression == CompressionMethod.INT8:
            compressed = TensorSerializer._quantize_int8(np_arr)
            metadata.compression = CompressionMethod.INT8
            metadata.compressed_size_bytes = len(compressed)
            return compressed, metadata

        elif compression == CompressionMethod.ZLIB:
            compressed = zlib.compress(raw_bytes, level=1)  # Fast compression
            metadata.compression = CompressionMethod.ZLIB
            metadata.compressed_size_bytes = len(compressed)
            return compressed, metadata

        elif compression == CompressionMethod.INT8_ZLIB:
            quantized = TensorSerializer._quantize_int8(np_arr)
            compressed = zlib.compress(quantized, level=1)
            metadata.compression = CompressionMethod.INT8_ZLIB
            metadata.compressed_size_bytes = len(compressed)
            return compressed, metadata

        elif compression == CompressionMethod.TOP_K:
            compressed = TensorSerializer._top_k_sparsify(np_arr, k_fraction=0.1)
            metadata.compression = CompressionMethod.TOP_K
            metadata.compressed_size_bytes = len(compressed)
            return compressed, metadata

        # Fallback
        metadata.compressed_size_bytes = len(raw_bytes)
        return raw_bytes, metadata

    @staticmethod
    def deserialize(data: bytes, metadata: ActivationMetadata):
        """
        Deserialize bytes back to a PyTorch tensor.
        Reverses whatever compression was applied.
        """
        import torch
        import numpy as np

        compression = metadata.compression

        if compression == CompressionMethod.NONE:
            np_arr = np.frombuffer(data, dtype=np.float32 if "32" in metadata.dtype else np.float16)

        elif compression == CompressionMethod.INT8:
            np_arr = TensorSerializer._dequantize_int8(data)

        elif compression == CompressionMethod.ZLIB:
            raw = zlib.decompress(data)
            np_arr = np.frombuffer(raw, dtype=np.float32 if "32" in metadata.dtype else np.float16)

        elif compression == CompressionMethod.INT8_ZLIB:
            quantized = zlib.decompress(data)
            np_arr = TensorSerializer._dequantize_int8(quantized)

        elif compression == CompressionMethod.TOP_K:
            np_arr = TensorSerializer._de_sparsify_top_k(data, metadata.shape)

        else:
            np_arr = np.frombuffer(data, dtype=np.float32)

        np_arr = np_arr.reshape(metadata.shape)
        tensor = torch.from_numpy(np_arr.copy())

        # Convert back to original dtype
        if metadata.dtype == "float16":
            tensor = tensor.half()
        elif metadata.dtype == "bfloat16":
            tensor = tensor.bfloat16()

        return tensor

    @staticmethod
    def _quantize_int8(arr) -> bytes:
        """Quantize float array to int8 with scale factor. ~2x compression."""
        import numpy as np
        scale = max(abs(arr.max()), abs(arr.min()), 1e-10)
        quantized = np.clip(arr / scale * 127, -127, 127).astype(np.int8)
        # Pack: 4 bytes scale + int8 data
        return struct.pack('f', float(scale)) + quantized.tobytes()

    @staticmethod
    def _dequantize_int8(data: bytes):
        """Dequantize int8 back to float32."""
        import numpy as np
        scale = struct.unpack('f', data[:4])[0]
        quantized = np.frombuffer(data[4:], dtype=np.int8)
        return (quantized.astype(np.float32) / 127.0 * scale)

    @staticmethod
    def _top_k_sparsify(arr, k_fraction: float = 0.1) -> bytes:
        """Keep only top-k% of values by magnitude. Extreme compression."""
        import numpy as np
        flat = arr.flatten()
        k = max(1, int(len(flat) * k_fraction))
        indices = np.argpartition(np.abs(flat), -k)[-k:]
        values = flat[indices].astype(np.float32)
        # Pack: 4 bytes k + 4 bytes total_len + (indices + values)
        packed = struct.pack('II', k, len(flat))
        packed += indices.astype(np.uint32).tobytes()
        packed += values.tobytes()
        return packed

    @staticmethod
    def _de_sparsify_top_k(data: bytes, shape: List[int]):
        """Reconstruct array from top-k sparse representation."""
        import numpy as np
        k, total_len = struct.unpack('II', data[:8])
        offset = 8
        indices = np.frombuffer(data[offset:offset + k * 4], dtype=np.uint32)
        offset += k * 4
        values = np.frombuffer(data[offset:offset + k * 4], dtype=np.float32)
        result = np.zeros(total_len, dtype=np.float32)
        result[indices] = values
        return result

    @staticmethod
    def choose_compression(size_bytes: int, direction: TransferDirection) -> CompressionMethod:
        """
        Auto-select compression based on tensor size and direction.
        
        Rules:
        - < 1 MB: no compression (overhead not worth it)
        - 1-100 MB: INT8 (fast, 2x compression)
        - 100 MB - 1 GB: INT8 + zlib (3-4x compression)
        - > 1 GB: INT8 + zlib (essential at this scale)
        - Backward pass: always at least INT8 (gradients compress well)
        """
        if size_bytes < COMPRESSION_THRESHOLD_BYTES:
            return CompressionMethod.NONE
        if direction == TransferDirection.BACKWARD:
            # Gradients compress very well
            if size_bytes > 100 * 1024 * 1024:
                return CompressionMethod.INT8_ZLIB
            return CompressionMethod.INT8
        # Forward activations
        if size_bytes > 100 * 1024 * 1024:
            return CompressionMethod.INT8_ZLIB
        return CompressionMethod.INT8


# ══════════════════════════════════════════════════════════════
# ACTIVATION ROUTER
# ══════════════════════════════════════════════════════════════

class ActivationRouter:
    """
    Routes activations between pipeline stages.
    
    Each pipeline group has one ActivationRouter that manages all
    activation transfers for all microbatches within that group.
    
    The router handles:
    1. Serializing tensors → bytes
    2. Chunking large transfers
    3. Sending via registered transport (WebSocket, TCP, etc.)
    4. Reassembling chunks on the receiving end
    5. Deserializing bytes → tensors
    6. Transfer verification (checksums)
    7. Bandwidth monitoring
    """

    def __init__(self, pipeline_group_id: str):
        self.pipeline_group_id = pipeline_group_id
        self.pending_transfers: Dict[str, ActivationMetadata] = {}
        self.chunk_buffers: Dict[str, Dict[int, bytes]] = {}  # transfer_id → {chunk_idx → data}
        self.completed_transfers: List[ActivationMetadata] = []
        self._send_fn: Optional[Callable] = None  # Registered transport function
        self._receive_callbacks: Dict[str, Callable] = {}  # transfer_id → callback

        # Stats
        self.total_bytes_sent: int = 0
        self.total_bytes_received: int = 0
        self.total_transfers: int = 0
        self.total_transfer_time: float = 0.0

    def register_transport(self, send_fn: Callable):
        """
        Register the transport function for sending data.
        
        send_fn(target_miner_id: str, message: dict) → awaitable
        
        This decouples the router from the transport layer.
        Could be WebSocket, TCP, HTTP, or any async transport.
        """
        self._send_fn = send_fn

    async def send_activation(
        self,
        tensor,
        metadata: ActivationMetadata,
        compression: Optional[CompressionMethod] = None,
    ) -> ActivationMetadata:
        """
        Send an activation tensor to the target miner.
        
        1. Serialize and compress
        2. Split into chunks if needed
        3. Send via registered transport
        4. Return metadata with transfer stats
        """
        if not self._send_fn:
            raise RuntimeError("No transport registered — call register_transport() first")

        # Auto-select compression if not specified
        if compression is None:
            est_size = 1
            for dim in metadata.shape:
                est_size *= dim
            est_size *= 2  # fp16
            compression = TensorSerializer.choose_compression(est_size, metadata.direction)

        # Serialize
        data, metadata = TensorSerializer.serialize(tensor, compression)
        metadata.pipeline_group_id = self.pipeline_group_id

        # Calculate chunks
        num_chunks = max(1, (len(data) + MAX_MESSAGE_BYTES - 1) // MAX_MESSAGE_BYTES)
        metadata.num_chunks = num_chunks

        # Send metadata header first
        metadata.transfer_start = time.time()
        header_msg = {
            "type": "activation_header",
            "metadata": metadata.to_dict(),
        }
        await self._send_fn(metadata.target_miner_id, header_msg)

        # Send data chunks
        for i in range(num_chunks):
            start = i * MAX_MESSAGE_BYTES
            end = min(start + MAX_MESSAGE_BYTES, len(data))
            chunk = ActivationChunk(
                transfer_id=metadata.transfer_id,
                chunk_index=i,
                total_chunks=num_chunks,
                data=data[start:end],
                is_last=(i == num_chunks - 1),
            )
            await self._send_fn(metadata.target_miner_id, chunk.to_message())

        metadata.transfer_end = time.time()

        # Update stats
        self.total_bytes_sent += metadata.compressed_size_bytes
        self.total_transfers += 1
        self.total_transfer_time += metadata.transfer_time
        self.pending_transfers[metadata.transfer_id] = metadata

        logger.debug(
            f"Sent activation {metadata.transfer_id}: "
            f"{metadata.direction.value} stage {metadata.source_stage}→{metadata.target_stage}, "
            f"mb={metadata.microbatch_index}, "
            f"{metadata.original_size_bytes / 1e6:.1f}MB→{metadata.compressed_size_bytes / 1e6:.1f}MB "
            f"({metadata.compression.value}, {metadata.compression_ratio:.1f}x), "
            f"{metadata.transfer_time * 1000:.0f}ms"
        )

        return metadata

    async def receive_header(self, msg: Dict[str, Any]) -> ActivationMetadata:
        """Process an incoming activation header. Prepares to receive chunks."""
        meta_dict = msg["metadata"]
        metadata = ActivationMetadata(
            transfer_id=meta_dict["transfer_id"],
            execution_id=meta_dict.get("execution_id", ""),
            pipeline_group_id=meta_dict.get("pipeline_group_id", ""),
            direction=TransferDirection(meta_dict["direction"]),
            microbatch_index=meta_dict["microbatch_index"],
            source_stage=meta_dict["source_stage"],
            target_stage=meta_dict["target_stage"],
            source_miner_id=meta_dict.get("source_miner_id", ""),
            target_miner_id=meta_dict.get("target_miner_id", ""),
            shape=meta_dict["shape"],
            dtype=meta_dict["dtype"],
            original_size_bytes=meta_dict["original_size_bytes"],
            compressed_size_bytes=meta_dict["compressed_size_bytes"],
            compression=CompressionMethod(meta_dict["compression"]),
            num_chunks=meta_dict["num_chunks"],
            checksum=meta_dict.get("checksum", ""),
        )
        metadata.transfer_start = time.time()
        self.pending_transfers[metadata.transfer_id] = metadata
        self.chunk_buffers[metadata.transfer_id] = {}
        return metadata

    async def receive_chunk(self, msg: Dict[str, Any]) -> Optional[Tuple[Any, ActivationMetadata]]:
        """
        Process an incoming activation chunk.
        
        Returns (tensor, metadata) when all chunks received, None otherwise.
        """
        chunk = ActivationChunk.from_message(msg)
        transfer_id = chunk.transfer_id

        if transfer_id not in self.chunk_buffers:
            self.chunk_buffers[transfer_id] = {}

        self.chunk_buffers[transfer_id][chunk.chunk_index] = chunk.data

        # Check if all chunks received
        metadata = self.pending_transfers.get(transfer_id)
        if not metadata:
            logger.warning(f"Received chunk for unknown transfer {transfer_id}")
            return None

        if len(self.chunk_buffers[transfer_id]) < metadata.num_chunks:
            return None  # Still waiting for more chunks

        # All chunks received — reassemble
        all_data = b""
        for i in range(metadata.num_chunks):
            all_data += self.chunk_buffers[transfer_id][i]

        metadata.transfer_end = time.time()

        # Verify checksum if no lossy compression
        if metadata.compression not in (CompressionMethod.INT8, CompressionMethod.INT8_ZLIB, CompressionMethod.TOP_K):
            actual_checksum = hashlib.sha256(all_data).hexdigest()[:16]
            if metadata.checksum and actual_checksum != metadata.checksum:
                logger.error(
                    f"Checksum mismatch for transfer {transfer_id}! "
                    f"Expected {metadata.checksum}, got {actual_checksum}"
                )

        # Deserialize
        tensor = TensorSerializer.deserialize(all_data, metadata)

        # Cleanup
        del self.chunk_buffers[transfer_id]
        del self.pending_transfers[transfer_id]
        self.completed_transfers.append(metadata)
        self.total_bytes_received += metadata.compressed_size_bytes

        logger.debug(
            f"Received activation {transfer_id}: "
            f"{metadata.direction.value} stage {metadata.source_stage}→{metadata.target_stage}, "
            f"mb={metadata.microbatch_index}, "
            f"{metadata.transfer_time * 1000:.0f}ms, {metadata.bandwidth_mbps:.0f} Mbps"
        )

        return tensor, metadata

    def get_stats(self) -> Dict[str, Any]:
        """Router statistics."""
        avg_bw = 0
        if self.completed_transfers:
            bws = [t.bandwidth_mbps for t in self.completed_transfers if t.bandwidth_mbps > 0]
            avg_bw = sum(bws) / len(bws) if bws else 0

        return {
            "pipeline_group_id": self.pipeline_group_id,
            "total_transfers": self.total_transfers,
            "pending_transfers": len(self.pending_transfers),
            "total_bytes_sent_mb": round(self.total_bytes_sent / 1e6, 1),
            "total_bytes_received_mb": round(self.total_bytes_received / 1e6, 1),
            "total_transfer_time_sec": round(self.total_transfer_time, 2),
            "avg_bandwidth_mbps": round(avg_bw, 1),
            "completed_transfers": len(self.completed_transfers),
        }


# ══════════════════════════════════════════════════════════════
# HELPER: Estimate activation sizes for planning
# ══════════════════════════════════════════════════════════════

def estimate_activation_size(
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    dtype_bytes: int = 2,
) -> Dict[str, Any]:
    """
    Estimate activation tensor size between pipeline stages.
    
    Returns size info dict for capacity planning.
    
    Examples:
      1B model:   batch=8, seq=4096, hidden=2048  → 128 MB
      7B model:   batch=4, seq=8192, hidden=4096  → 256 MB
      70B model:  batch=2, seq=16384, hidden=8192 → 512 MB
      405B model: batch=1, seq=32768, hidden=16384 → 1.07 GB
      Future 10T: batch=1, seq=65536, hidden=32768 → 4.29 GB
    """
    raw_size = batch_size * seq_length * hidden_size * dtype_bytes
    int8_size = batch_size * seq_length * hidden_size * 1 + 4  # +4 for scale
    int8_zlib_size = int(int8_size * 0.7)  # ~30% further compression typical

    return {
        "batch_size": batch_size,
        "seq_length": seq_length,
        "hidden_size": hidden_size,
        "raw_size_bytes": raw_size,
        "raw_size_mb": round(raw_size / 1e6, 1),
        "raw_size_gb": round(raw_size / 1e9, 3),
        "int8_size_mb": round(int8_size / 1e6, 1),
        "int8_zlib_size_mb": round(int8_zlib_size / 1e6, 1),
        "transfer_time_1gbps_sec": round(int8_size * 8 / 1e9, 2),
        "transfer_time_10gbps_sec": round(int8_size * 8 / 10e9, 3),
    }
