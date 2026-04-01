"""
GRADIENT COMPRESSOR
===================

Top-K sparsification with error feedback buffer for distributed LLM training.
This is the REAL gradient compression — not torch.std() pseudocode.

Algorithm:
1. Accumulate local gradient into error feedback buffer
2. Select Top-K largest magnitude values from the buffer
3. Send only those K (index, value) pairs over the network
4. Subtract sent values from the buffer (error feedback)
5. The unsent residual carries forward to the next round

This achieves 100-1000x compression while preserving convergence,
proven by: Aji & Heafield 2017, Stich et al. 2018, Karimireddy et al. 2019.

STATUS: PRODUCTION IMPLEMENTATION
UPDATED: 2026-04-01
PURPOSE: Gradient compression for distributed miner training
"""

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import torch — graceful fallback if not available
try:
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available — gradient compressor will use pure-Python fallback")


@dataclass
class CompressedGradient:
    """A compressed gradient: only the Top-K values and their indices."""
    indices: List[int]
    values: List[float]
    original_size: int
    k: int
    gradient_hash: str
    layer_name: str = ""

    @property
    def compressed_size(self) -> int:
        return len(self.indices)

    @property
    def compression_ratio(self) -> float:
        if self.compressed_size == 0:
            return 0.0
        return self.original_size / self.compressed_size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "indices": self.indices,
            "values": self.values,
            "original_size": self.original_size,
            "k": self.k,
            "gradient_hash": self.gradient_hash,
            "layer_name": self.layer_name,
            "compressed_size": self.compressed_size,
            "compression_ratio": self.compression_ratio,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompressedGradient":
        return cls(
            indices=data["indices"],
            values=data["values"],
            original_size=data["original_size"],
            k=data["k"],
            gradient_hash=data["gradient_hash"],
            layer_name=data.get("layer_name", ""),
        )


class GradientCompressor:
    """
    Top-K gradient compressor with error feedback buffer.
    
    Each miner maintains one compressor per model layer.
    The error feedback buffer ensures that small but important
    gradient updates are not permanently lost — they accumulate
    across rounds until they become large enough to be in the Top-K.
    """

    def __init__(self, compression_ratio: float = 0.01):
        """
        Args:
            compression_ratio: Fraction of gradient to keep (0.01 = Top 1% = 100x compression)
        """
        self.compression_ratio = compression_ratio
        self._error_buffers: Dict[str, Any] = {}

    def compress(self, gradient: Any, layer_name: str = "default") -> CompressedGradient:
        """
        Compress a gradient tensor using Top-K with error feedback.
        
        Args:
            gradient: Gradient tensor (torch.Tensor or list of floats)
            layer_name: Name of the model layer (for error buffer tracking)
            
        Returns:
            CompressedGradient with Top-K indices and values
        """
        if HAS_TORCH and torch.is_tensor(gradient):
            return self._compress_torch(gradient, layer_name)
        else:
            return self._compress_python(gradient, layer_name)

    def decompress(self, compressed: CompressedGradient) -> Any:
        """
        Decompress a gradient back to full size (sparse → dense).
        
        Args:
            compressed: CompressedGradient to decompress
            
        Returns:
            Full-size gradient (torch.Tensor or list)
        """
        if HAS_TORCH:
            return self._decompress_torch(compressed)
        else:
            return self._decompress_python(compressed)

    def _compress_torch(self, gradient: "torch.Tensor", layer_name: str) -> CompressedGradient:
        """Top-K compression using PyTorch."""
        flat = gradient.flatten().float()
        n = flat.numel()
        k = max(1, int(n * self.compression_ratio))

        # Add error feedback buffer
        if layer_name in self._error_buffers:
            flat = flat + self._error_buffers[layer_name]

        # Top-K by absolute magnitude
        abs_vals = flat.abs()
        topk_vals, topk_indices = torch.topk(abs_vals, k)

        # Get actual (signed) values at those indices
        selected_values = flat[topk_indices]

        # Update error feedback: subtract what we sent
        error = flat.clone()
        error[topk_indices] = 0.0
        self._error_buffers[layer_name] = error

        # Hash for integrity verification
        hash_input = json.dumps({
            "indices": topk_indices.tolist(),
            "values": selected_values.tolist(),
        }, sort_keys=True).encode()
        gradient_hash = hashlib.sha256(hash_input).hexdigest()

        return CompressedGradient(
            indices=topk_indices.tolist(),
            values=selected_values.tolist(),
            original_size=n,
            k=k,
            gradient_hash=gradient_hash,
            layer_name=layer_name,
        )

    def _compress_python(self, gradient: List[float], layer_name: str) -> CompressedGradient:
        """Top-K compression using pure Python (fallback)."""
        flat = list(gradient) if not isinstance(gradient, list) else gradient
        n = len(flat)
        k = max(1, int(n * self.compression_ratio))

        # Add error feedback buffer
        if layer_name in self._error_buffers:
            buf = self._error_buffers[layer_name]
            flat = [flat[i] + buf[i] for i in range(n)]

        # Top-K by absolute magnitude
        indexed = [(i, flat[i], abs(flat[i])) for i in range(n)]
        indexed.sort(key=lambda x: x[2], reverse=True)
        top_k = indexed[:k]

        indices = [x[0] for x in top_k]
        values = [x[1] for x in top_k]

        # Update error feedback
        error = list(flat)
        for idx in indices:
            error[idx] = 0.0
        self._error_buffers[layer_name] = error

        # Hash for integrity
        hash_input = json.dumps({
            "indices": indices,
            "values": values,
        }, sort_keys=True).encode()
        gradient_hash = hashlib.sha256(hash_input).hexdigest()

        return CompressedGradient(
            indices=indices,
            values=values,
            original_size=n,
            k=k,
            gradient_hash=gradient_hash,
            layer_name=layer_name,
        )

    def _decompress_torch(self, compressed: CompressedGradient) -> "torch.Tensor":
        """Decompress to a dense torch.Tensor."""
        result = torch.zeros(compressed.original_size)
        indices = torch.tensor(compressed.indices, dtype=torch.long)
        values = torch.tensor(compressed.values, dtype=torch.float32)
        result[indices] = values
        return result

    def _decompress_python(self, compressed: CompressedGradient) -> List[float]:
        """Decompress to a dense list."""
        result = [0.0] * compressed.original_size
        for idx, val in zip(compressed.indices, compressed.values):
            result[idx] = val
        return result

    def reset_buffers(self):
        """Reset all error feedback buffers (e.g., at epoch boundary)."""
        self._error_buffers.clear()
        logger.info("Gradient compressor error buffers reset")

    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics about error feedback buffers."""
        stats = {}
        for layer_name, buf in self._error_buffers.items():
            if HAS_TORCH and torch.is_tensor(buf):
                stats[layer_name] = {
                    "size": buf.numel(),
                    "norm": buf.norm().item(),
                    "max_abs": buf.abs().max().item(),
                    "nonzero_ratio": (buf != 0).float().mean().item(),
                }
            else:
                norm = math.sqrt(sum(v * v for v in buf))
                max_abs = max(abs(v) for v in buf) if buf else 0
                nonzero = sum(1 for v in buf if v != 0) / len(buf) if buf else 0
                stats[layer_name] = {
                    "size": len(buf),
                    "norm": norm,
                    "max_abs": max_abs,
                    "nonzero_ratio": nonzero,
                }
        return stats


def verify_gradient_hash(compressed: CompressedGradient) -> bool:
    """Verify the integrity of a compressed gradient."""
    hash_input = json.dumps({
        "indices": compressed.indices,
        "values": compressed.values,
    }, sort_keys=True).encode()
    expected = hashlib.sha256(hash_input).hexdigest()
    return expected == compressed.gradient_hash


# Global compressor instance (default 1% = 100x compression)
gradient_compressor = GradientCompressor(compression_ratio=0.01)
