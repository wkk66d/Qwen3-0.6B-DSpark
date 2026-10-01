"""Preallocated in-place KV cache.

transformers' `DynamicCache` calls `torch.cat` on every step, allocating a fresh
tensor of the full cache length for each of the 36 layers, twice per layer.  On a
12 GB card the caching allocator ends up *reserving* far more than the physical
VRAM (measured: ~2.8 MB reserved per cached token, i.e. 19x the real KV size),
after which WDDM silently pages GPU memory to host RAM and decode throughput
collapses ~6x (12.1 -> 1.9 tok/s at ~3.2k context).

This cache writes K/V into fixed buffers and hands attention *views* of them, so
steady-state decoding allocates nothing.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


class InplaceKVCache(DynamicCache):
    """Dynamic-semantics cache backed by fixed preallocated buffers."""

    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int,
                 max_len: int, dtype=torch.bfloat16, device: str = "cuda"):
        super().__init__()
        self.num_layers = int(num_layers)
        self.max_len = int(max_len)
        self.layers = [DynamicLayer() for _ in range(self.num_layers)]
        self.buf_k = [torch.zeros(1, num_kv_heads, self.max_len, head_dim,
                                  dtype=dtype, device=device) for _ in range(self.num_layers)]
        self.buf_v = [torch.zeros(1, num_kv_heads, self.max_len, head_dim,
                                  dtype=dtype, device=device) for _ in range(self.num_layers)]
        self._start = 0

    # ------------------------------------------------------------------ #
    def begin(self, start_pos: int):
        """Tell the cache where the next forward's tokens start (avoids a sync)."""
        self._start = int(start_pos)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               layer_idx: int, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        start = self._start
        L = key_states.shape[2]
        end = start + L
        if end > self.max_len:
            raise RuntimeError(f"KV cache overflow: {end} > {self.max_len}")
        self.buf_k[layer_idx][:, :, start:end, :] = key_states
        self.buf_v[layer_idx][:, :, start:end, :] = value_states
        layer = self.layers[layer_idx]
        layer.keys = self.buf_k[layer_idx][:, :, :end, :]
        layer.values = self.buf_v[layer_idx][:, :, :end, :]
        layer.is_initialized = True
        layer.dtype = key_states.dtype
        layer.device = key_states.device
        return layer.keys, layer.values

    # crop() from DynamicCache slices the stored views; because those views point
    # into the buffers, the slice is another view -- no copy, no allocation.
    def reset(self):
        self._start = 0
        for layer in self.layers:
            layer.keys = torch.tensor([], dtype=torch.bfloat16, device=self.buf_k[0].device)
            layer.values = torch.tensor([], dtype=torch.bfloat16, device=self.buf_k[0].device)
            layer.is_initialized = False

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.buf_k + self.buf_v)
