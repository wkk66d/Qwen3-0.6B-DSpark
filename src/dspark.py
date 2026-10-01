"""DSpark draft model (block-parallel backbone + sequential Markov head).

Reimplemented for this experiment from the published DSpark semantics
(deepseek-ai/dspark_qwen3_4b_block7 config + the vLLM `qwen3_dspark` reference
architecture).  Only the pieces needed for single-stream inference are built.

Key semantics
-------------
* Backbone: 5 Qwen3 decoder layers with *non-causal* attention over the query
  block, attending to (a) a "context" of projected target hidden states and
  (b) the query block itself.
* Context KV is projected, k-normed and RoPE'd at the context's *absolute*
  positions, then stored.  Query K/V are RoPE'd at the query's absolute
  positions.
* DSpark is next-token indexed: the hidden state at query slot i predicts the
  token at absolute position (anchor_pos + 1 + i).  With `sample_from_anchor`
  the query block is  [anchor, mask * (K-1)]  and yields K drafts.
* Markov head: base_logits[i] + markov_w2(markov_w1(prev_token)) sampled left to
  right, so draft k+1 is conditioned on draft k.
* Confidence head: per-slot acceptance probability estimate.

The drafter's `embed_tokens` / `lm_head` are frozen copies of the target's
(verified bit-identical), so they are not loaded here -- the caller passes the
target's embedding weight in.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # torch>=2.4
    from safetensors.torch import load_file as _sf_load
except Exception:  # pragma: no cover
    _sf_load = None


# --------------------------------------------------------------------------- #
# RoPE (Qwen3 / HF "default" non-interleaved rotary)
# --------------------------------------------------------------------------- #
def build_rope_cache(head_dim: int, base: float, positions: torch.Tensor, dtype):
    """positions: [T] long -> cos/sin of shape [T, head_dim]."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device) / head_dim)
    )
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, D]; cos/sin: [T, D]."""
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return x * cos + rotate_half(x) * sin


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #
class DSparkMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DSparkRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.variance_epsilon = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dt) * self.weight)


class DSparkAttention(nn.Module):
    def __init__(self, cfg: dict, layer_idx: int):
        super().__init__()
        h = cfg["hidden_size"]
        nh = cfg["num_attention_heads"]
        nkv = cfg["num_key_value_heads"]
        hd = cfg["head_dim"]
        self.layer_idx = layer_idx
        self.num_heads = nh
        self.num_kv_heads = nkv
        self.head_dim = hd
        self.num_kv_groups = nh // nkv
        self.scaling = hd ** -0.5
        self.q_proj = nn.Linear(h, nh * hd, bias=False)
        self.k_proj = nn.Linear(h, nkv * hd, bias=False)
        self.v_proj = nn.Linear(h, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nh * hd, h, bias=False)
        self.q_norm = DSparkRMSNorm(hd, eps=cfg["rms_norm_eps"])
        self.k_norm = DSparkRMSNorm(hd, eps=cfg["rms_norm_eps"])

    # --- context path: only K/V are needed (no queries) ---
    def context_kv(self, ctx: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """ctx: [1, L, h] projected context -> k, v of [1, nkv, L, hd]."""
        b, L, _ = ctx.shape
        k = self.k_proj(ctx).view(b, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(ctx).view(b, L, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        k = apply_rope(k, cos, sin)
        return k, v

    # --- query path ---
    def query(self, q_in: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        b, T, _ = q_in.shape
        q = self.q_proj(q_in).view(b, T, self.num_heads, self.head_dim)
        k = self.k_proj(q_in).view(b, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(q_in).view(b, T, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        return q, k, v


class DSparkDecoderLayer(nn.Module):
    def __init__(self, cfg: dict, layer_idx: int):
        super().__init__()
        self.self_attn = DSparkAttention(cfg, layer_idx)
        self.mlp = DSparkMLP(cfg["hidden_size"], cfg["intermediate_size"])
        self.input_layernorm = DSparkRMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.post_attention_layernorm = DSparkRMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])

    def forward(self, hidden, q_cos, q_sin, ctx_k, ctx_v, causal=False):
        """hidden: [1, T, h] query states. ctx_k/ctx_v: [1, nkv, L, hd] (already complete)."""
        residual = hidden
        x = self.input_layernorm(hidden)
        q, k, v = self.self_attn.query(x, q_cos, q_sin)
        if ctx_k is not None and ctx_k.shape[2] > 0:
            k_all = torch.cat([ctx_k, k], dim=2)
            v_all = torch.cat([ctx_v, v], dim=2)
        else:
            k_all, v_all = k, v
        if self.self_attn.num_kv_groups > 1:
            k_all = k_all.repeat_interleave(self.self_attn.num_kv_groups, dim=1)
            v_all = v_all.repeat_interleave(self.self_attn.num_kv_groups, dim=1)
        out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=None,
                                             dropout_p=0.0, is_causal=False,
                                             scale=self.self_attn.scaling)
        out = out.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], -1)
        out = self.self_attn.o_proj(out)
        hidden = residual + out
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        return residual + hidden


class VanillaMarkovHead(nn.Module):
    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def embed(self, token_ids):
        return self.markov_w1(token_ids.long())

    def bias(self, token_ids):
        return self.markov_w2(self.markov_w1(token_ids.long()))


class ConfidenceHead(nn.Module):
    """Mirror of speculators.models.dspark.model_definitions.ConfidenceHead.

    A bare Linear(input_dim, 1) is enough: measured on our trained drafter the
    head outputs a cleanly graded per-slot profile (slot means fall from ~0.64 to
    ~0.27, monotone decreasing, spread 0.49 versus a true spread of 0.39). An
    earlier "the head is flat" reading came from a probe bug -- conf_seen holds
    block_size values per round, not K=7 -- and is retracted.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        return self.proj(x).squeeze(-1)


# --------------------------------------------------------------------------- #
# Full drafter
# --------------------------------------------------------------------------- #
class DSparkDrafter(nn.Module):
    """Loads a `Qwen3DSparkModel` checkpoint; shares embed/lm_head with the target."""

    def __init__(self, config_path: str):
        super().__init__()
        with open(config_path, encoding="utf-8") as f:
            self.cfg = json.load(f)
        cfg = self.cfg
        self.vocab_size = cfg["vocab_size"]
        self.hidden_size = cfg["hidden_size"]
        self.block_size = int(cfg["block_size"])
        self.mask_token_id = int(cfg["mask_token_id"])
        self.target_layer_ids = list(cfg["target_layer_ids"])
        self.markov_rank = int(cfg.get("markov_rank", 0))
        self.markov_head_type = str(cfg.get("markov_head_type", "vanilla"))
        self.rope_theta = float(cfg.get("rope_parameters", {}).get("rope_theta", 1e6))
        self.head_dim = int(cfg["head_dim"])
        self.num_layers = int(cfg["num_hidden_layers"])

        self.layers = nn.ModuleList([DSparkDecoderLayer(cfg, i) for i in range(self.num_layers)])
        self.norm = DSparkRMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.fc = nn.Linear(len(self.target_layer_ids) * cfg["hidden_size"], cfg["hidden_size"], bias=False)
        self.hidden_norm = DSparkRMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.markov_head = VanillaMarkovHead(cfg["vocab_size"], self.markov_rank) if self.markov_rank > 0 else None
        self.confidence_head = None
        self.confidence_head_with_markov = bool(cfg.get("confidence_head_with_markov", False))
        if cfg.get("enable_confidence_head", False):
            dim = cfg["hidden_size"] + (self.markov_rank if self.confidence_head_with_markov else 0)
            self.confidence_head = ConfidenceHead(dim)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(cls, model_dir: str, dtype=torch.bfloat16, device="cuda"):
        cfg_path = os.path.join(model_dir, "config.json")
        model = cls(cfg_path)
        sd_path = os.path.join(model_dir, "model.safetensors")
        sd = _sf_load(sd_path, device="cpu")
        model._load_ckpt(sd)
        del sd
        model = model.to(device=device, dtype=dtype).eval()
        # the confidence head is fp32 in the checkpoint (and in vLLM); keep it fp32
        if model.confidence_head is not None:
            model.confidence_head = model.confidence_head.float()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def _load_ckpt(self, sd: dict):
        own = dict(self.state_dict())
        mapped = {}
        skipped = []
        for k, v in sd.items():
            if k in ("embed_tokens.weight", "lm_head.weight"):
                skipped.append(k)
                continue
            if k in own:
                if tuple(own[k].shape) != tuple(v.shape):
                    raise RuntimeError(f"shape mismatch for {k}: ckpt {tuple(v.shape)} vs model {tuple(own[k].shape)}")
                mapped[k] = v
            else:
                skipped.append(k)
        missing = [k for k in own if k not in mapped]
        # The per-slot confidence bias postdates the original checkpoints and
        # defaults to zeros, so its absence is not an error -- it just reproduces
        # the pre-patch behaviour exactly.
        optional = {"confidence_head.slot"}
        hard_missing = [k for k in missing if k not in optional]
        if hard_missing:
            raise RuntimeError(f"missing drafter weights: {hard_missing[:8]}")
        for k in missing:
            mapped[k] = own[k]
        self.load_state_dict(mapped, strict=True)
        self.skipped_ckpt_keys = skipped

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def project_context(self, hidden_list):
        """hidden_list: list of [1, L, h] target hidden states (one per target layer id)."""
        x = torch.cat(hidden_list, dim=-1)
        return self.hidden_norm(self.fc(x))

    @torch.inference_mode()
    def context_kv(self, ctx_proj: torch.Tensor, start_pos: int):
        """Returns per-layer (k, v) for a chunk of projected context."""
        L = ctx_proj.shape[1]
        positions = torch.arange(start_pos, start_pos + L, device=ctx_proj.device)
        cos, sin = build_rope_cache(self.head_dim, self.rope_theta, positions, ctx_proj.dtype)
        out = []
        for layer in self.layers:
            out.append(layer.self_attn.context_kv(ctx_proj, cos, sin))
        return out

    @torch.inference_mode()
    def draft(self, anchor_id: int, anchor_pos: int, K: int, kv_ctx_list, embed_weight,
              ban_token: Optional[int] = None):
        """Produce K draft tokens (and their Markov-corrected logits).

        anchor_id : token at absolute position `anchor_pos`
        K         : number of draft tokens (>=1).  Query block = [anchor, mask*(K-1)].
        kv_ctx_list : per-layer (k, v) context caches covering positions [0, anchor_pos)
        ban_token : if set, that id is removed from every draft position's distribution.
                    Used so an *unverified* draft cannot close the reasoning phase --
                    otherwise the lossy arm degenerates into "no CoT" instead of
                    "lossy CoT".
        """
        assert 1 <= K <= self.block_size, f"K={K} outside [1, {self.block_size}]"
        device = embed_weight.device
        q_ids = torch.full((1, K), self.mask_token_id, dtype=torch.long, device=device)
        q_ids[0, 0] = int(anchor_id)
        q_pos = torch.arange(anchor_pos, anchor_pos + K, device=device)
        noise = F.embedding(q_ids, embed_weight)                       # [1, K, h]
        q_cos, q_sin = build_rope_cache(self.head_dim, self.rope_theta, q_pos, noise.dtype)

        hidden = noise
        for i, layer in enumerate(self.layers):
            k_ctx, v_ctx = kv_ctx_list[i]
            hidden = layer(hidden, q_cos, q_sin, k_ctx, v_ctx)
        hidden = self.norm(hidden)

        base_logits = F.linear(hidden, embed_weight)                   # tied lm_head
        # ---- sequential Markov sweep ----
        prev = torch.tensor([int(anchor_id)], dtype=torch.long, device=device)
        step_logits = []
        drafts = []
        for i in range(K):
            lg = base_logits[:, i, :]
            if self.markov_head is not None:
                lg = lg + self.markov_head.bias(prev)
            if ban_token is not None:
                lg = lg.clone()
                lg[:, int(ban_token)] = float("-inf")
            step_logits.append(lg)
            nxt = torch.argmax(lg, dim=-1)
            drafts.append(nxt)
            prev = nxt
        draft_ids = torch.stack(drafts, dim=1)                          # [1, K]
        corrected = torch.stack(step_logits, dim=1)                     # [1, K, V]
        return draft_ids, corrected, hidden

    @torch.inference_mode()
    def confidence(self, hidden: torch.Tensor, prev_ids: torch.Tensor):
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            emb = self.markov_head.embed(prev_ids).to(hidden.dtype)
            x = torch.cat([hidden, emb], dim=-1)
        else:
            x = hidden
        return torch.sigmoid(self.confidence_head(x.float())).float()
