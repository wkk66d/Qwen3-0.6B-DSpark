"""Convert ShareGPT (and optionally GSM8K) into speculator-format rows.

Emits `{"input_ids": [...], "loss_mask": [...]}` JSONL, which is the branch of
`speculators prepare-data` that needs no vLLM render endpoint.  The loss mask is
built turn by turn: for each assistant turn we render the prefix with
`add_generation_prompt=True` and mask exactly the tokens the assistant adds, so
multi-turn conversations get every assistant span masked and every user/system
span zeroed.

Sources come from speculators.data_generation.configs.DATASET_CONFIGS, so the
data matches what the official recipe would have used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
# This box sets NO_PROXY to a value containing "[::1]", which httpx (used by the
# `datasets` hub client, unlike huggingface_hub's downloader) fails to parse:
# "Invalid port: ':1]'".  Bypass lists are irrelevant here -- hf-mirror is
# reachable directly -- so drop them rather than let a malformed entry abort.
for _v in ("NO_PROXY", "no_proxy"):
    os.environ.pop(_v, None)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_06B = r"E:\Semantic\data\hf\Qwen3-0.6B"


def to_messages(conv):
    """ShareGPT `conversations` -> chat messages.

    In Aeala/ShareGPT_Vicuna_unfiltered each element is a JSON *string* like
    '{"from":"human","value":"..."}', not a dict; other mirrors use real dicts.
    """
    out = []
    for turn in conv:
        if isinstance(turn, str):
            try:
                turn = json.loads(turn)
            except Exception:
                continue
        if not isinstance(turn, dict):
            continue
        who = (turn.get("from") or turn.get("role") or "").lower()
        val = turn.get("value") or turn.get("content") or ""
        if not isinstance(val, str) or not val:
            continue
        if who in ("human", "user"):
            out.append({"role": "user", "content": val})
        elif who in ("gpt", "assistant", "chatgpt"):
            out.append({"role": "assistant", "content": val})
        elif who == "system":
            out.append({"role": "system", "content": val})
    return out


def row_to_messages(row, cfg):
    """Dispatch on the dataset's shape rather than assuming ShareGPT."""
    if cfg.name == "gsm8k":
        q, a = row.get("question"), row.get("answer")
        if not q or not a:
            return []
        return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    conv = row.get("conversations") or row.get("conversation")
    return to_messages(conv) if conv else []


def render(tokenizer, msgs, enable_thinking):
    """Full token ids + mask, masking exactly the assistant-generated spans."""
    ids, mask = [], []
    pending = []
    for m in msgs:
        if m["role"] != "assistant":
            pending.append(m)
            continue
        # ShareGPT sometimes has consecutive assistant turns, or opens with one;
        # there is nothing for the template to attach a reply to, so skip it.
        if not any(p["role"] != "assistant" for p in pending):
            continue
        prompt = tokenizer.apply_chat_template(
            pending, tokenize=True, add_generation_prompt=True,
            enable_thinking=enable_thinking)
        if hasattr(prompt, "keys"):
            prompt = prompt["input_ids"]
        prompt = list(prompt)
        body = tokenizer(m["content"], add_special_tokens=False)["input_ids"]
        # drop what the prompt already covered, then append the masked turn
        common = 0
        for a, b in zip(ids, prompt):
            if a != b:
                break
            common += 1
        ids = prompt
        mask = mask[:common] + [0] * (len(prompt) - common)
        ids = ids + body
        mask = mask + [1] * len(body)
        pending = [m]
    # trailing non-assistant content (e.g. a final user turn) without a reply
    if pending and pending[-1]["role"] != "assistant":
        tail = tokenizer.apply_chat_template(
            pending, tokenize=True, add_generation_prompt=False,
            enable_thinking=enable_thinking)
        if hasattr(tail, "keys"):
            tail = tail["input_ids"]
        tail = list(tail)
        if len(tail) > len(ids):
            ids = ids + tail[len(ids):]
            mask = mask + [0] * (len(tail) - len(ids))
    return ids, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sharegpt")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seq-length", type=int, default=4096)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "sharegpt_spec.jsonl"))
    ap.add_argument("--tokenizer", default=TARGET_06B)
    ap.add_argument("--min-masked", type=int, default=32)
    a = ap.parse_args()

    from datasets import load_dataset
    from speculators.data_generation.configs import DATASET_CONFIGS
    from transformers import AutoTokenizer

    cfg = DATASET_CONFIGS[a.source]
    print(f"[build] {a.source} <- {cfg.hf_path} split={cfg.split}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    ds = load_dataset(cfg.hf_path, cfg.subset, split=cfg.split, streaming=True) \
        if cfg.subset else load_dataset(cfg.hf_path, split=cfg.split, streaming=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t0 = time.time()
    n = dropped = 0
    with open(a.out, "w", encoding="utf-8") as fh:
        for row in ds:
            if n >= a.n:
                break
            msgs = row_to_messages(row, cfg)
            if not msgs or not any(m["role"] == "assistant" for m in msgs):
                dropped += 1
                continue
            try:
                ids, mask = render(tok, msgs, enable_thinking=True)
            except Exception:
                dropped += 1
                continue
            if len(ids) > a.seq_length:
                ids, mask = ids[:a.seq_length], mask[:a.seq_length]
            if sum(mask) < a.min_masked:
                dropped += 1
                continue
            fh.write(json.dumps({"input_ids": ids, "loss_mask": mask}) + "\n")
            n += 1
            if n % 250 == 0:
                print(f"[build] {n} rows  {time.time()-t0:.0f}s", flush=True)
    print(f"[build] wrote {n} rows to {a.out} (dropped {dropped}) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
