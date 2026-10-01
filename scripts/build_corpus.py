"""Build an expanded speculator training corpus + a source-disjoint test set.

Why: the first drafter trained on 6258 rows (5000 ShareGPT + 1258 GSM8K, 3.89M
tokens) and then plateaued the moment we continued on the *same content* -- the
model had already seen it ~4.5 times, so more epochs could not add information.
The fix is more data, not more epochs.

Layout produced here (matching `train/data.py`'s positional split):

    data/mixed3_spec.jsonl   train rows ++ test rows   (order is irrelevant --
                                                      prepare-data permutes it)
    data/test2_spec.jsonl    the held-out rows, used by reorder_for_split.py to
                             restore the boundary after prepare-data shuffles

The test set is drawn ONLY from `mlabonne/open-perfectblend`, which contributes
nothing to training, so the two are source-disjoint by construction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
for _v in ("NO_PROXY", "no_proxy"):
    os.environ.pop(_v, None)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from build_sharegpt_data import render, to_messages  # noqa: E402

TARGET_06B = r"E:\Semantic\data\hf\Qwen3-0.6B"


def messages_from(row, kind):
    """Normalise a row from any of the sources into chat messages."""
    if kind == "gsm8k":
        q, a = row.get("question"), row.get("answer")
        return [{"role": "user", "content": q}, {"role": "assistant", "content": a}] \
            if q and a else []
    if kind == "ultrachat":
        out = []
        for m in row.get("messages") or []:
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant", "system") and content:
                out.append({"role": role, "content": content})
        return out
    # sharegpt / open-perfectblend
    return to_messages(row.get("conversations") or row.get("conversation") or [])


def harvest(kind, hf_path, subset, split, want, tok, seq_length, min_masked):
    from datasets import load_dataset
    ds = load_dataset(hf_path, subset, split=split, streaming=True) if subset \
        else load_dataset(hf_path, split=split, streaming=True)
    rows, n, dropped = [], 0, 0
    t0 = time.time()
    for row in ds:
        if n >= want:
            break
        msgs = messages_from(row, kind)
        if not msgs or not any(m["role"] == "assistant" for m in msgs):
            dropped += 1
            continue
        try:
            ids, mask = render(tok, msgs, enable_thinking=True)
        except Exception:
            dropped += 1
            continue
        if len(ids) > seq_length:
            ids, mask = ids[:seq_length], mask[:seq_length]
        if sum(mask) < min_masked:
            dropped += 1
            continue
        rows.append({"input_ids": ids, "loss_mask": mask})
        n += 1
    print(f"[corpus] {kind:11s} {n:6d} rows kept, {dropped:6d} dropped "
          f"({time.time()-t0:.0f}s)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "mixed3_spec.jsonl"))
    ap.add_argument("--test-out", default=os.path.join(ROOT, "data", "test2_spec.jsonl"))
    ap.add_argument("--sharegpt", type=int, default=5000)
    ap.add_argument("--gsm8k", type=int, default=7000)
    ap.add_argument("--ultrachat", type=int, default=6000)
    ap.add_argument("--test-rows", type=int, default=1800)
    ap.add_argument("--seq-length", type=int, default=4096)
    ap.add_argument("--min-masked", type=int, default=32)
    ap.add_argument("--tokenizer", default=TARGET_06B)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    plan = [
        ("sharegpt",   "Aeala/ShareGPT_Vicuna_unfiltered", None, "train",   a.sharegpt),
        ("gsm8k",      "openai/gsm8k",                     "main", "train", a.gsm8k),
        ("ultrachat",  "HuggingFaceH4/ultrachat_200k",     None, "train_sft", a.ultrachat),
    ]
    train = []
    for kind, path, sub, split, want in plan:
        train += harvest(kind, path, sub, split, want, tok, a.seq_length, a.min_masked)

    # test set: a source that contributes NOTHING to training
    test = harvest("perfectblend", "mlabonne/open-perfectblend", None, "train",
                   a.test_rows, tok, a.seq_length, a.min_masked)

    with open(a.test_out, "w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r) + "\n")
    with open(a.out, "w", encoding="utf-8") as f:
        for r in train + test:
            f.write(json.dumps(r) + "\n")

    tk = sum(len(r["input_ids"]) for r in train)
    print(f"[corpus] train {len(train)} rows / {tk/1e6:.2f}M tokens")
    print(f"[corpus] test  {len(test)} rows (source-disjoint)")
    print(f"[corpus] wrote {a.out} ({len(train)+len(test)} rows) and {a.test_out}")
    print("[corpus] next: prepare-data, then reorder_for_split.py "
          f"--test-jsonl {a.test_out}")


if __name__ == "__main__":
    main()
