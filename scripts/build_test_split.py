"""Build a source-independent test split for early stopping on E[L].

`train/dataloader.py` derives its splits from `train_data_ratio` over ONE corpus,
and `train/data.py:181-184` makes that split a *contiguous positional slice*:

    split_idx = int(len(data) * train_ratio)
    train = rows [0, split_idx)      val = rows [split_idx, len(data))

So the val set is whatever sits at the END of the corpus.  Our current corpus is
[5000 ShareGPT][1258 GSM8K], so the last 626 rows -- the "val" split -- are GSM8K,
and GSM8K also appears inside the training portion.  That val set is held out but
NOT source-independent.

This script appends rows from a different corpus (`mlabonne/open-perfectblend`,
a math/code/general blend) so the val slice is a genuinely different source, then
solves for the exact `train_data_ratio` that puts the boundary at the end of the
original training rows rather than somewhere inside them.

    python scripts/build_test_split.py --test-rows 700
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
TEST_SOURCE = "mlabonne/open-perfectblend"


def build_test_rows(n_want, seq_length, min_masked, tokenizer):
    from datasets import load_dataset
    ds = load_dataset(TEST_SOURCE, split="train", streaming=True)
    rows, n, dropped = [], 0, 0
    t0 = time.time()
    for row in ds:
        if n >= n_want:
            break
        msgs = to_messages(row.get("conversations") or [])
        if not msgs or not any(m["role"] == "assistant" for m in msgs):
            dropped += 1
            continue
        try:
            ids, mask = render(tokenizer, msgs, enable_thinking=True)
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
    print(f"[test] {n} test rows from {TEST_SOURCE} (dropped {dropped}) "
          f"in {time.time()-t0:.0f}s")
    return rows


def solve_ratio(n_train, n_total):
    """Smallest-deviation ratio whose positional cut lands exactly at n_train."""
    exact = n_train / n_total
    for delta in [i / 1e6 for i in range(0, 2000)]:
        for cand in (exact + delta, exact - delta):
            if 0.0 < cand < 1.0 and int(n_total * cand) == n_train:
                return cand
    raise SystemExit(f"no ratio near {exact} lands the cut at {n_train}/{n_total}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default=os.path.join(ROOT, "data", "mixed_spec.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "mixed2_spec.jsonl"))
    ap.add_argument("--test-jsonl", default=os.path.join(ROOT, "data", "test_spec.jsonl"))
    ap.add_argument("--test-rows", type=int, default=700)
    ap.add_argument("--seq-length", type=int, default=4096)
    ap.add_argument("--min-masked", type=int, default=32)
    ap.add_argument("--tokenizer", default=TARGET_06B)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    with open(a.train_jsonl, encoding="utf-8") as f:
        train_lines = [l for l in f if l.strip()]
    n_train = len(train_lines)
    print(f"[test] {n_train} existing training rows from {a.train_jsonl}")

    test_rows = build_test_rows(a.test_rows, a.seq_length, a.min_masked, tok)
    with open(a.test_jsonl, "w", encoding="utf-8") as f:
        for r in test_rows:
            f.write(json.dumps(r) + "\n")

    with open(a.out, "w", encoding="utf-8") as f:
        for l in train_lines:
            f.write(l if l.endswith("\n") else l + "\n")
        for r in test_rows:
            f.write(json.dumps(r) + "\n")

    n_total = n_train + len(test_rows)
    ratio = solve_ratio(n_train, n_total)
    print(f"[test] corpus {a.out}: {n_total} rows "
          f"(train {n_train} + independent test {len(test_rows)})")
    print(f"[test] train_data_ratio = {ratio!r}  -> "
          f"split_idx = int({n_total} * {ratio!r}) = {int(n_total*ratio)}")
    print(f"[test] use:  --train-data-ratio {ratio!r}")
    with open(os.path.join(ROOT, "data", "test_ratio.json"), "w", encoding="utf-8") as f:
        json.dump({"n_train": n_train, "n_test": len(test_rows),
                   "n_total": n_total, "train_data_ratio": ratio,
                   "test_source": TEST_SOURCE}, f, indent=1)


if __name__ == "__main__":
    main()
