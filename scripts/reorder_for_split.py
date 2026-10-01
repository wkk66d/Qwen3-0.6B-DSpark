"""Reorder a prepared corpus so the positional train/val split is source-clean.

`speculators prepare-data` does NOT preserve row order -- it is a pure permutation
of its input (verified: identical multisets, different order).  Since
`train/data.py:181-184` takes a *positional* slice,

    split_idx = int(len(data) * train_ratio)
    train = rows [0, split_idx)      val = rows [split_idx, len(data))

a shuffled corpus gives a random split.  That is held-out, but it is not
source-independent: in our run it put 629 of 700 intended test rows into TRAIN.

So we reorder the prepared Arrow dataset ourselves -- every training row first,
every independent-test row last -- and report the exact ratio that puts the cut
at the boundary.  Hidden states are indexed by row position, so they MUST be
regenerated after this; `hs_{i}` no longer lines up otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from datasets import load_from_disk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def key(ids):
    return tuple(int(x) for x in ids)


def solve_ratio(n_train, n_total):
    exact = n_train / n_total
    for i in range(0, 4000):
        for cand in (exact + i / 1e6, exact - i / 1e6):
            if 0.0 < cand < 1.0 and int(n_total * cand) == n_train:
                return cand
    raise SystemExit("no ratio lands the cut at %d/%d" % (n_train, n_total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", default=os.path.join(ROOT, "data", "mix2_prep2"))
    ap.add_argument("--test-jsonl", default=os.path.join(ROOT, "data", "test_spec.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "mix3_prep"))
    a = ap.parse_args()

    ds = load_from_disk(a.prepared)
    test_keys = set()
    for line in open(a.test_jsonl, encoding="utf-8-sig"):
        line = line.strip()
        if line:
            test_keys.add(key(json.loads(line)["input_ids"]))
    print(f"[reorder] {len(ds)} prepared rows, {len(test_keys)} test-row keys")

    test_idx, train_idx, dropped = [], [], 0
    seen_test = set()
    for i in range(len(ds)):
        k = key(ds[i]["input_ids"])
        if k in test_keys:
            if k in seen_test:
                # The same content occurs in both a training source and the test
                # source (open-perfectblend overlaps ShareGPT/ultrachat/GSM8K in a
                # handful of rows).  Keep the test copy and drop the duplicate
                # entirely rather than let it sit on both sides of the split.
                dropped += 1
                continue
            seen_test.add(k)
            test_idx.append(i)
        else:
            train_idx.append(i)
    print(f"[reorder] matched {len(test_idx)} unique test rows, {len(train_idx)} train rows"
          f"{f', dropped {dropped} train/test duplicates' if dropped else ''}")
    if len(test_idx) != len(test_keys):
        print(f"[reorder] WARNING: {len(test_idx)}/{len(test_keys)} test keys present")

    order = train_idx + test_idx
    ds2 = ds.select(order)
    ds2.save_to_disk(a.out)

    n_train, n_total = len(train_idx), len(order)
    ratio = solve_ratio(n_train, n_total)
    print(f"[reorder] wrote {a.out}: {n_total} rows "
          f"(train {n_train} then test {len(test_idx)})")
    print(f"[reorder] train_data_ratio = {ratio!r} -> split_idx = "
          f"{int(n_total * ratio)}")
    with open(os.path.join(ROOT, "data", "mix3_ratio.json"), "w", encoding="utf-8") as f:
        json.dump({"n_train": n_train, "n_test": len(test_idx), "n_total": n_total,
                   "train_data_ratio": ratio}, f, indent=1)

    # verify the boundary really is clean
    got = {key(ds2[i]["input_ids"]) for i in range(n_train, n_total)}
    leak = len(got & set(key(ds2[i]["input_ids"]) for i in range(n_train)))
    print(f"[reorder] VERIFY: val rows are test rows: {got == test_keys}; "
          f"overlap with train: {leak}")


if __name__ == "__main__":
    main()
