"""Validate generated hidden states with the trainer's own checker.

`speculators.data_generation.offline.check_hidden_states` is the authority on
the format: it requires `token_ids` and `hidden_states` keys, requires
`token_ids` to equal the preprocessed row exactly, requires
`len(tokens) == hidden_states.shape[0]`, and rejects NaN/Inf.  Running it here
means a format mistake surfaces now rather than hours into a training run.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "mix_prep"))
    ap.add_argument("--hs", default=os.path.join(ROOT, "data", "hs_smoke"))
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--expect-taps", type=int, default=4,
                    help="tap axis must be len(aux taps) + 1. `train/data.py` splits "
                         "[:, :-1] into the `fc` features and [:, -1] into "
                         "verifier_last_hidden_states, so a file with only the aux "
                         "taps yields (n-1)*H features and a shape error in `fc`. "
                         "check_hidden_states does NOT check this.")
    a = ap.parse_args()

    from datasets import load_from_disk
    from safetensors.torch import load_file
    from speculators.data_generation.offline import check_hidden_states

    ds = load_from_disk(a.data)
    n_ok = 0
    for i in range(min(a.limit, len(ds))):
        path = os.path.join(a.hs, "hs_%d.safetensors" % i)
        if not os.path.exists(path):
            print("[val] missing %s" % path)
            continue
        d = load_file(path)
        toks = ds[i]["input_ids"].tolist()
        check_hidden_states(d, toks)
        hs = d["hidden_states"]
        if a.expect_taps and hs.shape[1] != a.expect_taps:
            raise ValueError(
                f"tap axis is {hs.shape[1]}, expected {a.expect_taps} "
                f"(= len(aux taps) + 1 for verifier_last_hidden_states). "
                f"check_hidden_states would NOT have caught this.")
        print("[val] row %d OK  hidden=%s dtype=%s taps=%d tokens=%d"
              % (i, tuple(hs.shape), hs.dtype, hs.shape[1], len(toks)))
        n_ok += 1
    print("[val] %d/%d rows validated against check_hidden_states"
          % (n_ok, min(a.limit, len(ds))))


if __name__ == "__main__":
    main()
