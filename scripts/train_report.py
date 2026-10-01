"""Summarise a DSpark drafter training run from its per-epoch val_metrics.json."""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ["accept_rate_epoch", "accept_len_epoch", "full_acc_epoch",
        "confidence_pred_mean_epoch", "confidence_abs_error_epoch", "loss_epoch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=os.path.join(ROOT, "output", "dspark_06b", "run1"))
    a = ap.parse_args()

    epochs = sorted(
        int(d) for d in os.listdir(a.run)
        if d.isdigit() and os.path.exists(os.path.join(a.run, d, "val_metrics.json")))
    if not epochs:
        print(f"[rep] no val_metrics.json under {a.run}")
        return 2

    hdr = ["epoch"] + [k.replace("_epoch", "") for k in KEYS]
    print("  ".join(f"{h:>10}" for h in hdr))
    rows = []
    for e in epochs:
        with open(os.path.join(a.run, str(e), "val_metrics.json"), encoding="utf-8") as f:
            d = json.load(f)
        rows.append([e] + [d.get(k, float("nan")) for k in KEYS])
        print("  ".join(f"{v:>10.4f}" if isinstance(v, float) else f"{v:>10d}" for v in rows[-1]))

    if len(rows) >= 2:
        print("\n[rep] accept_len: %.4f -> %.4f (%+.4f over %d epochs)"
              % (rows[0][2], rows[-1][2], rows[-1][2] - rows[0][2], len(rows) - 1))
        # per-position accuracy at the final epoch, if present
        with open(os.path.join(a.run, str(epochs[-1]), "val_metrics.json"), encoding="utf-8") as f:
            d = json.load(f)
        pos = [(k, d[k]) for k in sorted(d) if k.startswith("position_")]
        if pos:
            print("[rep] final per-position acc: "
                  + ", ".join(f"p{k.split('_')[1]}={v:.3f}" for k, v in pos))
    return 0


if __name__ == "__main__":
    sys.exit(main())
