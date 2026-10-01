"""Per-epoch validation metrics for a training run, with the early-stop verdict."""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=os.path.join(ROOT, "output", "dspark_06b", "run3"))
    a = ap.parse_args()

    eps = sorted(int(d) for d in os.listdir(a.run)
                 if d.isdigit() and os.path.exists(os.path.join(a.run, d, "val_metrics.json")))
    if not eps:
        print("[run] no completed epochs")
        return 1
    print(f"{'epoch':>5} {'E[L]':>8} {'accept':>8} {'full_acc':>9} {'loss':>8} {'best':>8}")
    best, best_ep, since = float("-inf"), None, 0
    for e in eps:
        with open(os.path.join(a.run, str(e), "val_metrics.json"), encoding="utf-8") as f:
            m = json.load(f)
        el = m["accept_len_epoch"]
        if el > best + 1e-4:
            best, best_ep, since = el, e, 0
        else:
            since += 1
        print(f"{e:>5} {el:>8.4f} {m['accept_rate_epoch']:>8.4f} "
              f"{m['full_acc_epoch']:>9.4f} {m['loss_epoch']:>8.4f} {best:>8.4f}")
    print(f"\n[run] best E[L]={best:.4f} at epoch {best_ep}; "
          f"{since} epoch(s) without improvement since")
    return 0


if __name__ == "__main__":
    sys.exit(main())
