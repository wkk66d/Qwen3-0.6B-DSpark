"""Confidence-head training trajectory for a run.

The C arms of the main experiment accept draft tokens with NO verification, gated
only by this head's score.  So its calibration is not a diagnostic -- it IS the
mechanism's correctness.  The released 4B drafter's head was badly overconfident
(score 0.8-0.9 -> 0.38 actual acceptance), so the question here is whether a head
we trained ourselves calibrates better.

Reported per epoch:
  conf_loss   BCE-style training term for the head
  pred_mean   mean predicted acceptance probability
  actual      realised acceptance rate (accept_rate_epoch)
  bias        pred_mean - actual      (0 is perfectly calibrated on average)
  mae         confidence_abs_error_epoch
  cumprod     confidence_cumprod_bias_epoch
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(run):
    eps = []
    for d in sorted(os.listdir(run)):
        p = os.path.join(run, d, "val_metrics.json")
        if d.isdigit() and os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                eps.append((int(d), json.load(f)))
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=["run4", "run1"])
    a = ap.parse_args()

    for name in a.runs:
        run = os.path.join(ROOT, "output", "dspark_06b", name)
        if not os.path.isdir(run):
            continue
        eps = load(run)
        if not eps:
            continue
        print(f"\n=== {name} ({len(eps)} epochs) ===")
        print(f"{'ep':>3} {'conf_loss':>10} {'pred_mean':>10} {'actual':>8} "
              f"{'bias':>8} {'mae':>8} {'cumprod':>10}")
        for e, m in eps:
            pm = m.get("confidence_pred_mean_epoch")
            ac = m.get("accept_rate_epoch")
            print(f"{e:>3} {m.get('confidence_loss_epoch', float('nan')):>10.4f} "
                  f"{pm:>10.4f} {ac:>8.4f} "
                  f"{(pm - ac) if (pm is not None and ac is not None) else float('nan'):>8.4f} "
                  f"{m.get('confidence_abs_error_epoch', float('nan')):>8.4f} "
                  f"{m.get('confidence_cumprod_bias_epoch', float('nan')):>10.5f}")
        first, last = eps[0][1], eps[-1][1]
        print(f"[{name}] conf_loss {first.get('confidence_loss_epoch'):.4f} -> "
              f"{last.get('confidence_loss_epoch'):.4f} | "
              f"bias {first.get('confidence_pred_mean_epoch') - first.get('accept_rate_epoch'):+.4f} -> "
              f"{last.get('confidence_pred_mean_epoch') - last.get('accept_rate_epoch'):+.4f}")


if __name__ == "__main__":
    sys.exit(main())
