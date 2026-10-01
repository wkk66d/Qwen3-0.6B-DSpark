"""Print the step-level trajectory of a training log, in blocks.

Answers "is the run degrading?" without eyeballing three sampled lines: shows
train loss / accept-rate per block of steps so a running-average artefact can be
told apart from a genuine regression.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse(path):
    loss = acc = None
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = re.search(r"train/loss=([\d.eE+-]+)", ln)
            if m:
                loss = float(m.group(1))
            m = re.search(r"train/accept_rate=([\d.eE+-]+)", ln)
            if m:
                acc = float(m.group(1))
            m = re.search(r"global_step=(\d+)", ln)
            if m and loss is not None:
                rows.append((int(m.group(1)), loss, acc))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=os.path.join(ROOT, "results", "train_es_run2.log"))
    ap.add_argument("--block", type=int, default=50)
    a = ap.parse_args()

    rows = parse(a.log)
    if not rows:
        print("[trace] no step lines found")
        return 2
    print(f"[trace] {os.path.basename(a.log)}: {len(rows)} points, "
          f"steps {rows[0][0]}..{rows[-1][0]}")
    print(f"{'step':>7} {'loss':>8} {'acc':>7}")
    for i, (s, l, ac) in enumerate(rows):
        if i % a.block == 0 or i == len(rows) - 1:
            print(f"{s:>7} {l:>8.3f} {('%.3f' % ac) if ac is not None else '   -':>7}")
    ls = [l for _, l, _ in rows]
    acs = [x for _, _, x in rows if x is not None]
    print(f"\n[trace] loss  min={min(ls):.3f} max={max(ls):.3f} last={ls[-1]:.3f}")
    if acs:
        print(f"[trace] acc   min={min(acs):.3f} max={max(acs):.3f} last={acs[-1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
