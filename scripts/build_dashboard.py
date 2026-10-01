"""Collect training state for the web dashboard.

Everything is read from artefacts that already exist, so the dashboard never
becomes a second source of truth:

  training      output/dspark_06b/<run>/<epoch>/val_metrics.json   (per-epoch)
                output/dspark_06b/<run>/<epoch>/training_state.json (global_step)
  early stop    the launcher's log line "[es] epoch N: E[L]=..." 
  benchmark     results/raw/{bench}_{mode}[_{tag}].jsonl          (per-item)

`E[L]` is `accept_len_epoch`: the expected number of tokens accepted per target
forward call.  `serve_dashboard.py` calls `collect()` on every /data.json
request, so the page is live without a refresh loop.
"""

from __future__ import annotations

import glob
import json
import os
import re
import statistics as st
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "output", "dspark_06b")
# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def collect_run(run_dir, cur_step=None):
    """Per-epoch metrics plus a self-calibrating in-epoch step counter and ETA.

    Steps/epoch and seconds/epoch are DERIVED, not assumed: the trainer records
    `global_step` in each epoch's training_state.json, so consecutive deltas give
    the exact epoch length, and checkpoint-directory mtimes give its wall time.
    We deliberately do not use `profile/step_ms` for the ETA -- on this run it
    reports 830-1090 ms while 5150 steps actually take 55 min (= 640 ms/step), so
    it overstates by 30-70%.
    """
    name = os.path.basename(run_dir)
    # NUMERIC sort. `sorted(os.listdir(...))` is lexicographic, which puts epoch 10
    # between 1 and 2 -- that scrambled the x-axis of the E[L] chart and made
    # `epochs[-1]` (the "latest") report epoch 9 while 10 was the newest.
    ep_nums = sorted(int(d) for d in os.listdir(run_dir) if d.isdigit())
    epochs = []
    for n in ep_nums:
        d = str(n)
        vm = os.path.join(run_dir, d, "val_metrics.json")
        if not os.path.exists(vm):
            continue
        try:
            with open(vm, encoding="utf-8") as f:
                m = json.load(f)
        except Exception:
            continue
        pos = [m[k] for k in sorted(m) if re.fullmatch(r"position_\d+_acc_epoch", k)]
        epochs.append({
            "epoch": int(d),
            "el": m.get("accept_len_epoch"),
            "accept_rate": m.get("accept_rate_epoch"),
            "full_acc": m.get("full_acc_epoch"),
            "loss": m.get("loss_epoch"),
            "conf_pred": m.get("confidence_pred_mean_epoch"),
            "conf_err": m.get("confidence_abs_error_epoch"),
            "positions": pos,
        })

    # global_step and mtime per completed epoch
    gstep, mtime = {}, {}
    for n in ep_nums:
        d = str(n)
        ep_dir = os.path.join(run_dir, d)
        sp = os.path.join(ep_dir, "training_state.json")
        if os.path.exists(sp):
            try:
                with open(sp, encoding="utf-8") as f:
                    gstep[int(d)] = json.load(f).get("global_step")
            except Exception:
                pass
        mtime[int(d)] = os.path.getmtime(ep_dir)

    done = sorted(gstep)
    epoch_steps = None
    deltas = [gstep[b] - gstep[a] for a, b in zip(done, done[1:])
              if gstep.get(a) and gstep.get(b)]
    if deltas:
        epoch_steps = int(st.median(deltas))
    elif done and gstep.get(done[0]):
        epoch_steps = int(gstep[done[0]])

    # Epoch wall time from CONSECUTIVE checkpoint-dir deltas, not from a span
    # measured since the run started.  A resume rewrites train_command.txt /
    # run.yaml, so the "since start" span covered only the resumed epochs while
    # still being divided by the total epoch count -- it reported 383 s for an
    # epoch that really takes ~4230 s.  The median also discards the one inflated
    # gap caused by the idle time between a run stopping and being resumed.
    epoch_secs = None
    gaps = [mtime[b] - mtime[a] for a, b in zip(done, done[1:])]
    if gaps:
        epoch_secs = st.median(gaps)
    elif done:
        starts = [os.path.getmtime(os.path.join(run_dir, f))
                  for f in ("train_command.txt", "run.yaml")
                  if os.path.exists(os.path.join(run_dir, f))]
        if starts and mtime[done[0]] > min(starts):
            epoch_secs = mtime[done[0]] - min(starts)

    state = {}
    for d in reversed(sorted(os.listdir(run_dir))):
        sp = os.path.join(run_dir, d, "training_state.json") if d.isdigit() else None
        if sp and os.path.exists(sp):
            try:
                with open(sp, encoding="utf-8") as f:
                    state = json.load(f)
            except Exception:
                pass
            break

    # In-epoch progress + ETA.
    #
    # The log's global_step TRAILS the checkpoint by a few steps, because it does
    # not advance while validation runs.  At an epoch boundary that makes
    # `cur_step - start_step` briefly NEGATIVE, and an earlier version dropped the
    # whole block on that condition -- so the ETA vanished for the entire
    # validation phase.  Handle the boundary explicitly instead.
    progress = None
    if (cur_step and done and epoch_steps and gstep.get(done[-1]) is not None):
        start_step = gstep[done[-1]]
        local = cur_step - start_step
        if local < 0:
            # epoch just finished; validation is running and the log has not
            # caught up to the checkpoint's final step yet
            progress = {"phase": "validating", "epoch": done[-1],
                        "local_step": epoch_steps, "epoch_steps": epoch_steps,
                        "frac": 1.0, "eta_secs": None, "epoch_secs": epoch_secs}
        elif local <= epoch_steps * 1.05:
            frac = min(1.0, local / max(1, epoch_steps))
            eta = (epoch_steps - local) * (epoch_secs / epoch_steps) if epoch_secs else None
            progress = {"phase": "running", "epoch": len(done),
                        "local_step": local, "epoch_steps": epoch_steps,
                        "frac": frac, "eta_secs": eta, "epoch_secs": epoch_secs}
    return {"name": name, "epochs": epochs, "state": state,
            "epoch_steps": epoch_steps, "epoch_secs": epoch_secs,
            "progress": progress}


def collect_training(cur_step=None):
    runs = []
    if os.path.isdir(OUTDIR):
        for d in sorted(os.listdir(OUTDIR)):
            p = os.path.join(OUTDIR, d)
            if os.path.isdir(p) and any(x.isdigit() for x in os.listdir(p)):
                runs.append(collect_run(p, cur_step=cur_step))
    runs = [r for r in runs if r["epochs"]]
    return {"runs": runs, "early_stop": collect_early_stop()}


def collect_early_stop():
    """Parse the launcher's [es] lines -- the only place patience is tracked."""
    logs = sorted(glob.glob(os.path.join(ROOT, "results", "train_es*.log")),
                  key=os.path.getmtime, reverse=True)
    for lg in logs:
        try:
            with open(lg, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception:
            continue
        best, best_epoch, since, patience, stopped = None, None, 0, None, False
        for ln in lines:
            m = re.search(r"patience=(\d+)", ln)
            if m:
                patience = int(m.group(1))
            m = re.search(r"\[es\] epoch (\d+): E\[L\]=([\d.]+)\s+best=([\d.]+)\s+"
                          r"\(epoch (\S+)\)\s+(.*)", ln)
            if m:
                best, best_epoch = float(m.group(3)), m.group(4)
                since = 0 if "improved" in m.group(5) else int(
                    re.search(r"\((\d+)/", m.group(5)).group(1)) if re.search(r"\((\d+)/", m.group(5)) else since
            if "EARLY STOP" in ln:
                stopped = True
        if best is not None or patience is not None:
            return {"log": os.path.basename(lg), "best_el": best,
                    "best_epoch": best_epoch, "since": since,
                    "patience": patience, "stopped": stopped}
    return None


def _parse_log(path):
    out, loss, acc = [], None, None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return out
    for ln in lines:
        m = re.search(r"train/loss=([\d.eE+-]+)", ln)
        if m:
            loss = float(m.group(1))
        m = re.search(r"train/accept_rate=([\d.eE+-]+)", ln)
        if m:
            acc = float(m.group(1))
        m = re.search(r"global_step=(\d+)", ln)
        if m and loss is not None:
            out.append({"step": int(m.group(1)), "loss": loss, "acc": acc})
    return out


def collect_progress(run_name=None, max_points=500):
    """Step-level train loss/accept-rate for `run_name`.

    Per-epoch val metrics only move once an hour, so the trainer's own log gives
    the live trace.  Two things this has to get right:

    * **Continuations vs sibling branches.** A run interrupted and resumed
      *forwards* writes a new log whose steps continue past the old one -- those
      should be concatenated.  But a run restarted from an *earlier* checkpoint
      writes a log whose steps OVERLAP an abandoned branch's.  Merging by step
      number alone cannot tell these apart, and it let a dead branch's higher
      steps pin the chart's right edge: the live run reached 55640 while the
      abandoned one had already written 57890, so the plot never advanced.
      We therefore anchor on the live log (newest mtime) and keep older logs only
      where their steps are strictly below the live log's first step.

    * **No line-window truncation.** A `readlines()[-20000:]` window silently cut
      ~10000 steps off long runs; we read fully and downsample afterwards.
    """
    pat = f"*{run_name}*.log" if run_name else "train*.log"
    logs = [p for p in glob.glob(os.path.join(ROOT, "results", pat))
            if os.path.getsize(p) > 0 and os.path.basename(p).startswith("train")]
    if not logs:
        # Fall back to any training log. On a fresh checkout there may be none at
        # all -- `max()` on an empty sequence raises, so return no progress
        # rather than crashing the whole dashboard payload.
        logs = [p for p in glob.glob(os.path.join(ROOT, "results", "train*.log"))
                if os.path.getsize(p) > 0]
    if not logs:
        return []
    logs.sort(key=os.path.getmtime)              # oldest -> newest

    live_pts = _parse_log(logs[-1])
    frontier = min((p["step"] for p in live_pts), default=float("inf"))
    merged = {}
    for lg in logs[:-1]:
        for pt in _parse_log(lg):
            if pt["step"] < frontier:            # only genuine history
                merged[pt["step"]] = pt
    for pt in live_pts:
        merged[pt["step"]] = pt                  # live data always wins

    out = [merged[s] for s in sorted(merged)]
    if len(out) > max_points:
        stride = len(out) // max_points + 1
        out = out[::stride]
    return out


def _newest_run_name():
    """The run with the most recently written epoch directory."""
    best, best_t = None, -1
    if not os.path.isdir(OUTDIR):
        return None
    for d in os.listdir(OUTDIR):
        p = os.path.join(OUTDIR, d)
        if not os.path.isdir(p):
            continue
        subs = [os.path.join(p, x) for x in os.listdir(p) if x.isdigit()]
        if not subs:
            continue
        t = max(os.path.getmtime(s) for s in subs)
        if t > best_t:
            best, best_t = d, t
    return best




def collect():
    """Training-only payload for the dashboard (`/` and `/data.json`).

    The benchmark half of the original collector was removed, so `training` and
    `progress` are the whole contract. `active_run` still tells the page which run
    to feature rather than relying on alphabetical ordering of run names.
    """
    run = _newest_run_name()
    prog = collect_progress(run)
    cur = prog[-1]["step"] if prog else None
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "active_run": run,
        "training": collect_training(cur_step=cur),
        "progress": prog,
    }


if __name__ == "__main__":
    print(json.dumps(collect(), indent=1))
