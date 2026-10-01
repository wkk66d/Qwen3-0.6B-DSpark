"""Launch DSpark-0.6B drafter training from the preprocessed offline data.

Two modes:

  --smoke   tiny sizes, a couple of steps: proves the whole trainer path works
            (model build, hidden-state loading, draft forward, loss, optimizer,
            checkpoint write) in ~minutes instead of discovering a breakage
            after an hour of the real run.
  --full    the real settings.

Sizing is chosen for a 12 GB card, NOT copied from the official recipe.  The
recipe's `total_seq_len=4096` + `max_anchors=3072` means ~24.6k draft queries
attending over 4096 keys; with `--draft-attn-impl sdpa` that can materialise
~3.2 GiB of attention scores.  The memory-efficient `simple_flex_attention` path
needs torch.compile, which cannot link Triton on this box.  See TRAINING_PLAN.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFIER = r"E:\Semantic\data\hf\Qwen3-0.6B"
DATA = os.path.join(ROOT, "data", "mix_prep")
HS = os.path.join(ROOT, "data", "hs")

MODES = {
    # mode:    (total_seq_len, max_anchors, epochs, extra)
    "smoke":   (512, 32, 1, ["--max-steps", "3"]),
    "probe":   (2048, 256, 1, ["--max-steps", "20"]),
    "full":    (2048, 256, 5, []),
    "full1":   (2048, 256, 1, []),
    "full2":   (4096, 512, 5, []),
    # Continuation.  Re-run with the SAME --tag so the save path matches: the
    # trainer finds the last checkpoint, sets current_epoch = previous_epoch + 1,
    # and runs range(current_epoch, epochs), restoring optimizer and scheduler
    # state -- a true continuation, not a warm restart.  Confirmed in
    # trainer.py:238-292 and :692.
    "cont":    (2048, 256, 10, []),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODES), default="smoke")
    ap.add_argument("--hs", default=HS)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--tag", default=None, help="checkpoint subdir suffix")
    ap.add_argument("--from-pretrained", default=None,
                    help="warm-start weights from a checkpoint dir (fresh optimizer)")
    ap.add_argument("--train-data-ratio", type=float, default=None,
                    help="positional train/val cut; see scripts/build_test_split.py")
    ap.add_argument("--patience", type=int, default=0,
                    help="0 = train all epochs; N>0 = early-stop on E[L] after N "
                         "epochs without improvement on the held-out split")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the mode's epoch count. NOTE: the linear LR "
                         "schedule derives total_steps = epochs * steps_per_epoch "
                         "(trainer.py:155), so RAISING this on a resumed run "
                         "re-heats the LR to (1 - step/total) of peak instead of "
                         "continuing the anneal. Verify the first resumed epoch's "
                         "E[L] before trusting a long extension.")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="learning rate. A warm start restarts the optimizer state "
                         "(default optimizer is Muon, momentum 0.95, "
                         "muon_lr=10x lr), which on already-converged weights "
                         "produced a transient regression at 3e-4: run2 went "
                         "0.914 -> ~1.8 train loss within 400 steps. Lower it for "
                         "continuations.")
    ap.add_argument("--scheduler-type", default="linear",
                    choices=["linear", "cosine", "none"],
                    help="LR schedule. 'linear' derives num_training_steps = "
                         "num_epochs * steps_per_epoch (trainer.py:155) and bakes it "
                         "into the scheduler closure, so RAISING --epochs on a "
                         "resume leaves the restored step counter intact but "
                         "enlarges the denominator -- the LR jumps back up. "
                         "Measured on run4: epoch 9 ended at LR 0.0, epoch 10 ran at "
                         "1.693e-4 (Muon 1.693e-3) and E[L] fell 2.3369 -> 1.9487. "
                         "Use 'none' (constant LR) to continue a finished anneal.")
    ap.add_argument("--optimizer", default="muon", choices=["muon", "adamw"],
                    help="'adamw' for a head-only fine-tune: Muon defaults to "
                         "muon_lr = 10*lr, which is far too aggressive for a small "
                         "MLP sitting on frozen features.")
    ap.add_argument("--min-delta", type=float, default=1e-4)
    a = ap.parse_args()

    seq, anchors, epochs, extra = MODES[a.mode]
    if a.epochs is not None:
        epochs = a.epochs
    tag = a.tag or a.mode
    save = os.path.join(ROOT, "output", "dspark_06b", tag)

    cmd = [
        sys.executable, "-m", "speculators.train",
        "--verifier-name-or-path", VERIFIER,
        "--data-path", a.data,
        "--hidden-states-backend", "file",
        "--hidden-states-path", a.hs,
        "--save-path", save,
        "--speculator-type", "dspark",
        "--target-layer-ids", "2", "14", "25",
        "--block-size", "8",
        "--total-seq-len", str(seq),
        "--max-anchors", str(anchors),
        "--markov-rank", "256",
        "--markov-head-type", "vanilla",
        "--enable-confidence-head",
        "--confidence-head-with-markov",
        "--confidence-head-alpha", "1.0",
        "--loss-implementation", "eager",
        "--draft-attn-impl", "sdpa",
        "--epochs", str(epochs),
        "--lr", repr(a.lr),
        "--optimizer", a.optimizer,
        "--scheduler-type", a.scheduler_type,
        "--on-missing", "raise",
        "--log-freq", "5",
        "--checkpoint-freq", "1",
        *extra,
    ]
    # `--from-pretrained` supplies the decoder shape, and the trainer rejects the
    # DECODER_SHAPING_FLAGS alongside it (resolution.py:56-65, :330-337):
    # num_layers, draft_arch, draft_hidden_act, sliding_window,
    # full_attention_indices.  Only --num-layers applies here; the DSpark-specific
    # options (taps, markov rank, block size) are still allowed and must match the
    # checkpoint, which they do.
    if a.from_pretrained:
        cmd += ["--from-pretrained", a.from_pretrained]
    else:
        cmd += ["--num-layers", "3"]
    if a.train_data_ratio is not None:
        cmd += ["--train-data-ratio", repr(a.train_data_ratio)]
    print(f"[train] mode={a.mode} seq={seq} anchors={anchors} epochs={epochs}")
    print(f"[train] data={a.data}\n[train] hs  ={a.hs}\n[train] save={save}")
    if a.from_pretrained:
        print(f"[train] warm start from {a.from_pretrained}")
    print("[train] cmd: " + " ".join(cmd), flush=True)

    # The trainer wraps the draft forward in torch.compile.  Inductor then shells
    # out to the C++ compiler, and on this box PATH resolves gcc to Dev-Cpp's
    # MinGW, which cannot link against the MSVC-built CPython:
    #   undefined reference to `__imp_PyExc_TypeError'  -> InductorError
    # Disabling dynamo makes torch.compile a no-op so the step runs eagerly.
    # This is the same toolchain wall that made torch.compile unusable for the
    # inference engine; there it cost only speed, here it is fatal.
    env = dict(os.environ)
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["TORCH_COMPILE_DISABLE"] = "1"

    if a.patience <= 0:
        r = subprocess.run(cmd, cwd=ROOT, env=env)
        print(f"[train] exit {r.returncode}")
        return r.returncode

    # ---- early stopping on E[L] -------------------------------------------- #
    # The trainer has no early stopping (only save_best, and on val *loss*), so
    # we own the subprocess and watch the per-epoch val_metrics.json it writes.
    # The metric is accept_len_epoch: the expected number of tokens accepted per
    # target call, i.e. E[L].  It is measured on the held-out split, which
    # scripts/build_test_split.py arranges to be a DIFFERENT SOURCE from training.
    print(f"[es] early stopping on accept_len_epoch, patience={a.patience}, "
          f"min_delta={a.min_delta}", flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env)
    best, best_epoch, since = float("-inf"), None, 0
    seen = set()
    try:
        while proc.poll() is None:
            time.sleep(20)
            if not os.path.isdir(save):
                continue
            for d in sorted(os.listdir(save)):
                if not d.isdigit() or d in seen:
                    continue
                vm = os.path.join(save, d, "val_metrics.json")
                if not os.path.exists(vm):
                    continue
                time.sleep(3)          # let the writer finish
                try:
                    with open(vm, encoding="utf-8") as f:
                        m = json.load(f)
                except Exception:
                    continue
                seen.add(d)
                el = m.get("accept_len_epoch")
                if el is None:
                    continue
                if el > best + a.min_delta:
                    best, best_epoch, since = el, int(d), 0
                    tag_ = "improved"
                else:
                    since += 1
                    tag_ = f"no improvement ({since}/{a.patience})"
                print(f"[es] epoch {d}: E[L]={el:.4f}  best={best:.4f} "
                      f"(epoch {best_epoch})  {tag_}", flush=True)
                if since >= a.patience:
                    print(f"[es] EARLY STOP: no E[L] improvement for {since} epochs; "
                          f"best E[L]={best:.4f} at epoch {best_epoch}. "
                          f"Stopping training.", flush=True)
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[es] training terminated by early stopping", flush=True)
        return 0

    print(f"[es] training finished on its own (exit {proc.returncode}); "
          f"best E[L]={best:.4f} at epoch {best_epoch}", flush=True)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
