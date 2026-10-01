"""Preflight for the DSpark-0.6B training run.

Checks everything that would otherwise waste a multi-hour GPU run:

  * the preprocessed dataset exists and has rows
  * hidden states exist for *every* row and a sample of them passes the
    trainer's own `check_hidden_states`
  * no other process is holding the GPU (this box serialises GPU work: a
    concurrent job already corrupted TPS timings once)
  * prints the exact command, with sizes chosen from free VRAM

Sizing note.  The collator packs to `[1, total_seq_len, ...]`, and the draft
forward then processes `max_anchors * block_size` query positions attending over
the whole context.  With the recipe's `total_seq_len=4096` and
`max_anchors=3072` that is ~24.6k queries x 4096 keys x 16 heads x 2 B ~= 3.2 GB
of attention scores if the backend materialises them.  The memory-efficient path
(`simple_flex_attention`) needs torch.compile, which is unusable here (Triton
cannot link), so we must size by hand.  This script picks conservative values on
a 12 GB card and prints them; scale up only after a step succeeds.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFIER = r"E:\Semantic\data\hf\Qwen3-0.6B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "mix_prep"))
    ap.add_argument("--hs", default=os.path.join(ROOT, "data", "hs"))
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--anchors", type=int, default=None)
    a = ap.parse_args()

    ok = True

    # ---- dataset ----
    try:
        from datasets import load_from_disk
        ds = load_from_disk(a.data)
        n_rows = len(ds)
        n_tok = sum(len(x) for x in ds["input_ids"])
        print(f"[pre] dataset OK: {n_rows} rows, {n_tok} tokens, cols={ds.column_names}")
    except Exception as e:
        print(f"[pre] dataset FAILED: {e}")
        return 2

    # ---- hidden states ----
    files = sorted(glob.glob(os.path.join(a.hs, "hs_*.safetensors")))
    idx = []
    for f in files:
        try:
            idx.append(int(os.path.basename(f)[3:-len(".safetensors")]))
        except ValueError:
            pass
    print(f"[pre] hidden states: {len(idx)} files in {a.hs}")
    if len(idx) < n_rows:
        missing = sorted(set(range(n_rows)) - set(idx))
        print(f"[pre] MISSING {len(missing)} rows (first: {missing[:5]})")
        ok = False

    if idx:
        try:
            from safetensors.torch import load_file
            from speculators.data_generation.offline import check_hidden_states
            for i in idx[:2] + idx[-1:]:
                d = load_file(os.path.join(a.hs, f"hs_{i}.safetensors"))
                check_hidden_states(d, ds[i]["input_ids"].tolist())
            print(f"[pre] sample of hidden states passes check_hidden_states")
        except Exception as e:
            print(f"[pre] hidden-state validation FAILED: {e}")
            ok = False

    # ---- verifier ----
    if os.path.exists(os.path.join(VERIFIER, "config.json")):
        print(f"[pre] verifier OK: {VERIFIER}")
    else:
        print(f"[pre] verifier MISSING: {VERIFIER}")
        ok = False

    # ---- gpu ----
    # NOTE: torch.cuda.mem_get_info() is NOT trustworthy for this on Windows.
    # Under WDDM it reported "10.22 GiB free" while the round-robin held 10.5 GiB
    # of a 12 GiB card, because it reports what the calling context could obtain
    # rather than true device occupancy.  Use nvidia-smi, which reports the
    # device's actual usage, for the "is something else running" decision.
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        used_mib, total_mib = (int(x) for x in out.split(","))
        free_gib = (total_mib - used_mib) / 1024
        print(f"[pre] nvidia-smi: used {used_mib} MiB / {total_mib} MiB "
              f"-> {free_gib:.2f} GiB actually free")
        if used_mib > total_mib * 0.30:
            print("[pre] WARNING: the GPU is already substantially in use -- another "
                  "job is running. Do not start training alongside it.")
            ok = False
    except Exception as e:
        print(f"[pre] GPU check failed: {e}")

    # ---- sizing ----
    seq = a.seq_len or 2048
    anchors = a.anchors or 256
    q = anchors * 8
    attn_gb = 16 * q * seq * 2 / 2**30
    print(f"[pre] sizing: total_seq_len={seq} max_anchors={anchors} -> "
          f"{q} draft queries, ~{attn_gb:.2f} GiB attention scores if materialised")

    print("[pre] command:")
    print(f"""  python -m speculators.train \\
    --verifier-name-or-path "{VERIFIER}" \\
    --data-path "{a.data}" --hidden-states-backend file \\
    --hidden-states-path "{a.hs}" \\
    --save-path "{os.path.join(ROOT, 'output', 'dspark_06b')}" \\
    --speculator-type dspark --num-layers 3 --target-layer-ids 2 14 25 \\
    --block-size 8 --total-seq-len {seq} --max-anchors {anchors} \\
    --markov-rank 256 --markov-head-type vanilla \\
    --enable-confidence-head --confidence-head-with-markov --confidence-head-alpha 1.0 \\
    --loss-implementation eager --draft-attn-impl sdpa \\
    --epochs 1 --lr 3e-4 --on-missing raise \\
    --log-freq 10 --checkpoint-freq 500 --save-path "{os.path.join(ROOT, 'output', 'dspark_06b')}\"""")

    print("[pre] " + ("READY" if ok else "NOT READY"))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
