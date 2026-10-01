"""Generate the verifier hidden states the speculator trainer consumes.

Replaces `speculators generate-offline-data`, which requires a vLLM server
(no Windows support).  The format is defined by hs_connectors.FileTransfer:
`<out>/hs_{i}.safetensors` holding a dict with

    token_ids     : int64 [T]        -- must equal the preprocessed row exactly
    hidden_states : [T, num_taps, H] -- the tapped decoder-layer outputs

`speculators.data_generation.offline.check_hidden_states` validates all three of
those properties, so we mirror them precisely.

Layer convention: `layers[i].register_forward_hook`, i.e. the OUTPUT of decoder
layer i -- the same convention src/engine.py uses, so a drafter trained on these
files is consistent with our own inference engine.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_06B = r"E:\Semantic\data\hf\Qwen3-0.6B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=TARGET_06B)
    ap.add_argument("--data", required=True, help="preprocessed dataset dir")
    ap.add_argument("--out", required=True, help="dir for hs_{i}.safetensors")
    ap.add_argument("--taps", type=int, nargs="+", default=[2, 14, 25],
                    help="AUX tap layers; the verifier's final layer is appended")
    ap.add_argument("--final-tap", type=int, default=None,
                    help="layer whose output is the verifier's last hidden state "
                         "(default: the final decoder layer)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from datasets import load_from_disk
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, a.dtype)
    ds = load_from_disk(a.data)
    n = len(ds) if a.limit is None else min(a.limit, len(ds))

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=dtype, attn_implementation="sdpa").to(a.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_layers = len(model.model.layers)
    final = a.final_tap if a.final_tap is not None else n_layers - 1
    # `train/data.py` splits the tap axis as [:, :-1] -> the features fed to `fc`
    # and [:, -1] -> `verifier_last_hidden_states` (normed + lm_head for the tv
    # loss).  So the file must carry len(taps) + 1 entries, the extra last one
    # being the verifier's FINAL layer output.  Writing only the aux taps makes
    # `fc` receive (len(taps)-1)*H features and fail with a shape error.
    all_taps = list(a.taps) + [final]
    if final in a.taps:
        raise SystemExit(f"final tap {final} duplicates an aux tap in {a.taps}")
    print(f"[hs] {n} rows from {a.data}; aux taps={a.taps} + final={final} "
          f"(model has {n_layers} layers) -> tap axis = {len(all_taps)}", flush=True)

    captured: dict[int, torch.Tensor] = {}
    handles = []
    for i in all_taps:
        def mk(idx):
            def hook(module, inputs, output):
                captured[idx] = output[0] if isinstance(output, tuple) else output
            return hook
        handles.append(model.model.layers[i].register_forward_hook(mk(i)))

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    n_done = 0
    for i in range(n):
        out_path = os.path.join(a.out, f"hs_{i}.safetensors")
        if os.path.exists(out_path):
            n_done += 1
            continue
        row = ds[i]
        ids = row["input_ids"]
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
        ids = ids.to(torch.long)
        T = int(row.get("seq_len", len(ids)))
        ids = ids[:T]
        if T == 0:
            continue
        try:
            with torch.inference_mode():
                model(ids[None].to(a.device), use_cache=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[hs] OOM at row {i} (T={T}); skipping", flush=True)
            continue
        hs = torch.stack([captured[k][0] for k in all_taps], dim=1)  # [T, taps+1, H]
        save_file({"token_ids": ids.cpu(),
                   "hidden_states": hs.to(dtype).cpu().contiguous()}, out_path)
        n_done += 1
        if n_done % 200 == 0:
            el = time.time() - t0
            print(f"[hs] {n_done}/{n}  {el:.0f}s  eta {el/n_done*(n-n_done):.0f}s", flush=True)

    for h in handles:
        h.remove()
    print(f"[hs] wrote {n_done} files to {a.out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
