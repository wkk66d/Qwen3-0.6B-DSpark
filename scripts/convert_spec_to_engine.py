"""Convert a speculators-trained drafter into the layout `src/dspark.py` loads.

The two formats differ only in their *config*:

  speculators (models/dflash/config.py)
      transformer_layer_config : nested PretrainedConfig holding hidden_size,
                                 num_attention_heads, head_dim, rms_norm_eps, ...
      aux_hidden_state_layer_ids, draft_vocab_size, target_hidden_size, ...
  src/dspark.py
      the same values flattened to the top level, with the tap layers under
      `target_layer_ids` and a `num_hidden_layers` key.

The *weights* need no remapping: the backbone is HF's own `Qwen3DecoderLayer`
(same parameter names our `DSparkAttention`/`DSparkMLP` use), the heads are
`markov_head.markov_w{1,2}` / `confidence_head.proj`, and for a full-vocab draft
`lm_head.weight` is excluded on save so the head is shared with the verifier --
exactly our engine's assumption.

The converter self-verifies: it instantiates `DSparkDrafter` from the converted
config and runs the real `_load_ckpt`, which raises if any weight our forward
needs is absent.  A conversion that "succeeds" without that check is worthless.

    python scripts/convert_spec_to_engine.py <trained_ckpt_dir> <out_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

FLATTEN = [
    "hidden_size", "intermediate_size", "num_attention_heads",
    "num_key_value_heads", "head_dim", "rms_norm_eps", "rope_parameters",
    "rope_theta", "hidden_act", "attention_bias", "attention_dropout",
]

# keys that belong to the verifier or to a reduced-vocab scheme, never to our model
DROP_PREFIX = ("verifier_lm_head", "verifier_norm", "t2d", "d2t")
DROP_EXACT = ("lm_head.weight", "embed_tokens.weight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="speculators checkpoint dir (contains config.json)")
    ap.add_argument("dst", help="output dir for src/dspark.py")
    ap.add_argument("--verifier", default=r"E:\Semantic\data\hf\Qwen3-0.6B",
                    help="verifier dir, for vocab_size and any missing top-level fields")
    a = ap.parse_args()

    from safetensors.torch import load_file, save_file

    with open(os.path.join(a.src, "config.json"), encoding="utf-8") as f:
        sc = json.load(f)
    tlc = sc.get("transformer_layer_config") or {}

    # the verifier supplies vocab_size and anything the draft config omits
    vc = {}
    vpath = os.path.join(a.verifier, "config.json")
    if os.path.exists(vpath):
        with open(vpath, encoding="utf-8") as f:
            vc = json.load(f)

    for k in FLATTEN:
        if k not in sc and k in tlc:
            sc[k] = tlc[k]
        if k not in sc and k in vc:
            sc[k] = vc[k]

    # rope: our engine reads cfg["rope_parameters"]["rope_theta"]
    rp = sc.get("rope_parameters")
    if not isinstance(rp, dict):
        rp = {}
    if "rope_theta" not in rp:
        rp["rope_theta"] = sc.get("rope_theta") or vc.get("rope_theta") \
            or (vc.get("rope_parameters") or {}).get("rope_theta") or 1_000_000.0
    sc["rope_parameters"] = rp

    taps = sc.get("aux_hidden_state_layer_ids") or sc.get("target_layer_ids")
    if not taps:
        raise SystemExit("no aux_hidden_state_layer_ids / target_layer_ids in config")
    sc["target_layer_ids"] = list(taps)
    sc["num_hidden_layers"] = int(sc.get("num_layers") or tlc.get("num_hidden_layers") or 0)
    sc["vocab_size"] = int(sc.get("vocab_size") or vc.get("vocab_size") or 151936)
    dvs = sc.get("draft_vocab_size")
    vocab = int(sc.get("vocab_size") or vc.get("vocab_size") or 151936)
    # The trainer fills draft_vocab_size in as the full verifier vocab when no
    # --draft-vocab-size was requested, so only a value *smaller* than that means
    # a genuinely reduced-vocab draft (which needs d2t handling our engine lacks).
    if dvs is not None and int(dvs) != vocab:
        raise SystemExit(
            f"draft_vocab_size={dvs} != verifier vocab_size={vocab}: this is a "
            "reduced-vocab draft and src/dspark.py has no d2t mapping. Retrain "
            "without --draft-vocab-size.")
    sc["draft_vocab_size"] = None
    sc.setdefault("markov_head_type", "vanilla")
    sc.setdefault("enable_confidence_head", True)
    sc.setdefault("confidence_head_with_markov", True)

    sd = load_file(os.path.join(a.src, "model.safetensors"))
    kept, dropped = {}, []
    for k, v in sd.items():
        if k in DROP_EXACT or k.startswith(DROP_PREFIX):
            dropped.append(k)
            continue
        kept[k] = v

    os.makedirs(a.dst, exist_ok=True)
    with open(os.path.join(a.dst, "config.json"), "w", encoding="utf-8") as f:
        json.dump(sc, f, indent=1)
    save_file(kept, os.path.join(a.dst, "model.safetensors"))

    print(f"[conv] {len(kept)} tensors kept, {len(dropped)} dropped: {dropped}")
    print(f"[conv] config: hidden={sc.get('hidden_size')} layers={sc['num_hidden_layers']} "
          f"taps={sc['target_layer_ids']} block={sc.get('block_size')} "
          f"markov_rank={sc.get('markov_rank')} vocab={sc['vocab_size']}")

    # ---- self-verify with the real loader ----
    from dspark import DSparkDrafter
    m = DSparkDrafter(os.path.join(a.dst, "config.json"))
    m._load_ckpt(kept)          # raises on any weight our forward needs
    print(f"[conv] VERIFIED: DSparkDrafter._load_ckpt accepted all "
          f"{len(kept)} tensors; skipped={m.skipped_ckpt_keys}")
    print(f"[conv] wrote {a.dst}")


if __name__ == "__main__":
    main()
