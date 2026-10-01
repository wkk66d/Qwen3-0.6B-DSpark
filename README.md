# Qwen3-0.6B-DSpark

[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-wkk66d%2FQwen3--0.6B--DSpark-yellow)](https://huggingface.co/wkk66d/Qwen3-0.6B-DSpark)
[![base model](https://img.shields.io/badge/base%20model-Qwen3--0.6B-blue)](https://huggingface.co/Qwen/Qwen3-0.6B)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Training pipeline and trained weights for a **DSpark-style speculative-decoding
drafter** built from scratch for [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B).

This repository covers the model side only: how the corpus was built, how the
target's hidden states were extracted, how the drafter was trained, how it is
converted for inference, and the dashboard used to watch training. It ships the
finished drafter in `weights/` and mirrors it on the Hub.

## Released model

The trained drafter is published at
**[🤗 wkk66d/Qwen3-0.6B-DSpark](https://huggingface.co/wkk66d/Qwen3-0.6B-DSpark)**
(128.1 M params, 244 MiB bf16, 40 tensors). The Hub copy and
`weights/dspark_06b_engine_r4/` here are the same checkpoint.

```python
from speculators import SpeculatorModel

drafter = SpeculatorModel.from_pretrained("wkk66d/Qwen3-0.6B-DSpark")
```

The published `config.json` names its verifier as `Qwen/Qwen3-0.6B` rather than a
local path, so it loads unchanged on any machine.

Measured on a single RTX 4070, GPQA Diamond, 6000-token thinking budget:

| | tok/s | speed-up |
|---|---|---|
| `Qwen/Qwen3-0.6B` alone | 33.1 | 1.00× |
| + this drafter (K=7, verified) | **67.5** | **2.04×** |

---

## The drafter

| | |
|---|---|
| target | `Qwen/Qwen3-0.6B` — 28 layers, hidden 1024, tied embeddings |
| drafter | 3 layers, block size 8, Markov head (rank 256), confidence head |
| hidden-state taps | layers **2, 14, 25** |
| parameters | 128.1 M (244 MiB in bf16) |
| training corpus | 18,000 rows / **8.71 M tokens** |
| held-out split | 1,800 rows, **source-disjoint** from training |
| epochs | 10 (early stopping armed at patience 3, never triggered) |
| **E[L] on held-out** | 1.6083 → **2.3487**, improving every epoch |

`E[L]` is `accept_len_epoch` — the expected number of draft tokens accepted per
target call. It is the training objective that matters: it is exactly the
speed-up factor a verifier will realise.

### Training curve

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| E[L] | 1.608 | 1.714 | 1.809 | 1.851 | 1.934 | 1.997 | 2.067 | 2.159 | 2.246 | **2.337** |
| accept rate | 0.217 | 0.250 | 0.268 | 0.273 | 0.294 | 0.309 | 0.319 | 0.335 | 0.352 | 0.366 |
| full-block acc | 0.292 | 0.318 | 0.335 | 0.344 | 0.359 | 0.371 | 0.383 | 0.398 | 0.412 | 0.424 |
| val loss | 1.777 | 1.658 | 1.572 | 1.528 | 1.473 | 1.427 | 1.378 | 1.330 | 1.280 | 1.249 |

Monotone improvement across all ten epochs. A continuation run (epoch 10) added
only +0.012, so the corpus is the binding constraint, not the number of epochs.

### Confidence head calibration

The confidence head predicts a per-slot acceptance probability. Measured on
held-out data it tracks the real per-slot structure: predicted slot means fall
from ≈0.64 to ≈0.27, monotone decreasing, against a true profile of
0.686 → 0.293. Its spread (0.49) actually exceeds the realised spread (0.39).

The head is a bare `nn.Linear(input_dim, 1)` — 1281 parameters, no hidden layer.
That is enough, because the per-slot signal is linearly available in the
drafter's block hidden states. If you probe it, note that the engine's
`conf_seen` accumulates `block_size` values per round, **not** K drafts; slicing
with the wrong stride averages the structure away and makes the head look flat.

---

## Corpus

Three sources, chosen for diversity rather than volume:

| source | rows |
|---|---|
| `Aeala/ShareGPT_Vicuna_unfiltered` | 5,000 |
| `openai/gsm8k` | 7,000 |
| `HuggingFaceH4/ultrachat_200k` | 6,000 |

The held-out set comes **only** from `mlabonne/open-perfectblend`, which
contributes nothing to training, so the split is source-disjoint rather than
merely held out. This matters: a random split of a small mixed corpus tells you
almost nothing about generalisation.

---

## Pipeline

```bash
# 0. build the corpus (train rows + a source-disjoint held-out set)
python scripts/build_corpus.py --out data/mixed3_spec.jsonl --test-out data/test2_spec.jsonl

# 1. tokenise and mask into the speculators format
python -m speculators prepare-data --model <target> --data data/mixed3_spec.jsonl \
       --output data/mix4_prep --seq-length 4096 --max-samples 21000

# 2. RESTORE THE SPLIT -- prepare-data permutes rows; see below
python scripts/reorder_for_split.py --prepared data/mix4_prep \
       --test-jsonl data/test2_spec.jsonl --out data/mix5_prep

# 3. extract target hidden states at the three taps (GPU, ~30 min)
python scripts/gen_hidden_states.py --data data/mix5_prep --out data/hs5 --taps 2 14 25

# 4. optional pre-flight checks before committing GPU hours
python scripts/preflight_train.py

# 5. train, with early stopping on the held-out E[L]
python scripts/train_dspark_06b.py --mode cont --tag run4 \
       --data data/mix5_prep --hs data/hs5 \
       --train-data-ratio 0.909072539907052 --patience 3

# 6. convert the trained checkpoint into the inference format
python scripts/convert_spec_to_engine.py output/dspark_06b/run4/10 \
       output/dspark_06b_engine_r4

# 7. watch it
python scripts/serve_dashboard.py --port 8765
```

`train_dspark_06b.py` owns the trainer subprocess and implements the early
stopping the trainer itself lacks: it polls each epoch's `val_metrics.json`,
tracks `accept_len_epoch`, and terminates on N epochs without improvement.

### Three things that will bite you

**`speculators prepare-data` permutes its input.** Verified by multiset
comparison — identical content, different order. `train/data.py` takes a
*positional* slice (`split_idx = int(len(data) * train_ratio)`), so ordering the
input file accomplishes nothing. In an early run this leaked **629 of 700
intended held-out rows into training**. `reorder_for_split.py` reorders the
*prepared* Arrow dataset instead (train rows first, held-out rows last) and
solves for the exact ratio that lands the cut on the boundary. Because hidden
states are indexed by row position, they must be regenerated afterwards.

**Raising `--epochs` on a resumed run re-heats the learning rate.** The linear
schedule computes `num_training_steps = num_epochs × steps_per_epoch` at
construction and does **not** store it in the scheduler state dict. Restoring a
checkpoint restores the step counter but not the denominator, so a resumed run
comes back at `(1 - step/total)` of peak instead of where the anneal left it.
Measured: epoch 9 ended at LR **0.0**; after raising `--epochs` to 25 the next
epoch ran at **1.693e-4** and E[L] fell 2.3369 → 1.9487. Resume a finished
anneal with `--scheduler-type none` and a small constant LR instead.

**The tap axis must be `len(aux_taps) + 1`.** `train/data.py` uses `[:, :-1]`
for the `fc` features and `[:, -1]` for `verifier_last_hidden_states`, and
`check_hidden_states` does **not** validate the tap count — a mismatch surfaces
much later as `a and b must have same reduction dim`. `gen_hidden_states.py`
appends the final layer tap automatically and prints the resolved axis.

---

## Environment

* Windows, single RTX 4070 (12 GB, sm_89), Python 3.11, PyTorch 2.10.0+cu128.
* [`speculators`](https://github.com/vllm-project/speculators) 0.8.0 for training.
  Install with `--no-deps`: a plain `pip install speculators` replaces torch with
  a 2.13 wheel. Three local patches were needed — see
  [`docs/speculators-patches.md`](docs/speculators-patches.md).
* `torch.compile` cannot link on this machine (MinGW/MSVC toolchain conflict), so
  the launcher sets `TORCHDYNAMO_DISABLE=1` and `TORCH_COMPILE_DISABLE=1`.

---

## Layout

```
src/dspark.py                    drafter definition and checkpoint loader
src/kvcache.py                   KV cache used by the drafter
scripts/build_sharegpt_data.py   raw source -> chat rows
scripts/build_corpus.py          assemble the training corpus + held-out set
scripts/build_test_split.py      solve for the split ratio
scripts/reorder_for_split.py     repair the positional split
scripts/gen_hidden_states.py     target hidden states at the taps
scripts/validate_hs.py           verify tap alignment before training
scripts/preflight_train.py       pre-training sanity checks
scripts/train_dspark_06b.py      training launcher + E[L] early stopping
scripts/run_metrics.py           per-epoch validation summary
scripts/train_report.py          training curves
scripts/trace_train.py           step-level loss/accept trace
scripts/confidence_trace.py      confidence-head calibration over epochs
scripts/convert_spec_to_engine.py  trained checkpoint -> inference format
scripts/build_dashboard.py       dashboard data collector
scripts/serve_dashboard.py       dashboard server
web/                             training dashboard (no build step, no CDN)
weights/dspark_06b_engine_r4/    trained drafter (Git LFS), mirrored on the Hub
```

## Files on the Hub

| file | |
|---|---|
| `model.safetensors` | the drafter, 244 MiB, LFS |
| `config.json` | `DSparkDraftModel`, taps [2,14,25], block 8, Markov rank 256 |
| `README.md` | model card |

## Credit

The drafter architecture, the Markov head, and the confidence-head objective
come from the DSpark / DFlash work; this repository's contribution is the
small-target training pipeline and the trained weights.
