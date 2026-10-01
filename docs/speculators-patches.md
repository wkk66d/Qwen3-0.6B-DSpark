# Local patches to `speculators` 0.8.0

The training framework needed four changes to run on this machine. Patches 1–2
are Windows compatibility; patches 3–4 are experiment switches that are **inert
by default**, kept only because they document what was tried.

## 1. Windows symlink guards — `train/trainer.py`, `train/checkpointer.py`

`os.symlink` raises `WinError 1314` without Developer Mode. The checkpoint
symlink calls are wrapped in `try/except` and write a `.txt` pointer instead.
Cosmetic: the checkpoints themselves are unaffected.

## 2. `fcntl` shim

`speculators.utils.hs_connectors` imports `fcntl`, which is Unix-only. A shim at
`site-packages/fcntl.py` implements `flock` / `LOCK_EX` / `LOCK_SH` / `LOCK_UN`
on top of `msvcrt.locking`.

## 3. `DSPARK_CONF_TARGET` — `models/dspark/metrics.py`

Default `tv` (upstream). Setting it to `argmax` supervises the confidence head
against the target model's greedy token instead of the distributional overlap
`1 - d_TV`.

**Outcome: not needed, and not used.** The trained head predicts distributional
overlap; a deterministic threshold gate consumes argmax agreement, so the two
differ in principle. In practice the head already produces a correctly graded
per-slot profile (spread 0.49 against a true 0.39, monotone), so re-supervising
it only shifted the level (mean 0.50 → 0.64 against a true 0.42) and worsened
calibration. The released `tv` objective was kept.

## 4. `DSPARK_FREEZE_BACKBONE` — `train/trainer.py`

Sets `requires_grad=False` on every parameter except the confidence head, so the
head can be re-supervised without disturbing an already-trained drafter. Safe
because `split_named_params_for_muon` skips frozen parameters
(`train/optimizers.py:56`), leaving only the head in the optimizer groups.

Used with `--optimizer adamw --lr 1e-3 --scheduler-type none` for head-only
fine-tunes — about one epoch instead of ten.

---

## Why #3 and #4 exist, and why you should ignore them

Both were motivated by a measurement that turned out to be **wrong**: that the
confidence head was "flat across draft slots" — spread 0.016 against a true
spread of 0.378.

The error was in the *probe*, not the head. `step_conf` appends `block_size`
values per round to `eng.conf_seen` (the whole block, anchor included), while the
gate consumes only the first K. Slicing that list with stride `K = 7` mixes
adjacent rounds together and averages any structure away. The giveaway was in the
numbers and was missed: the sample counts were divisible by 8, not by 7.

Measured correctly, the head was never flat. Per-slot means fall from ≈0.64 to
≈0.27, monotone decreasing, spread 0.49 versus a true 0.39 — better than the
realised profile.

The general lesson, which is why this file is kept: **an aggregate that averages
over slots cannot reveal slot structure, and a broken per-slot measurement is
worse than no measurement, because it looks like evidence.** Two conclusions and
two training runs were built on that artifact before it was caught.
