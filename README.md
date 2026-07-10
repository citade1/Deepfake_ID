# Representation Geometry for Real-vs-Generated Image Detection

An investigation, not a product: **can the geometry of a task-neutral vision
backbone's representations separate real from AI-generated images — and if so,
which geometric property carries the signal, and does it generalize?**

Backbone: **CLIP ViT-B/32** (never trained on real/fake labels). Data:
**Tiny-GenImage** (8 generators, per-image generator tag). The value is the
investigation — what works, what doesn't, and why — each experiment motivated
by a paper.

## Plan

**Stage 1 — Does LID add signal?**
Baseline: a probe on the frozen CLS features (the raw representation). Augmented:
the same probe **plus a local intrinsic dimension (LID) signal** taken at the
intermediate layer where representation ID peaks (Ansuini et al.). If LID adds
nothing over raw features, it is not the geometric descriptor this task needs.

**Stage 2 — Failure modes (clean probe vs. LID model).**
1. **JPEG / resize robustness** — the "it's just texture" test (Pope et al.).
2. **Cross-generator hold-out** — train on some generators, test on unseen ones.
3. **Trained vs. untrained backbone ablation** — random-init ViT; if geometry
   still separates, we are reading raw statistics, not learned semantic
   structure (Ansuini et al.).

**Stage 3 — Subspace reinterpretation (Effort-influenced).**
LID summarizes *local density* via nearest-neighbor distance ratios. The
real/fake signal, though, may live in a low-dimensional *subspace* of the
representation (a near-perfect linear probe is early evidence). Stage 3 explores
an intrinsic-dimension notion framed as **subspace structure** rather than
neighbor density — in the spirit of *Effort*, not a reimplementation of it.

> **Two different "intrinsic dimensions" (a trap to avoid).** Li et al.,
> [*Measuring the Intrinsic Dimension of Objective Landscapes*](https://arxiv.org/abs/1804.08838),
> measure ID in **parameter space** — the smallest random weight-subspace in
> which a network still solves a task. That is a property of the *objective
> landscape*, not of the data manifold. LID / TwoNN / Ansuini measure ID of the
> **data representation** via neighbor geometry. Same name, different space. The
> subspace intuition motivating Stage 3 comes from the former and transfers by
> analogy (few dimensions suffice), not by direct application.

## Findings

Setup: frozen CLIP ViT-B/32, Tiny-GenImage, cosine geometry. AUC means over
**5 seeds** unless marked † (single-seed diagnostic). Each table names the
experiment that produced it; raw logs are kept under `log/` (untracked).

**ID profile across depth** † (`experiments/id_profile.py`) — Ansuini's
hunchback replicates on CLIP, and real images consistently occupy a *higher*
local dimension than fakes:

| layer (of 12) | 1 | 3 | **7 (peak)** | 8 | 9 | 12 |
| --- | --- | --- | --- | --- | --- | --- |
| TwoNN ID, all | 24.1 | 15.2 | **26.4** | 25.1 | 23.8 | 18.8 |
| ID real / fake | 26.1 / 19.5 | 15.0 / 13.1 | 28.7 / 22.6 | 28.7 / 21.3 | 27.2 / 19.7 | 19.9 / 15.2 |

LID is taken at layer 7 (ID peak); the raw probe at layer 12 (standard readout).

**Stage 1 — does LID add signal?** (`experiments/stage1_ablation.py`, error bars
`experiments/multiseed.py`) — no. The signal lives in raw feature *directions*,
not local density; the population-level ID gap above does not survive to
per-sample:

| probe input | test AUC (in-dist) |
| --- | --- |
| raw CLS (768-d) | 0.9733 ± 0.0003 |
| LID only (20-d) | 0.6257 ± 0.0002 |
| raw + LID | 0.9748 ± 0.0003 |

**Stage 2.1 — JPEG robustness** (`experiments/jpeg_robustness.py`, error bars
`experiments/multiseed.py`) — ranking survives compression (largely semantic
signal, not fragile texture); the 0.5-threshold accuracy degrades
(0.92 → ~0.72 †), so calibration, not separability, is what compression breaks:

| JPEG quality | clean | 95 | 75 | 50 | 30 | 15 |
| --- | --- | --- | --- | --- | --- | --- |
| raw AUC | 0.973 | 0.927 | 0.956 | 0.946 | 0.918 | 0.910 |

**Stage 2.2 — cross-generator, leave-one-generator-out** †
(`experiments/cross_generator.py`; mean is 5-seed from
`experiments/selective_lid.py`) — training on 6 generators, testing on the 7th.
Partial generalization: a stable mean of **0.906 ± 0.003** vs 0.974 in-dist,
with the architecturally distinct generators (VQDM, ADM) hardest:

| held-out | ADM | BigGAN | GLIDE | Midjourney | SD15 | VQDM | Wukong | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw AUC | 0.852 | 0.970 | 0.992 | 0.913 | 0.934 | **0.790** | 0.884 | **0.906** |
| raw+LID AUC | 0.859 | 0.979 | 0.993 | 0.911 | 0.936 | 0.790 | 0.889 | 0.908 |

**Stage 2.3 — trained vs. untrained backbone**
(`experiments/untrained_ablation.py`, error bars `experiments/multiseed.py`) —
a random-init CLIP barely separates: the signal is training-induced semantics,
not raw image statistics (the ~0.11-above-chance residual is architectural):

| backbone | raw | LID | raw+LID |
| --- | --- | --- | --- |
| trained CLIP | 0.9733 ± 0.0003 | 0.6257 ± 0.0002 | 0.9748 ± 0.0003 |
| random init | 0.6143 ± 0.0026 | 0.5385 ± 0.0018 | 0.6201 ± 0.0014 |

**Stage 2.4 — selective LID** (`experiments/selective_lid.py`) — LID's residual
signal is real but *localized*: out-of-distribution it concentrates on the
raw-uncertain tail, yet does not move the detector-level numbers. A
confidence-gated detector (`utils/selective.py`, raw+LID re-scores the least
confident 20%) stays within noise overall:

| metric (held-out generators, 5 seeds) | value |
| --- | --- |
| AUC gain on hardest 10% (by raw confidence) | **+0.067 ± 0.020** |
| AUC gain on hardest 20% | +0.039 ± 0.012 |
| deployed AUC, raw → selective | 0.906 → 0.908 (± 0.002, within noise) |
| deployed accuracy, raw → selective | 0.824 → 0.828 (± 0.003) |

## Layout

```
experiments/id_profile.py       TwoNN ID per CLIP layer -> picks the LID layer
experiments/stage1_ablation.py  raw vs LID-only vs raw+LID
experiments/jpeg_robustness.py  JPEG robustness sweep
experiments/cross_generator.py  leave-one-generator-out generalization
experiments/untrained_ablation.py  random-init vs trained backbone
experiments/selective_lid.py    confidence-gated selective LID (OOD, multi-seed)
experiments/multiseed.py        multi-seed error bars for fixed-feature runs
utils/lid_estimator.py          k-NN log-ratio LID + TwoNN estimators
utils/selective.py              confidence-gated SelectiveLID detector
utils/dataloader_helper.py      Tiny-GenImage splits + JPEG collate
```

## Status

Stage 1 done (LID refuted globally, small signal on low-confidence cases);
Stage 2 done and multi-seeded (JPEG, cross-generator LOO, untrained backbone,
selective LID). Stage 3 (Effort-style subspace) next.
