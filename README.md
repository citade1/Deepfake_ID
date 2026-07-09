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

**Stage 1 — LID adds no signal** (CLIP ViT-B/32, in-distribution, cosine):

| arm | test AUC |
| --- | --- |
| raw CLS | 0.974 |
| LID only | 0.626 |
| raw + LID | 0.975 |

The signal lives in raw feature *directions*, not local density — LID alone is
near-chance and adds nothing to the raw probe. The global ID gap (real ≈28 vs
fake ≈21 at the mid-network ID peak, layer 8/12) does not survive to the
per-sample level.

**Stage 2, JPEG** — the raw detector is compression-robust in *ranking* (AUC
0.97 → ~0.91 at quality 15) but accuracy degrades (0.92 → ~0.72): the signal is
largely semantic, not fragile texture. Cross-generator is the decisive next test.

## Layout

```
experiments/id_profile.py       TwoNN ID per CLIP layer -> picks the LID layer
experiments/stage1_ablation.py  raw vs LID-only vs raw+LID
experiments/jpeg_robustness.py  JPEG robustness sweep
utils/lid_estimator.py          k-NN log-ratio LID + TwoNN estimators
utils/dataloader_helper.py      Tiny-GenImage splits + JPEG collate
```

## Status

Stage 1 done (LID refuted as a per-sample signal); Stage 2 JPEG done.
Cross-generator hold-out and the untrained-backbone ablation are next.
