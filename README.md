# The geometry of AI-generated-image detection in CLIP space

An investigation — not a detector. **Which geometric property of a task-neutral
vision backbone's features separates real from AI-generated images, and how does
it generalize across generator families?** Backbone: **CLIP ViT-B/32**, frozen,
cosine geometry. A small head is the only thing trained. Every number below is a
mean over 5 seeds unless marked † (single-seed diagnostic).

## Key findings

1. **Local intrinsic dimension (LID) is not the signal.** A raw CLS linear probe
   reaches AUC 0.98; LID alone 0.83; raw+LID ≈ raw. Adding LID to a probe helps
   nothing — and a confidence-gated "selective LID" detector, which helped
   slightly on the pilot, *hurts* on Community Forensics (net −105 of 2680 gated
   cases). LID is refuted, in- and out-of-distribution.

2. **Cross-architecture generalization is asymmetric.** Train a probe on one
   generator family, test on another. Train-**modern** (latent-diffusion / flow /
   commercial) → test-**GAN** collapses to **0.726 ± 0.021**; train-**old**
   (GAN / pixel-diffusion) → test-**modern** holds at **0.886 ± 0.032**. It is
   architecture distance, not chronology, that governs transfer.

3. **The modern "fakeness" signal is essentially one-dimensional.** A single
   shared direction in CLIP space detects held-out latent-diffusion / flow /
   commercial families at ~0.93–0.96 — as well as the full 768-d probe (0.96–0.97).
   GAN sits on a near-orthogonal axis with a *smaller* real→fake shift (0.19 vs
   ~0.30), so restricting to the modern axis drops GAN detection to 0.60.

4. **The asymmetry is fully explained by geometry:** modern fakes are a large
   shift on a shared axis (loud → caught even by a mismatched detector); GAN is a
   small shift in a distinct direction (quiet + off-axis → missed by
   modern-trained detectors).

**Takeaway.** Within a generator paradigm the detectable signal is trivially
low-dimensional, so no clever inductive bias is needed. Across paradigms the
directions are near-orthogonal and cannot be synthesized from one another — you
need *examples* of the new architecture (data), not a bias. This is a
mechanistic account of *why* generator-agnostic detection succeeds or fails, not
a new detector.

## Community Forensics results (the study)

Data: [Community Forensics](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small)
(labels and families derived from `model_name`, the reliable field — the dataset's
`architecture`/`label` columns are inconsistent). Families span the generations
GAN → PixelDiff → LatentDiff/SD → Flow/FLUX → Commercial (~300 generators).

**Cross-architecture transfer matrix** (`experiments/cf_matrix.py`,
`figures/cf_matrix.png`) — rows train, cols test, raw-probe AUC:

| train \ test | GAN | PixDiff | LatDiff | Flow | Comm |
| --- | --- | --- | --- | --- | --- |
| **GAN** | 0.94 | 0.96 | 0.90 | 0.88 | 0.84 |
| **PixDiff** | 0.87 | 0.99 | 0.94 | 0.90 | 0.87 |
| **LatDiff** | **0.73** | 0.89 | 0.99 | 0.92 | 0.98 |
| **Flow** | **0.74** | 0.87 | 0.97 | 0.99 | 0.99 |
| **Comm** | **0.72** | 0.81 | 0.98 | 0.96 | 0.99 |

**Direction geometry** (`experiments/cf_directions.py`) — per family the
real→fake shift `d_F = μ_fake(F) − μ_real`. GAN is isolated (mean off-diagonal
cosine 0.58; GAN↔modern ≈ 0.30–0.43) and has the smallest shift; one shared axis
(mean direction) detects each family:

| | GAN | PixDiff | LatDiff | Flow | Comm |
| --- | --- | --- | --- | --- | --- |
| shift magnitude | **0.193** | 0.254 | 0.297 | 0.289 | 0.309 |
| 1-D shared-axis AUC | **0.745** | 0.911 | 0.957 | 0.957 | 0.956 |

**Subspace generalization, leave-one-generation-out** (`experiments/cf_subspace.py`):

| held-out | full 768-d | 4-D subspace | 1-D shared axis |
| --- | --- | --- | --- |
| GAN | 0.811 ± 0.027 | 0.646 ± 0.010 | 0.600 ± 0.016 |
| PixelDiff | 0.929 ± 0.006 | 0.894 ± 0.011 | 0.741 ± 0.144 |
| LatentDiff | 0.975 ± 0.002 | 0.954 ± 0.005 | 0.934 ± 0.072 |
| Flow | 0.961 ± 0.006 | 0.955 ± 0.003 | 0.928 ± 0.061 |
| Commercial | 0.970 ± 0.004 | 0.963 ± 0.002 | 0.944 ± 0.030 |

## Pilot (Tiny-GenImage)

The pilot on Tiny-GenImage (7 generators) motivated the move to CF. It already
showed LID adds no signal (raw 0.973, LID 0.626, raw+LID 0.975), partial
cross-generator generalization (leave-one-out raw 0.906 ± 0.003), and that a
random-init CLIP barely separates (0.61) — the signal is training-induced
semantics. A "selective LID" gain on hard cases (+0.067 AUC on the hardest 10%)
appeared here but **did not replicate on CF**, which is why LID is finally
refuted. Pilot scripts are in git history; the reusable pipeline below is CF.

> **A trap worth stating.** Li et al.,
> [*Measuring the Intrinsic Dimension of Objective Landscapes*](https://arxiv.org/abs/1804.08838),
> measure ID in **parameter space** (smallest weight-subspace that still solves a
> task) — not the **data-representation** ID that LID / TwoNN / Ansuini measure.
> Same name, different space; the two do not transfer directly.

## Pipeline & reproduction

Extract features once, then re-draw and analyse many times (a `--seed` is a fresh
sample → robustness).

```
experiments/build_maps.py       (bootstrap) scan shard metadata -> checkpoints/cf_*.json maps
experiments/prepare_cf.py       download CF shards -> per-shard CLIP feature cache (resumable)
experiments/compose_cf.py       draw a labelled, generator-balanced dataset  (--seed)
experiments/cf_generational.py  in-distribution + leave-one-generation-out (raw / LID / raw+LID)
experiments/cf_matrix.py        train x test architecture generalization matrix
experiments/cf_directions.py    per-family real->fake direction geometry
experiments/cf_subspace.py      how many fakeness directions suffice (LOGO)
experiments/cf_ensemble.py      ensemble of per-family probes vs one pooled probe
experiments/cf_twonn.py         global manifold ID (TwoNN) of real vs each family
experiments/cf_finetune.py      LayerNorm fine-tune (LOGO): representation vs decision geometry
utils/cf_data.py                load a composed dataset
utils/heads.py                  MLP probe: fit + AUC
utils/lid_estimator.py          k-NN log-ratio LID + TwoNN estimators
utils/figs.py                   save results as JSON + 300-dpi PNG
```

The study runs across five ViT-Base backbones (`utils/backbones.py`: clip, siglip2,
dinov2, mae, vit) — one shard download feeds all five. Analyses take `--backbone`;
fine-tuning uses CLIP B/16 only.

```bash
python experiments/build_maps.py                 # one-time: rebuild checkpoints/cf_*.json (reads metadata only)
python experiments/prepare_cf.py                 # download shards once -> per-backbone feature caches
for bb in clip siglip2 dinov2 mae vit; do
  python experiments/compose_cf.py  --backbone $bb --seed 0
  python experiments/cf_matrix.py   --backbone $bb --seeds 5
  python experiments/cf_directions.py --backbone $bb --seeds 5
  python experiments/cf_subspace.py --backbone $bb --seeds 5
  python experiments/cf_twonn.py    --backbone $bb --seeds 5
done
python experiments/cf_finetune.py --holdout all --seeds 3  # CLIP B/16 only; pools disk-cached
```

Each experiment writes `figures/<name>_<backbone>.json` and a paper-ready
`figures/<name>_<backbone>.png`. Raw run logs go to `log/` (untracked).

## Status — preliminary exploration

Honest assessment: this is a **characterization/analysis**, not a discovery. The pieces
mostly confirm known or definitional facts (frozen-CLIP generalization; fine-tuning hurts
transfer; generative-vs-discriminative = Fisher LDA; the cross-generator "transfer
predictor" is near-definitional for a linear detector). The least-trivial single fact:
across five pretraining objectives (CLIP / SigLIP2 / DINOv2 / MAE / ViT), cross-generator
detection is a **1-D whitened phenomenon** — one closed-form axis `w = Σ⁻¹ d` matches a
full MLP on held-out generators and beats it on the hardest family (GAN); extra capacity
gives no OOD benefit. See `report.md` for the full journey and `notes.txt` for running notes.

The durable *next* direction is a genuinely new question — **localization + attribution**
("which region, which generator"; DINOv2-central, needs region-mask data) — not a better
passive binary detector.
