# The geometry of cross-generator AI-image detection

When can a frozen vision backbone's features tell real from AI-generated images *without the
backbone ever being trained on fakes* — and when does that break on a generator it has never
seen? This is a study of **why** generator-agnostic detection succeeds or fails, measured
geometrically, rather than a proposal for a new detector.

**TL;DR** — On this dataset and these five backbones, a **single closed-form direction**
(`Σ⁻¹(μ_fake − μ_real)`, no training) detects held-out generators about as well as a trained
MLP probe on the same features. Where transfer does fail, it tracks *direction mismatch*
rather than model capacity: GAN-family fakes sit on a nearly separate axis from modern ones.
Whether this reflects something general about frozen features or something specific to
Community Forensics is not settled by these experiments.

## Setup

- **Data**: [Community Forensics](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small),
  ~300 generators grouped into five families in rough chronological order —
  **GAN → PixelDiff → LatentDiff → Flow → Commercial**. Labels and families come from
  `model_name`; the dataset's own `architecture`/`label` columns are inconsistent.
- **Backbones** (all ViT-Base, frozen): CLIP and SigLIP2 *(image–text contrastive)*,
  DINOv2 and MAE *(self-supervised)*, ViT *(ImageNet-supervised)*. Running every analysis on
  all five is what separates "a property of CLIP" from "a property of strong ViT features".
- **Protocol**: **leave-one-generation-out** — a detector is fit on four families and scored
  on the fifth, so every number below is on generators never seen during fitting.
  Mean over 5 seeds.

### The two directions being compared

Let `μ_real`, `μ_fake` be class means in feature space and `Σ` the within-class covariance.

| | definition | intuition |
| --- | --- | --- |
| **raw direction** `d` | `μ_fake − μ_real` | the straight line between the two class means |
| **cleaned direction** `w` | `Σ⁻¹d` | the same line after dividing out the variance that content and style contribute — i.e. Fisher's linear discriminant, computed in closed form, no gradient steps |

## Results

Detection AUC on **held-out generators** (mean over the five families):

| backbone | pretraining | raw `d` | **cleaned `w`** | trained MLP | hardest family (GAN): `w` vs MLP |
| --- | --- | --- | --- | --- | --- |
| CLIP | image–text | 0.918 | **0.983** | 0.983 | **0.977** vs 0.966 |
| SigLIP2 | image–text | 0.915 | **0.973** | 0.977 | 0.973 vs 0.972 |
| DINOv2 | self-sup. | 0.883 | **0.953** | 0.960 | 0.953 vs 0.954 |
| MAE | self-sup. | 0.874 | **0.946** | 0.941 | **0.914** vs 0.899 |
| ViT | supervised | 0.869 | **0.931** | 0.933 | **0.917** vs 0.910 |

*(`experiments/cf_whiten.py`; MLP = 2-layer head on the same 768-d features)*

Reading the table:

1. **Whitening helps; extra capacity does not.** `d → w` gains +0.06–0.07 AUC on every
   backbone, while `w →` MLP gains ~nothing: the closed-form direction stays within ±0.007
   of the trained probe on average and comes out ahead on GAN — the hardest held-out
   family — for 3 of 5 backbones. So on this benchmark the useful structure is captured by
   a whitened mean difference, and a 2-layer head adds no out-of-distribution robustness on
   top of it. (Whether a larger probe or more training data would change this is untested.)
2. **Not specific to vision-language pretraining.** The same pattern appears in
   self-supervised and supervised backbones, so it is not an artifact of CLIP's image–text
   objective. Image–text models do hold a few points of absolute accuracy over the others.

Two supporting results:

- **Local intrinsic dimension is not the signal** (`cf_generational.py`). The project started
  from the hypothesis that fakes occupy a locally simpler manifold. Measured per-image LID
  reaches 0.90 AUC against 0.99 for the raw features, and adding it to the features changes
  nothing. What separates the classes is *where the two clouds sit*, not how locally complex
  they are. (Set-level intrinsic dimension does differ — real ≈ 18.5 vs 9.7–18.0 across the
  fake families by TwoNN — but that is a property of the set, not a per-image cue.)
- **Failures are a direction problem** (`cf_matrix.py`, `cf_finetune.py`). Transfer within the
  modern cluster (LatentDiff/Flow/Commercial) is ~0.98, but a detector fit on modern
  generators drops to 0.82–0.92 on GAN, whose real→fake shift points along a nearly separate
  axis. Fine-tuning the backbone (LayerNorm-only) *increases* representation alignment yet
  *degrades* held-out transfer — a geometric account of why universal detectors keep the
  backbone frozen.

<details>
<summary>Cross-family transfer matrix and per-detector failure overlap</summary>

Rows = family fit on, columns = family scored on (CLIP B/16, AUC):

| fit \ scored | GAN | PixDiff | LatDiff | Flow | Comm |
| --- | --- | --- | --- | --- | --- |
| **GAN** | 1.00 | 0.97 | 0.94 | 0.93 | 0.88 |
| **PixDiff** | 0.96 | 1.00 | 0.94 | 0.94 | 0.88 |
| **LatDiff** | 0.90 | 0.93 | 1.00 | 0.92 | 0.98 |
| **Flow** | 0.92 | 0.93 | 0.98 | 0.99 | 0.98 |
| **Comm** | **0.82** | 0.83 | 0.98 | 0.95 | 0.99 |

**Which images each detector misses** (`cf_gallery.py`, in-distribution, single sample).
The cleaned direction and the MLP miss largely the *same* fakes (Jaccard 0.66) — expected,
since both read the same seen-fake axis. Mahalanobis distance-to-real misses *different* ones
(Jaccard 0.11–0.15): it is weak where they are strong (PixelDiff) and strong where they are
weak (Commercial), so the two families of method are complementary. Only 5 of 47 errors are
missed by all three, and those are photorealistic scenes with no visible artifact.
*Error counts are small — a diagnostic, not a measured claim.*

</details>

## Running it

Features are extracted once per backbone, then re-drawn and analyzed many times
(`--seed` = a fresh draw). Every experiment writes `figures/<name>_<backbone>.json`
alongside a 300-dpi PNG.

```bash
python experiments/build_maps.py                          # one-time: shard metadata -> checkpoints/cf_*.json
python experiments/prepare_cf.py                          # download shards once -> per-backbone feature caches
python experiments/compose_cf.py --backbone clip --seed 0  # draw a generator-balanced dataset
python experiments/cf_whiten.py  --backbone clip --seeds 5 # headline result
python -m pytest tests/                                   # label-mapping sanity checks
```

Other analyses take the same `--backbone` / `--seeds` flags: `cf_matrix` (transfer matrix),
`cf_directions` and `cf_subspace` (direction geometry), `cf_generational` (LID),
`cf_twonn` (intrinsic dimension), `cf_ensemble`, `cf_finetune` (LayerNorm fine-tune),
`cf_gallery` (failure cases). Shared code lives in `utils/`: `geometry.py` defines `d` and
`w`, `backbones.py` the five feature extractors, `heads.py` the MLP probe.

## Scope

A characterization study, not a proposed detector. One observation that first looked like a
result — that geometric alignment predicts cross-generator transfer
(`cf_transfer_predictor.py`) — is close to a restatement of what a linear detector *is*, so
it is kept as a consistency check rather than a finding.

Follow-ups were scoped against the 2025–26 literature before committing to any: localization,
agentic detection, explanation-faithfulness verification, and whole-image generator
attribution are all actively worked on, the last including its geometric formulation
(Riemannian-Geometric Fingerprints generalizes the mean-difference idea used here). The
remaining gap is **localized attribution** — which region *and* which generator — which needs
a dataset that does not exist yet.
