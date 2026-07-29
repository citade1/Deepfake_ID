# The geometry of cross-generator AI-image detection

When can a frozen vision backbone's features tell real from AI-generated images *without the
backbone ever being trained on fakes* — and when does that break on a generator it has never
seen? A characterization of **why** generator-agnostic detection succeeds or fails, not a
proposed detector.

**TL;DR** — On this dataset and these five backbones, a single closed-form direction
(`Σ⁻¹(μ_fake − μ_real)`, no training) detects held-out generators about as well as a trained
MLP probe on the same features. Where transfer fails it tracks *direction mismatch* rather
than model capacity: GAN-family fakes sit on a nearly separate axis from modern ones.

## Setup

- **Data**: [Community Forensics](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small)
  — ~300 generators in five families, roughly chronological:
  **GAN → PixelDiff → LatentDiff → Flow → Commercial**. Labels come from `model_name`; the
  dataset's own `architecture`/`label` columns are inconsistent.
- **Backbones** (ViT-Base, frozen): CLIP, SigLIP2 *(image–text)*, DINOv2, MAE
  *(self-supervised)*, ViT *(supervised)* — five pretraining objectives, so a result that
  holds across all of them is not a property of CLIP alone.
- **Protocol**: leave-one-generation-out — fit on four families, score on the fifth. Every
  number is on generators unseen during fitting, averaged over 5 seeds.

With `μ_real`, `μ_fake` the class means and `Σ` the within-class covariance, the comparison is
between the **raw direction** `d = μ_fake − μ_real` and the **whitened direction** `w = Σ⁻¹d`
(Fisher's linear discriminant — the same line with content/style variance divided out).

## Results

Detection AUC on **held-out generators**, mean over the five families:

| backbone | pretraining | raw `d` | **whitened `w`** | trained MLP | hardest family (GAN): `w` vs MLP |
| --- | --- | --- | --- | --- | --- |
| CLIP | image–text | 0.918 | **0.983** | 0.983 | **0.977** vs 0.966 |
| SigLIP2 | image–text | 0.915 | **0.973** | 0.977 | 0.973 vs 0.972 |
| DINOv2 | self-sup. | 0.883 | **0.953** | 0.960 | 0.953 vs 0.954 |
| MAE | self-sup. | 0.874 | **0.946** | 0.941 | **0.914** vs 0.899 |
| ViT | supervised | 0.869 | **0.931** | 0.933 | **0.917** vs 0.910 |

*(`cf_whiten.py`; MLP = 2-layer head on the same 768-d features)*

- **Whitening helps, capacity does not.** `d → w` gains +0.06–0.07 everywhere; `w →` MLP
  gains ~nothing (within ±0.007 on average, and `w` wins on GAN for 3 of 5 backbones). On
  this benchmark the useful structure is a whitened mean difference.
- **Not a vision-language effect.** The same pattern holds for self-supervised and supervised
  backbones; image–text pretraining buys a few points of absolute accuracy, not the effect.
- **Local intrinsic dimension is not the signal** (`cf_generational.py`). The project started
  from the hypothesis that fakes are locally simpler. Per-image LID reaches 0.90 AUC against
  0.99 for the raw features, and adding it changes nothing — what separates the classes is
  *where* the two clouds sit, not their local complexity.
- **Failures are directional** (`cf_matrix.py`, `cf_finetune.py`). Transfer inside the modern
  cluster is ~0.98, but a detector fit on modern generators drops to 0.82–0.92 on GAN, whose
  real→fake shift points along a nearly separate axis. LayerNorm fine-tuning *increases*
  representation alignment while *degrading* held-out transfer — a geometric reason universal
  detectors keep the backbone frozen.

<details>
<summary>Transfer matrix and per-detector failure overlap</summary>

Rows = fit on, columns = scored on (CLIP B/16, AUC):

| fit \ scored | GAN | PixDiff | LatDiff | Flow | Comm |
| --- | --- | --- | --- | --- | --- |
| **GAN** | 1.00 | 0.97 | 0.94 | 0.93 | 0.88 |
| **PixDiff** | 0.96 | 1.00 | 0.94 | 0.94 | 0.88 |
| **LatDiff** | 0.90 | 0.93 | 1.00 | 0.92 | 0.98 |
| **Flow** | 0.92 | 0.93 | 0.98 | 0.99 | 0.98 |
| **Comm** | **0.82** | 0.83 | 0.98 | 0.95 | 0.99 |

`cf_gallery.py` (in-distribution, single sample): the whitened direction and the MLP miss
largely the *same* fakes (Jaccard 0.66) — both read the same seen-fake axis — while
Mahalanobis distance-to-real misses different ones (0.11–0.15), weak where they are strong
(PixelDiff) and strong where they are weak (Commercial). Only 5 of 47 errors are missed by
all three. *Small counts — a diagnostic, not a measured claim.*

</details>

## Running it

Features are extracted once per backbone, then re-drawn and analyzed many times
(`--seed` = a fresh draw). Each experiment writes `figures/<name>_<backbone>.json` and a PNG.

```bash
python experiments/build_maps.py                           # one-time: shard metadata
python experiments/prepare_cf.py                           # download shards -> feature caches
python experiments/compose_cf.py --backbone clip --seed 0   # generator-balanced dataset
python experiments/cf_whiten.py  --backbone clip --seeds 5  # headline result
python -m pytest tests/
```

Other analyses share the `--backbone` / `--seeds` flags: `cf_matrix`, `cf_directions`,
`cf_subspace`, `cf_generational`, `cf_twonn`, `cf_ensemble`, `cf_finetune`, `cf_gallery`.
Shared code is in `utils/` — `geometry.py` (the directions), `backbones.py` (feature
extraction), `heads.py` (MLP probe).

One caveat worth flagging in the repo itself: `cf_transfer_predictor.py` tests whether
geometric alignment predicts cross-generator transfer. It does, but for a linear score that
is close to true by construction, so it is kept as a consistency check rather than a result.
