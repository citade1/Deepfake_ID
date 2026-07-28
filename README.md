# The geometry of cross-generator AI-image detection

**Preliminary exploration, not a new detector.** When does a frozen vision backbone's
features let you tell real from AI-generated images *without ever training on fakes*, and
when does that break on a generator you have never seen? The aim is to *characterize* why
generator-agnostic detection succeeds or fails, geometrically.

Backbones are **frozen**; only a small head or a closed-form direction is fit. Data:
[Community Forensics](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small)
(labels and families derived from `model_name`, its reliable field; `architecture`/`label`
are inconsistent). Families span the generations **GAN → PixelDiff → LatentDiff → Flow →
Commercial** (~300 generators). Every number is a mean over 5 seeds unless noted.

## Setup

The study repeats the same analysis across **five ViT-Base backbones with different
pretraining objectives** (`utils/backbones.py`): CLIP and SigLIP2 (vision-language),
DINOv2 and MAE (self-supervised), ViT (supervised). One shard download feeds all five, so
the geometry claims are not an artifact of any single model. Evaluation is always on
generators held out of training (leave-one-generation-out).

## Key findings

1. **The generalizable signal is one "cleaned" direction.** Take the line from the average
   real image to the average fake image, then divide out how much the features naturally
   wobble with content and style (whitening; `w = Σ⁻¹d`, see `utils/geometry.py`). This
   single closed-form direction detects *held-out* generators about as well as a full
   neural-net probe — and beats it on the hardest family (GAN). Extra capacity buys nothing
   out of distribution.

   | backbone | raw direction | **cleaned direction (1-D)** | full MLP probe |
   | --- | --- | --- | --- |
   | CLIP | 0.918 | **0.983** | 0.983 |
   | SigLIP2 | 0.915 | **0.973** | 0.977 |
   | DINOv2 | 0.883 | **0.953** | 0.960 |
   | MAE | 0.874 | **0.946** | 0.941 |
   | ViT (supervised) | 0.869 | **0.931** | 0.933 |

   *(held-out AUC, mean over 5 families; `experiments/cf_whiten.py`)*

2. **It is not specific to vision-language pretraining.** The same effect appears in all
   five backbones — language (CLIP/SigLIP2), self-supervised (DINOv2/MAE), and supervised
   (ViT). Language models get a small absolute edge, not the phenomenon.

3. **Local intrinsic dimension (LID) is not the signal.** A raw probe reaches 0.99; LID
   alone 0.90; combining them adds nothing. The signal is *where* the two groups sit, not
   how locally complex they are (`experiments/cf_generational.py`).

4. **Crossing generator paradigms is the hard part, and GAN is the outlier.** Transfer
   between families is uneven; the weakest case is detecting GAN with a detector trained on
   modern generators, because GAN sits on a nearly separate direction. Fine-tuning the
   backbone aligns representations but *degrades* this cross-family transfer — a geometric
   reason universal detectors keep the backbone frozen (`cf_matrix.py`, `cf_finetune.py`).

**Honest takeaway.** Most of this confirms known or definitional facts (frozen-CLIP
generalization; fine-tuning hurts transfer; the "cleaned direction" is Fisher's linear
discriminant). The least-trivial single fact is that cross-generator detection is a
**1-D whitened phenomenon** across five pretraining objectives. This is a characterization,
not a discovery — see *Status* below.

## Results

**Cross-architecture transfer** (`cf_matrix.py`, CLIP B/16) — rows train, cols test, AUC:

| train \ test | GAN | PixDiff | LatDiff | Flow | Comm |
| --- | --- | --- | --- | --- | --- |
| **GAN** | 1.00 | 0.97 | 0.94 | 0.93 | 0.88 |
| **PixDiff** | 0.96 | 1.00 | 0.94 | 0.94 | 0.88 |
| **LatDiff** | 0.90 | 0.93 | 1.00 | 0.92 | 0.98 |
| **Flow** | 0.92 | 0.93 | 0.98 | 0.99 | 0.98 |
| **Comm** | **0.82** | 0.83 | 0.98 | 0.95 | 0.99 |

Modern→GAN is the soft spot (0.82–0.92); within the modern cluster transfer is ~0.98.

**Manifold intrinsic dimension** (`cf_twonn.py`, TwoNN, layer 12): real ≈ 19.3 vs fakes
16.7–17.7 — a modest, set-level gap (not a per-image detector, consistent with finding 3).

## Pipeline & reproduction

Extract features once, then re-draw and analyze many times (`--seed` = a fresh sample).

```
experiments/build_maps.py             (bootstrap) scan shard metadata -> checkpoints/cf_*.json
experiments/prepare_cf.py             download shards once -> per-backbone feature caches
experiments/compose_cf.py             draw a labelled, generator-balanced dataset (--backbone --seed)
experiments/cf_whiten.py              cleaned direction vs raw vs full MLP, held-out  (headline)
experiments/cf_matrix.py              train x test transfer matrix
experiments/cf_directions.py          per-family real->fake direction geometry
experiments/cf_subspace.py            how many fakeness directions suffice (LOGO)
experiments/cf_generational.py        in-distribution + LOGO (raw / LID / raw+LID)
experiments/cf_twonn.py               manifold intrinsic dimension (TwoNN) per family
experiments/cf_ensemble.py            ensemble of per-family probes vs one pooled probe
experiments/cf_transfer_predictor.py  can geometry predict transfer? (near-definitional; see Status)
experiments/cf_finetune.py            LayerNorm fine-tune (LOGO): representation vs decision
utils/backbones.py                    the five ViT-Base backbones + feature extraction
utils/geometry.py                     the raw (d) and cleaned (w) fakeness directions
utils/cf_data.py                      load a dataset; map model_name -> family
utils/heads.py                        MLP probe: fit + AUC
utils/lid_estimator.py                LID + TwoNN intrinsic-dimension estimators
utils/figs.py                         save results as JSON + 300-dpi PNG
```

```bash
python experiments/build_maps.py                 # one-time: rebuild checkpoints/cf_*.json (metadata only)
python experiments/prepare_cf.py                 # download shards once -> per-backbone feature caches
for bb in clip siglip2 dinov2 mae vit; do
  python experiments/compose_cf.py   --backbone $bb --seed 0
  python experiments/cf_whiten.py    --backbone $bb --seeds 5
  python experiments/cf_matrix.py    --backbone $bb --seeds 5
  python experiments/cf_directions.py --backbone $bb --seeds 5
  python experiments/cf_subspace.py  --backbone $bb --seeds 5
  python experiments/cf_twonn.py     --backbone $bb --seeds 5
done
python experiments/cf_finetune.py --holdout all --seeds 3   # CLIP B/16 only; image pools disk-cached
python -m pytest tests/                            # label-mapping sanity checks
```

Each experiment writes `figures/<name>_<backbone>.json` and a paper-ready
`figures/<name>_<backbone>.png`. Raw run logs go to `log/` (untracked).

## Status — preliminary exploration

This is a **characterization/analysis**, honestly not a discovery: the pieces mostly
confirm known or definitional results, and the "geometry predicts cross-generator transfer"
observation (`cf_transfer_predictor.py`) is close to a restatement of what a linear detector
*is* — kept in the repo as a consistency check, not a result.

The durable *next* question is a genuinely new one — **localization + attribution** ("which
region is fake, and which generator made it"). Pixel-mask localization datasets are now
plentiful (COCO-Inpaint, OpenSDI, GIM, DiffSeg30k, …) and DINOv2-based localization already
exists (DinoLizer); the open niche is **localized attribution** (per-region generator
identity), for which no dataset yet exists — it would have to be built from multi-generator
inpainting. That, not a better passive binary detector, is where representation geometry
could still say something new.
