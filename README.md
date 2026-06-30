# Deepfake Detection via Multi-Layer Local Intrinsic Dimensionality

Detect deepfake images and videos by measuring how the **local intrinsic
dimension (LID)** of a frozen Vision Transformer's features changes across its
depth — instead of fine-tuning the backbone on visual artifacts.

## Results

> **Status: pre-training.** The pipeline runs end to end, but the classifier has
> not yet been trained on the full dataset, so no metrics are reported yet.
> In-distribution test accuracy / AUC will be added here after the first run.

## Quickstart

Requires **Python 3.10**. `dlib` (used for face detection at inference) needs
CMake and a C++ toolchain — on macOS: `brew install cmake`.

```bash
pip install -r requirements.txt

# The training dataset is gated — accept its terms on the HF page, then:
huggingface-cli login
```

**Train** the MLP head (the ViT backbone stays frozen):

```bash
python train.py                # full run, early stopping on validation AUC
python train.py --epochs 1     # quick smoke test
```

This saves `checkpoints/lid_classifier.pth` (the trained head + the frozen LID
reference bank).

**Infer** on a folder of images/videos:

```bash
# put .jpg/.png/.mp4/.avi files (no sub-folders) in ./data
python inference.py            # writes predictions to submission.csv
```

> Runs on CPU as-is. Training only updates a small MLP — the heavy ViT is
> forward-only — so CPU is workable, though a CUDA/MPS device is faster.

## How it works

Most detectors fine-tune a classifier on visual features, asking *what* a face
looks like. This project instead asks *how the geometry of the representation is
structured*: my hypothesis is that GAN/diffusion-generated images occupy a
different manifold than real photographs, measurable as a shift in **local
intrinsic dimension** — real images tending toward higher-dimensional, more
complex neighborhoods, generated ones collapsing to lower-dimensional, more
regular ones.

```
ViT (frozen)  →  start / middle / end layer CLS tokens
                        ↓ LID vs. fixed real-image reference bank (k-dim each)
              concat (3k-dim) → MLP → Real / Fake
```

- **LID estimation** uses a *k*-NN log-ratio (hill) estimator (Amsaleg et al.,
  2015); TwoNN (Facco et al., 2017) provides diagnostic global-dimension logs.
  The per-point log-ratio vector is what the shallow MLP discriminates.
- **Reference bank.** LID is a property of a point relative to a data manifold,
  so it is measured against a *fixed* bank of real-image features — a held-out
  subset of real images, built once and frozen, disjoint from the images being
  classified. This makes an image's LID feature a deterministic function of the
  image alone (single-image inference included), and keeps the bank part of the
  *feature transform* rather than the trained classifier.
- **Backbone.** `prithivMLmods/Deep-Fake-Detector-v2-Model`, fully frozen; only
  the MLP head trains. Data is
  [`prithivMLmods/Deepfake-vs-Real-60K`](https://huggingface.co/datasets/prithivMLmods/Deepfake-vs-Real-60K)
  (~30k real / ~30k fake, mapped to `0=Real, 1=Fake`).

## Project layout

```
train.py                    train the MLP head, build + save the reference bank
inference.py                face detection (dlib), video sampling, batched LID inference
utils/model.py              MultiLayerLIDModel: frozen ViT + LID features + MLP
utils/lid_estimator.py      k-NN log-ratio LID and TwoNN estimators
utils/dataloader_helper.py  HF dataset loading, label mapping, train/val/test/reference splits
```

## Limitations & future work

- **Cross-generator generalization is unverified.** The core idea —
  manifold-level, generator-agnostic detection — needs a leave-one-generator-out
  evaluation (train on a subset of generators, test on held-out ones, against a
  CLS-feature baseline). That experiment is not yet run; current scope is
  in-distribution only.
- The reference bank is drawn from a single real-image distribution; its effect
  on calibration across domains is not yet characterized.

## Status

Active development.
