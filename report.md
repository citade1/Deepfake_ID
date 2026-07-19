# Representation Geometry of AI-Generated-Image Detection — Working Report

A concrete log of the question, the intuition behind each experiment, the design, the
result, and what it means. Terser running notes live in `notes.txt`; this is the
detailed version. Two phases: an **exploratory phase** on frozen CLIP ViT-B/32 (now
treated as preliminary), and a **consolidated thesis** on ViT-B/16 across five backbones.

---

## 0. Question and stance

**What geometric property of a frozen vision backbone's features separates real from
AI-generated images, and what governs whether that signal generalizes to *unseen*
generators?** Priority: understanding / inductive bias over leaderboard performance.
Passive detection is a Bitter-Lesson arms race; the durable value is a *mechanistic*
account (and, downstream, attribution).

Backbones are **frozen**; only a small head or a closed-form axis is fit. Data:
**Community Forensics** (`OwensLab/CommunityForensics-Small` + `-Eval`), labels and
generator families derived from the reliable `model_name` field. Families span the
generations **GAN → PixelDiff → LatentDiff → Flow → Commercial**.

---

## 1. Setup

- **Features**: unit-normalized CLS token (mean-pool for SigLIP) at layers 7 and 12.
  All ViT-Base backbones give 768-d; ViT-L gives 1024-d. Within a backbone the geometry
  is compared on identical images (same seed draws the same rows across backbones).
- **Pipeline**: `build_maps.py` (shard metadata) → `prepare_cf.py` (one download,
  features for every backbone) → `compose_cf.py` (balanced draw) → analyses.
- **Backbones** (`utils/backbones.py`), chosen to span *training objectives*:
  CLIP-B/16 (lang. contrastive), SigLIP2-B/16 (sigmoid lang.), DINOv2-B/14 (self-sup
  distillation), MAE-B/16 (self-sup masked reconstruction), ViT-B/16 (supervised).
  Plus CLIP-L/14 as a headline. The point is a *generality* axis, not "best backbone".

---

## 2. Experiment sequence

### 2.1 LID is not the signal  (`cf_generational.py`)
- **Intuition**: if fakes lie on a lower-dimensional / differently-curved manifold, a
  per-sample **local intrinsic dimension** (k-NN log-ratio) should separate real/fake.
- **Result (B/32)**: raw CLS probe 0.98; LID alone 0.83; raw+LID ≈ raw. LID adds nothing.
- **Meaning — the key reframe**: LID/TwoNN/Ansuini measure **within-manifold** geometry
  (local density, curvature), which is nonlinear and PCA-invisible. The detection signal
  is a **between-manifold displacement** `d = μ_fake − μ_real`, which is *linear* and
  ~1-dimensional. Different quantities. This within/between distinction organizes
  everything that follows: the signal lives in a *linear offset*, not in *manifold shape*.

### 2.2 The fakeness signal is ~1-dimensional  (`cf_directions.py`)
- **Intuition**: if it is a between-manifold offset, one direction should carry it.
- **Result (B/32)**: a single shared axis (mean of per-family `d_F`) detects held-out
  modern families at 0.93–0.96 (≈ full 768-d probe). GAN sits on a near-orthogonal axis
  (cosine 0.30–0.43 to modern) with a *smaller* shift, so the modern axis drops GAN to ~0.60.
- **Meaning**: "fakeness" is essentially 1-D *within a generator paradigm*; across
  paradigms the directions are near-orthogonal and cannot be synthesized from one another.

### 2.3 Cross-architecture transfer is asymmetric  (`cf_matrix.py`, `cf_directions.py`)
- **Result (B/32)**: train-modern→test-GAN ≈ 0.73; train-old→test-modern ≈ 0.89. The
  asymmetry = shift **magnitude** (modern loud, GAN quiet) × **direction** (orthogonal).
- **Meaning**: it is architecture *distance*, not chronology, that governs transfer. A
  detector generalizes to B iff B's shift lies along the direction the detector reads.
  (This sentence is the seed of the transfer-predictor thesis, §2.9.)

### 2.4 How many directions generalize?  (`cf_subspace.py`)
- Leave-one-generation-out: full-768-d vs a ≤4-D family-direction subspace vs a 1-D
  shared axis. Modern families: 1-D ≈ full. GAN: even the subspace can't reach it.
- **Meaning**: within seen diversity the effective dimension is tiny; the failure is a
  *missing direction*, which no amount of seen data along other directions supplies.

### 2.5 Ensemble ≠ improvement  (`cf_ensemble.py`)
- **Idea tested** (an ETRI hypothesis): combine per-generator probes to beat pooling.
- **Result (B/32)**: pooled 0.977 ≈ ensemble-max 0.976 ≈ ensemble-mean 0.974. No gain.
- **Meaning**: a high-capacity pooled MLP already spans all training directions; a
  decision-level ensemble is the same information. Cleverness doesn't beat capacity+data.

### 2.6 Manifold ID gap is real but weak  (`cf_twonn.py`)
- Global TwoNN ID (a *set* statistic): real ≈ 19.3 vs fakes 16.7–17.7 at layer 12
  (Commercial ≈ real, most diverse). A modest between-class ID gap.
- **Meaning**: consistent with §2.1 — the signal is in manifold *structure at the set
  level*, not in a per-image feature, so it is not directly a detector.

### 2.7 Fine-tuning: representation aligns, decision does not  (`cf_finetune.py`)
- **Setup**: LayerNorm-only fine-tune on real-vs-fake, hold one family out, source-balanced
  real (FFHQ faces so GAN stays hard), with early stopping. Frozen vs tuned readouts.
- **Q1 (generative vs discriminative axis)** — the mechanism. Fisher LDA: the boundary
  normal is `w ∝ Σ⁻¹ d`, i.e. the generative axis `d` **whitened** by within-class
  covariance. Isotropic Σ → w = d; anisotropic Σ → w rotates off d.
  Result: after tuning, held-out families' **generative** alignment cos(d_held, d_seen)
  rises (0.65→0.9) but their **discriminative** alignment cos(d_held, w) *falls*
  (0.35→0.15). The representation "aligns" on the generative axis while moving *off* the
  axis the boundary actually reads.
- **Q2 (distance-to-real)**: a frozen Mahalanobis-to-real detector already catches
  held-out families at 0.85–0.99 with no fake training; fine-tuning *destroys* this
  (drops to 0.6–0.75) because it distorts the real distribution to serve the seen boundary.
- **Meaning — a mechanistic account of why universal detectors freeze the backbone**:
  fine-tuning specializes the whole geometry (boundary *and* real covariance) to seen
  generators; transfer degrades under both readouts even as unseen fakes superficially
  align. Overfitting is systemic, not one boundary.
- **Caveat**: tuned values are MPS-nondeterministic (large seed variance); frozen values
  are deterministic. The robust claim is directional (tuning does not help transfer).

### 2.8 Whitening ablation — nuisance removal that works  (`cf_whiten.py`)
- **Intuition** (from Q1): the useful "remove unnecessary directions" is not "keep only
  w" (that equals a linear probe) but **whitening** — divide out the high-variance
  *content* directions (Σ⁻¹) so the low-variance fakeness signal separates. That is
  exactly `w = Σ⁻¹ d`.
- **Result (B/16, 5 seeds, clean real split)**: the whitened 1-D axis beats the raw 1-D
  axis on **every** held-out family, largest on the hard ones:
  CLIP GAN 0.907→0.977, PixelDiff 0.870→0.986; DINOv2 GAN 0.871→0.953; MAE GAN 0.817→0.914.
  Average gain ≈ **+0.06–0.07 held-out AUC across all backbones**. (vs a full MLP: see §3.)
- **Meaning**: a concrete, interpretable generalization gain — one whitened direction.

### 2.9 Transfer-failure predictor — the unifying claim  (`cf_transfer_predictor.py`)
- **Claim**: the *label-free* geometric alignment **cos(d_B, w_A)** — how a new
  generator B's shift lines up with the discriminative axis a detector trained on A
  reads — predicts the actual A→B transfer AUC, and flags the failures.
- **Result (B/16, 5 seeds, off-diagonal transfer pairs)**:

  | backbone | pretraining | Spearman (w_A) | Spearman (d_A, baseline) |
  |---|---|---|---|
  | CLIP | lang. contrastive | 0.81 | 0.77 |
  | SigLIP2 | sigmoid lang. | 0.93 | 0.80 |
  | DINOv2 | self-sup distill | 0.91 | 0.71 |
  | MAE | self-sup masked | 0.75 | 0.86 |
  | ViT | supervised | 0.80 | 0.80 |

  The predictor works across **all five** backbones. The whitened axis w predicts better
  than the raw axis d for 3/5 (lang. + DINOv2), comparable for MAE/ViT.
- **Meaning**: geometry predicts, without labels, whether a detector will generalize to a
  new generator.
- **HONEST CAVEAT (this is not a discovery).** For a linear scorer `s(x)=w_A·x`, generator
  B's mean score is `≈ w_A·d_B ∝ cos(d_B, w_A)·|d_B|`, so AUC(A→B) is monotone in
  cos(d_B, w_A) essentially *by construction*. The predictor is close to a restatement of
  "what a linear detector is" / "overfitting lives in the boundary normal w", not an
  empirical finding. It functions as a **consistency check** on the geometry, not a result.
  The residual non-trivial bit (a *linear* w predicts a *nonlinear* MLP's transfer) is the
  same message as §2.8/§3: nonlinearity/capacity add nothing to OOD.

### 2.10 Cross-backbone generality — does it need vision-language pretraining?
- Both the predictor (§2.9) and the whitening gain (§2.8) **replicate across all five
  backbones** — language (CLIP/SigLIP2), self-supervised (DINOv2/MAE), supervised (ViT).
- Language pretraining gives a modest *absolute-detectability* edge (whitened GAN ≈ 0.97
  for lang. vs 0.91–0.95 for the rest) but does **not** create the phenomenon.
- **Meaning**: the whitened-fakeness-axis story is a **general property of strong ViT
  representations**, not an artifact of CLIP's image-text alignment. This preempts the
  obvious "isn't this just CLIP?" objection.

---

## 3. What this actually is (honest reassessment)

This body of work is a **characterization / analysis**, not a discovery. Almost every
piece confirms something already known or definitional: frozen-CLIP generalization
(Ojha 2023), fine-tuning-hurts-transfer (the motivation of frozen detectors),
generative-vs-discriminative = Fisher LDA (textbook), and the transfer predictor is
near-definitional for a linear detector (§2.9 caveat). The 1-D fakeness axis and the
asymmetry are re-framings of known phenomena.

The **least-trivial single fact** worth stating:

> Across five pretraining objectives, cross-generator fake detection is a **1-D whitened
> phenomenon**: one closed-form axis `w = Σ⁻¹ d` matches a full MLP on held-out
> generators and *beats* it on the hardest family (GAN); extra capacity / nonlinearity
> gives no OOD benefit and sometimes hurts.

That is an honest, mildly-surprising **negative characterization** (capacity doesn't
help) — workshop-tier at best, and even it is close to expected from "1-D axis +
overfitting". It is not a discovery.

**Recommendation**: treat §§2.1–2.10 as **preliminary exploration**. A real contribution
requires a genuinely new question — the localization + attribution direction (§6), or a
question not answerable by "read off the linear geometry".

**Full-MLP baseline (`cf_whiten.py`, 3-way, B/16, 5 seeds)** — one whitened axis vs a
full 768/1024-d MLP probe on held-out generators (whitened-w 1-D / full-MLP):

| backbone | GAN (hardest) | 5-family mean |
|---|---|---|
| CLIP | 0.977 / 0.969  (**w wins**) | 0.983 / 0.983 |
| SigLIP2 | 0.973 / 0.972 | 0.973 / 0.977 |
| DINOv2 | 0.953 / 0.954 | 0.953 / 0.960 |
| MAE | 0.914 / 0.899  (**w wins**) | 0.946 / 0.941 |
| ViT | 0.917 / 0.910  (**w wins**) | 0.931 / 0.933 |

The closed-form **1-D whitened axis matches the trained full MLP within ±0.007 mean AUC**,
and **beats it on the hardest held-out family (GAN) for 3/5 backbones**. Interpretation
(consistent with §2.7): the MLP can overfit to seen-generator features, while the
constrained whitened axis is more robust off-axis. The generalizable fakeness signal is
**one whitened direction**; extra capacity gives no OOD benefit and sometimes hurts.

---

## 4. Honest limitations
- §§2.1–2.7 were measured on CLIP **B/32** (preliminary; to be re-run on B/16 for a paper).
  The thesis experiments (§§2.8–2.10) are on **B/16 × 5 backbones**.
- Single dataset composition, moderate sizes (5 seeds, tight std).
- Whitening/LDA is **textbook**; novelty is the *deepfake-transfer application* and the
  *detector/predictor coupling*, not the whitening operation itself.
- Must be positioned against the **label-free OOD-accuracy-prediction** literature
  (below) — that is the real prior art for §2.9.

---

## 5. Positioning / reading list
- **Frozen-CLIP universal detection**: Ojha et al. CVPR 2023; Wang et al. CVPR 2020;
  Corvi et al. 2023; Cozzolino et al. 2024 (CLIP).
- **Label-free OOD accuracy prediction (prior art for §2.9)**: Garg et al. ICLR 2022
  (ATC); Guillory et al. ICCV 2021; Deng & Zheng CVPR 2021; Baek et al. NeurIPS 2022.
- **Attribution (the next project)**: Yu et al. ICCV 2019 (GAN fingerprints);
  Sha et al. CCS 2023 (DE-FAKE).
- **ID / geometry**: Ansuini et al. NeurIPS 2019; Pope et al. ICLR 2021; Facco et al. 2017.
- **Localization (next project)**: Guillaro et al. TruFor CVPR 2023; Oquab et al. DINOv2 2023.
- **Method**: Khosla et al. NeurIPS 2020 (supervised contrastive).

---

## 6. Next project — localization / attribution (data survey)

### Landscape (from a 2026 survey + ant-research/Awesome-AIGC-Image-Video-Detection)
- **Localization ("which region is fake") — CROWDED.** Many recent pixel-mask datasets:
  COCO-Inpaint (258K, 6 inpainters incl. Flux-Fill), OpenSDI (300K), NeXT-IMDL (558K,
  multi-turn), DiffSeg30k (30K), BR-Gen (150K), RealHD (730K+, inpaint+face-swap),
  GIM (1M, ImageNet), TGIF2, Inpaint32K. DINO is already used here (**DinoLizer**, 2025);
  methods: FakeShield, DiffSeg, CTNet. Entering pure localization = SOTA competition.
- **Image-level attribution ("which generator") — MATURE.** ImageAttributionBench
  (31 generators: 17 open + 14 commercial), GenImage (8), WildFake; LIDA "Attribution as
  Retrieval". Standard benchmarks exist.
- **Classic manipulation localization** (splice/copy-move): CASIA, Columbia, NIST16,
  IMD2020, DEFACTO. Different domain — trains poorly to diffusion inpainting.

### The gap (confirmed empty by the comprehensive ant-research repo)
- **Localized ATTRIBUTION — per-region generator identity — has no dataset.** Existing sets
  give (region mask) OR (image-level generator), never "region A by model α, region B by
  model β". This is the open niche; it needs data construction (compose multi-generator
  inpaints on OpenSDI/COCO-Inpaint bases, which carry image-level generator labels + masks).
- Motivation: "AI detectors overrely on GLOBAL artifacts" (2026) — local signal is needed.

### Two entry points
- **(a) Data-first**: build a localized-attribution set from OpenSDI/COCO-Inpaint. Novel, heavy.
- **(b) Pilot-first (recommended)**: before building data, test whether our **whitened axis
  extends to patch level** — estimate μ_real, Σ_real from real patches, project each patch
  (DINOv2 dense features) onto w, get a per-patch fakeness map, score vs GT masks
  (IoU/AUC) on OpenSDI or DiffSeg30k. Cheap; tells us if the geometry transfers to dense
  before committing to data construction. Success → (a); failure → rethink.

### Contrastive real/fake encoder
- Our fine-tune result predicts it overfits to seen generators; only interesting if designed
  around the low-dim shared axis. Low priority.
