# What each figure shows

Every file is `<name>_<backbone>.png` with a matching `.json` holding the numbers.
All are produced by `scripts/reproduce.sh analysis` (plus `figures` for the signature).

| file | what it plots | what to look for |
|---|---|---|
| `signature_clip` | held-out AUC along two ladders: method (`d` → `w` → MLP) and dimension (1-D → 5-D → 768-D), one line per family | five families flatten after the whitened axis; **GAN stays low until the full 768-d probe** |
| `cf_whiten_*` | the same three method arms as bars, per held-out family | the `d` → `w` jump is large everywhere; `w` → MLP is flat except on GAN |
| `cf_matrix_*` | train-family × test-family AUC heatmap | off-diagonal asymmetry — Commercial→GAN 0.60 vs LatDiff/Flow/Commercial mutually 0.87–0.97 |
| `cf_subspace_*` | full 768-d vs 5-D subspace vs 1-D shared axis, per held-out family | GAN collapses to 0.64 in the 5-D span of the other families; LatDiff loses only 0.03 |
| `cf_directions_*` | left: per-family shift magnitude ‖μ_fake−μ_real‖. right: 1-D shared-axis AUC | both are **in-sample** descriptions of the draw, not held-out claims |
| `cf_separability_*` | asymmetric effective-separability matrix, axis A vs family B | the same asymmetry as `cf_matrix`, in pure geometry with no probe |
| `cf_twonn_*` | global TwoNN intrinsic dimension per group, layers 7 and 12 | every fake family sits below real (19.9); the ordering does *not* match transfer difficulty |
| `cf_generational_*` | raw / LID / raw+LID, per held-out family | LID alone is far below raw, and adding it moves the total within noise |
| `cf_ensemble_*` | pooled probe vs ensemble-max vs ensemble-mean, per test set | the ensemble never beats the single pooled probe |
| `cf_transfer_predictor_*` | geometric alignment cos(d_B, w_A) against actual A→B AUC | a consistency check only — the quantity is close to true by construction |

`provenance.txt` records the commit, package versions and device of the run that produced these.
