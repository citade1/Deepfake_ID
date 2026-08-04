#!/usr/bin/env bash
# Reproduce the study end to end on a fresh machine.
#   scripts/reproduce.sh [provenance|maps|extract|compose|analysis|figures|all]
# Every stage is resumable: maps checkpoint per shard, extract skips cached shards,
# compose and the analyses simply overwrite their outputs.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-$HOME/miniforge3/envs/cv/bin/python}
export PYTHONPATH="$PWD"

# These four are load-bearing, not cosmetic:
#   DISABLE_XET       the Xet chunk downloader stalls on a slow link with no timeout
#   DOWNLOAD_TIMEOUT  the 10s default is shorter than a 1.8GB shard takes at 2MB/s
#   KMP_DUPLICATE_LIB_OK / OMP_NUM_THREADS   keep torch+MPS stable on macOS
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

BACKBONES=(clip siglip2 dinov2 mae vit)
SEEDS=(0 1 2 3 4)
ANALYSES=(whiten transfer directions subspace twonn alignment lid ensemble)

provenance() {
    mkdir -p figures
    {
        date -u +"utc %Y-%m-%dT%H:%M:%SZ"
        echo "commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
        $PY -c 'import sys,torch,transformers,pyarrow,huggingface_hub as h
print(f"python {sys.version.split()[0]} torch {torch.__version__} "
      f"transformers {transformers.__version__} pyarrow {pyarrow.__version__} hub {h.__version__}")
print("device", "mps" if torch.backends.mps.is_available() else "cpu")'
    } | tee figures/provenance.txt
}

maps()     { $PY -m pipeline.build_maps; }              # re-run until it reports 599/599
extract()  { $PY -m pipeline.extract; }                 # ~22GB download, resumable per shard

compose()  {
    for b in "${BACKBONES[@]}"; do
        for s in "${SEEDS[@]}"; do $PY -m pipeline.compose --backbone "$b" --seed "$s"; done
    done
}

analysis() {
    for b in "${BACKBONES[@]}"; do
        for m in "${ANALYSES[@]}"; do $PY -m "analysis.$m" --backbone "$b" --seeds "${#SEEDS[@]}"; done
    done
}

figures() { $PY scripts/signature_figure.py clip; }   # the one figure the README leads with


case "${1:-all}" in
    provenance) provenance ;;
    maps)       maps ;;
    extract)    extract ;;
    compose)    compose ;;
    analysis)   analysis ;;
    figures)    figures ;;
    all)        provenance; maps; extract; compose; analysis; figures ;;
    *) echo "usage: $0 [provenance|maps|extract|compose|analysis|figures|all]" >&2; exit 1 ;;
esac
