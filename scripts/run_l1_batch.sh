#!/usr/bin/env bash
# Batch L1: ANN seed 1, poi SNN T=4 e T=1 seed 0 (la ANN seed 0 e' runs/go2_l1_annD_s0, config identica)
cd "$(dirname "$0")/.."
scripts/run_l1.sh ann 0 1
scripts/run_l1.sh snn 4 0
scripts/run_l1.sh snn 1 0
echo BATCH DONE
