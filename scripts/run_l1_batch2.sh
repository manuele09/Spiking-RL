#!/usr/bin/env bash
# Seed aggiuntivi L1 (config D): ANN s2, SNN T=4 s1,s2, SNN T=1 s1,s2. I run completati sono saltati.
cd "$(dirname "$0")/.."
scripts/run_l1.sh ann 0 2
scripts/run_l1.sh snn 4 1
scripts/run_l1.sh snn 4 2
scripts/run_l1.sh snn 1 1
scripts/run_l1.sh snn 1 2
echo BATCH2 DONE
