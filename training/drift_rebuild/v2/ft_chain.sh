#!/usr/bin/env bash
# W2c warm-start finetune row on v2 (q8_finetune_row protocol: every 3rd build
# from C0, window 6, lr 5e-5, 5 epochs, init from the previous checkpoint).
cd "$(dirname "$0")/../.."
PY=${PY:-python}
B=($(${PY:-python} -c "import json;print(' '.join(b for b in json.load(open('drift2026/patch_index.json'))['builds'] if b.startswith('2.55')))"))
prev=drift_v2/models/w2c_cut08_s42.pt
for p in $(seq 11 3 43); do
  name=$(printf "w2c_ft_cut%02d_s42" $p)
  if [ ! -f drift_v2/results/w2c_ft/$name.json ]; then
    $PY -u drift_rebuild/v2/run.py drift_rebuild/r2_train_vf.py --name $name --features cumulative_prev \
      --regime window --window 6 --seed 42 --cutoff-build ${B[$p]} --init-from $prev --lr 5e-5 \
      --max-epochs 5 --skip-sanity --results-subdir w2c_ft || exit 1
  fi
  prev=drift_v2/models/$name.pt
done
