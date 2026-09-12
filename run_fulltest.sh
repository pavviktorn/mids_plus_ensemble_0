#!/usr/bin/env bash
# Full labelled test of heldout3/frames across all 4 GPUs (4 shards), then merge -> results.txt.
cd /datasets/work/vLLM/temp/mids_plus_ensemble_0 || exit 1
. ../mids_plus/.venv/bin/activate
# cap per-process CPU threads so 4 parallel shards don't oversubscribe (SVD load is multithreaded)
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6
export TOKENIZERS_PARALLELISM=false
D="${1:-/datasets/work/vLLM/temp/mids_plus/runs/real/heldout3/frames}"
N=4
rm -f results_shard*.txt results_shard*.txt.tally.json DONE_FULLTEST
PIDS=()
for k in $(seq 1 $N); do
  gpu=$((k-1))
  CUDA_VISIBLE_DEVICES=$gpu python -u inference.py --test --dir "$D" --shard "$k/$N" \
    --results "results_shard${k}.txt" --batch-size 64 > "shard${k}.log" 2>&1 &
  PIDS+=($!)
  echo "launched shard $k/$N on GPU $gpu (pid ${PIDS[-1]})"
done
for p in "${PIDS[@]}"; do wait "$p"; done
echo "all shards done; merging..."
python merge_shards.py --shards results_shard1.txt results_shard2.txt results_shard3.txt results_shard4.txt --out results.txt
touch DONE_FULLTEST
echo "=== FULLTEST DONE ==="
