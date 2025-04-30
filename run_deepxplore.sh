#!/bin/bash

SEEDS=100 # coverage가 너무 낮은 것 같아 seeds img를 많이 늘려봄
GRAD_ITERS=50
THRESHOLD=0.05 # coverage가 너무 낮은 것 같아 theshold를 많이 줄여봄
START_POINT="0 0"
OCCL_SIZE="10 10"

TRANSFORMATIONS=("no" "light" "occl" "blackout")
WEIGHT_DIFFS=(1.0)
WEIGHT_NCS=(0.5)
STEPS=(0.01)
TARGETS=(0 1)

mkdir -p logs

for TRANS in "${TRANSFORMATIONS[@]}"; do
  for WD in "${WEIGHT_DIFFS[@]}"; do
    for WNC in "${WEIGHT_NCS[@]}"; do
      for STEP in "${STEPS[@]}"; do
        for TM in "${TARGETS[@]}"; do
          LOGFILE="logs/${TRANS}_wd${WD}_wnc${WNC}_step${STEP}_t${TM}.log"
          echo "Running $TRANS | diff=$WD | nc=$WNC | step=$STEP | tgt=$TM"
          
          python gen_diff_two_models.py \
            "$TRANS" "$WD" "$WNC" "$STEP" "$SEEDS" "$GRAD_ITERS" "$THRESHOLD" \
            -t "$TM" -sp $START_POINT -occl_size $OCCL_SIZE > "$LOGFILE" 2>&1
          
          echo "Done -> $LOGFILE"
        done
      done
    done
  done
done
