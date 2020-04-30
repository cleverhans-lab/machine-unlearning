#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1

if [[ ! -f general-report.csv ]]; then
    echo "nb_shards,nb_requests,accuracy,retraining_time" > general-report.csv
fi

for j in {0..15}; do
    r=$((${j}*${shards}/5))
    acc=$(python aggregation.py --strategy uniform --container "${shards}" --shards "${shards}" --dataset datasets/purchase/datasetfile --label "${r}")
    cat containers/"${shards}"/times/shard-*:"${r}".time > "containers/${shards}/times/times"
    time=$(python time.py --container "${shards}" | awk -F ',' '{print $1}')
    echo "${shards},${r},${acc},${time}" >> general-report.csv
done
