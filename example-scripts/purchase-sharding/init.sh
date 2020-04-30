#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1
    
if [[ ! -d "containers/${shards}" ]] ; then
    mkdir "containers/${shards}"
    mkdir "containers/${shards}/cache"
    mkdir "containers/${shards}/times"
    mkdir "containers/${shards}/outputs"
    echo 0 > "containers/${shards}/times/null.time"
fi

python distribution.py --shards "${shards}" --distribution uniform --container "${shards}" --dataset datasets/purchase/datasetfile --label 0

for j in {1..15}; do
    r=$((${j}*${shards}/5))
    python distribution.py --requests "${r}" --distribution uniform --container "${shards}" --dataset datasets/purchase/datasetfile --label "${r}"
done
