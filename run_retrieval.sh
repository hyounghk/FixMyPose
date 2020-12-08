#!/bin/sh
dataset=fixmypose

# Name of the model, used in snapshot
name=${model}_2pixel

task=speaker
if [ -z "$1" ]; then
    gpu=0
else
    gpu=$1
fi

log_dir=$dataset/$task/$name
mkdir -p snap_retrieval/$log_dir
mkdir -p log_retrieval/$dataset/$task
cp $0 snap_retrieval/$log_dir/run.bash
cp -r src_retrieval snap_retrieval/$log_dir/src

CUDA_VISIBLE_DEVICES=$gpu stdbuf -i0 -o0 -e0 python src_retrieval/main.py --output snap_retrieval/$log_dir \
    --maxInput 100  --worker 4 --train speaker --dataset $dataset \
    --batchSize 15 --hidDim 512 --dropout 0.5 \
    --seed 5555 \
    --optim adam --lr 1e-4 --epochs 50 \
    | tee log_retrieval/$log_dir.log
