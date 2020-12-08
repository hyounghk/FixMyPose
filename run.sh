#!/bin/sh

dataset=fixmypose

# Main metric to use
metric=CIDEr


# Name of the model, used in snapshot
name=${model}_2pixel

task=speaker
if [ -z "$1" ]; then
    gpu=0
else
    gpu=$1
fi

log_dir=$dataset/$task/$name
mkdir -p snap/$log_dir
mkdir -p log/$dataset/$task
cp $0 snap/$log_dir/run.bash
cp -r src snap/$log_dir/src

CUDA_VISIBLE_DEVICES=$gpu stdbuf -i0 -o0 -e0 python src/main.py --output snap/$log_dir \
    --maxInput 40 --metric $metric  --worker 4 --train speaker --dataset $dataset \
    --batchSize 45 --hidDim 512 --dropout 0.5 \
    --seed 9595 \
    --optim adam --lr 1e-4 --epochs 500 \
    | tee log/$log_dir.log
