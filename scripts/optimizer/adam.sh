#!/bin/bash

# 设置固定参数


cd /home/xwang/code/Tight_Generalization_bound/optimizer_test

# 循环不同的层数


echo "Running classification Phase Task"
python classification.py \
        --n_qubits 6 \
        --n_layers 20 \
        --n_samples 2000 \
        --n_epochs 100 \
        --n_repeats 10 \
        --data_type "phase" \
        --optimizer "adam"



echo "All experiments completed!"