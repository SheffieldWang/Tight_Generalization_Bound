#!/bin/bash

# 设置固定参数


cd /home/xwang/code/Tight_Generalization_bound/label_random

# 循环不同的种子值
for seed in {0..9}
do
    echo "Running classification Phase Task with seed $seed"
    python classification.py \
            --n_qubits 6 \
            --n_layers 20 \
            --n_samples 1000 \
            --n_epochs 100 \
            --n_repeats 1 \
            --data_type "classification_phase" \
            --alpha 1 \
            --optimizer "adam"  \
            --seed $seed
done


echo "All experiments completed!"