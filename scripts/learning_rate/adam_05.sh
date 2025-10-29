#!/bin/bash

# 设置固定参数


cd /home/xwang/code/Tight_Generalization_bound/learning_rate

# 循环不同的样本数量
sample_sizes=(250)

echo "Running classification Phase Task with different sample sizes"
for n_samples in "${sample_sizes[@]}"; do
    echo "Running with n_samples = $n_samples"
    python classification.py \
            --n_qubits 6 \
            --n_layers 20 \
            --n_samples $n_samples \
            --n_epochs 100 \
            --n_repeats 10 \
            --data_type "phase" \
            --optimizer "adam" \
            --lr 0.5
done


echo "All experiments completed!"