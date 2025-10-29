#!/bin/bash

# 设置固定参数


cd /home/xwang/code/Tight_Generalization_bound/classification_phase

# 循环不同的样本数量
sample_sizes=(2000)

echo "Running classification Phase Task with different sample sizes"
for n_samples in "${sample_sizes[@]}"; do
    echo "Running with n_samples = $n_samples"
    python classification.py \
            --n_qubits 6 \
            --n_layers 100 \
            --n_samples $n_samples \
            --n_epochs 100 \
            --n_repeats 10 \
            --data_type "phase"
done


echo "All experiments completed!"