#!/bin/bash

# 设置固定参数


cd /home/xwang/code/Tight_Generalization_bound/regression_special

# 循环不同的量子比特数

echo "Running Regression Special Task"

for n_qubits in 12
do
    echo "Running with n_qubits = $n_qubits"
    python regression.py \
            --n_qubits $n_qubits \
            --n_layers 20 \
            --n_samples 2000 \
            --n_epochs 100 \
            --n_repeats 10 \
            --data_type "regression_special"
done

echo "All experiments completed!"