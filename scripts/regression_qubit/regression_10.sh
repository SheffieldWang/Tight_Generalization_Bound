#!/bin/bash

# 设置固定参数


cd /home/xwang/code/Tight_Generalization_bound/regression

# 循环不同的样本数量
for n_samples in 10 50 500 1000 1500
do
    echo "Running Regression Phase Task with n_samples = $n_samples"
    python regression.py \
            --n_qubits 10 \
            --n_layers 20 \
            --n_samples $n_samples \
            --n_epochs 100 \
            --n_repeats 10 \
            --data_type "regression"
done


echo "All experiments completed!"