#!/bin/bash

# 设置固定参数


cd /home/xwang/code/Tight_Generalization_bound/regression_datasets

# 循环不同的层数

for i in {1..9}
do
    echo "Running Regression Task for dataset_index $i"
    python regression.py \
            --n_qubits 6 \
            --n_layers 20 \
            --n_samples 10 \
            --n_epochs 100 \
            --n_repeats 1 \
            --data_type "regression" \
            --dataset_index $i
done

echo "All experiments completed!"