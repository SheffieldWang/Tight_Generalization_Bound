#!/bin/bash

# 设置固定参数


cd /home/qcql/Quantum/Quantum_Tight_Generalization_Bound/classification_phase

# 循环不同的层数


echo "Running classification Phase Task"
python classification.py \
        --n_qubits 6 \
        --n_layers 2000 \
        --n_samples 2000 \
        --n_epochs 100 \
        --n_repeats 10 \
        --data_type "classification_phase" 



echo "All experiments completed!"