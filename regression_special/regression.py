import sys
import os
current_dir = os.getcwd()  # 获取当前工作目录
parent_dir = os.path.dirname(current_dir)  # 获取父目录
sys.path.append(parent_dir)

import pennylane as qml

import jax 
import jax.numpy as jnp  
import optax
from flax import nnx

import argparse
import pickle


import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from datasets_utils import get_quantum_dataloaders
from model import QuantumClassifier
from train_utils import RegressionTrainer,mse_loss_fn
from metric import Metrics,MetricComputer

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)


class ExperimentManager:
    def __init__(self, config):
        self.config = config
        self.n_repeats = config['n_repeats']
        self.results_df = pd.DataFrame()
        self.metrics = Metrics()
        
        self.setup_config()
        
        
    def setup_config(self):
        self.metrics.register_metric("loss",split="train",index_type="repeat")
        self.metrics.register_metric("error",split="train",index_type="repeat")
        self.metrics.register_metric("loss",split="test",index_type="repeat")
        self.metrics.register_metric("error",split="test",index_type="repeat")
        
    def run_experiments(self):
        """运行多次实验"""
        for i in range(self.n_repeats):
            print(f"\nRunning experiment {i+1}/{self.n_repeats}")
            observable = qml.PauliZ(0)
            for n_qubit in range(1,n_qubits):
                observable = observable @ qml.PauliZ(n_qubit)
            # 每次实验使用不同的随机种子
            seed = i
            print(seed)
            qnet = QuantumClassifier(ansatz_type="CNOT_Hamiltonian",n_qubits=n_qubits,n_layers=n_layers,measurement_type="hamiltonian",hamiltonian=observable,seed=seed)
            trainer = RegressionTrainer(self.config,qnet,train_loader,test_loader,mse_loss_fn)
            _= trainer.train()
            train_loss,train_error = trainer.get_train_metrics()
            test_loss,test_error = trainer.get_test_metrics()
            self.metrics.update("loss", train_loss, split="train", index_type="repeat")
            self.metrics.update("error", train_error, split="train", index_type="repeat")
            self.metrics.update("loss", test_loss, split="test", index_type="repeat")
            self.metrics.update("error", test_error, split="test", index_type="repeat")
            
            
            
        train_results = self.metrics.get_metrics(split='train')
        train_values = train_results['values']
        train_stats = train_results['stats']

        test_results = self.metrics.get_metrics(split='test')
        test_values = test_results['values']
        test_stats = test_results['stats']

        # Create DataFrame with results
        data = {'Experiment': range(1, n_repeats+1)}

        # Add train metrics
        for key in train_values.keys():
            data[f'Train {key.capitalize()}'] = train_values[key]

        # Add test metrics  
        for key in test_values.keys():
            data[f'Test {key.capitalize()}'] = test_values[key]

        self.results_df = pd.DataFrame(data)

        
        print("\n" + "="*50)
        print("Train Statistics".center(50))
        print("="*50)
        for metric in train_stats.keys():
            print(f"\n{metric.capitalize()}:")
            print("-"*30)
            for stat_name, value in train_stats[metric].items():
                print(f"{stat_name.capitalize():>15}: {value:.4f}")

        print("\n" + "="*50)
        print("Test Statistics".center(50))
        print("="*50)
        for metric in test_stats.keys():
            print(f"\n{metric.capitalize()}:")
            print("-"*30) 
            for stat_name, value in test_stats[metric].items():
                print(f"{stat_name.capitalize():>15}: {value:.4f}")



        # 保存结果
        self.save_results()
        
    
    def save_results(self):
        """保存实验结果"""
        
        # 保存DataFrame为CSV
        csv_path = f'results/experiment_results_{self.config["project_name"]}_{self.config["group_name"]}.csv'
        os.makedirs('results', exist_ok=True)
        self.results_df.to_csv(csv_path, index=False)
        
        # 保存所有数据（包括参数）到pickle文件
        full_results = {
            'config': self.config,
            'results_df': self.results_df,
        }
        pickle_path = f'results/full_results_{self.config["project_name"]}_{self.config["group_name"]}.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(full_results, f)
        
        print(f"Results saved to {csv_path} and {pickle_path}")   

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Quantum Classification Model Parameters')
    parser.add_argument('--n_qubits', type=int, default=6,
                        help='Number of qubits (default: 6)')
    parser.add_argument('--n_layers', type=int, default=20,
                        help='Number of quantum circuit layers (default: 20)') 
    parser.add_argument('--n_samples', type=int, default=2000,
                        help='Number of samples (default: 2000)')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size (default: 200)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate (default: 0.005)')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--n_repeats', type=int, default=10,
                        help='Number of repeats of experiment (default: 10)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--data_type', type=str, default="regression_special",
                        help='Data type (default: phase)')
    parser.add_argument('--optimizer', type=str, default="adam",
                        help='Optimizer (default: adam)')
    
    
    args = parser.parse_args()

    n_qubits = args.n_qubits
    n_layers = args.n_layers  
    n_samples = args.n_samples
    batch_size = args.batch_size
    lr = args.lr
    n_epochs = args.n_epochs
    n_repeats = args.n_repeats
    seed = args.seed
    data_type = args.data_type
    optimizer = args.optimizer
    # 实验配置
    config = {
        'n_qubits': n_qubits,
        'n_layers': n_layers,
        'n_samples':n_samples,
        'batch_size': batch_size,
        'learning_rate': lr,
        'n_epochs': n_epochs,
        'n_repeats': n_repeats,
        'seed':seed,
        'optimizer':optimizer,
        'project_name': f'regression_{data_type}',
        'group_name': f'qubits_{n_qubits}_layers_{n_layers}_samples_{n_samples}'
    }

    
    train_loader, test_loader = get_quantum_dataloaders(n_qubits=n_qubits,n_samples=n_samples,batch_size=batch_size,data_type=data_type)


    experiment_manager = ExperimentManager(config)
    experiment_manager.run_experiments()

    print(f"Regression_{data_type} Task for Qubit Number: {n_qubits} and Samples Number: {n_samples} completed!")
