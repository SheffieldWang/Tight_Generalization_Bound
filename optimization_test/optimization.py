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
from train_utils import hinge_loss_fn
from metric import Metrics,MetricComputer

import wandb
import os 
import logging
from datetime import datetime

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

@nnx.jit(static_argnames=['loss_fn'])
def train_step(model,optimizer,batch,loss_fn):
    data,target = batch
    (loss,predictions), grads = nnx.value_and_grad(loss_fn,has_aux=True)(model,data,target)
    optimizer.update(grads)
    return loss,predictions

@nnx.jit(static_argnames=['loss_fn'])
def eval_step(model,batch,loss_fn):
    data,target = batch
    (loss,predictions), _ = nnx.value_and_grad(loss_fn,has_aux=True)(model,data,target)
    return loss,predictions

class ClassificationTrainer:
    def __init__(self, config,model,train_loader,test_loader,loss_fn):
        """
        初始化训练器
        
        参数:
        - config: 配置字典
        - model: 要训练的模型
        - metrics: Metrics实例,用于记录指标
            由于Python对象是通过引用传递的,对self.metrics的修改会反映到输入的metrics对象上
        """
        self.config = config
        self.epochs = config['n_epochs']
        self.model = model
        self.best_model = None
        self.best_accuracy = 0.0
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.metrics = Metrics()  # metrics对象通过引用传递,内部修改会影响原对象
        self.metric_computer = MetricComputer()

        
        self.setup_config()
        self.setup_logging()

        
    def setup_model(self):
        """初始化模型"""
        
    def setup_logging(self):
        # 配置日志的基本设置:
        # - 日志级别设为INFO
        # - 日志格式包含时间、日志级别和具体消息
        # - 只输出到控制台
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )

        # 获取当前模块的logger实例
        self.logger = logging.getLogger(__name__)

    def setup_config(self):
        """设置优化器和学习率调度器"""
        self.optimizer = nnx.Optimizer(self.model, optax.adam(learning_rate=self.config['learning_rate']))
        
        self.metrics.register_metric("loss",split="train",index_type="epoch")
        self.metrics.register_metric("accuracy",split="train",index_type="epoch")
        self.metrics.register_metric("loss",split="test",index_type="epoch")
        self.metrics.register_metric("accuracy",split="test",index_type="epoch")

            
    def run_epoch(self, epoch: int, split: str = "train") -> None:
        """执行一个完整的训练或评估 epoch

        Args:
            epoch: 当前 epoch 序号
            split: 执行模式，可选 "train" 或 "test"
        """
        # 根据模式选择数据加载器
        data_loader = self.train_loader if split == "train" else self.test_loader
        
        total_loss = 0.0
        total_accuracy = 0.0
        desc = f"{split.capitalize()} Epoch {epoch+1}/{self.epochs}"
        
        with tqdm(data_loader, desc=desc, postfix={"loss": 0.0}, leave=False) as pbar:
            for batch_idx, batch in enumerate(pbar):
                _, target = batch
                
                # 区分训练/评估的前向传播
                if split == "train":
                    loss, logits = train_step(self.model, self.optimizer, batch,self.loss_fn)
                else:
                    loss, logits = eval_step(self.model, batch,self.loss_fn)  # 假设存在 eval_step
                
                # 计算指标
                accuracy = self.metric_computer.compute_accuracy(logits,target)
                

                
                # 更新进度条
                pbar.set_postfix({"loss": loss, "accuracy": accuracy})
                total_loss += loss
                total_accuracy += accuracy
        
        # 计算平均指标并记录
        avg_loss = total_loss/len(data_loader)
        avg_accuracy = total_accuracy/len(data_loader)
        self.metrics.update("loss", avg_loss, split=split, index_type="epoch")
        self.metrics.update("accuracy", avg_accuracy, split=split, index_type="epoch")



    def train(self):
        """完整训练流程"""
        
        # 初始化 wandb
        wandb.init(project=self.config["project_name"], config=self.config)
        
        for epoch in tqdm(range(self.config['n_epochs']), desc='Training epochs', leave=True):
            # 训练阶段
            self.run_epoch(epoch,"train")
            self.run_epoch(epoch,"test")
            # 记录指标
            wandb.log({
                "train_loss": self.metrics.get_metric("loss",split="train",index=epoch),
                "train_accuracy": self.metrics.get_metric("accuracy",split="train",index=epoch),
                "test_loss": self.metrics.get_metric("loss",split="test",index=epoch),
                "test_accuracy": self.metrics.get_metric("accuracy",split="test",index=epoch)
            })
        wandb.finish()  
        

        train_loss = self.metrics.get_metric('loss', split='train', index=-1)   
        train_accuracy = self.metrics.get_metric('accuracy', split='train', index=-1)
        self.logger.info(f"Training finished. Final train loss: {train_loss:.6f}")
        self.logger.info(f"Final train accuracy: {train_accuracy:.6f}")
        
        
        # 计算训练集和测试集的最终损失和错误率
        # return self.best_model,self.best_error
        return self.model,self.metrics
    

class ExperimentManager:
    def __init__(self, config):
        self.config = config
        self.n_repeats = config['n_repeats']
        self.metrics_list = []

    
        
    def run_experiments(self):
        """运行多次实验"""
        for i in range(self.n_repeats):
            print(f"\nRunning experiment {i+1}/{self.n_repeats}")
            observable = qml.PauliZ(0)
            # 每次实验使用不同的随机种子
            seed = i
            qnet = QuantumClassifier(ansatz_type="CNOT_Amp",n_qubits=n_qubits,n_layers=n_layers,measurement_type="hamiltonian",hamiltonian=observable,seed=seed)
            trainer = ClassificationTrainer(self.config,qnet,train_loader,test_loader,hinge_loss_fn)
            _,metrics = trainer.train()
            self.metrics_list.append(metrics)

            





        # 保存结果
        self.save_results()
        
    
    def save_results(self):
        """保存实验结果"""
        
        # 保存所有数据（包括参数）到pickle文件
        os.makedirs('results', exist_ok=True)
        full_results = {
            'config': self.config,
            'metrics_list': self.metrics_list
        }
        pickle_path = f'results/full_results_{self.config["project_name"]}_{self.config["group_name"]}.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(full_results, f)
        
        print(f"Results saved to {pickle_path}")

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
    parser.add_argument('--data_type', type=str, default="phase",
                        help='Data type (default: phase)')
    
    
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
        'project_name': f'optimization_test_{data_type}',
        'group_name': f'loop_epochs_{n_epochs}'
    }

    
    train_loader, test_loader = get_quantum_dataloaders(n_qubits=n_qubits,n_samples=n_samples,batch_size=batch_size,type="phase")


    experiment_manager = ExperimentManager(config)
    experiment_manager.run_experiments()

    print(f"Classification_{data_type} Task for Qubit Number: {n_qubits} and Samples Number: {n_samples} completed!")
