import jax 
import jax.numpy as jnp  
import optax
from flax import nnx
from .loss_fn import mse_loss_fn
from metric import Metrics,MetricComputer
from tqdm.auto import tqdm
import wandb
import os 
import logging
from datetime import datetime
from datasets_utils import is_dataloader_shuffled



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



class RegressionTrainer:
    def __init__(self, config,model,train_loader,test_loader,loss_fn,optimizer='adam'):
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
        self.best_error = float('inf')
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.metrics = Metrics()  # metrics对象通过引用传递,内部修改会影响原对象
        
        self.setup_config()
        self.setup_logging()
        self.use_wandb = self.config.get('use_wandb', False)

        
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
        if self.config['optimizer'] == 'adam':
            self.optimizer = nnx.Optimizer(self.model, optax.adam(learning_rate=self.config['learning_rate']))
        elif self.config['optimizer'] == 'sgd':
            self.optimizer = nnx.Optimizer(self.model, optax.sgd(learning_rate=self.config['learning_rate']))
        elif self.config['optimizer'] == 'rmsprop':
            self.optimizer = nnx.Optimizer(self.model, optax.rmsprop(learning_rate=self.config['learning_rate']))
        elif self.config['optimizer'] == 'adagrad':
            self.optimizer = nnx.Optimizer(self.model, optax.adagrad(learning_rate=self.config['learning_rate']))
        elif self.config['optimizer'] == 'lion':
            self.optimizer = nnx.Optimizer(self.model, optax.lion(learning_rate=self.config['learning_rate']))
        else:
            raise ValueError(f"Invalid optimizer: {self.config['optimizer']}")
        
        self.metrics.register_metric("loss",split="train",index_type="epoch")
        self.metrics.register_metric("error",split="train",index_type="epoch")
        self.metrics.register_metric("loss",split="test",index_type="epoch")
        self.metrics.register_metric("error",split="test",index_type="epoch")

            
    def run_epoch(self, epoch: int, split: str = "train",data_loader=None) -> None:
        """执行一个完整的训练或评估 epoch

        Args:
            epoch: 当前 epoch 序号
            split: 执行模式，可选 "train" 或 "test"
        """
        # 根据模式选择数据加载器
        if data_loader is None:
            data_loader = self.train_loader if split == "train" else self.test_loader
        else:
            data_loader = data_loader
        
        total_loss = 0.0
        total_error = 0.0
        desc = f"{split.capitalize()} Epoch {epoch+1}/{self.epochs}"
        
        with tqdm(data_loader, desc=desc, postfix={"loss": 0.0}, leave=False) as pbar:
            for batch_idx, batch in enumerate(pbar):
                _, target = batch
                
                # 区分训练/评估的前向传播
                if split == "train":
                    loss, outputs = train_step(self.model, self.optimizer, batch,self.loss_fn)
                else:
                    loss, outputs = eval_step(self.model, batch,self.loss_fn)  # 假设存在 eval_step
                
                # 计算指标
                error = jnp.mean(jnp.abs(outputs-target))
                

                
                # 更新进度条
                pbar.set_postfix({"loss": loss, "error": error})
                total_loss += loss
                total_error += error
        
        # 计算平均指标并记录
        avg_loss = total_loss/len(data_loader)
        avg_error = total_error/len(data_loader)
        self.metrics.update("loss", avg_loss, split=split, index_type="epoch")
        self.metrics.update("error", avg_error, split=split, index_type="epoch")

        return avg_loss,avg_error

    def train(self):
        """完整训练流程"""
        
        # 初始化 wandb
        wandb.init(project=self.config["project_name"], config=self.config)
        
        for epoch in tqdm(range(self.config['n_epochs']), desc='Training epochs', leave=True):
            # 训练阶段
            self.run_epoch(epoch,"train")
            

            
            # 记录指标
            wandb.log({
                "train_loss": self.metrics.get_metric("loss",split="train",index=epoch),
                "train_error": self.metrics.get_metric("error",split="train",index=epoch)
            })
        wandb.finish()  
            

        train_loss = self.metrics.get_metric('loss', split='train', index=-1)
        train_error = self.metrics.get_metric('error', split='train', index=-1)
        self.logger.info(f"Training finished. Final train loss: {train_loss:.6f}")
        self.logger.info(f"Final train error: {train_error:.6f}")
        
        # 计算训练集和测试集的最终损失和错误率
        # return self.best_model,self.best_error
        return self.model,train_loss,train_error
    
    def get_train_metrics(self):
        """完整训练流程"""
        
        train_loss,train_error = self.run_epoch(0,"test",self.train_loader)

        self.logger.info(f"Training finished. Final train loss: {train_loss:.6f}")
        self.logger.info(f"Final train error: {train_error:.6f}")
        
        return train_loss,train_error
    
    def get_test_metrics(self):
        """完整训练流程"""
        
        test_loss,test_error = self.run_epoch(0,"test",self.test_loader)
            
        self.logger.info(f"Training finished. Final test loss: {test_loss:.6f}")
        self.logger.info(f"Final test error: {test_error:.6f}")
        
        return test_loss,test_error
    

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
        self.use_wandb = self.config.get('use_wandb', False)

        
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
        if self.config['optimizer'] == 'adam':
            self.optimizer = nnx.Optimizer(self.model, optax.adam(learning_rate=self.config['learning_rate']))
        elif self.config['optimizer'] == 'sgd':
            self.optimizer = nnx.Optimizer(self.model, optax.sgd(learning_rate=self.config['learning_rate']))
        elif self.config['optimizer'] == 'rmsprop':
            self.optimizer = nnx.Optimizer(self.model, optax.rmsprop(learning_rate=self.config['learning_rate']))
        elif self.config['optimizer'] == 'adagrad':
            self.optimizer = nnx.Optimizer(self.model, optax.adagrad(learning_rate=self.config['learning_rate']))
        elif self.config['optimizer'] == 'lion':
            self.optimizer = nnx.Optimizer(self.model, optax.lion(learning_rate=self.config['learning_rate']))
        else:
            raise ValueError(f"Invalid optimizer: {self.config['optimizer']}")
        
        self.metrics.register_metric("loss",split="train",index_type="epoch")
        self.metrics.register_metric("accuracy",split="train",index_type="epoch")
        self.metrics.register_metric("loss",split="test",index_type="epoch")
        self.metrics.register_metric("accuracy",split="test",index_type="epoch")

            
    def run_epoch(self, epoch: int, split: str = "train",data_loader=None) -> None:
        """执行一个完整的训练或评估 epoch

        Args:
            epoch: 当前 epoch 序号
            split: 执行模式，可选 "train" 或 "test"
        """
        # 根据模式选择数据加载器
        if data_loader is None:
            data_loader = self.train_loader if split == "train" else self.test_loader
        else:
            data_loader = data_loader
        
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
        
        return avg_loss,avg_accuracy



    def train(self):
        """完整训练流程"""
        
        # 初始化 wandb
        wandb.init(project=self.config["project_name"], config=self.config)
        
        for epoch in tqdm(range(self.config['n_epochs']), desc='Training epochs', leave=True):
            # 训练阶段
            train_loss,train_accuracy = self.run_epoch(epoch,"train",self.train_loader)
            # 记录指标
            wandb.log({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy
            })
        wandb.finish()  
        
        
        
        # 计算训练集和测试集的最终损失和错误率
        # return self.best_model,self.best_error
        return self.model
   
   
    def get_train_metrics(self):
        """完整训练流程"""
        
        train_loss,train_accuracy = self.run_epoch(0,"test",self.train_loader)

        self.logger.info(f"Training finished. Final train loss: {train_loss:.6f}")
        self.logger.info(f"Final train accuracy: {train_accuracy:.6f}")
        
        return train_loss,train_accuracy
    
    def get_test_metrics(self):
        """完整训练流程"""
        
        test_loss,test_accuracy = self.run_epoch(0,"test",self.test_loader)
            
        self.logger.info(f"Training finished. Final test loss: {test_loss:.6f}")
        self.logger.info(f"Final test accuracy: {test_accuracy:.6f}")
        
        return test_loss,test_accuracy