import pennylane as qml
import jax
import jax.numpy as jnp
import warnings 
from flax import nnx
import numpy as np
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

# Create multiple diagonal matrices with different multipliers
n_qubits = 3
dim = 2**n_qubits  # Dimension of the matrix (8x8 for 3 qubits)

# Create matrices with different multipliers (2 through 10)
H = {}

for multiplier in range(2, 15):
    # Create diagonal values (each value is a multiple of the multiplier)
    diagonal_values = jnp.array([multiplier * i for i in range(dim)])
    
    # Create the diagonal matrix
    diagonal_matrix = jnp.diag(diagonal_values)
    
    # Store the matrix in the dictionary
    H[multiplier] = diagonal_matrix
    
def H_encoding(x,diagonal_matrix):
    """
    Creates a quantum gate e^{-ix H} where H is a diagonal matrix.
    
    Args:
        diagonal_matrix: The diagonal matrix H
        x: The evolution parameter (default: 1.0)
    
    Returns:
        A quantum operation representing e^{-ix H}
    """
    # Extract the diagonal values
    diagonal_values = jnp.diag(diagonal_matrix)
    
    # Create the evolution matrix directly
    phases = -1j * x * diagonal_values
    evolution_matrix = jnp.diag(jnp.exp(phases))
    
    # Define a function that applies the unitary
    def diagonal_evolution_op(wires):
        qml.QubitUnitary(evolution_matrix, wires=wires)
    
    return diagonal_evolution_op


class BaseQuantumCircuit(nnx.Module):
    def __init__(self, 
                 n_qubits=1,
                 device_name='default.qubit', 
                 interface='jax',
                 measurement_type='probs',
                 hamiltonian=None,
                 measure_wires=None):
        """
        Initialize the quantum model.

        Parameters:
        - n_qubits (int): Number of qubits.
        - n_reps (int): Number of repetitions.
        - n_layers (int): Number of layers per repetition.
        - device_name (str): Name of the quantum device.
        - interface (str): Interface type, e.g., 'jax'.
        """      
        self.n_qubits = n_qubits
        self.interface = interface
        self.device = qml.device(device_name, wires=self.n_qubits)
        self.measurement_type = measurement_type
        self.hamiltonian = hamiltonian 
        self.measure_wires = measure_wires

        self._validate_measurement_settings()
        
    def _validate_measurement_settings(self):
        """验证测量设置的有效性"""
        if self.measurement_type not in ['probs', 'hamiltonian', 'density', 'state']:
            raise ValueError("Measurement type must be 'probs', 'hamiltonian', 'density' or 'state'")
        if self.measurement_type == 'hamiltonian' and self.hamiltonian is None:
            raise ValueError("Hamiltonian is not provided")
        if (self.measurement_type == 'probs' or self.measurement_type == 'density') and self.measure_wires is None:
            warnings.warn("Measure wires are not provided, using all qubits")
            self.measure_wires = range(self.n_qubits)    


    def __call__(self,x):
        if self.measurement_type == 'probs':
            batched_probs = jax.vmap(self.probs, in_axes=(0), out_axes=0)
            return batched_probs(x)
        elif self.measurement_type == 'hamiltonian':
            batched_expvalH = jax.vmap(self.expvalH, in_axes=(0), out_axes=0)
            return batched_expvalH(x)
        elif self.measurement_type == 'density':
            batched_density = jax.vmap(self.density_matrix, in_axes=(0), out_axes=0)
            return batched_density(x)
        elif self.measurement_type == 'state':
            batched_state = jax.vmap(self.state, in_axes=(0), out_axes=0)
            return batched_state(x) 
        else:
            raise ValueError("Invalid measurement type")
        

    def ansatz(self, x):
        """需要在子类中实现的抽象方法"""
        raise NotImplementedError("Ansatz must be implemented in subclass")
    
              
                    
    def draw_circuit(self, x):
        """
        绘制量子线路图

        参数:
        - x (array): 输入数据

        返回:
        - 量子线路图
        """
        @qml.qnode(self.device, interface=self.interface)
        def circuit_to_draw(x):
            self.ansatz(x)
            return qml.probs(wires=0)

        # 使用 PennyLane 的 draw_mpl() 函数绘制线路图
        # style: 图表样式，可选 'default'或'pennylane'
        # decimals: 数值显示的小数位数
        # fontsize: 字体大小
        # max_length: 最大显示长度
        fig = qml.draw_mpl(circuit_to_draw, 
                          style='pennylane',
                          decimals=2,
                          fontsize=13)(x)
        return fig

    def probs(self, x):
        """
        Calculate and return measurement probabilities.

        Parameters:
        - x (array): Input data
        - wires (int or list): Quantum wires to measure, defaults to 0

        Returns:
        - array: Measurement probabilities
        """
            
        @qml.qnode(self.device, interface=self.interface)
        def qnode_probs(x):
            self.ansatz(x)
            return qml.probs(wires=self.measure_wires)
        return qnode_probs(x)
    
    def expvalH(self, x):
        """
        Calculate and return the measurement expectation value.

        Args:
            - x (array): Input data

        Returns:
            - array: Measurement expectation value
        """
            
        @qml.qnode(self.device, interface=self.interface)
        def qnode_expvalH(x):
            self.ansatz(x)
            return qml.expval(self.hamiltonian)
        return qnode_expvalH(x)

    def state(self, x):
        """
        Get and return the quantum state.

        Args:
            - x (array): Input data

        Returns:
            - array: Quantum state
        """
        @qml.qnode(self.device, interface=self.interface)
        def qnode_state(x):
            self.ansatz(x)
            return qml.state()
        return qnode_state(x)


    def density_matrix(self, x):
        """
        Get and return the density matrix.

        Args:
            - x (array): Input data

        Returns:
            - array: Density matrix
        """
        @qml.qnode(self.device, interface=self.interface)
        def qnode_density(x):
            self.ansatz(x)
            return qml.density_matrix(wires=self.measure_wires)
        return qnode_density(x)


class ParameterizedQuantumCircuit(BaseQuantumCircuit):
    """具有可训练参数的量子电路基类"""
    def __init__(self, 
                 n_qubits=1, 
                 **kwargs):
        super().__init__(n_qubits, **kwargs)
        self.params = None  # 将在子类中初始化
    def update_params(self, new_params):
        """
        更新模型参数

        参数:
        - new_params: 新的参数值，需要与self.params的形状相匹配
        """
        if self.params is None:
            raise ValueError("参数尚未初始化")
        if new_params.shape != self.params.shape:
            raise ValueError(f"参数形状不匹配。期望形状: {self.params.shape}，实际形状: {new_params.shape}")
        self.params = nnx.Param(new_params)
        
    def get_params(self):
        """
        获取当前模型参数

        返回:
        - 当前模型参数的副本
        """
        if self.params is None:
            raise ValueError("参数尚未初始化")
        return jax.numpy.array(self.params)
    
    
    def save_params(self, filepath):
        """
        保存模型参数到文件

        参数:
        - filepath: 保存参数的文件路径，建议使用.npy格式
        """
        if self.params is None:
            raise ValueError("参数尚未初始化")
        
        params_numpy = jnp.array(self.params)
        jnp.save(filepath, params_numpy)


class CNOTCircuit_Amp(ParameterizedQuantumCircuit):
    """回归模型的量子电路"""
    def __init__(self, 
                 n_qubits=1, 
                 n_layers=1, 
                 seed=0,
                 **kwargs):
        super().__init__(n_qubits,**kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers  
        self.shape = (self.n_layers, self.n_qubits, 3)
        key = nnx.Rngs(seed).params()
        
        self.params = nnx.Param(jax.random.normal(key, self.shape))

    def ansatz(self, x):
        """实现回归模型的ansatz"""
        x = x.reshape(-1)
        qml.AmplitudeEmbedding(x, wires=range(self.n_qubits),normalize=True,pad_with=0.0)
        for l in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RZ(self.params[l][i][0], wires=i)
                qml.RY(self.params[l][i][1], wires=i)
                qml.RZ(self.params[l][i][2], wires=i)
            if self.n_qubits > 1:
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i,(i+1)%self.n_qubits])
            qml.Barrier(wires=range(self.n_qubits), only_visual=True)


class CNOTCircuit_angle(ParameterizedQuantumCircuit):
    """回归模型的量子电路"""
    def __init__(self, 
                 n_qubits=1, 
                 n_layers=1, 
                 seed=0,
                 **kwargs):
        super().__init__(n_qubits,**kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers  
        self.shape = (self.n_layers, self.n_qubits, 3)
        key = nnx.Rngs(seed).params()
        
        self.params = nnx.Param(jax.random.normal(key, self.shape))

    def ansatz(self, x):
        """实现回归模型的ansatz"""
        
        # Encoding 
        for i in range(self.n_qubits):
            qml.RY(x[i], wires=i)
        
        for l in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RZ(self.params[l][i][0], wires=i)
                qml.RY(self.params[l][i][1], wires=i)
                qml.RZ(self.params[l][i][2], wires=i)
            if self.n_qubits > 1:
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i,(i+1)%self.n_qubits])
            qml.Barrier(wires=range(self.n_qubits), only_visual=True)


class CNOTCircuit_Hamiltonian(ParameterizedQuantumCircuit):
    """回归模型的量子电路"""
    def __init__(self, 
                 n_qubits=1, 
                 n_layers=1, 
                 seed=0,
                 **kwargs):
        super().__init__(n_qubits,**kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers  
        self.shape = (self.n_layers, self.n_qubits, 3)
        key = nnx.Rngs(seed).params()
        
        self.params = nnx.Param(jax.random.normal(key, self.shape))

    def ansatz(self, x):
        """实现回归模型的ansatz"""
        
        # Encoding 
        for i in range(self.n_qubits-2):
            H_encoding(x[i],H[2+i])(wires=[i,i+1,i+2])
        
        for l in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RZ(self.params[l][i][0], wires=i)
                qml.RY(self.params[l][i][1], wires=i)
                qml.RZ(self.params[l][i][2], wires=i)
            if self.n_qubits > 1:
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i,(i+1)%self.n_qubits])
            qml.Barrier(wires=range(self.n_qubits), only_visual=True)

class QCNNCircuit(ParameterizedQuantumCircuit):
    """QCNN量子电路"""
    def __init__(self,
                 n_qubits=10,
                 n_layers=4,
                 seed=0,
                 **kwargs):
        super().__init__(n_qubits, **kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        key = nnx.Rngs(seed).params()
        
        self.embed_params = nnx.Param(jax.random.normal(key))
        
        # 输入验证

        if n_layers > int(jnp.ceil(jnp.log2(n_qubits))):
            raise ValueError("Too many layers for given qubit count")
        # 计算每层所需参数数量
        # 池化层参数
        
        self._initialize_parameter_shapes()
        self._initialize_parameters(key)
        
    def _initialize_parameter_shapes(self):
        """预计算各层参数尺寸"""
        self.param_shapes = {
            'rotation': (self.n_qubits, 3),
            'conv': [],
            'pool': []
        }
        
        current_qubits = self.n_qubits
        for _ in range(self.n_layers):
            # 卷积层参数: (qubits-1)*9
            conv_shape = (max((current_qubits - 1), 1) , 15)
            self.param_shapes['conv'].append(conv_shape)
            
            # 池化层参数: qubits/2
            pool_shape =  (max(current_qubits // 2,1),1)
            self.param_shapes['pool'].append(pool_shape)
            
            current_qubits //= 2
            


    def _initialize_parameters(self,key):
        # 旋转层参数
        key, subkey = jax.random.split(key)
        self.rotation_params = nnx.Param(jax.random.normal(
            subkey, self.param_shapes['rotation']
        ))
        
        # 逐层初始化卷积/池化参数
        self.conv_params = []
        self.pool_params = []
        
        for l in range(self.n_layers):
            # 卷积参数
            key, subkey = jax.random.split(key)
            conv_init = nnx.Param(jax.random.normal(
                subkey, shape=self.param_shapes['conv'][l],
            ))
            self.conv_params.append(conv_init)
            
            # 池化参数
            key, subkey = jax.random.split(key)
            pool_init = nnx.Param(jax.random.normal(
                subkey, shape=self.param_shapes['pool'][l],
            ))
            self.pool_params.append(pool_init)
            
            
    def rotation_block(self, params, wires):
        for n in range(len(wires)):
            qml.RX(params[n][0], wires=wires[n])
            qml.RZ(params[n][1], wires=wires[n])
            qml.RX(params[n][2], wires=wires[n])

    def convolutional_layer(self, params, wires):
        n_wires = len(wires)
        assert n_wires >= 2, "Circuit too small!"

        # 不可以改成 n_wires -2
        for n in range(n_wires):
            if n % 2 == 0 and n < n_wires-1:
                qml.RX(params[n][9], wires=wires[n])
                qml.RZ(params[n][10], wires=wires[n])
                qml.RX(params[n][11], wires=wires[n])
                qml.RX(params[n][12], wires=wires[n+1])
                qml.RZ(params[n][13], wires=wires[n+1])
                qml.RX(params[n][14], wires=wires[n+1])
                qml.IsingXX(params[n][0], wires=[wires[n], wires[n+1]])
                qml.IsingYY(params[n][1], wires=[wires[n], wires[n+1]])
                qml.IsingZZ(params[n][2], wires=[wires[n], wires[n+1]])
                qml.RX(params[n][3], wires=wires[n])
                qml.RZ(params[n][4], wires=wires[n])
                qml.RX(params[n][5], wires=wires[n])
                qml.RX(params[n][6], wires=wires[n+1])
                qml.RZ(params[n][7], wires=wires[n+1])
                qml.RX(params[n][8], wires=wires[n+1])

        for n in range(n_wires-1):
            if n % 2 == 1:
                qml.RX(params[n][9], wires=wires[n])
                qml.RZ(params[n][10], wires=wires[n])
                qml.RX(params[n][11], wires=wires[n])
                qml.RX(params[n][12], wires=wires[n+1])
                qml.RZ(params[n][13], wires=wires[n+1])
                qml.RX(params[n][14], wires=wires[n+1])
                qml.IsingXX(params[n][0], wires=[wires[n], wires[n+1]])
                qml.IsingYY(params[n][1], wires=[wires[n], wires[n+1]])
                qml.IsingZZ(params[n][2], wires=[wires[n], wires[n+1]])
                qml.RX(params[n][3], wires=wires[n])
                qml.RZ(params[n][4], wires=wires[n])
                qml.RX(params[n][5], wires=wires[n])
                qml.RX(params[n][6], wires=wires[n+1])
                qml.RZ(params[n][7], wires=wires[n+1])
                qml.RX(params[n][8], wires=wires[n+1])




    def pooling_layer(self, params, wires):
        n_wires = len(wires)
        assert n_wires >= 2, "Circuit too small!"

        for n in range(n_wires):
            if n % 2 == 1:
                m_outcome = qml.measure(wires[n])
                qml.cond(m_outcome, qml.RX)(params[n][0], wires=wires[n-1])

    def conv_and_pool_block(self,conv_params,pool_params,wires):
        wires = range(self.n_qubits)
        for l in range(self.n_layers):
            self.convolutional_layer(conv_params[l], wires)
            self.pooling_layer(pool_params[l], wires)
            wires = wires[::2]

    def ansatz(self, x):
        """Implement QCNN ansatz"""
        x = x.reshape(-1)
        
        qml.AmplitudeEmbedding(features=x, wires=range(self.n_qubits), pad_with=self.embed_params, normalize=True)
        # qml.Barrier(wires=range(self.n_qubits), only_visual=True)
        self.rotation_block(self.rotation_params, range(self.n_qubits))
        self.conv_and_pool_block(self.conv_params, self.pool_params, range(self.n_qubits))
        
        
class DataReuploadingCircuit(ParameterizedQuantumCircuit):
    """数据重上传量子电路"""
    def __init__(self,
                 n_qubits=1,
                 n_reps=1,
                 n_layers=1,
                 seed=0,
                 **kwargs):
        super().__init__(n_qubits,**kwargs)
        self.n_qubits = n_qubits
        self.n_reps = n_reps
        self.n_layers = n_layers
        self.shape = (self.n_reps, self.n_layers, self.n_qubits, 3)
        key = nnx.Rngs(seed).params()
        self.params = nnx.Param(jax.random.normal(key, self.shape))

    def ansatz(self, x):
        x = x.reshape(self.n_qubits,self.n_layers,3)
        for p in range(self.n_reps):
            for l in range(self.n_layers):
                for n in range(self.n_qubits):
                    qml.Rot(x[l][n][0], x[l][n][1], x[l][n][2], wires=n)
                    qml.Rot(self.params[p][l][n][0], self.params[p][l][n][1], self.params[p][l][n][2], wires=n)
                if self.n_qubits > 1:
                    for n in range(self.n_qubits):
                        qml.CNOT(wires=[n,(n+1)%self.n_qubits])
                qml.Barrier(wires=range(self.n_qubits), only_visual=True)






class QuantumClassifier(nnx.Module):
    def __init__(self, ansatz_type="general", **kwargs):
        super().__init__()
        
        
        # 根据ansatz_type选择适当的电路类
        circuit_classes = {
            'CNOT_Amp': CNOTCircuit_Amp,
            'CNOT_angle': CNOTCircuit_angle,
            'QCNN': QCNNCircuit,
            'DataReuploading': DataReuploadingCircuit,
            'CNOT_Hamiltonian': CNOTCircuit_Hamiltonian
        }

        CircuitClass = circuit_classes.get(ansatz_type, BaseQuantumCircuit)
        self.quantum_circuit = CircuitClass(**kwargs)

        
    def __call__(self, x):
        return self.quantum_circuit(x)