import numpy as np
from scipy.optimize import minimize

# 假设我们有一个2维量子系统（量子比特）
dim = 2**2

# 构建一个幺正矩阵作为我们的量子操作
U = np.array([[1, 0, 0, 0],
              [0, 0, -1, 0],
              [0, 1, 0, 0],
              [0, 0, 0, 1]])

# 定义去极化噪声模型
def depolarizing_channel(rho, p=0.1):
    """
    Applies a depolarizing channel to a density matrix rho with noise parameter p.
    """
    dim = rho.shape[0]
    identity = np.eye(dim)
    return (1-p)*rho + p/dim*identity

# 模拟理想量子操作和噪声的影响
def apply_quantum_operation(rho):
    rho = np.dot(U, rho)
    rho = np.dot(rho, U.conj().T)
    return depolarizing_channel(rho)

# 生成理想和带噪声的数据
ideal_data = [np.outer(b, b.conj()) for b in np.eye(dim)]
noisy_data = [apply_quantum_operation(rho) for rho in ideal_data]

# 定义最大似然估计的目标函数
def likelihood_function(choi_matrix_flat, noisy_data):
    # 将 choi_matrix_flat 转换回矩阵形式
    choi_matrix = choi_matrix_flat.reshape((dim, dim))
    
    # 计算似然值
    likelihood = 0
    for rho in noisy_data:
        estimated_rho = np.dot(choi_matrix, rho)
        likelihood += np.linalg.norm(estimated_rho - rho)**2
    return likelihood

# 初始化 Choi 矩阵为单位矩阵的向量化形式
initial_choi_flat = np.eye(dim).flatten()

# 执行最大似然估计
result = minimize(likelihood_function, initial_choi_flat, args=(noisy_data,), method='L-BFGS-B')

# 恢复 Choi 矩阵
estimated_choi_matrix = result.x.reshape((dim, dim))
print("Estimated Quantum Operation Matrix:\n", estimated_choi_matrix)