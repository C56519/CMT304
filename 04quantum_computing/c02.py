from qutip import *
import numpy as np
from scipy.optimize import minimize
from qutip.qip.operations import cnot, rz

# 初始化系统和基态
num_qubits = 4
ket00 = tensor(basis(2, 0), basis(2, 0))
basis_states = [tensor(basis(2, i), basis(2, j)) for i in range(2) for j in range(2)]

def construct_F(theta):
    # 确保 theta 是单一浮点数，并直接用作 rz 的参数
    rotation = rz(theta[0])  # 使用 theta[0] 以确保其为单个值
    F_matrix = tensor(rotation, qeye(2))  # 只在第一个量子比特上应用旋转
    return F_matrix

CNOT1 = tensor(cnot(), qeye(2))
CNOT2 = tensor(qeye(2), cnot())

def run_circuit(F, input_state):
    F_inv = F.dag()
    initial_state = tensor(input_state, ket00)
    state_after_F = F * initial_state
    state_after_CNOT1 = CNOT1 * state_after_F
    state_after_CNOT2 = CNOT2 * state_after_CNOT1
    final_state = F_inv * state_after_CNOT2
    return final_state

def cost_function(theta):
    F = construct_F(theta)
    cost = 0
    for ketX in basis_states:
        final_state = run_circuit(F, ketX)
        cost -= fidelity(final_state.ptrace([0, 1]), ketX) ** 2
    return cost

initial_theta = np.array([0.1])  # 以数组形式初始化，确保传入的是单一浮点数
result = minimize(cost_function, x0=initial_theta, method='L-BFGS-B')

optimized_theta = result.x
optimized_F = construct_F(optimized_theta)
print("优化后的 F 门:")
print(optimized_F)
