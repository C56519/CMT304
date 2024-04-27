from qutip import *
from qutip.qip.operations import cnot

# 初始化基础态 |00>
ket00 = tensor(basis(2, 0), basis(2, 0))

# 定义基础态集合
basis_states = [tensor(basis(2, i), basis(2, j)) for i in range(2) for j in range(2)]

# 假设 F 和 F_inv（这里为简化起见假设是单位矩阵）
F = tensor(qeye(2), qeye(2), qeye(2), qeye(2))
F_inv = F.dag()

# 定义 CNOT 门
CNOT1 = cnot(N=4, control=0, target=2)
CNOT2 = cnot(N=4, control=1, target=3)

# 准备分析结果的数据结构
analysis_results = []

# 遍历所有基础态
for ketX in basis_states:
    # 构建初始状态 |x>|00>
    initial_state = tensor(ketX, ket00)

    # 通过电路
    state_after_F = F * initial_state
    state_after_CNOT1 = CNOT1 * state_after_F
    state_after_CNOT2 = CNOT2 * state_after_CNOT1
    final_state = F_inv * state_after_CNOT2

    # 提取 |A> 和 |B> 的状态
    state_A = ptrace(final_state, [0, 1])
    state_B = ptrace(final_state, [2, 3])

    # 存储分析结果
    analysis_results.append({
        'input_state': ketX,
        'output_state_A': state_A,
        'output_state_B': state_B
    })

# 打印分析结果
for result in analysis_results:
    print("Input State |x>: ", result['input_state'])
    print("Output State |A>: ", result['output_state_A'])
    print("Output State |B>: ", result['output_state_B'])
    print("\n")