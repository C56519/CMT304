from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import state_fidelity, process_fidelity
from qiskit_experiments.library import ProcessTomography

# 构建量子电路
qc = QuantumCircuit(4)

# 添加未知操作 F 的占位符
# 为了执行过程层析，我们不定义具体的 F
qc.append(F_gate, [0, 1])

# 添加两个 CNOT 门
qc.cx(0, 2)
qc.cx(1, 3)

# 添加逆操作 F^-1 的占位符
qc.append(F_gate_inverse, [0, 1])

# 定义模拟器
simulator = Aer.get_backend('aer_simulator')

# 执行过程层析
exp = ProcessTomography(qc)
data = exp.run(simulator).block_for_results()

# 通过拟合数据来重建操作 F
results = data.analysis_results()
fitted_process = results['process_matrix'].data