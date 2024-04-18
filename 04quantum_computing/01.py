from qutip import *
from qutip.qip.operations import *
import numpy as np
import matplotlib.pyplot as plt

# 创建量子基态
# basis(n, m) 创建长度为n的列向量，并规定第m个元素值1，其余值为0
# 这里创建了量子比特的两个基态  |0⟩ 和 |1⟩
Zero = basis(2,0)
One = basis(2,1)
print(Zero,"\n",One)
# 使用Bloch球展示状态
b = Bloch()
b.add_states([Zero,One])
b.show()

# 创建新的量子叠加态：使用基态 |0⟩ 和 |1⟩ 来形成不同的量子叠加态
# .unit() 归一化，保证量子态的总概率密度为1
psi_x = (Zero + (1+0j) * One).unit()    # ∣ψx⟩
psi_y = (Zero + (0+1j) * One).unit()    # ∣ψy⟩
psi_w = (Zero + (1+1j) * One).unit()    # ∣ψw⟩

b = Bloch()
b.add_states([psi_x,psi_y,psi_w])
b.show()

# 使用Hadamard门（H门）来创建新的量子态从标准基态 |0⟩ 和 |1⟩
H = 1/np.sqrt(2) * Qobj([[1,1],[1,-1]]) # or H = snot()
Psi0 = H * ket("0") # or Zero
Psi1 = H * ket("1") # or One

b = Bloch()
b.add_states([ket("0"),ket("1"),Psi0,Psi1])
b.show()

# 定义三个Pauli矩阵
# 通过基本量子门（以及它们的平方根）来控制和变换量子态
X = sigmax()
Y = sigmay()
Z = sigmaz()
Psi1 = Y * ket("0")
Psi2 = X.sqrtm() * Psi1
Psi3 = Z.sqrtm() * Psi2

b = Bloch()
b.add_states([ket("0"),Psi1,Psi2,Psi3])
b.show()

plt.show()