import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jit, vmap, random
import numpy as np
from scipy.ndimage import gaussian_filter1d

# 1 读取文件
data = np.loadtxt('measurements.csv', delimiter=',')
# 整理inputs和labels
time = jnp.array(data[:, 0], dtype=jnp.float32).reshape(-1, 1)
voltage = jnp.array(data[:, 1], dtype=jnp.float32).reshape(-1, 1)

sigma = 10  # 高斯核的标准差
time = gaussian_filter1d(time.flatten(), sigma).reshape(-1, 1)
voltage = gaussian_filter1d(voltage.flatten(), sigma).reshape(-1, 1)

# 超参数
neurons = 200           # 每层神经元个数
lr = 0.002              # 学习率
training_num = 20000    # 训练次数

# 2 构建神经网络
def nn(params, x):
    # W权重矩阵 b偏置向量
    W1, b1, W2, b2, W3, b3, W4, b4 = params
    # 三层隐藏层
    # 第一层：h1 = tanh(x ⋅ W1 + b1)
    h1 = jnp.tanh(jnp.dot(x, W1) + b1)
    # 第二层：h2 = tanh(h1 · W2 + b2)
    h2 = jnp.tanh(jnp.dot(h1, W2) + b2)
    # 第三层：h3 = tanh(h2 · W3 + b3)
    h3 = jnp.tanh(jnp.dot(h2, W3) + b3)
    # 输出层
    y = jnp.dot(h3, W4) + b4
    return y


# 3 损失函数
def loss(params, x, y):
    y_pred = nn(params, x)
    # 均方误差MSE
    return jnp.mean((y - y_pred)**2)

# 4 初始化三个隐藏层的参数
inp_size = 1
output_size = 1
# 按照正态分布随机分发参数的值
key = random.PRNGKey(0)
key, *subkeys = random.split(key, 9)

params = [
    random.normal(subkeys[0], (inp_size, neurons)), jnp.zeros(neurons),
    random.normal(subkeys[1], (neurons, neurons)), jnp.zeros(neurons),
    random.normal(subkeys[2], (neurons, neurons)), jnp.zeros(neurons),
    random.normal(subkeys[3], (neurons, output_size)), jnp.zeros(output_size)
]


# 5 效率优化
# 将损失函数和损失函数的参数编译成高效JIT编译版本，提高代码执行效率
c_loss = jit(loss)
d_loss = jit(grad(loss))
# 向量化神经网络nn的预测函数，允许批量输入神经网络，并行计算，提高效率
v_nn = jit(vmap(nn, (None, 0)))

# 6 梯度下降找最优解
def update_params(params, x, y, lr):
    grads = d_loss(params, x, y)
    params = [param - lr * grad for param, grad in zip(params, grads)]
    return params

# 7 运行梯度下降
err = []
for i in range(training_num):
    # 计算每次训练的损失存入列表
    err.append(c_loss(params, time, voltage))
    # 更新参数
    params = update_params(params, time, voltage, lr)
    # 每500次迭代打印损失值
    if i % 500 == 0:
        print(f"Iteration {i}: Loss = {err[-1]}")
# 记录训练结束时的最终损失值
err.append(c_loss(params, time, voltage))

# 8 画图
# Plot loss and predictions
plt.figure(figsize=(14, 7))

# Loss plot
plt.subplot(1, 2, 1)
plt.semilogy(err)
plt.title("Training Loss")

# Prediction plot
plt.subplot(1, 2, 2)
plt.plot(time, voltage, label="Ground Truth")
plt.plot(time, v_nn(params, time), label="Predictions", linestyle='dashed')
plt.title("Predictions vs. Ground Truth")
plt.legend()

plt.tight_layout()
plt.show()