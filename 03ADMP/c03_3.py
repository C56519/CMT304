import numpy as np
import jax.numpy as jnp
from jax import grad, random

# 加载数据集
file_path = 'measurements.csv'
data = np.loadtxt(file_path, delimiter=',')
time = np.array(data[:, 0])
voltage = np.array(data[:, 1])

steps = data.shape[0]
dt = time[1] - time[0]
LEARNING_RATE = 0.0005  # 学习率

# ODE求解器, 传入参数计算x处的预测电压值
def ode_solver(params, x0, dx0):
    a, b = params
    x, dx = x0, dx0
    voltage_pred = [x0]
    for ti in range(1, steps):
        ddx = b * x - a * dx
        dx += ddx * dt
        x += dx * dt
        voltage_pred.append(x)
    return jnp.array(voltage_pred)

# 损失函数
def loss(params, x0, dx0, voltage):
    voltage_pred = ode_solver(params, x0, dx0)
    return jnp.mean((voltage - voltage_pred) ** 2)

# 初始化参数
key = random.PRNGKey(0)
params = random.uniform(key, (2,), minval=-0.5, maxval=0.5)

# 更新参数
# 使用jax库计算梯度
d_loss = grad(loss)
def update_params(params, x0, dx0, voltage):
    grads = d_loss(params, x0, dx0, voltage)
    params -= LEARNING_RATE * grads
    return params

# Run: 运行梯度下降
# 每次迭代都求出均方误差
for i in range(100):
    err = loss(params, voltage[0], 0, voltage)
    print(f"Iter {i}: error={err:.4f} - parameters=[{params[0]:.4f},{params[1]:.4f}]")
    params = update_params(params, voltage[0], 0, voltage)
