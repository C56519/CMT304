import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jit, vmap, random
import numpy as np

# 1 Load file.
data = np.loadtxt('measurements.csv', delimiter=',')
# Organising inputs and labels.
time = jnp.array(data[:, 0], dtype=jnp.float32).reshape(-1, 1)
voltage = jnp.array(data[:, 1], dtype=jnp.float32).reshape(-1, 1)

# Hyperparameters
NEURONS = 200           # Number of neurons per layer.
LEARNING_RATE = 0.002   # Learning rate.
TRAINING_NUM = 10000    # Number of trainings.

# 3 Building the structure of a neural network.
def nn(params, x):
    """
    The basic structure of a neural network.
    :param params: Random parameter list.
    :param x: Inputs(time).
    :return: Outputs(Predicted voltage).
    """
    # W weight matrix, b bias vector
    W1, b1, W2, b2, W3, b3, W4, b4 = params
    # hidden layer
    # First layer: h1 = tanh(x ⋅ W1 + b1)
    h1 = jnp.tanh(jnp.dot(x, W1) + b1)
    # Second layer: h2 = tanh(h1 · W2 + b2)
    h2 = jnp.tanh(jnp.dot(h1, W2) + b2)
    # Third layer: h3 = tanh(h2 · W3 + b3)
    h3 = jnp.tanh(jnp.dot(h2, W3) + b3)
    # Output layer
    y = jnp.dot(h3, W4) + b4
    return y

# 4 Loss Function
def loss(params, x, y):
    """
    Use the neural network to predict the voltage values and calculate the mean square error for this training.
    :param params: Random parameter list.
    :param x: Inputs(time).
    :param y: Outputs(True voltage).
    :return: The mean square error value for this training.
    """
    y_pred = nn(params, x)
    # Mean Square Error MSE
    return jnp.mean((y - y_pred)**2)

# 5 Initialise the parameters of the hidden layer
inp_size = 1
output_size = 1
# Randomly distribute the values of the parameters according to the normal distribution
key = random.PRNGKey(0)
key, *subkeys = random.split(key, 9)
params = [
    random.normal(subkeys[0], (inp_size, NEURONS)), jnp.zeros(NEURONS),
    random.normal(subkeys[1], (NEURONS, NEURONS)), jnp.zeros(NEURONS),
    random.normal(subkeys[2], (NEURONS, NEURONS)), jnp.zeros(NEURONS),
    random.normal(subkeys[3], (NEURONS, output_size)), jnp.zeros(output_size)
]

# 6 Efficiency optimisation
# Compile to an efficient JIT compilation version to improve code efficiency
c_loss = jit(loss)
d_loss = jit(grad(loss))
# Allows batch input to neural networks and parallel computation for increased efficiency
v_nn = jit(vmap(nn, (None, 0)))

# 7 Update the parameters
def update_params(params, x, y):
    """
    Execute the gradient descent algorithm and update the parameters, to find the optimal solution.
    :param params: Random parameter list.
    :param x: Inputs(time).
    :param y: Outputs(True voltage).
    :return: Updated parameter list.
    """
    grads = d_loss(params, x, y)
    params = [param - LEARNING_RATE * grad for param, grad in zip(params, grads)]
    return params

# 8 Running programme
err = []
for i in range(TRAINING_NUM):
    # Computing the loss for each training and store them into a list.
    err.append(c_loss(params, time, voltage))
    # Updating parameters
    params = update_params(params, time, voltage)
    # Print loss values every 500 iterations
    if i % 500 == 0:
        print(f"Iteration {i}: Loss = {err[-1]}")
err.append(c_loss(params, time, voltage))

# 9 Drawing
plt.figure(figsize=(14, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.semilogy(err, color='blue')
plt.xlabel('Number of trainings')
plt.ylabel('Loss')
plt.title("Training loss")

# V plot
plt.subplot(1, 2, 2)
plt.plot(time, voltage, label="True voltage", alpha=0.6, color='red')
plt.plot(time, v_nn(params, time), label="Predicted voltage", linestyle='dashed', color='green')
plt.xlabel('Time(s)')
plt.ylabel('Voltage(V)')
plt.title("Predicted voltage vs. True voltage")
plt.legend()

plt.tight_layout()
plt.show()