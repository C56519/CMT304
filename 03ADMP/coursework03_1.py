import numpy as np
import jax.numpy as jnp
from jax import grad, jit
import matplotlib.pyplot as plt

# 0 Hyperparameters
LEARNING_RATE = 0.01
TRAINING_NUMBERS = 30000
# alpha, A, B, beta
INITIAL_PARAMTERS = [0.0, 1.0, 0.1, 9.0]

# 1 Read the CSV file and load the data.
data = np.loadtxt('measurements.csv', delimiter=',', skiprows=1)
time = data[:, 0]
voltage = data[:, 1]

# 2 Define the circuit equation.
def circuit_equation(t, params):
    alpha, A, B, beta = params
    # V(t) = e^(αt) * (A*cos(βt) + B*sin(βt))
    vt = jnp.exp(alpha * t) * (A * jnp.cos(beta * t) + B * jnp.sin(beta * t))
    return vt

# 3 Loss function.
def loss_function(params, t, actual_v):
    predicted_v = circuit_equation(t, params)
    # Using mean square error.
    loss = jnp.mean((predicted_v - actual_v) ** 2)
    return loss

# 4 Parameter optimiser to update parameters.
# 4.1 Computing the gradient of the loss function.
gradient_function = jit(grad(loss_function))
# 4.2 Define the parameter update function.
def parameters_update(params, time, voltage): 
    current_gradient = gradient_function(params, time, voltage)
    new_params = params - LEARNING_RATE * current_gradient
    return new_params

# 4.3 Perform optimization.
loss_history = []
params = jnp.array(INITIAL_PARAMTERS)
for i in range(TRAINING_NUMBERS):
    params = parameters_update(params, time, voltage)
    current_loss = loss_function(params, time, voltage)
    loss_history.append(current_loss)
    if (i+1) % 500 == 0 or (i+1) == TRAINING_NUMBERS:
        print(f"Iteration number {i+1}: Loss = {current_loss}")

# 5 Print the final paremeters.
final_params = params
# Generate fitted values using the final parameters.
voltage_prediction = circuit_equation(time, final_params)
# Print the final parameters.
print(f"Final paremeters:")
print(f"α: {final_params[0]}")
print(f"A: {final_params[1]}")
print(f"B: {final_params[2]}")
print(f"β: {final_params[3]}")
print(f"Final equation: v(t) = e^({final_params[0]:.2f}t) * ({final_params[1]:.2f}*cos({final_params[3]:.2f}t) + {final_params[2]:.2f}*sin({final_params[3]:.2f}t))")

# 6 Drawing.
plt.figure(figsize=(14, 6))
# Loss plot.
plt.subplot(1, 2, 1)
plt.semilogy(loss_history, color='blue')
plt.xlabel('Number of trainings')
plt.ylabel('Loss')
plt.title("Training loss")

# Votage plot.
plt.subplot(1, 2, 2)
plt.plot(time, voltage, label='True Votage', color='red', alpha=0.6)
plt.scatter(time, voltage_prediction, label="Predicted Voltage", s=2, color='green')
plt.title('True Voltage vs Predicted voltage')
plt.xlabel('Time(s)')
plt.ylabel('Voltage(V)')
plt.title("Predicted voltage")
plt.legend()

plt.tight_layout()
plt.show()