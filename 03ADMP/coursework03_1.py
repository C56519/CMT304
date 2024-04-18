import numpy as np
import jax.numpy as jnp
from jax import grad, jit
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt

# 0 Hyperparameters
LEARNING_RATE = 0.001
TRAINING_NUMBERS = 4000
# A, gamma, omega, phi, B
INITIAL_PARAMTERS = [1.0, 1.0, 9.0, 0.0, 0.0]

# 1 Read the CSV file and load the data.
data = np.loadtxt('measurements.csv', delimiter=',', skiprows=1)
time = data[:, 0]
voltage = data[:, 1]

# 2 Define the circuit equation.
def circuit_equation(t, params):
    A, gamma, omega, phi, B = params
    # v(t) = A * e^(−γt) * cos(ωt+ϕ) + B
    equation = A * jnp.exp(-gamma * t) * jnp.cos(omega * t + phi) + B
    return equation

# 3 Loss function.
def loss_function(params, t, actual_v):
    predicted_v = circuit_equation(t, params)
    # Using mean square error.
    loss = jnp.mean((predicted_v - actual_v) ** 2)
    return loss

# 4 Parameter optimiser to update parameters.
# 4.1 Create an optimizer.
initial_params, update_params, get_new_params = optimizers.adam(step_size=LEARNING_RATE)
opt_state = initial_params(INITIAL_PARAMTERS)
# 4.2 Computing the gradient of the loss function.
gradient_function = jit(grad(loss_function))
# 4.3 Define the parameter update function.
def paramters_update_function(i, opt_state, time, voltage):
    params = get_new_params(opt_state)
    current_gradient = gradient_function(params, time, voltage)
    new_opt_state = update_params(i, current_gradient, opt_state)
    return new_opt_state, params, current_gradient

# 4.4 Perform optimization.
loss_history = []
for i in range(TRAINING_NUMBERS):
    opt_state, current_params, current_gradient = paramters_update_function(i, opt_state, time, voltage)
    current_loss = loss_function(current_params, time, voltage)
    loss_history.append(current_loss)
    if i % 500 == 0 or i == TRAINING_NUMBERS:
        print(f"Iteration number {i}: Loss = {current_loss}.")

# 5 Print the final paremeters.
final_params = get_new_params(opt_state)
# Generate fitted values using the final parameters.
fitted_voltage = circuit_equation(time, final_params)
# Print the final parameters.
print(f"Final paremeters:")
print(f"A: {final_params[0]}")
print(f"gamma: {final_params[1]}")
print(f"omega: {final_params[2]}")
print(f"phi: {final_params[3]}")
print(f"B: {final_params[4]}")

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
plt.scatter(time, fitted_voltage, label="Predicted Voltage", s=2, color='green')
plt.title('True Voltage vs Predicted voltage')
plt.xlabel('Time(s)')
plt.ylabel('Voltage(V)')
plt.title("Predicted voltage")
plt.legend()

plt.tight_layout()
plt.show()