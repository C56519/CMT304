import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit

# 1 Load the dataset and divide it into input and lable lists.
file_path = 'measurements.csv'
data = np.loadtxt(file_path, delimiter=',')
time = np.array(data[:, 0])
voltage = np.array(data[:, 1])
# Transform into the format demanded by tensorflow.
time = time.reshape(-1, 1)

# 2 Hyperparameters
MODEL_LEARNING_RATE = 0.001     # Learning rate of the optimiser
NEURONS_NUMBERS = 100           # Number of neurons
TRAINING_EPOCHS = 1000          # Number of training epochs
VALIDATION_RATES = 0.2          # Rate of dividing the validation set

# 3 Building Neural Networks.
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(NEURONS_NUMBERS, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(NEURONS_NUMBERS, activation='relu'),
        tf.keras.layers.Dense(NEURONS_NUMBERS, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=MODEL_LEARNING_RATE)
    model.compile(loss = 'mse',
                  optimizer = optimizer,
                  metrics=['mae', 'mse'])
    return model


# Start to build the model.
model = build_model()

# 4 Train the model.
history = model.fit(
    time, voltage, epochs = TRAINING_EPOCHS,
    validation_split = VALIDATION_RATES,
    verbose = 1
)

# 5 Test
evaluation = model.evaluate(time, voltage, verbose=0)
print(evaluation)
predictions = model.predict(time)
# Save prediction results locally.
np.savetxt('predictions_1.csv', predictions)

# 6 Draw graphs
# (1) Graph of loss changes during training
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# (3) Graph of prediction and actual voltage comparison
plt.figure(figsize=(14, 6))
plt.scatter(time, predictions, label='Predicted Voltage', color='red', s=2)
plt.scatter(time, voltage, label='True Voltage', alpha=0.6, color='blue', s=2)
plt.title('Predicted Voltage & True Voltage')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.legend()
plt.show()