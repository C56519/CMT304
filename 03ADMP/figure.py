import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-7, 7)

# Apply the tanh function to these values
y = np.tanh(x)

# Create the plot
plt.figure()
plt.plot(x, y, label='tanh function', color='green')

# Add title and labels
plt.title('Tanh Function')
plt.xlabel('x')
plt.ylabel('y')

# Add a legend
plt.legend()

# Show the plot
#plt.grid(True)
plt.show()