import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

np.random.seed(40)

# Generate random data
x = [i for i in range(1,6)]
y = np.random.rand(5)

# Create scatter plot
fig, ax = plt.subplots()
sc = ax.scatter(x, y, label="Data")
plt.xlabel('Embedding')
plt.ylabel('Tensor Value')
plt.xlim(0, 6)
plt.xticks(np.arange(1, 6))
plt.grid(axis='x', linestyle='--')
plt.title('Data with Sign-Controlled Laplacian Noise Added')

# Add laplacian noise
controlled = [-0.06, 0.2, -0.05, 0.1, 0.04]
yn = y + controlled

# Add arrows
for i in range(len(x)):
    plt.arrow(x[i], y[i], 0, controlled[i], head_width=0.3, head_length=0.02, fc='black', ec='black')

# Add data centers
yc = np.random.rand(5)
ax.scatter(x, yc, label="Group Center", color="green", s=100)
# plt.legend()

# Define update function for animation
def update(frame):
    if frame == 0:
        return sc,
    else:
        for i in range(len(x)):
            y[i] += controlled[i] * 0.05
        sc.set_offsets(np.c_[x, y])
        return sc,

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(20), blit=True)

# Save animation as GIF
ani.save('scatter_animation.gif', writer='pillow', fps=10)

# plt.show()

