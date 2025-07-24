import numpy as np
import matplotlib.pyplot as plt


r = [0.3, 0.7, 1.0, 1.5]
energy = [-0.5378574003221939, -2.130342309122369, -2.2451909580830285, -2.24502331716132]
theta = [100, 80, 40, 300]
phi = [180, 200, 220, 80]
gamma = [0, 20, 80, 120]
color = ['red', 'green', 'blue', 'pink']

z = np.linspace(0, 2 * np.pi, 100)

plt.figure(figsize=(8, 5))
# Parametric equations for a circle centered at (0, r)
for i in range(0, len(r)):
    x = r[i] * np.cos(z)
    y = r[i] * np.sin(z) + r[3]
    plt.plot(x, y, label=f'E = {energy[i]}', color = color[i])

for i in range(0, len(r)):
    plt.plot(0, r[3] - r[i], marker='o', color=color[i], markersize=8)
    plt.plot(r[i] * np.cos(theta[i] * np.pi / 180), r[i] * np.sin(theta[i] * np.pi / 180) + r[3], color = color[i], marker='o', markersize=8)
    plt.plot(r[i] * np.cos(phi[i] * np.pi / 180), r[i] * np.sin(phi[i] * np.pi / 180) + r[3], color = color[i], marker='o', markersize=8)
    plt.plot(r[i] * np.cos(gamma[i] * np.pi / 180), r[i] * np.sin(gamma[i] * np.pi / 180) + r[3], color = color[i], marker='o', markersize=8)

plt.title(f'All Geometries')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(f'final.png', dpi=300)
plt.close()