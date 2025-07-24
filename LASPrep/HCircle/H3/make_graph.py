import numpy as np
import matplotlib.pyplot as plt



r = [0.3, 0.7, 1.0, 1.5]
energy = [-1.1336074000628167, -1.590978084940520, -1.6260886593151958, -1.6350640247502377]
theta = [148, 60, 68, 104]
phi = [32, 124, 224, 300]
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

plt.title(f'All Geometries')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(f'final.png', dpi=300)
plt.close()