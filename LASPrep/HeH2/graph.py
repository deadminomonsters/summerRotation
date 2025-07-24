import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import re

# Arrays to hold data
distances = []
all_singlets = []  # list of lists
all_triplets = []  # list of lists

# Regex patterns
distance_pattern = re.compile(r"Distance\s*=\s*([0-9.]+)")
energy_line_pattern = re.compile(r"E\s*=\s*(-?[0-9.]+)\s*S\^2\s*=\s*([0-9.]+)")

# Temporary holders
singlet = []
triplet = []

# Read and parse the file
with open('job.out', 'r') as f:
    for line in f:
        dist_match = distance_pattern.search(line)
        energy_match = energy_line_pattern.search(line)

        # Found a new distance → store old values, reset for new block
        if dist_match:
            if singlet or triplet:
                all_singlets.append(singlet)
                all_triplets.append(triplet)
                singlet = []
                triplet = []

            b = float(dist_match.group(1))
            distances.append(b)

        # Found an energy line
        elif energy_match:
            energy = 27.2114 * float(energy_match.group(1))
            s2 = float(energy_match.group(2))

            if abs(s2 - 0.0) < 1e-3:
                singlet.append(energy)
            elif abs(s2 - 2.0) < 1e-3:
                triplet.append(energy)

    # Handle last block after file ends
    if singlet or triplet:
        all_singlets.append(singlet)
        all_triplets.append(triplet)



fig, ax = plt.subplots(figsize=(6, 8))

# Loop over each distance
for i, d in enumerate(distances):
    # Offset singlets to the left, triplets to the right
    x_singlet = d
    x_triplet = d

    # Plot singlet levels
    for energy in all_singlets[i]:
        ax.hlines(energy, x_singlet - 0.02, x_singlet + 0.02, color='blue', linewidth=2)

    # Plot triplet levels
    for energy in all_triplets[i]:
        ax.hlines(energy, x_triplet - 0.02, x_triplet + 0.02, color='red', linewidth=2)

# Labeling and axis setup
ax.set_xlabel('Distance (Å)', fontsize=14)
ax.set_ylabel('Energy (eV)', fontsize=14)
ax.set_title('Singlet and Triplet Energy Spectra vs. Distance', fontsize=15)

# Add legend manually
custom_lines = [Line2D([0], [0], color='blue', lw=2),
                Line2D([0], [0], color='red', lw=2)]
ax.legend(custom_lines, ['Singlet', 'Triplet'])

# Optional: tighter y-limits
all_energies = sum(all_singlets, []) + sum(all_triplets, [])
ax.set_ylim(min(all_energies) - 0.1, max(all_energies) + 0.1)

plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('E_spectra.png', dpi=300)
plt.close()


#J
J = []
for i, d in enumerate(distances):
    J.append(219474 * (all_triplets[i][0] - all_singlets[i][0]) / 2 )

plt.figure(figsize=(8, 5))
plt.plot(distances, J, marker='o', linestyle='-', color='purple', label='J coupling')

# Labeling
plt.xlabel('Distance (Å)', fontsize=14)
plt.ylabel('J (Hartree)', fontsize=14)
plt.title('J Coupling vs Distance', fontsize=15)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# Optional: Save
plt.savefig('J.png', dpi=300)

plt.tight_layout()
plt.close()