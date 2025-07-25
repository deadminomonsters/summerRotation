import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import re

# Arrays to hold data
distances = []
scf_energy = []
las_energy = []
all_singlets = []  # list of lists
all_triplets = []  # list of lists
all_quintets = []

# Regex patterns
distance_pattern = re.compile(r"Distance:\s*([0-9.]+)")
scf_energy_line = re.compile(r"converged\sSCF\senergy\s*=\s(-?[0-9.]+)")
las_energy_line = re.compile(r"LASCI\sE\s*=\s(-?[0-9.]+)")
cas_energy_line = re.compile(r"E\s*=\s*(-?[0-9.]+)\s*S\^2\s*=\s*(-?\d+(?:\.\d+)?)")

# Temporary holders
singlet = []
triplet = []
quintet = []

# Read and parse the file
with open('job.out', 'r') as f:
    for line in f:
        dist_match = distance_pattern.search(line)
        scf_match = scf_energy_line.search(line)
        las_match = las_energy_line.search(line)
        cas_match = cas_energy_line.search(line)

        # Found a new distance → store old values, reset for new block
        if dist_match:
            if singlet or triplet:
                all_singlets.append(singlet)
                all_triplets.append(triplet)
                all_quintets.append(quintet)
                singlet = []
                triplet = []
                quintet = []

            b = float(dist_match.group(1))
            distances.append(b)
        
        # Found an energy line
        elif scf_match:
            scf_energy.append(27.2114 * float(scf_match.group(1)))


        elif las_match:
            las_energy.append(27.2114 * float(las_match.group(1)))


        elif cas_match:
            energy = 27.2114 * float(cas_match.group(1))
            s2 = float(cas_match.group(2))

            if abs(s2 - 0.0) < 1e-3:
                singlet.append(energy)
            elif abs(s2 - 2.0) < 1e-3:
                triplet.append(energy)
            elif abs(s2 - 6.0) < 1e-3:
                quintet.append(energy)

    # Handle last block after file ends
    if singlet or triplet:
        all_singlets.append(singlet)
        all_triplets.append(triplet)
        all_quintets.append(quintet)


cas_energy = []
for i, xi in enumerate(distances):
    cas_energy.append(all_singlets[i][0])


plt.figure(figsize=(6, 4))
plt.plot(distances, scf_energy, 'o-', label='SCF Energy', color='green')
plt.plot(distances, las_energy, 'o-', label='LAS Energy', color='blue')
plt.plot(distances, cas_energy, 's-', label='CAS Energy', color='red')

plt.xlabel('Distance (Å)', fontsize=14)
plt.ylabel('Energy (eV)', fontsize=14)
plt.title('PES for Two H2 Molecules', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('PES.png', dpi=300)
plt.close()


# #J
# J = []
# for i, d in enumerate(distances):
#     J.append(219474 * (all_triplets[i][0] - all_singlets[i][0]) / 2 )

# plt.figure(figsize=(8, 5))
# plt.plot(distances, J, marker='o', linestyle='-', color='purple', label='J coupling')

# # Labeling
# plt.xlabel('Distance (Å)', fontsize=14)
# plt.ylabel('J (Hartree)', fontsize=14)
# plt.title('J Coupling vs Distance', fontsize=15)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend()

# # Optional: Save
# plt.savefig('J.png', dpi=300)

# plt.tight_layout()
# plt.close()