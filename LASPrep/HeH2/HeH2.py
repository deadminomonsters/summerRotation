from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas
from pyscf.csf_fci import csf_solver
from pyscf.csf_fci.csfstring import count_all_csfs
import numpy as np
import matplotlib.pyplot as plt

def run(b):


    mol = gto.M(
        atom = f'H 0 0 {-b}; He 0 0 0; H 0 0 {b}',
        basis = 'ccpvdz',
        charge = 0,
        spin = 0,
        verbose = 4,
        max_memory = 40000
    )

    ncas = 2     # Number of active orbitals
    nelecas = 2   # Number of active electrons
    mf = scf.RHF(mol).run()


    #Set up solvers for singlet
    solvers = csf_solver(mol, smult=1)
    
    solvers.nroots = 3
    # Weights
    weights = np.ones(solvers.nroots) / solvers.nroots
    mo_coeff = avas.kernel(mf, ['H 1S'],minao=mol.basis)[2]
    #Run CASSCF
    sing = mcscf.CASSCF(mf, ncas, nelecas)
    sing = mcscf.state_average_mix_(sing, [solvers,], weights)
    sing.kernel(mo_coeff)
    sing_e = sing.e_states

    
    #Set up solvers for triplet
    solvers = csf_solver(mol, smult=3)
    solvers.nroots = 1
    # Weights and Coef
    weights = np.ones(solvers.nroots) / solvers.nroots
    mo_coeff = avas.kernel(mf, ['H 1S'],minao=mol.basis)[2]
    #Run CASSCF
    tri = mcscf.CASSCF(mf, ncas, nelecas)
    tri = mcscf.state_average_mix_(tri, [solvers,], weights)
    tri.kernel(mo_coeff)
    tri_e = tri.e_states

    return sing_e, tri_e

x = [i/10 for i in range(5, 31, 1)]
sin_e = []
tri_e = []
J = []


for b in x:
    sing, tri = (run(b))
    sin_e.append(sing)
    tri_e.append(tri)
    J.append(219474.6 * (tri[0]-sing[0])/2)

with open('energies.txt', 'w') as fout:
    for i, xi in enumerate(x):
        fout.write(f'{xi}:    {sin_e[i]}    {tri_e[i]}\n')



# Convert Hartree to eV
hartree_to_ev = 27.2114
sin_e_ev = [[e * hartree_to_ev for e in row] for row in sin_e]
tri_e_ev = [[e * hartree_to_ev for e in row] for row in tri_e]


#Plot Energy Spectra
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlabel('Bond Distance (Å)', fontsize=12)
ax.set_ylabel('Energy (eV)', fontsize=12)
ax.set_title('Singlet and Triplet Energy Levels', fontsize=14)

# Plot singlet energy levels as horizontal lines
for i, dist in enumerate(x):
    for energy in sin_e_ev[i]:
        ax.hlines(energy, dist - 0.05, dist + 0.05, color='blue', label='Singlet' if i == 0 else "")

# Plot triplet energy levels as horizontal lines
for i, dist in enumerate(x):
    for energy in tri_e_ev[i]:
        ax.hlines(energy, dist - 0.05, dist + 0.05, color='red', label='Triplet' if i == 0 else "")

# Avoid duplicate legend entries
handles, labels = ax.get_legend_handles_labels()
ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())

plt.tight_layout()
plt.savefig("energy_levels.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(x, J, marker='o', linestyle='-', color='purple')
plt.xlabel('Bond Distance (Å)', fontsize=12)
plt.ylabel('J Value (cm^-1)', fontsize=12)
plt.title('J vs Bond Distance', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('J_vs_distance.png', dpi=300)
plt.close()