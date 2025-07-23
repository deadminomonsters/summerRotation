from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas
from pyscf.csf_fci import csf_solver
from pyscf.csf_fci.csfstring import count_all_csfs
import numpy as np
import matplotlib.pyplot as plt

def run(r, a, b):


    mol = gto.M(
        atom = f'H 0 0 0; H {r * np.cos(a)} {r * np.sin(a) + r} 0 ; H {r * np.cos(b)} {r * np.sin(b) + r} 0',
        basis = 'ccpvdz',
        charge = 0,
        spin = 1,
        max_memory = 40000
    )

    ncas = 3     # Number of active orbitals
    nelecas = 3   # Number of active electrons
    mf = scf.RHF(mol).run()


    #Set up solvers for singlet
    solvers = csf_solver(mol, smult=2)
    solvers.nroots = 2
    # Weights
    weights = np.ones(solvers.nroots) / solvers.nroots
    mo_coeff = avas.kernel(mf, ['H 1S'],minao=mol.basis)[2]
    #Run CASSCF
    sing = mcscf.CASSCF(mf, ncas, nelecas)
    sing = mcscf.state_average_mix_(sing, [solvers,], weights)
    sing.kernel(mo_coeff)
    sing_e = sing.e_states

    
    #Set up solvers for triplet
    solvers = csf_solver(mol, smult=4)
    solvers.nroots = count_all_csfs(3, 3, 0, 4)
    # Weights and Coef
    weights = np.ones(solvers.nroots) / solvers.nroots
    mo_coeff = avas.kernel(mf, ['H 1S'],minao=mol.basis)[2]
    #Run CASSCF
    tri = mcscf.CASSCF(mf, ncas, nelecas)
    tri = mcscf.state_average_mix_(tri, [solvers,], weights)
    tri.kernel(mo_coeff)
    tri_e = tri.e_states

    
    return min(sing_e[0], tri_e[0])

r = .7
theta = [i for i in range(0, 360, 4)]
phi = [i for i in range(0, 360, 4)]
lowest_e = 100
lowest_th_ph = [0, 0]


cutoff = 8

for a in theta:
    for b in phi:
        if abs(a-b)>cutoff and abs(a-270)>cutoff and abs(b-270)>cutoff:
            th = a * np.pi / 180
            ph = b * np.pi / 180
            curr_e = run(r, th, ph)
            if curr_e < lowest_e:
                lowest_e = curr_e
                lowest_th_ph[0] = a
                lowest_th_ph[1] = b
                print(f'Energy: {lowest_e}      Theta: {a}      Phi: {b}')



z = np.linspace(0, 2 * np.pi, 100)

# Parametric equations for a circle centered at (0, r)
x = r * np.cos(z)
y = r * np.sin(z) + r

plt.figure(figsize=(5, 5))
plt.plot(x, y, label=f'E = {lowest_e}')
plt.plot(0, 0, marker='o', color='red', markersize=8)
plt.plot(r * np.cos(lowest_th_ph[0] * np.pi / 180), r * np.sin(lowest_th_ph[0] * np.pi / 180) + r, label=f'Theta = {lowest_th_ph[0]}', marker='o', markersize=8)
plt.plot(r * np.cos(lowest_th_ph[1] * np.pi / 180), r * np.sin(lowest_th_ph[1] * np.pi / 180) + r, label=f'Phi = {lowest_th_ph[1]}', marker='o', markersize=8)
plt.gca().set_aspect('equal')  # Equal aspect ratio
plt.title(f'Optimal Geometry for R={r}')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'optGeom{r}.png', dpi=300)
plt.close()