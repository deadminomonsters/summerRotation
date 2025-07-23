from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas
from pyscf.csf_fci import csf_solver
from pyscf.csf_fci.csfstring import count_all_csfs
import numpy as np
import matplotlib.pyplot as plt

def run(r, a, b, c):


    mol = gto.M(
        atom = f'H 0 0 0; H {r * np.cos(a)} {r * np.sin(a) + r} 0 ; H {r * np.cos(b)} {r * np.sin(b) + r} 0; H {r * np.cos(c)} {r * np.sin(c) + r} 0',
        basis = 'ccpvdz',
        charge = 0,
        spin = 0,
        max_memory = 40000
    )

    ncas = 4     # Number of active orbitals
    nelecas = 4   # Number of active electrons
    mf = scf.RHF(mol).run()


    #Set up solvers for singlet
    solvers = csf_solver(mol, smult=1)
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
    solvers = csf_solver(mol, smult=3)
    solvers.nroots = 2
    # Weights and Coef
    weights = np.ones(solvers.nroots) / solvers.nroots
    mo_coeff = avas.kernel(mf, ['H 1S'],minao=mol.basis)[2]
    #Run CASSCF
    tri = mcscf.CASSCF(mf, ncas, nelecas)
    tri = mcscf.state_average_mix_(tri, [solvers,], weights)
    tri.kernel(mo_coeff)
    tri_e = tri.e_states

    #Set up solvers for triplet
    solvers = csf_solver(mol, smult=5)
    solvers.nroots = count_all_csfs(4, 4, 0, 5)
    # Weights and Coef
    weights = np.ones(solvers.nroots) / solvers.nroots
    mo_coeff = avas.kernel(mf, ['H 1S'],minao=mol.basis)[2]
    #Run CASSCF
    quad = mcscf.CASSCF(mf, ncas, nelecas)
    quad = mcscf.state_average_mix_(quad, [solvers,], weights)
    quad.kernel(mo_coeff)
    quad_e = quad.e_states

    
    return min(sing_e[0], tri_e[0], quad_e[0])

r = .3
theta = [i for i in range(0, 360, 20)]
phi = [i for i in range(0, 360, 20)]
gam = [i for i in range(0, 360, 20)]
lowest_e = 100
lowest_th_ph_ga = [0, 0, 0]

cutoff = 4

for a in theta:
    for b in phi:
        for c in gam:
            if abs(a-b)>cutoff and abs(a-c)>cutoff and abs(c-b)>cutoff:
                if abs(a-270)>cutoff and abs(b-270)>cutoff and abs(c-270)>cutoff:
                    th = a * np.pi / 180
                    ph = b * np.pi / 180
                    ga = c * np.pi / 180
                    curr_e = run(r, th, ph, ga)
                    if curr_e < lowest_e:
                        lowest_e = curr_e
                        lowest_th_ph_ga[0] = a
                        lowest_th_ph_ga[1] = b
                        lowest_th_ph_ga[2] = c
                        print(f'Energy: {lowest_e}      Theta: {a}      Phi: {b}        Gamma: {c}')



z = np.linspace(0, 2 * np.pi, 100)

# Parametric equations for a circle centered at (0, r)
x = r * np.cos(z)
y = r * np.sin(z) + r

plt.figure(figsize=(5, 5))
plt.plot(x, y, label=f'E = {lowest_e}')
plt.plot(0, 0, marker='o', color='red', markersize=8)
plt.plot(r * np.cos(lowest_th_ph_ga[0] * np.pi / 180), r * np.sin(lowest_th_ph_ga[0] * np.pi / 180) + r, label=f'Theta = {lowest_th_ph_ga[0]}', marker='o', markersize=8)
plt.plot(r * np.cos(lowest_th_ph_ga[1] * np.pi / 180), r * np.sin(lowest_th_ph_ga[1] * np.pi / 180) + r, label=f'Phi = {lowest_th_ph_ga[1]}', marker='o', markersize=8)
plt.plot(r * np.cos(lowest_th_ph_ga[2] * np.pi / 180), r * np.sin(lowest_th_ph_ga[2] * np.pi / 180) + r, label=f'Gamma = {lowest_th_ph_ga[2]}', marker='o', markersize=8)
plt.gca().set_aspect('equal')  # Equal aspect ratio
plt.title(f'Optimal Geometry for R={r}')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'optGeom{r}.png', dpi=300)
plt.close()