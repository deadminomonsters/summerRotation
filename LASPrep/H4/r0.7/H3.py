from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas
from pyscf.csf_fci import csf_solver
from pyscf.csf_fci.csfstring import count_all_csfs
import numpy as np
import matplotlib.pyplot as plt

def run(b):


        mol = gto.M(
        atom = f'H 0 0 0; H 0 {2*b} 0 ; H {b} {b} 0; H {-1*b} {b} 0',
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

r = 0.7
energy = run(r)

z = np.linspace(0, 2 * np.pi, 100)

# Parametric equations for a circle centered at (0, r)
x = r * np.cos(z)
y = r * np.sin(z) + r
plt.figure(figsize=(5, 5))
plt.plot(x, y, label=f'E = {energy}')
plt.plot(0, 0, marker='o', color='red', markersize=8)
plt.plot(0, 2*r, marker='o', color='red', markersize=8)
plt.plot(r, r, marker='o', color='red', markersize=8)
plt.plot(-1*r, r, marker='o', color='red', markersize=8)
plt.gca().set_aspect('equal')  # Equal aspect ratio
plt.title(f'Square for R={r}')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'optGeom{r}.png', dpi=300)
plt.close()