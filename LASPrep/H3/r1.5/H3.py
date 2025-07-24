from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas
from pyscf.csf_fci import csf_solver
from pyscf.csf_fci.csfstring import count_all_csfs
import numpy as np
import matplotlib.pyplot as plt

def run(b):


    mol = gto.M(
        atom = f'H 0 0 0; H 0 { b * (3 / 4)**(1/2)} {b/2}; H 0 0 {b}',
        basis = 'ccpvdz',
        charge = 0,
        spin = 1,
        verbose = 4,
        max_memory = 40000
    )

    ncas = 3     # Number of active orbitals
    nelecas = 3   # Number of active electrons
    mf = scf.RHF(mol).run()


    #Set up solvers for singlet
    solvers = csf_solver(mol, smult=2)
    solvers.nroots = count_all_csfs(3, 2, 1, 2)
    # Weights
    weights = np.ones(solvers.nroots) / solvers.nroots
    mo_coeff = avas.kernel(mf, ['H 1S'],minao=mol.basis)[2]
    #Run CASSCF
    sing = mcscf.CASSCF(mf, ncas, nelecas)
    sing = mcscf.state_average_mix_(sing, [solvers,], weights)
    sing.kernel(mo_coeff)
    sing_e = sing.e_states[0]

    
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
    tri_e = tri.e_states[0]

    from pyscf.tools import molden

    # Write cube files for each active orbital
    molden.from_mo(mol, f"triangle1.5.molden", sing.mo_coeff[:, sing.ncore:sing.ncore+sing.ncas])

    return min(sing_e, tri_e)

r = 1.5
d = (3 ** (1/2)) * r
energy = run(d)

z = np.linspace(0, 2 * np.pi, 100)

# Parametric equations for a circle centered at (0, r)
x = r * np.cos(z)
y = r * np.sin(z) + r

x1 = r * np.cos(np.pi/6)
x2 = -1 * x1
y1 = r * np.sin(np.pi/6) + r

plt.figure(figsize=(5, 5))
plt.plot(x, y, label=f'E = {energy}')
plt.plot(0, 0, marker='o', color='red', markersize=8)
plt.plot(x1, y1, marker='o', color='red', markersize=8)
plt.plot(x2, y1, marker='o', color='red', markersize=8)
plt.gca().set_aspect('equal')  # Equal aspect ratio
plt.title(f'Equilateral Triangle for R={r}')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'optGeom{r}.png', dpi=300)
plt.close()