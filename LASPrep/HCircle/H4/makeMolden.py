from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas
from pyscf.csf_fci import csf_solver
from pyscf.csf_fci.csfstring import count_all_csfs
import numpy as np
import matplotlib.pyplot as plt

r = [0.3, 0.7, 1.0, 1.5]
theta = [100, 80, 40, 300]
phi = [180, 200, 220, 80]
gamma = [0, 20, 80, 120]

mol = gto.M(
f'H 0 0 0; H {r[3] * np.cos(theta[3] * np.pi/180)} {r[3] * np.sin(theta[3] * np.pi/180) + r[3]} 0 ; H {r[3] * np.cos(phi[3] * np.pi/180)} {r[3] * np.sin(phi[3] * np.pi/180) + r[3]} 0; H {r[3] * np.cos(gamma[3] * np.pi/180)} {r[3] * np.sin(gamma[3] * np.pi/180) + r[3]} 0',
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

from pyscf.tools import molden

# Write cube files for each active orbital
molden.from_mo(mol, f"H4_1.5.molden", sing.mo_coeff[:, sing.ncore:sing.ncore+sing.ncas])
