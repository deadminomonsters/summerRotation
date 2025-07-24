from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas
from pyscf.csf_fci import csf_solver
from pyscf.csf_fci.csfstring import count_all_csfs
import numpy as np
import matplotlib.pyplot as plt

r = [0.3, 0.7, 1.0, 1.5]
theta = [148, 60, 68, 104]
phi = [32, 124, 224, 300]

mol = gto.M(
f'H 0 0 0; H {r[0] * np.cos(theta[0] * np.pi/180)} {r[0] * np.sin(theta[0] * np.pi/180) + r[0]} 0 ; H {r[0] * np.cos(phi[0] * np.pi/180)} {r[0] * np.sin(phi[0] * np.pi/180) + r[0]} 0',
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

from pyscf.tools import molden

# Write cube files for each active orbital
molden.from_mo(mol, f"H3_0.3.molden", sing.mo_coeff[:, sing.ncore:sing.ncore+sing.ncas])
