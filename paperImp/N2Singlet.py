from pyscf import gto, scf, mcscf
from pyscf.csf_fci import csf_solver
ncas = 6      # Number of active orbitals
nelecas = 6   # Number of active electrons
mol = gto.Mole()
mol.atom = 'N 0 0 0; N 0 0 1.098'
mol.basis = 'aug-cc-pvtz'
mol.charge = 0
mol.spin = 0
mol.verbose = 4
mol.max_memory = 40000
mol.build()
mf = scf.RHF(mol).run()

# Define your active space.
from pyscf.mcscf import avas
#mo_coeff = avas.kernel(mf, ['N 2p'],minao=mol.basis)[2]


# Define the CAS solver
solvers = csf_solver(mol, smult=1)
solvers.nroots = 4



# Weights
import numpy as np
weights = np.ones(solvers.nroots) / solvers.nroots
mc = mcscf.CASCI(mf, ncas, nelecas)
mc = mcscf.state_average_mix_(mc, [solvers,], weights)
#mc.kernel(mo_coeff)
mc.kernel()

exe = (mc.e_states[1]-mc.e_states[0]) * 27.2114

with open('energies.txt', 'a') as fout:
    fout.write(f'N2Sin:    {exe}\n')
