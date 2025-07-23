from pyscf import gto, scf, mcscf
from pyscf.csf_fci import csf_solver
ncas = 4     # Number of active orbitals
nelecas = 2   # Number of active electrons
mol = gto.Mole()
mol.atom = 'Be 0 0 0'
mol.basis = 'ccpvtz'
mol.charge = 0
mol.spin = 0
mol.verbose = 4
mol.max_memory = 40000
mol.build()
mf = scf.RHF(mol).run()



# Define the CAS solver
solvers = csf_solver(mol, smult=1)
solvers.nroots = 2

# Weights
import numpy as np
weights = np.ones(solvers.nroots) / solvers.nroots
mc = mcscf.CASCI(mf, ncas, nelecas)
mc = mcscf.state_average_mix_(mc, [solvers,], weights)
mc.kernel()

exe = (mc.e_states[1]-mc.e_states[0]) * 27.2114

with open('energies.txt', 'a') as fout:
    fout.write('Table 3\n')
    fout.write(f'BeSin:    {exe}\n')