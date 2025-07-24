from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas
from pyscf.csf_fci import csf_solver
from pyscf.csf_fci.csfstring import count_all_csfs
import numpy as np
import matplotlib.pyplot as plt
from pyscf.tools import molden

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

    solverstri = csf_solver(mol, smult=3)
    solverstri.nroots = 1

    # Weights
    weights = np.ones(solvers.nroots+ solverstri.nroots) / (solvers.nroots+ solverstri.nroots)
    mo_coeff = avas.kernel(mf, ['H 1s'],minao=mol.basis)[2]

    #Run CASSCF
    sing = mcscf.CASSCF(mf, ncas, nelecas)
    sing = mcscf.state_average_mix_(sing, [solvers,solverstri], weights)
    print(f'Distance = {b}')
    sing.kernel(mo_coeff)
    sing_e = sing.e_states

    return sing_e

x = [i/20 for i in range(10, 61, 1)]


for b in x:
    sing = (run(b))