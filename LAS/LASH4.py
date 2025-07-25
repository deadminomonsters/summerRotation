import sys
import numpy as np

from pyscf.tools import molden
from pyscf import gto, scf, mcscf, tools, dft, lib
from pyscf.mcscf import avas
from pyscf.csf_fci import csf_solver
from pyscf.csf_fci.csfstring import count_all_csfs

from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_dmet import localintegrals, dmet, fragments
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf import lassi

def run(b):
    
    dist = [.5, .6, 2.0]

    mol = gto.M(
        atom = f'H 0 0 0; H 0 .74 0; H 0 0 {b}; H 0 .74 {b}',
        basis = 'STO-6G',
        charge = 0,
        spin = 0,
        verbose = 4,
        max_memory = 40000
    )


    mf = scf.RHF(mol).run()


    # LASSCF Calculations
    las = LASSCF(mf,(2, 2),(2, 2),spin_sub=(1, 1))
    frag_atom_list = ([0, 1], [2, 3]) 

    # Coeff
    mo_coeff = avas.kernel(mf, ['H 1s'],minao=mol.basis)[2]       
    mo0 = las.localize_init_guess(frag_atom_list, mo_coeff)
    las.kernel(mo0)

    
    #CAS calculations
    
    ncas = 4     # Number of active orbitals
    nelecas = 4   # Number of active electrons

    #Set up solvers1 for singlet
    solvers1 = csf_solver(mol, smult=1)
    solvers1.nroots = 1
    # Weights
    weights = np.ones(solvers1.nroots) / (solvers1.nroots)

    #Run CASSCF
    cas = mcscf.CASSCF(mf, ncas, nelecas)
    cas = mcscf.state_average_mix_(cas, [solvers1,], weights)
    cas.kernel(mo_coeff)


    for i in dist:
        if b == i:
            molden.from_mo(mol, f"AVAS_{b}.molden", mo_coeff[:, las.ncore:las.ncore+las.ncas])
            molden.from_mo(mol, f"LAS_{b}.molden", las.mo_coeff[:, las.ncore:las.ncore+las.ncas])
            molden.from_mo(mol, f"CAS_{b}.molden", cas.mo_coeff[:, cas.ncore:cas.ncore+cas.ncas])



x = [i/10 for i in range(5, 31, 1)]


for b in x:
    print(f'Distance: {b}')
    run(b)