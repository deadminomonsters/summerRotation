from pyscf import gto, scf, mcscf

# 1. Define the molecule
mol = gto.Mole()
mol.atom = '''
O  0.000000  0.000000  0.000000
H  0.000000 -0.757000  0.587000
H  0.000000  0.757000  0.587000
'''
mol.basis = 'sto-3g'
mol.spin = 0   # singlet
mol.charge = 0
mol.build()

# 2. Perform Hartree-Fock (RHF) calculation
mf = scf.RHF(mol)
mf.kernel()

# 3. Run CASSCF
# active space: (ncas, nelecas)
# ncas = number of active orbitals
# nelecas = number of active electrons
ncas = 4
nelecas = 4

mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.kernel()

# Optional: print the active orbitals
mo = mc.mo_coeff
mo_active = mo[:, mc.ncore:mc.ncore+mc.ncas]
print("Active orbitals (MO coefficients):")
print(mo_active)

from pyscf.tools import molden

# Write cube files for each active orbital
molden.from_mo(mol, f"test.molden", mc.mo_coeff[:, mc.ncore:mc.ncore+mc.ncas])

