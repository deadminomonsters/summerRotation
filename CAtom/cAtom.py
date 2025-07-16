from pyscf import gto, scf, mcscf, fci

# 1. Molecule
mol = gto.Mole()
mol.atom = 'C 0 0 0'
mol.basis = '6-31g'
mol.charge = 0
mol.spin = 0
mol.build()

# 2. RHF
mf = scf.RHF(mol)
mf_energy = mf.kernel()
print('RHF Energy: ', mf_energy)

# 3. CASCI (4e,4o)
mc_ci = mcscf.CASCI(mf, ncas=4, nelecas=4)
mc_ci_energy = mc_ci.kernel()[0]
print('CASCI Energy: ', mc_ci_energy)

# 4. CASSCF (4e,4o)
mc = mcscf.CASSCF(mf, ncas=4, nelecas=4)
mc_energy = mc.kernel()[0]
print('CASSCF Energy: ', mc_energy)

# 5. FCI
cisolver = fci.FCI(mol, mf.mo_coeff)
fci_energy = cisolver.kernel()[0]
print("FCI energy: ", fci_energy)