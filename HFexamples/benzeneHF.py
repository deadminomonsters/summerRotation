from pyscf import gto, scf, mcscf
import basis_set_exchange

coor = '''
C  0.000000  1.40272  0.000000
H  0.000000  2.49029  0.000000
C -1.21479  0.70136  0.000000
H -2.15666  1.24515  0.000000
C -1.21479 -0.70136  0.000000
H -2.15666 -1.24515  0.000000
C  0.000000 -1.40272  0.000000
H  0.000000 -2.49029  0.000000
C  1.21479 -0.70136  0.000000
H  2.15666 -1.24515  0.000000
C  1.21479  0.70136  0.000000
H  2.15666  1.24515  0.000000
'''


#Building Benzene
benzene = gto.Mole()
benzene.atom = coor
benzene.basis = 'ccpvdz'
benzene.symmetry = True
benzene.spin = 0   # singlet
benzene.charge = 0
benzene.build()


# 2a. Perform RHF calculation
print("\nRHF Calculation")
rhf = scf.RHF(benzene)
rhf.verbose = 4
rhf.kernel()

#2b. Perform RHF with Newton Optimizer
print("\nRHF w/ Newton Optimizer")
mf_newton = scf.RHF(benzene).newton() 
mf_newton.verbose = 4
mf_newton.kernel()

#2c. Perform RHF w/ Density Fitting
print("\nRHF w/ Density Fitting")
rhf_df = scf.RHF(benzene).density_fit()
rhf_df.kernel()

# 3. Perform ROHF calculation
print("\nROHF Calculation")
rohf = scf.ROHF(benzene)
rohf.kernel()

# 4. Perform UHF calculation
print("\nUHF Calculation")
uhf = scf.UHF(benzene)
uhf.kernel()

# 5. Performing calculations with different basis sets
benzene = gto.Mole()
benzene.atom = coor
benzene.basis = {'H' : gto.load(basis_set_exchange.api.get_basis('cc-pV5Z', augment_diffuse=0, elements='H', fmt='nwchem'), 'H'),
                 'C' : gto.load(basis_set_exchange.api.get_basis('cc-pV5Z', augment_diffuse=0, elements='C', fmt='nwchem'), 'C'),}
benzene.symmetry = True
benzene.spin = 0   # singlet
benzene.charge = 0
benzene.build()

print("\nRHF Calculation w/ ccpvtz")
rhf = scf.RHF(benzene)
rhf.kernel()