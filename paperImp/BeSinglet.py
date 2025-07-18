from pyscf import gto, scf, mcscf

ncas = 4      # Number of active orbitals
nelecas = 2   # Number of active electrons



gs = gto.Mole()
gs.atom = 'Be 0 0 0'
gs.basis = 'ccpvtz'
gs.charge = 0
gs.spin = 0
gs.build()
mf = scf.RHF(gs).run()
mc = mcscf.CASCI(mf, ncas, nelecas)
mc.nroots = 3
ge = mc.kernel()[0]



es = gto.Mole()
es.atom = 'Be 0 0 0'
es.basis = 'ccpvtz'
charge = 0
es.spin = 0
es.build()
mf2 = scf.RHF(es).run()
mc2 = mcscf.CASCI(mf2, ncas, nelecas)
mc2.nroots = 3
ee = mc2.kernel()[0]

exe = (ee-ge) * 27.2114

with open('energies.txt', 'w') as fout:
    fout.write(f'Be:    {exe}')