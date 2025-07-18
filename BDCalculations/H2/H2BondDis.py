#!/usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import cc


def run(b):


    mol = gto.M(
        verbose = 5,
        atom = 'H 0 0 0; H 0 0 %f' % b,
        basis = 'ccpvdz',
        charge = 0,
        spin = 0,
        symmetry = True
    )

    #RHF
    rh = scf.RHF(mol)
    rh_energy = rh.kernel()

    #UHF
    uh = scf.UHF(mol)
    uh_energy = uh.kernel()

    #RCCSD
    rcc = cc.CCSD(rh)
    rcc.kernel()
    rcc_energy = rcc.e_tot


    #UCCSD 
    ucc = cc.UCCSD(uh)
    ucc.kernel()
    ucc_energy = ucc.e_tot

    #CASSCF (4e,4o)
    ca = mcscf.CASSCF(rh, ncas=2, nelecas=2)
    ca_energy =ca.kernel()[0]

    return rh_energy, uh_energy, rcc_energy, ucc_energy, ca_energy

x = [i/10 for i in range(5, 31, 1)] # makes 1->4

e1 = []
e2 = []
e3 = []
e4 = []
e5 = []

for b in x:
    rh_energy, uh_energy, rcc_energy, ucc_energy, ca_energy = run(b)

    e1.append(rh_energy)
    e2.append(uh_energy)
    e3.append(rcc_energy)
    e4.append(ucc_energy)
    e5.append(ca_energy)

e_zero = e5[-1]
e1 = e1 - e_zero
e2 = e2 - e_zero
e3 = e3 - e_zero
e4 = e4 - e_zero
e5 = e5 - e_zero

with open('h2-scan.txt', 'w') as fout:
    fout.write('RHF\tUHF\tRCCSD\tUCCSD\tCASSCF\n')
    for i, xi in enumerate(x):
        fout.write(f'{xi}\t{float(e1[i]):.6f}\t{float(e2[i]):.6f}\t{float(e3[i]):.6f} \t{float(e4[i]):.6f}\t{float(e5[i]):.6f} \n')


def harmonic(R, V0, Req, k):
    return V0 + 0.5 * k * (R - Req)**2

#For RHF
min_index = np.argmin(e1)
start = max(min_index - 5, 0)
end = min(min_index + 4, len(x))
x_fit = x[start:end]
e_fit = e1[start:end]
p0 = [min(e_fit), x[np.argmin(e_fit)], 0.5]
params, cov = curve_fit(harmonic, x_fit, e_fit, p0=p0)
V0, Req, k = params
# Compute vibrational frequency
mu = .504
conv = (2.6204819277 * (10**29) )/mu
c = 2.998 * 10**10
nu_cm1 = (1 / (2 * np.pi * c)) * np.sqrt(k * conv)
print('For RHF')
print(f'Spring Constant: {k}')
print(f'Vibrational Frequency: {nu_cm1}')
R_fit = np.linspace(min(x), max(x), 100)
E_fit = harmonic(R_fit, *params)
plt.plot(R_fit, E_fit, '-', label='RHF Fit ')

#For CCSD
min_index = np.argmin(e3)
start = max(min_index - 5, 0)
end = min(min_index + 4, len(x))
x_fit = x[start:end]
e_fit = e3[start:end]
p0 = [min(e_fit), x[np.argmin(e_fit)], 0.5]
params, cov = curve_fit(harmonic, x_fit, e_fit, p0=p0)
V0, Req, k = params
# Compute vibrational frequency
nu_cm1 = (1 / (2 * np.pi * c)) * np.sqrt(k * conv)
print('For CCSD')
print(f'Spring Constant: {k}')
print(f'Vibrational Frequency: {nu_cm1}')
R_fit = np.linspace(min(x), max(x), 100)
E_fit = harmonic(R_fit, *params)
plt.plot(R_fit, E_fit, '-', label='RCCSD Fit ')

#For CASSCF
min_index = np.argmin(e5)
start = max(min_index - 5, 0)
end = min(min_index + 4, len(x))
x_fit = x[start:end]
e_fit = e5[start:end]
p0 = [min(e_fit), x[np.argmin(e_fit)], 0.5]
params, cov = curve_fit(harmonic, x_fit, e_fit, p0=p0)
V0, Req, k = params
# Compute vibrational frequency
nu_cm1 = (1 / (2 * np.pi * c)) * np.sqrt(k * conv)
print('For CASSCF')
print(f'Spring Constant: {k}')
print(f'Vibrational Frequency: {nu_cm1}')
R_fit = np.linspace(min(x), max(x), 100)
E_fit = harmonic(R_fit, *params)
plt.plot(R_fit, E_fit, '-', label='CASSCF Fit ')


plt.plot(x, e1, label='RHF')
plt.plot(x, e2, label='UHF')
plt.plot(x, e3, label='RCCSD')
plt.plot(x, e4, label='UCCSD')
plt.plot(x, e5, label='CASSCF')
plt.ylim(-.4, .05)
plt.legend()
plt.savefig('bondDis.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free up memory