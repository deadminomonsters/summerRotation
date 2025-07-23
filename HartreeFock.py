import pyscf
import scipy
import numpy as np
from pyscf import gto

mol = gto.M(
    atom = f'O 0 0 0.11779; H 0 0.75545 -0.47116; H 0 -0.75545 -0.47116',
    basis = 'ccpvdz',
    charge = 0,
    spin = 0,
)

def get_hcore(mol):
    t = mol.intor_symmetric('int1e_kin')
    v = mol.intor_symmetric('int1e_nuc')
    hcore = t + v
    return hcore

def get_eri(mol):
    eri = mol.intor('int2e')
    return eri

def get_veff(mol, dm):
    eri = get_eri(mol)
    J = np.einsum('pqrs, qp->rs', eri, dm)
    K = np.einsum('pqrs, rq->ps', eri, dm)
    return J - 0.5*K

def construct_fock(mol, dm):
    hcore = get_hcore(mol)
    veff = get_veff(mol, dm)
    fock = hcore + veff
    return fock

def construct_dm(mol, mo_coeff):
    nocc = int(mol.nelectron//2)
    dm = 2. * np.dot(mo_coeff[:, :nocc], mo_coeff[:, :nocc].T)
    return dm

def get_energy(mol, dm):
    h1 = get_hcore(mol)
    veff = get_veff(mol, dm)
    energy = np.einsum('pq, qp->', h1, dm) \
        + 0.5 * np.einsum('pq, qp->', veff, dm) \
        + mol.energy_nuc()
    return energy

def generalized_eigval(fock, s):
    mo_energy, mo_coeff, = scipy.linalg.eigh(fock, s)
    return mo_energy, mo_coeff

def scf_procedure(mol, ethresh=1e-7, dmthresh=1e-7, maxiter=100):
    s = mol.intor_symmetric('int1e_ovlp')

    mo_coeff = np.zeros_like(s)
    dm = construct_dm(mol, mo_coeff)

    converge = False
    energy = 0
    for i in range(maxiter):
        fock = construct_fock(mol, dm)
        mo_energy, mo_coeff = generalized_eigval(fock, s)
        new_dm = construct_dm(mol, mo_coeff)
        new_energy = get_energy(mol, new_dm)
        print(f'Iteration: {i}       Energy: {new_energy}')
        if np.abs(energy - new_energy) < ethresh and np.linalg.norm(new_dm - dm) < dmthresh:
            print('Hola, SCF Converged')
            converge = True
            break
        dm = new_dm
        energy = new_energy

    if not converge:
        print('SCF has not converged')

    return energy, mo_coeff


energy = scf_procedure(mol)[0]
print(energy)