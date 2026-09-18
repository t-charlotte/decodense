#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
bonds module
"""

import numpy as np
import sys

from pyscf import gto, scf, dft, lo
from pyscf.pbc import dft as pbc_dft
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf
from pyscf.scf.hf import mulliken_pop

from pyscf.data import radii

from pyscf import tools as pyscf_tools

from .tools import make_rdm1, contract, dim, make_mbo

from typing import Union, Tuple

def bond_mbo(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT, pbc_scf.hf.RHF, pbc_dft.rks.RKS],
    mo_coeff: Tuple[np.ndarray, np.ndarray],
    mo_occ: Tuple[np.ndarray, np.ndarray],
    minao: str,
    pop_method: str,
    ndo: bool,
) -> Tuple[np.ndarray, np.ndarray]:

    # RHF reference
    if mo_occ[0].size == mo_occ[1].size:
        rhf = np.allclose(mo_coeff[0], mo_coeff[1]) and np.allclose(
            mo_occ[0], mo_occ[1]
        )
    else:
        rhf = False
    
    if isinstance(mol, pbc_gto.Cell):
        s = mol.pbc_intor("int1e_ovlp_sph")
    else:
        s = mol.intor_symmetric("int1e_ovlp")

    alpha, beta = dim(mo_occ)            # number of electrons

    if pop_method in ["iao","iaombo"]:
        # ndo assertion
        if ndo:
            raise NotImplementedError(
                "IAO-based populations for NDOs is not implemented"
            )
        pmol = lo.iao.reference_mol(mol, minao=minao)
        # generate the IAOs
        # TODO: implement unrestricted
        iao = []
        for i, spin_mo in enumerate((alpha,)):
            iao_spin = lo.iao.iao(mol, mo_coeff[i][:, spin_mo], minao=minao)
            iao.append(lo.vec_lowdin(iao_spin, s))
        # overlap matrix
        ovlp = np.eye(pmol.nao_nr()) # IAOs are orthonormal

    elif pop_method in ["mulliken","mullikenmbo"]:
        pmol = mol
        ovlp = s # overlap matrix as calculated before

    else:
        assert False, "Requested population method for Mayer Bond Orders NYI. Valid options: \"mulliken\" and \"iao\"."

    # end if pop_method

    # some useful numbers
    natm = pmol.natm                     # number of atoms
    nbonds = int(natm * (natm - 1) / 2)  # number of bonds (or atom pairs)
    ao_labels = pmol.ao_labels(fmt=None)
    n_ao = len(ao_labels)                # Number of AOs

    # AO -> atom indicator matrix, built once and reused by every atom-grouping step below
    ao_atom_idx = np.array([lbl[0] for lbl in ao_labels])
    atom_of_ao = np.zeros((n_ao, natm), dtype=np.float64)
    atom_of_ao[np.arange(n_ao), ao_atom_idx] = 1.0

    # per-atom AO index boundaries (AOs are ordered per-atom by pyscf), used for an
    # O(n_ao**2) segment-sum grouping instead of an O(n_ao**2 * natm) matmul below
    aoslices = pmol.aoslice_by_atom()
    ao_bounds = aoslices[:, 2].astype(np.intp)
    ao_empty_atom = aoslices[:, 2] == aoslices[:, 3]

    # generate the 1e RDM
    # for an RHF reference, beta is identical to alpha, so it is not recomputed
    if pop_method in ["mulliken","mullikenmbo"]:
        mo_a = mo_coeff[0][:, alpha]
    else: # iao
        mo_a = contract("ki,kl,lj->ij", iao[0], s, mo_coeff[0][:, alpha])
    mocc_a = mo_occ[0][alpha]
    rdm1_a = make_rdm1(mo_a, mocc_a)

    if rhf:
        rdm1_b = rdm1_a
    else:
        if pop_method in ["mulliken","mullikenmbo"]:
            mo_b = mo_coeff[1][:, beta]
        else: # iao
            mo_b = contract("ki,kl,lj->ij", iao[0], s, mo_coeff[1][:, beta])
        mocc_b = mo_occ[1][beta]
        rdm1_b = make_rdm1(mo_b, mocc_b)

    def _mbo_atom_to_bond(
        mbo: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns an array with normalized MBOs from atom to bond (atom_labels,bond_labels)
            and an array with (non-normalized) MBOs ordered according to bond_label.
        Requires an array of MBOs (atom_label_0, atom_label_1) as input.
        """

        # upper-triangle atom-pair indices
        a_idx, b_idx = np.triu_indices(natm, k=1)

        pair_vals = mbo[a_idx, b_idx]
        mbo_sorted = np.where(np.abs(pair_vals) > 1e-29, pair_vals, 0.0)

        mbo_AtoB = np.zeros([natm, nbonds], dtype=np.float64)
        bond_idx = np.arange(nbonds)
        mbo_AtoB[a_idx, bond_idx] = mbo_sorted
        mbo_AtoB[b_idx, bond_idx] = mbo_sorted

        # normalization
        row_sums = np.abs(mbo_AtoB.sum(axis=1))
        row_nonzero = ~np.all(mbo_AtoB == 0.0, axis=1)
        mbo_AtoB[row_nonzero] /= row_sums[row_nonzero, None]

        return mbo_AtoB, mbo_sorted
    # end def _mbo_atom_to_bond()

    def _get_mulpop(
        rdm1: np.ndarray,
        ovlp: np.ndarray,
    ) -> np.ndarray:
        """
        Mulliken population using the provided 1e-RDM and overlap matrix.
        """

        mulpop_ao = np.sum(rdm1 * ovlp.T, axis=1)
        return atom_of_ao.T @ mulpop_ao

    # end def _get_mulpop()

    def _get_mbo(
        rdm1_a: np.ndarray,
        rdm1_b: np.ndarray,
    ) -> np.ndarray:
        """
        Mayer bond order index between atoms A and B.
        """
        # B_{AB} = \sum_{\lambda \in A} \sum_{\omega \in B} (PS)_{\omega \lambda} (PS)_{\lambda \omega}
        # (https://doi.org/10.1016/0009-2614(83)80005-0)

        rdm1_mbo = rdm1_a + rdm1_b

        ps = rdm1_mbo @ ovlp

        interm_1 = ps * ps.T # (PS) dot (PS)

        # group rows/cols of interm_1 by atom (segment sum); equivalent to
        # atom_of_ao.T @ interm_1 @ atom_of_ao but O(n_ao**2) instead of
        # O(n_ao**2 * natm), since AOs are contiguous per atom
        row_grouped = np.add.reduceat(interm_1, ao_bounds, axis=0)
        mbo = np.add.reduceat(row_grouped, ao_bounds, axis=1)
        if ao_empty_atom.any():
            mbo[ao_empty_atom, :] = 0.0
            mbo[:, ao_empty_atom] = 0.0

        do_check = False # set manually for now
        if do_check:
            # NOTE: bond orders for same atom A sum up to twice the Mulliken population of A            
            mulpop = _get_mulpop(rdm1_mbo,ovlp) # get the mulliken population
            mulpop_check = np.sum(mbo,axis=0) * 0.5 # should be the same as mulpop
            assert np.allclose(mulpop, mulpop_check), "Deviations found in Mulliken atomic populations!"
            
            # alternative definition: subtract all bond orders except same atom from nuclear charges
            charges = mol.atom_charges()
            for a in range(natm):
                mbo[a][a] += charges[a] - 2*mulpop_check[a] # corresponds to charges[a] - ( sum mbo[a][b] over b )
            mulpop_check = np.sum(mbo,axis=0)
            assert np.allclose(charges, mulpop_check), "Deviations found in nuclear charges!"
        # end do_check

        # subtract mulliken population to get number of nonbonding electrons
        # NOTE: investigate validity for UHF
        # for a in range(natm):
        #     mbo[a][a] -= mulpop[a] # this should also equal mulpop - sum of Bab where b is not a

        return mbo
    # end _get_mbo()

    mbo = _get_mbo(
        rdm1_a, # alpha RDM1
        rdm1_b, # beta RDM1
        )
    mbo_AtoB, mbo_sorted = _mbo_atom_to_bond(mbo)

    return mbo_AtoB, mbo_sorted
# end def bond_mbo()

def orb_mbo(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT, pbc_scf.hf.RHF, pbc_dft.rks.RKS],
    mo_coeff: Tuple[np.ndarray, np.ndarray],
    mo_occ: Tuple[np.ndarray, np.ndarray],
    minao: str,
    pop_method: str,
    ndo: bool,
):

    # RHF reference
    if mo_occ[0].size == mo_occ[1].size:
        rhf = np.allclose(mo_coeff[0], mo_coeff[1]) and np.allclose(
            mo_occ[0], mo_occ[1]
        )
    else:
        rhf = False

    if isinstance(mol, pbc_gto.Cell):
        s = mol.pbc_intor("int1e_ovlp_sph")
    else:
        s = mol.intor_symmetric("int1e_ovlp")

    # Molecular dimensions
    alpha, beta = dim(mo_occ)

    # Max number of occupied spin-orbs
    n_spin = max(alpha.size, beta.size)

    # mol object projected into minao basis
    if pop_method in ["iao","iaombo"]:
        # ndo assertion
        if ndo:
            raise NotImplementedError(
                "IAO-based populations for NDOs is not implemented"
            )
        pmol = lo.iao.reference_mol(mol, minao=minao)
    elif pop_method in ["mulliken","mullikenmbo"]:
        pmol = mol
    else:
        assert False, "Requested population method for Mayer Bond Orders NYI. Valid options: \"mulliken\" and \"iao\"."

    # Number of atoms
    natm = pmol.natm

    # Number of atom pairs
    npairs = int(natm * (natm - 1) / 2)

    # AO labels
    ao_labels = pmol.ao_labels(fmt=None)

    # Number of AOs
    n_ao = len(ao_labels)

    # AO -> atom indicator matrix, built once and reused by every call to _get_mbo below
    ao_atom_idx = np.array([lbl[0] for lbl in ao_labels])
    atom_of_ao = np.zeros((n_ao, natm), dtype=np.float64)
    atom_of_ao[np.arange(n_ao), ao_atom_idx] = 1.0

    a_idx, b_idx = np.triu_indices(natm, k=1)

    # Overlap matrix
    if pop_method in ["mulliken","mullikenmbo"]:
        ovlp = s
    else: # iao
        ovlp = np.eye(pmol.nao_nr())


    def _get_mbo(
        mocc_k: float,
        atom_w_k: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns the Mayer bond orders for a single occupied (spin-)orbital

        a single orbital's contribution to the 1-RDM is rank-1,
        rdm1_k = mocc_k * outer(mo_k, mo_k), so (rdm1_k @ ovlp) is rank-1 too:
            w_k = mo_k * (ovlp @ mo_k)   (elementwise)
        and the atom-resolved bond-order matrix collapses to
            mbo = mocc_k**2 * outer(atom_w_k, atom_w_k)
        where atom_w_k[a] is w_k summed over the AOs on atom a. this avoids
        ever forming an (n_ao, n_ao) intermediate per orbital.
        """

        diag = mocc_k**2 * atom_w_k**2
        pair_vals = mocc_k**2 * atom_w_k[a_idx] * atom_w_k[b_idx]

        do_check = False
        if do_check:
            # Mulliken population
            mulpop = mocc_k * atom_w_k

            # NOTE: bond orders for same atom A sum up to the Mulliken population of A
            mbo_check = mocc_k**2 * np.outer(atom_w_k, atom_w_k)
            mulpop_check = np.sum(mbo_check,axis=0) # should be the same as mulpop
            assert np.allclose(mulpop, mulpop_check), "Deviations found in Mulliken atomic populations!"

        # sort bond orders:
        # first all non-bonding populations (len: natm)
        # then all atom pair bond orders (len: npairs)

        # also filter out values that are too small (< 1e-29)

        mbo = mocc_k**2 * np.outer(atom_w_k, atom_w_k)

        mbo_sorted = np.zeros((natm + npairs), dtype=np.float64)
        mbo_sorted[:natm] = np.where(np.abs(diag) > 1e-29, diag, 0.0)
        mbo_sorted[natm:] = np.where(np.abs(pair_vals) > 1e-29, 2 * pair_vals, 0.0)

        # normalization
        # note that the sum should equal 1 already,
        # meaning that normalization should not have any effect
        mbo_sorted = mbo_sorted/np.sum(mbo_sorted)

        return mbo, mbo_sorted
    # end def _get_mbo()


    # loop over spin, calculate the orb rdm1s and get their mbos

    mbo = []
    mbo_sorted = []
    rdm1 = None
    # for an RHF reference, beta is identical to alpha and is not recomputed
    rdm1_spin_alpha = None
    mbo_spin_alpha = None
    mbo_spin_sorted_alpha = None
    for i, spin_mo in enumerate((alpha,beta)):

        if i == 1 and rhf:
            rdm1 += rdm1_spin_alpha
            mbo.append(mbo_spin_alpha)
            mbo_sorted.append(mbo_spin_sorted_alpha)
            continue

        # Get mo_coefficients and occupation
        if pop_method in ["mulliken","mullikenmbo"]:
            mo = mo_coeff[i][:, spin_mo]
        else: # iao
            iao = lo.iao.iao(mol, mo_coeff[i][:, spin_mo], minao=minao)
            iao = lo.vec_lowdin(iao, s)
            mo = contract("ki,kl,lj->ij", iao, s, mo_coeff[i][:, spin_mo])
        # end if pop_method

        mocc = mo_occ[i][spin_mo]

        # accumulate the total (spin-summed) 1-RDM, needed for mbo_full below
        rdm1_spin = make_rdm1(mo, mocc)
        if rdm1 is None:
            rdm1 = np.zeros_like(rdm1_spin)
        rdm1 += rdm1_spin

        # Rank-1 shortcut (see _get_mbo docstring): for every orbital at once,
        # compute w_k = mo_k * (ovlp @ mo_k) and its atom-summed counterpart.
        Smo = ovlp @ mo
        w = mo * Smo
        atom_w = atom_of_ao.T @ w  # (natm, n_orb)

        mbo_spin = []
        mbo_spin_sorted = []
        # print("\nSpin " + str(i) + " :: \n")
        for j in range(mo.shape[1]):
            #print("\n Orbital " + str(j) + " :: \n")
            mbo_spin_j, mbo_spin_j_sorted = _get_mbo(mocc[j], atom_w[:, j])
            mbo_spin.append(mbo_spin_j)
            mbo_spin_sorted.append(mbo_spin_j_sorted)

            if False:
                # save the (rank-1) orbital rdm1 as a cube file
                pyscf_tools.cubegen.density(
                    pmol,
                    "rdm1_orb_" + str(j) + "_" + str(i) + ".cube",
                    mocc[j] * np.outer(mo[:, j], mo[:, j])
                )
            # end if
        # end loop over MOs

        if i == 0:
            rdm1_spin_alpha = rdm1_spin
            mbo_spin_alpha = mbo_spin
            mbo_spin_sorted_alpha = mbo_spin_sorted

        mbo.append(mbo_spin)
        mbo_sorted.append(mbo_spin_sorted)
    # end loop over spin

    def _mbo_atom_to_bond(
        mbo: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns an array with normalized MBOs from atom to bond -- dim = (atom_labels,bond_labels)
        """
        # a_idx, b_idx reused from the enclosing scope

        pair_vals = mbo[a_idx, b_idx]
        pair_vals = np.where(np.abs(pair_vals) > 1e-29, pair_vals, 0.0)

        mbo_AtoB = np.zeros([natm, npairs], dtype=np.float64)
        bond_idx = np.arange(npairs)
        mbo_AtoB[a_idx, bond_idx] = pair_vals
        mbo_AtoB[b_idx, bond_idx] = pair_vals

        # normalize, then scale by intra-atomic MBO (mbo[a][a] == diagonal of mbo)
        row_sums = np.abs(mbo_AtoB.sum(axis=1))
        row_nonzero = ~np.all(mbo_AtoB == 0.0, axis=1)
        diag = np.diag(mbo)
        mbo_AtoB[row_nonzero] /= row_sums[row_nonzero, None]
        mbo_AtoB[row_nonzero] *= diag[row_nonzero, None]

        return mbo_AtoB
    # end def _mbo_atom_to_bond()

    mbo_full = make_mbo(np.sum(rdm1, axis=-1), ovlp, natm, ao_labels)

    return mbo_sorted, mbo_full
# end def orb_mbo()