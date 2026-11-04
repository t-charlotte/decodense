#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
aap_properties module
"""

import numpy as np
from pyscf import gto, scf, dft, lo
from pyscf.pbc import dft as pbc_dft
from pyscf.pbc import scf as pbc_scf
from typing import Union, Optional

from .tools import dim, make_rdm1
from .decomp import CompKeys, PROP_KEYS
from .prop_tools import (
    _e_nuc_pairs,
    _v_nuc_atoms,
    _xc_energies,
    _orb_atom_occ,
    _tr_atom,
    _e_jk_atoms,
    _e_dft_x,
)


def a2ap_redistribute(
    res: dict[str, np.ndarray],
    a2ap_weights: np.ndarray,
    bond_mask: Optional[np.ndarray] = None,
    atom_ref: Optional[np.ndarray] = None,
    aap: bool = False,
) -> dict[str, np.ndarray]:
    """
    this function redistributes atom-wise (aap = False) or
    atom-and-atom-pair-wise (aap = True) energies over the bonds:
    1. (aap only) the energy of each atom pair that is not a bond
       is split equally over its two atoms
    2. the atom energies are redistributed over the bonds using a2ap_weights,
       and (aap only) the energies of the bonded atom pairs are added
    3. (if atom_ref is given) the isolated-atom energies are redistributed in the
       same way and subtracted from the total energy, saved under CompKeys.tot_rel
    only the keys in PROP_KEYS (see decomp.py) are redistributed, all other keys are discarded.
    returns:
    - the bond-wise results
    note: assumes that the rows of a2ap_weights are normalized, in order to achieve a
    lossless decomposition (atoms without bonds have zero weights, see bond_mbo)
    """
    natm, npairs = a2ap_weights.shape
    if bond_mask is None:
        bond_mask = np.ones(npairs, dtype=bool) # treat every atom pair as a bond
    nb = ~bond_mask

    # weights from atoms to bonds only
    a2b_weights = a2ap_weights[:, bond_mask]

    # atom and bond indices
    iu0, iu1 = np.triu_indices(natm, k=1)

    bond_res: dict[str, np.ndarray] = {}

    for k, v in res.items():
        if k not in PROP_KEYS:
            continue
        v = np.asarray(v, dtype=np.float64)
        if aap:
            if v.shape != (natm + npairs,):
                raise ValueError(
                    f"{k}: expected atom-and-atom-pair-wise array of shape "
                    f"({natm + npairs},), got {v.shape}"
                )
            e_atom, e_pair = v[:natm].copy(), v[natm:]
            # 1. split the energy of non-bonded atom pairs equally over both atoms
            e_atom += 0.5 * (
                np.bincount(iu0[nb], weights=e_pair[nb], minlength=natm)
                + np.bincount(iu1[nb], weights=e_pair[nb], minlength=natm)
            )
            e_bond = e_pair[bond_mask]
        else:
            if v.shape != (natm,):
                raise ValueError(
                    f"{k}: expected atom-wise array of shape ({natm},), got {v.shape}"
                )
            e_atom, e_bond = v, 0.0
        # 2. redistribute atom energies over the bonds
        bond_res[k] = e_atom @ a2b_weights + e_bond

    # 3. total energy relative to the isolated atoms
    if atom_ref is not None:
        bond_res[CompKeys.tot_rel] = bond_res[CompKeys.tot] - atom_ref @ a2b_weights

    # atom indices of the bonds
    bond_res[CompKeys.bonds] = np.column_stack((iu0[bond_mask], iu1[bond_mask]))

    return bond_res


# end def a2ap_redistribute()


def aap_prop_tot(
    mol: gto.Mole,
    mf: Union[
        scf.hf.SCF,
        dft.rks.KohnShamDFT,
        pbc_scf.hf.RHF,
        pbc_scf.uhf.UHF,
        pbc_dft.rks.RKS,
        pbc_dft.uks.UKS,
    ],
    mo_coeff: tuple[np.ndarray, np.ndarray],
    mo_occ: tuple[np.ndarray, np.ndarray],
    rdm1: Optional[np.ndarray],
    minao: str,
    pop_method: str,
    weights: list[np.ndarray],
) -> dict[str, Union[np.ndarray, list[np.ndarray]]]:
    """
    this function returns atom-and-atom-pair-decomposed mean-field properties
    """

    # restricted reference
    if mo_occ[0].size == mo_occ[1].size:
        restrict = np.allclose(mo_coeff[0], mo_coeff[1]) and np.allclose(
            mo_occ[0], mo_occ[1]
        )
    else:
        restrict = False

    # dft logical
    dft_calc = isinstance(mf, dft.rks.KohnShamDFT)

    # compute total 1-RDMs (AO basis), identical for both spins if restricted
    rdm1_from_mos = rdm1 is None
    if rdm1 is None:
        rdm1_a = make_rdm1(mo_coeff[0], mo_occ[0])
        rdm1 = np.array(
            [rdm1_a, rdm1_a if restrict else make_rdm1(mo_coeff[1], mo_occ[1])]
        )
        del rdm1_a
    if rdm1.ndim == 2:
        rdm1 = np.array([rdm1, rdm1]) * 0.5

    # mol object projected into minao basis
    if pop_method == "iao":
        pmol = lo.iao.reference_mol(mol, minao=minao)
    else:
        pmol = mol

    # molecular dimensions
    natm = pmol.natm
    npairs = natm * (natm - 1) // 2
    iu0, iu1 = np.triu_indices(natm, k=1)

    # nuclear repulsion property
    prop_nuc_rep = _e_nuc_pairs(pmol)[iu0, iu1]

    # orbital-to-atom occupation factors and occupied mo coefficients
    spin_mos = dim(mo_occ)
    F = _orb_atom_occ(mo_occ, weights, spin_mos, natm)
    c = [mo_coeff[i][:, spin_mo] for i, spin_mo in enumerate(spin_mos)]

    # no solvent contributions (yet) -> NYI

    # atom-wise kinetic energies and atom-atom energy matrices
    e_kin = _tr_atom(mol.intor_symmetric("int1e_kin"), c, F, restrict)
    nuc_att = np.array([_tr_atom(v, c, F, restrict) for v in _v_nuc_atoms(mol)])
    coul, exch = _e_jk_atoms(mol, mf, c, F, restrict)

    # xc energies of individual atoms
    if dft_calc:
        # "pure" xc energy and its nlc part (vv10)
        e_xc, e_xc_nlc = _xc_energies(
            mol, mf, mo_coeff, mo_occ, rdm1, weights, "mo", natm
        )
        if e_xc_nlc is None:
            e_xc_nlc = np.zeros(natm, dtype=np.float64)
        # exact exchange included in the DFT XC energy
        e_dft_x = _e_dft_x(mol, mf, rdm1, c, F, exch, restrict, rdm1_from_mos)
    # end if dft_calc

    # atom (diagonal) and atom-pair (off-diagonal) contributions
    zeros_ap = np.zeros(npairs, dtype=np.float64)

    def _a_ap(mat: np.ndarray) -> np.ndarray:
        """
        this function takes an atom-atom energy matrix mat (mat[K,L]: interaction
        of the electron cloud of atom K, or for nuc_att its nucleus, with the electron
        cloud of atom L) and returns the atom contributions mat[K,K] followed by the
        atom-pair contributions mat[K,L] + mat[L,K] (L < K)
        """
        return np.concatenate((np.diag(mat), mat[iu0, iu1] + mat[iu1, iu0]))

    prop: dict[str, Union[np.ndarray, list[np.ndarray]]] = {
        CompKeys.coul: _a_ap(coul),
        CompKeys.exch: _a_ap(exch),
        CompKeys.kin: np.concatenate((e_kin, zeros_ap)),
        CompKeys.nuc_att: _a_ap(nuc_att),
    }

    if dft_calc:
        prop[CompKeys.exch_DFT] = np.concatenate((e_dft_x, zeros_ap))
        prop[CompKeys.xc] = np.concatenate((e_xc, zeros_ap))
        prop[CompKeys.xc_nlc] = np.concatenate((e_xc_nlc, zeros_ap))

    # sum up electronic contributions
    prop[CompKeys.el] = sum(prop.values())

    if dft_calc:
        ex = prop.pop(CompKeys.exch)
        # atom contributions to the (full) exact exchange energy
        at_contribs = ex[:natm] + 0.5 * (
            np.bincount(iu0, weights=ex[natm:], minlength=natm)
            + np.bincount(iu1, weights=ex[natm:], minlength=natm)
        )
        # fractions (guard against atoms without any exchange contribution)
        e_xc_atom = (
            prop.pop(CompKeys.xc)
            + prop.pop(CompKeys.exch_DFT)
            + prop.pop(CompKeys.xc_nlc)
        )[:natm]
        fracs = np.divide(
            e_xc_atom,
            at_contribs,
            out=np.zeros(natm, dtype=np.float64),
            where=np.abs(at_contribs) > 1.0e-14,
        )
        # scaled quantities
        prop[CompKeys.xc] = np.concatenate(
            (fracs * ex[:natm], 0.5 * (fracs[iu0] + fracs[iu1]) * ex[natm:])
        )
        # re-sum electronic contributions
        del prop[CompKeys.el]
        prop[CompKeys.el] = sum(prop.values())
    # end if dft_calc

    prop[CompKeys.struct] = np.concatenate(
        (np.zeros(natm, dtype=np.float64), prop_nuc_rep)
    )
    prop[CompKeys.tot] = prop[CompKeys.el] + prop[CompKeys.struct]

    return {**prop}


# end aap_prop_tot
