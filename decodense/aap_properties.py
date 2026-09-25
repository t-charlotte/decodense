#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
aap_properties module
"""

import copy
import numpy as np
from pyscf import gto, scf, dft, df, lo, lib, solvent
from pyscf.dft import numint
from pyscf.pbc import dft as pbc_dft
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf
from pyscf.pbc.dft import numint as pbc_numint
from typing import List, Tuple, Dict, Union, Any, Optional

from .pbctools import ewald_e_nuc, get_nuc_pbc
from .tools import dim, make_rdm1, contract
from .decomp import CompKeys

# block size in _mm_pot()
BLKSIZE = 200

# max. number of atom-RDM1s per batched get_jk call (lower it to save memory)
JK_BLKSIZE = 16

def aap_prop_tot(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mf: Union[
        scf.hf.SCF,
        dft.rks.KohnShamDFT,
        pbc_scf.hf.RHF,
        pbc_scf.uhf.UHF,
        pbc_dft.rks.RKS,
        pbc_dft.uks.UKS,
    ],
    mo_coeff: Tuple[np.ndarray, np.ndarray],
    mo_occ: Tuple[np.ndarray, np.ndarray],
    rdm1: Optional[np.ndarray],
    minao: str,
    pop_method: str,
    prop_type: str,
    part: str,
    ndo: bool,
    gauge_origin: np.ndarray,
    weights: List[np.ndarray],
) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
    """
    this function returns atom-and-atom-pair-decomposed mean-field properties
    """

    # unsupported options
    if isinstance(mol, pbc_gto.Cell):
        raise NotImplementedError("AAP decomposition for periodic systems NYI!")
    if prop_type != "energy":
        raise NotImplementedError("AAP decomposition of dipoles NYI!")
    # if part != "atoms": # TODO: this should not be checked, because the part will be restructured later
    #     raise NotImplementedError(f"AAP decomposition for part = '{part}' NYI!")
    if ndo:
        raise NotImplementedError("AAP decomposition for NDOs NYI!")
    if any(hasattr(mf, attr) for attr in ("mm_mol", "with_solvent", "h1e_mmpol")):
        raise NotImplementedError("AAP decomposition for solvation energies NYI!")

    # restricted reference
    if mo_occ[0].size == mo_occ[1].size:
        restrict = np.allclose(mo_coeff[0], mo_coeff[1]) and np.allclose(
            mo_occ[0], mo_occ[1]
        )
    else:
        restrict = False

    # dft logical
    dft_calc = isinstance(mf, dft.rks.KohnShamDFT)

    # ao dipole integrals with specified gauge origin -> NYI
    ao_dip = None

    # compute total 1-RDMs (AO basis)
    if rdm1 is None:
        rdm1 = np.array(
            [make_rdm1(mo_coeff[0], mo_occ[0]), make_rdm1(mo_coeff[1], mo_occ[1])]
        )
    if rdm1.ndim == 2:
        rdm1 = np.array([rdm1, rdm1]) * 0.5

    # mol object projected into minao basis
    if pop_method in ["iao","iaombo"]:
        pmol = lo.iao.reference_mol(mol, minao=minao)
    else:
        pmol = mol

    # molecular dimensions
    alpha, beta = dim(mo_occ)

    natm = pmol.natm
    npairs = int(natm * (natm - 1) / 2)

    # effective atomic charges
    if part in ["atoms", "eda"]:
        charge_atom = (
            -(np.sum(weights[0], axis=0) + np.sum(weights[1], axis=0))
            + pmol.atom_charges()
        )
    else:
        charge_atom = 0.0

    # nuclear repulsion property
    prop_nuc_rep = _e_nuc_ap(pmol)

    # core hamiltonian
    kin, nuc, sub_nuc = _h_core(mol, mf)

    # no solvent contributions (yet) -> NYI

    # orbital-resolved xc energies
    if dft_calc:
        if hasattr(mf, "vk"):
            vk = copy.copy(mf.vk)
        else:
            vk = mf.get_k(mol=mol, dm=np.sum(rdm1, axis=0) if restrict else rdm1)
        # xc-type and ao_deriv
        xc_type, ao_deriv = _xc_ao_deriv(mf.xc)
        # update exchange operator wrt range-separated parameter and exact exchange
        # components
        vk = _vk_dft(mol, mf, mf.xc, np.sum(rdm1, axis=0) if restrict else rdm1, vk)
        # "pure" xc energy per orbital
        e_xc_orb = _xc_orb_energies(
            mol, mf, rdm1, mo_coeff, (alpha, beta), xc_type, ao_deriv
        )
        # nlc (vv10) energy per orbital
        e_xc_nlc_orb = None
        if dft.libxc.nlc_coeff(mf.xc) != ():
            nlc_pars = dft.libxc.nlc_coeff(mf.xc)[0][0]
            if mf.nlcgrids.coords is None:
                mf.nlcgrids.build(with_non0tab=True)
            ao_value_nlc = _ao_val(mol, mf.nlcgrids.coords, 1)
            _, _, rho_vv10 = _make_rho(ao_value_nlc, np.sum(rdm1, axis=0), "GGA")
            eps_xc_nlc = numint._vv10nlc(
                rho_vv10,
                mf.nlcgrids.coords,
                rho_vv10,
                mf.nlcgrids.weights,
                mf.nlcgrids.coords,
                nlc_pars,
            )[0]
            e_xc_nlc_orb = _orb_energies(
                ao_value_nlc[0], mf.nlcgrids.weights * eps_xc_nlc, mo_coeff, (alpha, beta)
            )
            del ao_value_nlc
    # end if dft_calc

    # perform decomposition
    prop: Dict[str, Union[np.ndarray, List[np.ndarray]]]

    spin_mos = (alpha, beta)

    # atom-specific 1-RDMs
    rdm1_atom, F = _atom_rdm1(mo_coeff, mo_occ, weights, spin_mos, natm)
    rdm1_atom_tot = (rdm1_atom[0] + rdm1_atom[1]).reshape(natm, -1)

    # atom-atom energy matrices
    coul, exch = _e_jk_atoms(mol, mf, rdm1_atom, rdm1_atom_tot, restrict)
    nuc_att = sub_nuc.reshape(natm, -1) @ rdm1_atom_tot.T
    e_kin = rdm1_atom_tot @ kin.ravel()

    # atom (diagonal) and atom-pair (off-diagonal) contributions,
    iu0, iu1 = np.triu_indices(natm, k=1)
    zeros_ap = np.zeros(npairs, dtype=np.float64)

    def _a_ap(mat: np.ndarray) -> np.ndarray:
        return np.concatenate((np.diag(mat), mat[iu0, iu1] + mat[iu1, iu0]))

    prop = {
        CompKeys.coul: _a_ap(coul),
        CompKeys.exch: _a_ap(exch),
        CompKeys.kin: np.concatenate((e_kin, zeros_ap)),
        CompKeys.nuc_att: _a_ap(nuc_att),
    }

    if dft_calc:
        # exact exchange included in the DFT XC energy
        if restrict:
            e_dft_x = -0.25 * (rdm1_atom_tot @ vk.ravel())
        else:
            e_dft_x = -0.5 * sum(
                rdm1_atom[i].reshape(natm, -1) @ vk[i].ravel() for i in range(2)
            )
        # "pure" DFT XC energy and its NLC part (vv10)
        e_xc = sum(e_xc_orb[i] @ F[i] for i in range(2))
        if e_xc_nlc_orb is not None:
            e_xc_nlc = sum(e_xc_nlc_orb[i] @ F[i] for i in range(2))
        else:
            e_xc_nlc = np.zeros(natm, dtype=np.float64)

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
            e_xc_atom, at_contribs,
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

    prop[CompKeys.struct] = np.concatenate((np.zeros(natm, dtype=np.float64), prop_nuc_rep))

    return {**prop, CompKeys.charge_atom: np.append(charge_atom, np.zeros((npairs), dtype=np.float64))}
# end aap_prop_tot

def bond_prop_tot(
    mbo: np.ndarray,
    aap_res: Dict[str, Any],
    smiles: str
):
    from rdkit import Chem

    # Generate RDKit molecule object
    params = Chem.SmilesParserParams()
    params.removeHs = False # do not remove hydrogen atoms

    mol = Chem.MolFromSmiles(smiles, params)
    mol = Chem.AddHs(mol)

    natm = mol.GetNumAtoms()
    nbond = mol.GetNumBonds()

    # Generate dictionary for bond label to bond index and atom labels
    # Structure: b_idces[b_label] = [b_idx, a_label_0, a_label_1]

    return
# end bond_prop_tot


def _e_nuc(mol: gto.Mole) -> np.ndarray:
    """
    this function returns the nuclear repulsion energy
    """
    # coordinates and charges of nuclei
    charges = mol.atom_charges()
    # internuclear distances (with self-repulsion removed)
    dist = gto.inter_distance(mol)
    dist[np.diag_indices_from(dist)] = 1e200
    return contract("i,ij,j->i", charges, 1.0 / dist, charges) * 0.5

def _e_nuc_ap(mol: gto.Mole) -> np.ndarray:
    """
    CHRT: function that returns the nuclear repulsion energy per atom pair
    """
    # coordinates and charges of nuclei
    charges = mol.atom_charges()
    # internuclear distances (with self-repulsion removed)
    dist = gto.inter_distance(mol)
    dist[np.diag_indices_from(dist)] = 1e200
    enuc = contract("i,ij,j->ij", charges, 1.0 / dist, charges)

    return enuc[np.triu_indices(mol.natm,k=1)]

def _atom_rdm1(
    mo_coeff: Tuple[np.ndarray, np.ndarray],
    mo_occ: Tuple[np.ndarray, np.ndarray],
    weights: List[np.ndarray],
    spin_mos: Tuple[np.ndarray, np.ndarray],
    natm: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    this function calculates:
    - the atom-specific RDM1s - shape (2, natm, nao, nao)
    - orbital-to-atom occupation factors F[i] - shape (nocc_i, natm)
    such that rdm1_atom[i,a] = sum_m F[i][m,a] |c_m><c_m|
    """
    nao = mo_coeff[0].shape[0]
    rdm1_atom = np.zeros((2, natm, nao, nao), dtype=np.float64)
    F = []
    for i, spin_mo in enumerate(spin_mos):
        if len(spin_mo) == 0:
            F.append(np.zeros((0, natm), dtype=np.float64))
            continue
        c = mo_coeff[i][:, spin_mo]
        w = np.asarray(weights[i])
        f = mo_occ[i][spin_mo][:, None] * w / np.sum(w, axis=1, keepdims=True)
        for a in range(natm):
            rdm1_atom[i, a] = (c * f[:, a]) @ c.T
        F.append(f)
    # end for
    return rdm1_atom, F
# end def _atom_rdm1

def _e_jk_atoms(
    mol: gto.Mole,
    mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT],
    rdm1_atom: np.ndarray,
    rdm1_atom_tot: np.ndarray,
    restrict: bool,
    blksize: int = JK_BLKSIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    this function calculates atom-atom Coulomb and exchange energy matrices:
    coul[a,b] = 1/2 tr(J[D_a] D_b)
    exch[a,b] = -1/2 sum_s tr(K[D_as] D_bs)
    """
    natm, nao = rdm1_atom.shape[1], rdm1_atom.shape[-1]
    coul = np.zeros((natm, natm), dtype=np.float64)
    exch = np.zeros((natm, natm), dtype=np.float64)
    for a0, a1 in lib.prange(0, natm, blksize):
        n = a1 - a0
        if restrict: # alpha and beta 1-RDMs are identical -> one J/K build per atom

            vj, vk = mf.get_jk(
                mol=mol, dm=rdm1_atom[0, a0:a1], with_j=True, with_k=True
            )
            vj, vk = vj.reshape(n, -1), vk.reshape(n, -1)
            
            coul[a0:a1] = vj @ rdm1_atom_tot.T # 1/2 tr(2 vj, D_tot)
            exch[a0:a1] = -vk @ rdm1_atom[0].reshape(natm, -1).T # -1/2 * 2 spins * tr(vk, D_alpha)

        else:

            vj, vk = mf.get_jk(
                mol=mol,
                dm=rdm1_atom[:, a0:a1].reshape(-1, nao, nao),
                with_j=True,
                with_k=True,
            )
            vj, vk = vj.reshape(2, n, -1), vk.reshape(2, n, -1)

            coul[a0:a1] = 0.5 * (vj[0] + vj[1]) @ rdm1_atom_tot.T
            exch[a0:a1] = -0.5 * sum(
                vk[i] @ rdm1_atom[i].reshape(natm, -1).T for i in range(2)
            )

        # end if
    # end for
    return coul, exch
# end def _e_jk_atoms

def _orb_energies(
    ao0: np.ndarray,
    weps: np.ndarray,
    mo_coeff: Tuple[np.ndarray, np.ndarray],
    spin_mos: Tuple[np.ndarray, np.ndarray],
) -> List[np.ndarray]:
    """
    this function calculates orbital energies e_m = sum_r weps(r) |phi_m(r)|^2
    for a given weighted energy density weps(r) = w(r) eps(r)
    """
    return [
        weps @ (ao0 @ mo_coeff[i][:, spin_mo]) ** 2
        for i, spin_mo in enumerate(spin_mos)
    ]

def _xc_orb_energies(
    mol: gto.Mole,
    mf: dft.rks.KohnShamDFT,
    rdm1: np.ndarray,
    mo_coeff: Tuple[np.ndarray, np.ndarray],
    spin_mos: Tuple[np.ndarray, np.ndarray],
    xc_type: str,
    ao_deriv: int,
) -> List[np.ndarray]:
    """
    this function calculates orbital xc energies e_m = sum_r w(r) eps_xc(r) |phi_m(r)|^2
    block-wise on the DFT grid (ao values are never stored for the full grid)
    """
    e_orb = [np.zeros(len(spin_mo), dtype=np.float64) for spin_mo in spin_mos]
    if xc_type == "HF":
        return e_orb
    # restricted rho from _make_rho: 1D for LDA, 2D otherwise
    rho_ndim_restricted = 1 if xc_type == "LDA" else 2
    max_memory = max(2000, mf.max_memory - lib.current_memory()[0])
    for ao, _, weight, _ in mf._numint.block_loop(
        mol, mf.grids, mol.nao_nr(), ao_deriv, max_memory=max_memory
    ):
        _, _, rho = _make_rho(ao, rdm1, xc_type)
        spin = 0 if rho.ndim == rho_ndim_restricted else 1
        eps_xc = dft.libxc.eval_xc(mf.xc, rho, spin=spin)[0]
        ao0 = ao if ao.ndim == 2 else ao[0]
        for i, e in enumerate(_orb_energies(ao0, weight * eps_xc, mo_coeff, spin_mos)):
            e_orb[i] += e
    return e_orb

def _dip_nuc(mol: gto.Mole, gauge_origin: np.ndarray) -> np.ndarray:
    """
    this function returns the nuclear contribution to the molecular dipole moment
    """
    # coordinates and formal/actual charges of nuclei
    coords = mol.atom_coords()
    form_charges = mol.atom_charges()
    return contract("i,ix->ix", form_charges, coords - gauge_origin)


def _h_core(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT, pbc_scf.RHF],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    this function returns the components of the core hamiltonian
    """
    if isinstance(mol, pbc_gto.Cell) and isinstance(
        mf, (pbc_scf.hf.RHF, pbc_scf.uhf.UHF)
    ):
        # kinetic integrals
        kin = mol.pbc_intor("int1e_kin")
        mydf = mf.with_df
        # individual atomic potentials
        sub_nuc = get_nuc_pbc(mol, mydf)
    else:
        # kinetic integrals
        kin = mol.intor_symmetric("int1e_kin")
        # individual atomic potentials
        sub_nuc = _get_nuc(mol)
    # total nuclear potential
    nuc = np.sum(sub_nuc, axis=0)
    return kin, nuc, sub_nuc


def _get_nuc(mol: gto.Mole) -> np.ndarray:
    """
    individual atomic potentials for molecules
    """
    # coordinates and charges of nuclei
    coords = mol.atom_coords()
    charges = mol.atom_charges()
    # individual atomic potentials
    sub_nuc = np.zeros([mol.natm, mol.nao_nr(), mol.nao_nr()], dtype=np.float64)
    for k in range(mol.natm):
        with mol.with_rinv_origin(coords[k]):
            sub_nuc[k] = -1.0 * mol.intor("int1e_rinv") * charges[k]
    return sub_nuc


def _solvent(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT, pbc_scf.RHF],
    rdm1: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    # initialize
    pot_solv, nuc_solv, vdW_solv = None, None, None

    # point charges
    if hasattr(mf, "mm_mol"):
        mm_mol = getattr(mf, "mm_mol", None)
        pot_solv, nuc_solv = _point_charges(mol, mm_mol)
    # pcm
    elif hasattr(mf, "with_solvent"):
        pot_solv, nuc_solv = _pcm(mol, rdm1, mf.with_solvent)
    # OpenMM polarizable embedding
    elif hasattr(mf, "h1e_mmpol"):
        # static contribution to one-electron Hamiltonian
        pot_solv = getattr(mf, "h1e_mmpol").copy()

        # static nuclear contribution
        nuc_solv = np.array(
            [
                mf.V_mm_at_nucl[i] * mol.atom_charges()[i]
                for i in range(len(mol.atom_charges()))
            ]
        )

        # QM-MM vdW potential
        vdW_solv = mf.ommp_qm_helper.vdw_energy_by_atom(mf.ommp_obj)

        # polarization contributions
        if hasattr(mf, "v_mmpol_d"):
            # IPD contribution to the Fock Matrix
            pot_solv += 0.5 * getattr(mf, "v_mmpol_d")

            # polarization contribution from the potential of the IPDs at the nuclei
            nuc_solv += 0.5 * np.array(
                [
                    mf.V_pol_at_nucl[i] * mol.atom_charges()[i]
                    for i in range(len(mol.atom_charges()))
                ]
            )

    return pot_solv, nuc_solv, vdW_solv


def _point_charges(mol: gto.Mole, mm_mol: gto.Mole) -> Tuple[np.ndarray, np.ndarray]:
    """
    this function returns the full mm potential and the nuclei interaction with the
    point charges (adapted from: qmmm/itrf.py:get_hcore() in PySCF)
    """
    # settings
    coords = mm_mol.atom_coords()
    charges = mm_mol.atom_charges()
    blksize = BLKSIZE
    # integrals
    intor = "int3c2e_cart" if mol.cart else "int3c2e_sph"
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, intor)
    # compute interaction potential
    nao = mol.nao_nr()
    mm_pot = np.zeros(nao * (nao + 1) // 2, dtype=np.float64)
    for i0, i1 in lib.prange(0, charges.size, blksize):
        fakemol = gto.fakemol_for_charges(coords[i0:i1])
        j3c = df.incore.aux_e2(mol, fakemol, intor=intor, aosym="s2ij", cintopt=cintopt)
        mm_pot += np.einsum("xk,k->x", j3c, -charges[i0:i1])
    mm_pot = lib.unpack_tril(mm_pot)
    # nuclei interaction with point charges
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()
    nuc_solv = np.zeros(len(mol.atom))
    mm_atom_charges = mm_mol.atom_charges()
    mm_atom_coords = mm_mol.atom_coords()
    for j in range(mol.natm):
        q2, r2 = atom_charges[j], atom_coords[j]
        r = lib.norm(r2 - mm_atom_coords, axis=1)
        nuc_solv[j] = q2 * np.sum(mm_atom_charges / r)
    return mm_pot, nuc_solv


def _pcm(
    mol: gto.Mole, rdm1: np.ndarray, solvent_model: solvent.PCM
) -> Tuple[np.ndarray, np.ndarray]:
    """
    this function returns the pcm potential matrix and the nuclei interaction with the
    solvent (adapted from: solvent/pcm.py:_get_vind() in PySCF)
    """
    surface = solvent_model.surface
    nao = mol.nao_nr()
    rdm1 = rdm1.reshape(-1, nao, nao)
    if rdm1.shape[0] == 2:
        rdm1 = (rdm1[0] + rdm1[1]).reshape(-1, nao, nao)
    # get the electronic part of the potential
    vmat_e = 0.5 * solvent_model._get_vind(np.sum(rdm1, axis=0))[1]
    # calculate the cavity surface charges
    K = solvent_model._intermediates["K"]
    R = solvent_model._intermediates["R"]
    v_grids_e = solvent_model._get_v(rdm1)
    v_grids_n = solvent_model.v_grids_n
    v_grids = v_grids_n - v_grids_e
    b = np.dot(R, v_grids.T)
    q = np.linalg.solve(K, b).T
    vK_1 = np.linalg.solve(K.T, v_grids.T)
    qt = np.dot(R.T, vK_1).T
    q_sym = (q + qt) / 2.0
    # get the nuclear part of the potential
    nuc_solv_pcm = np.zeros(mol.natm)
    for j in range(mol.natm):
        q2, r2 = mol.atom_charges()[j], mol.atom_coords()[j]
        r = lib.norm(r2 - surface["grid_coords"], axis=1)
        nuc_solv_pcm[j] = 0.5 * q2 * np.sum(q_sym / r)
    return vmat_e, nuc_solv_pcm


def _xc_ao_deriv(xc_func: str) -> Tuple[str, int]: #TODO: fix UnboundLocalError for xc="HF" or any unknown functional type
    """
    this function returns the type of xc functional and the level of ao derivatives
    needed
    """
    xc_type = dft.libxc.xc_type(xc_func)
    if xc_type in ["LDA", "HF"]:
        ao_deriv = 0
    elif xc_type in ["GGA", "NLC"]:
        ao_deriv = 1
    elif xc_type == "MGGA":
        ao_deriv = 2
    else:
        raise NotImplementedError(f"xc functional type {xc_type} not supported")
    return xc_type, ao_deriv


def _make_rho_interm1(
    ao_value: np.ndarray, rdm1: np.ndarray, xc_type: str
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    this function returns the rho intermediates (c0, c1) needed in _make_rho()
    (adpated from: dft/numint.py:eval_rho() in PySCF)
    """
    # determine dimensions based on xctype
    xctype = xc_type.upper()
    if xctype == "LDA" or xctype == "HF":
        ngrids, nao = ao_value.shape
    else:
        ngrids, nao = ao_value[0].shape
    # compute rho intermediate based on xctype
    if xctype == "LDA" or xctype == "HF":
        c0 = contract("ik,kj->ij", ao_value, rdm1)
        c1 = None
    elif xctype in ("GGA", "NLC"):
        c0 = contract("ik,kj->ij", ao_value[0], rdm1)
        c1 = None
    else:  # meta-GGA
        c0 = contract("ik,kj->ij", ao_value[0], rdm1)
        c1 = np.empty((3, ngrids, nao), dtype=np.float64)
        for i in range(1, 4):
            c1[i - 1] = contract("ik,jk->ij", ao_value[i], rdm1)
    return c0, c1


def _make_rho_interm2(
    c0: np.ndarray, c1: Optional[np.ndarray], ao_value: np.ndarray, xc_type: str
) -> np.ndarray:
    """
    this function returns rho from intermediates (c0, c1)
    (adpated from: dft/numint.py:eval_rho() in PySCF)
    """
    # determine dimensions based on xctype
    xctype = xc_type.upper()
    if xctype == "LDA" or xctype == "HF":
        ngrids = ao_value.shape[0]
    else:
        ngrids = ao_value[0].shape[0]
    # compute rho intermediate based on xctype
    if xctype == "LDA" or xctype == "HF":
        rho = contract("pi,pi->p", ao_value, c0)
    elif xctype in ("GGA", "NLC"):
        rho = np.empty((4, ngrids), dtype=np.float64)
        rho[0] = contract("pi,pi->p", c0, ao_value[0])
        for i in range(1, 4):
            rho[i] = contract("pi,pi->p", c0, ao_value[i]) * 2.0
    else:  # meta-GGA
        assert c1 is not None
        rho = np.empty((6, ngrids), dtype=np.float64)
        rho[0] = contract("pi,pi->p", ao_value[0], c0)
        rho[5] = 0.0
        for i in range(1, 4):
            rho[i] = contract("pi,pi->p", c0, ao_value[i]) * 2.0
            rho[5] += contract("pi,pi->p", c1[i - 1], ao_value[i])
        XX, YY, ZZ = 4, 7, 9
        ao_value_2 = ao_value[XX] + ao_value[YY] + ao_value[ZZ]
        rho[4] = contract("pi,pi->p", c0, ao_value_2)
        rho[4] += rho[5]
        rho[4] *= 2.0
        rho[5] *= 0.5
    return rho


def _make_rho(
    ao_value: np.ndarray, rdm1: np.ndarray, xc_type: str
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    this function returns important dft intermediates, e.g., energy density, grid
    weights, etc.
    """
    # rho corresponding to given 1-RDM
    if rdm1.ndim == 2:
        c0, c1 = _make_rho_interm1(ao_value, rdm1, xc_type)
        rho = _make_rho_interm2(c0, c1, ao_value, xc_type)
    else:
        if np.allclose(rdm1[0], rdm1[1]):
            c0, c1 = _make_rho_interm1(ao_value, rdm1[0] * 2.0, xc_type)
            rho = _make_rho_interm2(c0, c1, ao_value, xc_type)
        else:
            c0_a, c1_a = _make_rho_interm1(ao_value, rdm1[0], xc_type)
            c0_b, c1_b = _make_rho_interm1(ao_value, rdm1[1], xc_type)
            rho = np.stack(
                (
                    _make_rho_interm2(c0_a, c1_a, ao_value, xc_type),
                    _make_rho_interm2(c0_b, c1_b, ao_value, xc_type),
                )
            )
            c0 = c0_a + c0_b
            if c1_a is not None and c1_b is not None:
                c1 = c1_a + c1_b
            else:
                c1 = None
    return c0, c1, rho


def _vk_dft(
    mol: gto.Mole,
    mf: dft.rks.KohnShamDFT,
    xc_func: str,
    rdm1: np.ndarray,
    vk: np.ndarray,
) -> np.ndarray:
    """
    this function returns the appropriate dft exchange operator
    """
    # range-separated and exact exchange parameters
    ks_omega, ks_alpha, ks_hyb = mf._numint.rsh_and_hybrid_coeff(xc_func)
    # if hybrid func: compute vk
    if abs(ks_hyb) > 1e-10:
        if not hasattr(mf, "vk"):
            vk = mf.get_k(mol=mol, dm=rdm1)
        # scale amount of exact exchange
        vk *= ks_hyb
    else:
        vk_copy = np.copy(vk)
        vk = np.zeros_like(vk_copy)
    # range separated coulomb operator
    if abs(ks_omega) > 1e-10:
        vk_lr = mf.get_k(mol, rdm1, omega=ks_omega)
        vk_lr *= ks_alpha - ks_hyb
        if not hasattr(mf, "vk"):
            vk += vk_lr
        else:
            vk += np.sum(vk_lr, axis=0)
    return vk


def _ao_val(mol: gto.Mole, grids_coords: np.ndarray, ao_deriv: int) -> np.ndarray:
    """
    this function returns ao function values on the given grid
    """
    if not isinstance(mol, pbc_gto.Cell):
        return numint.eval_ao(mol, grids_coords, deriv=ao_deriv)
    else:
        return pbc_numint.eval_ao(mol, grids_coords, deriv=ao_deriv)
