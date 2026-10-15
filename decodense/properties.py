#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
properties module
"""

import copy
import numpy as np
from itertools import starmap
from pyscf import gto, scf, dft, df, lo, lib, solvent
from pyscf.pbc import dft as pbc_dft
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf
from typing import Union, Any, Optional

from .pbctools import ewald_e_nuc
from .tools import dim, make_rdm1, orbsym, contract
from .decomp import CompKeys
from .prop_tools import _h_core, _vk_dft, _xc_energies

# block size in _mm_pot()
BLKSIZE = 200


def prop_tot(
    mol: Union[gto.Mole, pbc_gto.Cell],
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
    prop_type: str,
    part_method: Optional[str],
    ndo: bool,
    gauge_origin: np.ndarray,
    weights: Optional[list[np.ndarray]],
    atom_ref: Optional[np.ndarray] = None,
) -> dict[str, Union[np.ndarray, list[np.ndarray]]]:
    """
    this function returns atom-decomposed and orbital-decomposed mean-field properties
    """
    # declare nested kernel functions in global scope
    global prop_atom_mo
    global prop_atom_ao
    global prop_orb

    # restricted reference
    if mo_occ[0].size == mo_occ[1].size:
        restrict = np.allclose(mo_coeff[0], mo_coeff[1]) and np.allclose(
            mo_occ[0], mo_occ[1]
        )
    else:
        restrict = False

    # dft logical
    dft_calc = isinstance(mf, dft.rks.KohnShamDFT)

    # ao dipole integrals with specified gauge origin
    if prop_type == "dipole":
        with mol.with_common_origin(gauge_origin):
            ao_dip = mol.intor_symmetric("int1e_r", comp=3)
    else:
        ao_dip = None

    # compute total 1-RDMs (AO basis)
    if rdm1 is None:
        rdm1 = rdm1_tot = np.array(
            [make_rdm1(mo_coeff[0], mo_occ[0]), make_rdm1(mo_coeff[1], mo_occ[1])]
        )
    else:
        rdm1_tot = np.array(
            [make_rdm1(mo_coeff[0], mo_occ[0]), make_rdm1(mo_coeff[1], mo_occ[1])]
        )
    if rdm1.ndim == 2:
        rdm1 = np.array([rdm1, rdm1]) * 0.5

    # mol object projected into minao basis
    if pop_method == "iao":
        pmol = lo.iao.reference_mol(mol, minao=minao)
    else:
        pmol = mol

    # molecular dimensions
    alpha, beta = dim(mo_occ)
    natm = pmol.natm

    # nuclear repulsion property
    if prop_type == "energy":
        if isinstance(mol, pbc_gto.Cell):
            prop_nuc_rep = ewald_e_nuc(mol)
        else:
            prop_nuc_rep = _e_nuc(pmol)
    elif prop_type == "dipole":
        prop_nuc_rep = _dip_nuc(pmol, gauge_origin)

    # core hamiltonian
    kin, nuc, sub_nuc = _h_core(mol, mf)

    # get possible solvent contributions
    pot_solv, nuc_solv, vdW_solv = _solvent(mol, mf, np.sum(rdm1, axis=0))

    # fock potential
    if hasattr(mf, "vj"):
        vj = copy.copy(mf.vj)
        if hasattr(mf, "vk"):
            vk = copy.copy(mf.vk)
        else:
            _, vk = mf.get_jk(
                mol=mol,
                dm=np.sum(rdm1, axis=0) if restrict else rdm1,
                with_j=False,
                with_k=not dft_calc,
            )
    else:
        vj, vk = mf.get_jk(
            mol=mol, dm=np.sum(rdm1, axis=0) if restrict else rdm1, with_k=not dft_calc
        )

    # xc energy contributions: atom-wise ("mo"/"ao") or orbital-wise
    e_xc: Optional[Union[np.ndarray, list[np.ndarray]]] = None
    e_xc_nlc: Optional[Union[np.ndarray, list[np.ndarray]]] = None

    if dft_calc:
        # ndo assertion
        if ndo:
            raise NotImplementedError(
                "NDOs for KS-DFT do not yield a lossless decomposition"
            )
        # update exchange operator wrt range-separated parameter and exact exchange
        # components
        vk = _vk_dft(mol, mf, mf.xc, np.sum(rdm1, axis=0) if restrict else rdm1, vk)
        # xc energy (incl. nlc (vv10)) from individual atoms or orbitals
        if prop_type == "energy":
            e_xc, e_xc_nlc = _xc_energies(
                mol, mf, mo_coeff, mo_occ, rdm1, weights, part_method, natm
            )

    # molecular dimensions
    alpha, beta = dim(mo_occ)

    # atomic labels
    if part_method == "ao":
        ao_labels = mol.ao_labels(fmt=None)

    def prop_atom_mo(atom_idx: int) -> dict[str, Any]:
        """
        this function returns MO-based atom-wise energy/dipole contributions
        """
        # init results
        res: dict[str, Union[float, np.ndarray]] = {}
        # atom-specific rdm1
        rdm1_atom = np.zeros_like(rdm1_tot)
        # loop over spins
        if prop_type == "energy" and not restrict:
            res[CompKeys.coul] = 0.0
            res[CompKeys.exch] = 0.0
        for i, spin_mo in enumerate((alpha, beta)):
            # loop over spin-orbitals
            for m, j in enumerate(spin_mo):
                # get orbital(s)
                orb = mo_coeff[i][:, j].reshape(mo_coeff[i].shape[0], -1)
                # orbital-specific rdm1
                rdm1_orb = make_rdm1(orb, mo_occ[i][j])
                # weighted contribution to rdm1_atom
                rdm1_atom[i] += (
                    rdm1_orb * weights[i][m][atom_idx] / np.sum(weights[i][m])
                )
            # coulumb & exchange energy associated with given atom
            if prop_type == "energy" and not restrict:
                res[CompKeys.coul] += _trace(
                    np.sum(vj, axis=0), rdm1_atom[i], scaling=0.5
                )
                res[CompKeys.exch] -= _trace(vk[i], rdm1_atom[i], scaling=0.5)
        # common energy contributions associated with given atom
        if prop_type == "energy":
            rdm1_atom_sum = np.sum(rdm1_atom, axis=0)
            if restrict:
                res[CompKeys.coul] = _trace(vj, rdm1_atom_sum, scaling=0.5)
                res[CompKeys.exch] = -_trace(vk, rdm1_atom_sum, scaling=0.25)
            res[CompKeys.kin] = _trace(kin, rdm1_atom_sum)
            res[CompKeys.nuc_att_loc] = _trace(nuc, rdm1_atom_sum, scaling=0.5)
            res[CompKeys.nuc_att_glob] = _trace(
                sub_nuc[atom_idx], np.sum(rdm1_tot, axis=0), scaling=0.5
            )
            if pot_solv is not None:
                res[CompKeys.solvent] = _trace(pot_solv, rdm1_atom_sum)
            if nuc_solv is not None:
                res[CompKeys.solvent] += nuc_solv[atom_idx]
            if vdW_solv is not None:
                res[CompKeys.solvent_vdw] = vdW_solv[atom_idx]
            # additional xc energy contribution
            if e_xc is not None:
                res[CompKeys.xc] = e_xc[atom_idx]
            if e_xc_nlc is not None:
                res[CompKeys.xc_nlc] = e_xc_nlc[atom_idx]
        elif prop_type == "dipole":
            res[CompKeys.el] = -_trace(ao_dip, np.sum(rdm1_atom, axis=0))
        # sum up electronic contributions
        if prop_type == "energy":
            res[CompKeys.el] = sum(res.values())
        return res

    def prop_atom_ao(atom_idx: int) -> dict[str, Any]:
        """
        this function returns AO-based atom-wise (EDA - energy density analysis)
        energy/dipole contributions
        """
        # init results
        res = {}
        # get AOs on atom k
        select = np.where([atom[0] == atom_idx for atom in ao_labels])[0]
        # common energy contributions associated with given atom
        if prop_type == "energy":
            if restrict:
                res[CompKeys.coul] = _trace(
                    vj[select], np.sum(rdm1_tot, axis=0)[select], scaling=0.5
                )
                res[CompKeys.exch] = -_trace(
                    vk[select], np.sum(rdm1_tot, axis=0)[select], scaling=0.25
                )
            else:
                res[CompKeys.coul] = 0.0
                res[CompKeys.exch] = 0.0
                # loop over spins
                for i, _ in enumerate((alpha, beta)):
                    res[CompKeys.coul] += _trace(
                        np.sum(vj, axis=0)[select], rdm1_tot[i][select], scaling=0.5
                    )
                    res[CompKeys.exch] -= _trace(
                        vk[i][select], rdm1_tot[i][select], scaling=0.5
                    )
            res[CompKeys.kin] = _trace(kin[select], np.sum(rdm1_tot, axis=0)[select])
            res[CompKeys.nuc_att_loc] = _trace(
                nuc[select], np.sum(rdm1_tot, axis=0)[select], scaling=0.5
            )
            res[CompKeys.nuc_att_glob] = _trace(
                sub_nuc[atom_idx], np.sum(rdm1_tot, axis=0), scaling=0.5
            )
            if pot_solv is not None:
                res[CompKeys.solvent] = _trace(
                    pot_solv[select], np.sum(rdm1_tot, axis=0)[select]
                )
            if nuc_solv is not None:
                res[CompKeys.solvent] += nuc_solv[atom_idx]
            if vdW_solv is not None:
                res[CompKeys.solvent_vdw] = vdW_solv[atom_idx]
            # additional xc energy contribution
            if e_xc is not None:
                res[CompKeys.xc] = e_xc[atom_idx]
            if e_xc_nlc is not None:
                res[CompKeys.xc_nlc] = e_xc_nlc[atom_idx]
        elif prop_type == "dipole":
            res[CompKeys.el] = -_trace(
                ao_dip[:, select], np.sum(rdm1_tot, axis=0)[select]
            )
        # sum up electronic contributions
        if prop_type == "energy":
            res[CompKeys.el] = sum(res.values())
        return res

    def prop_orb(spin_idx: int, orb_idx: int) -> dict[str, Any]:
        """
        this function returns orbital-wise energy/dipole contributions
        """
        # init res
        res = {}
        # get orbital(s)
        orb = mo_coeff[spin_idx][:, orb_idx].reshape(mo_coeff[spin_idx].shape[0], -1)
        # orbital-specific rdm1
        rdm1_orb = make_rdm1(orb, mo_occ[spin_idx][orb_idx])
        # total energy or dipole moment associated with given spin-orbital
        if prop_type == "energy":
            if restrict:
                res[CompKeys.coul] = _trace(vj, rdm1_orb, scaling=0.5)
                res[CompKeys.exch] = -_trace(vk, rdm1_orb, scaling=0.25)
            else:
                res[CompKeys.coul] = _trace(np.sum(vj, axis=0), rdm1_orb, scaling=0.5)
                res[CompKeys.exch] = -_trace(vk[spin_idx], rdm1_orb, scaling=0.5)
            res[CompKeys.kin] = _trace(kin, rdm1_orb)
            res[CompKeys.nuc_att] = _trace(nuc, rdm1_orb)
            if pot_solv is not None:
                res[CompKeys.solvent] = _trace(pot_solv, rdm1_orb)
            # additional xc energy contribution
            if e_xc is not None:
                res[CompKeys.xc] = e_xc[spin_idx][orb_idx]
            if e_xc_nlc is not None:
                res[CompKeys.xc_nlc] = e_xc_nlc[spin_idx][orb_idx]
        elif prop_type == "dipole":
            res[CompKeys.el] = -_trace(ao_dip, rdm1_orb)
        # sum up electronic contributions
        if prop_type == "energy":
            res[CompKeys.el] = sum(res.values())
        return res

    # perform decomposition
    prop: dict[str, Union[np.ndarray, list[np.ndarray]]]
    if part_method in ["mo", "ao"]:
        # domain
        domain = np.arange(pmol.natm)
        # execute kernel
        res = list(map(prop_atom_mo if part_method == "mo" else prop_atom_ao, domain))
        # init atom-specific energy or dipole arrays
        if prop_type == "energy":
            prop = {
                comp_key: np.zeros(pmol.natm, dtype=np.float64)
                for comp_key in res[0].keys()
            }
        elif prop_type == "dipole":
            prop = {
                comp_key: np.zeros([pmol.natm, 3], dtype=np.float64)
                for comp_key in res[0].keys()
            }
        # collect results
        for k, r in enumerate(res):
            for key, val in r.items():
                prop[key][k] = val
        if ndo:
            prop[CompKeys.struct] = np.zeros_like(prop_nuc_rep)
        else:
            prop[CompKeys.struct] = prop_nuc_rep
        prop[CompKeys.tot] = prop[CompKeys.el] + prop[CompKeys.struct]

        if (prop_type == "energy") and (atom_ref is not None):
            prop[CompKeys.atom_ref] = atom_ref
            prop[CompKeys.tot_rel] = prop[CompKeys.tot] - atom_ref

        return {**prop}
    else:  # orbs
        # domain: (spin, mo index in mo_coeff, position within the spin channel)
        # position is relevant for non-aufbau occupation
        domain = np.array(
            [
                (i, j, m)
                for i, orbs in enumerate((alpha, beta))
                for m, j in enumerate(orbs)
            ]
        )
        # execute kernel
        res = list(starmap(prop_orb, domain[:, :2]))  # type: ignore
        # init orbital-specific energy or dipole array
        if prop_type == "energy":
            prop = {
                comp_key: [np.zeros(alpha.size), np.zeros(beta.size)]
                for comp_key in res[0].keys()
            }
        elif prop_type == "dipole":
            prop = {
                comp_key: [
                    np.zeros([alpha.size, 3], dtype=np.float64),
                    np.zeros([beta.size, 3], dtype=np.float64),
                ]
                for comp_key in res[0].keys()
            }
        # collect results
        for k, r in enumerate(res):
            for key, val in r.items():
                prop[key][domain[k, 0]][domain[k, 2]] = val
        if ndo:
            prop[CompKeys.struct] = np.zeros_like(prop_nuc_rep)
        else:
            prop[CompKeys.struct] = prop_nuc_rep
        prop[CompKeys.tot] = prop[CompKeys.el]
        return {
            **prop,
            CompKeys.mo_occ: [mo_occ[0][alpha], mo_occ[1][beta]],
            CompKeys.orbsym: orbsym(mol, (mo_coeff[0][:, alpha], mo_coeff[1][:, beta])),
        }


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


def _dip_nuc(mol: gto.Mole, gauge_origin: np.ndarray) -> np.ndarray:
    """
    this function returns the nuclear contribution to the molecular dipole moment
    """
    # coordinates and formal/actual charges of nuclei
    coords = mol.atom_coords()
    form_charges = mol.atom_charges()
    return contract("i,ix->ix", form_charges, coords - gauge_origin)


def _solvent(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT, pbc_scf.RHF],
    rdm1: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    # initialize
    pot_solv, nuc_solv, vdW_solv = None, None, None

    # point charges
    if hasattr(mf, "mm_mol"):
        mm_mol = getattr(mf, "mm_mol", None)
        pot_solv, nuc_solv = _point_charges(mol, mm_mol)
    # pcm
    elif hasattr(mf, "with_solvent"):
        if not isinstance(mf.with_solvent, solvent.pcm.PCM):
            raise NotImplementedError(
                "decodense only supports PCM-type solvent models (C-PCM, IEF-PCM, "
                f"SS(V)PE, COSMO, SMD), not {type(mf.with_solvent).__name__}"
            )
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


def _point_charges(mol: gto.Mole, mm_mol: gto.Mole) -> tuple[np.ndarray, np.ndarray]:
    """
    this function returns the full mm potential and the nuclei interaction with the
    point charges (adapted from: qmmm/itrf.py:get_hcore() in PySCF)
    """
    # settings
    coords = mm_mol.atom_coords()
    charges = mm_mol.atom_charges()
    # integrals
    intor = "int3c2e_cart" if mol.cart else "int3c2e_sph"
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, intor)
    # compute interaction potential
    nao = mol.nao_nr()
    mm_pot = np.zeros(nao * (nao + 1) // 2, dtype=np.float64)
    for i0, i1 in lib.prange(0, charges.size, BLKSIZE):
        fakemol = gto.fakemol_for_charges(coords[i0:i1])
        j3c = df.incore.aux_e2(mol, fakemol, intor=intor, aosym="s2ij", cintopt=cintopt)
        mm_pot += np.einsum("xk,k->x", j3c, -charges[i0:i1])
    mm_pot = lib.unpack_tril(mm_pot)
    # nuclei interaction with point charges
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()
    nuc_solv = np.zeros(mol.natm)
    mm_atom_charges = mm_mol.atom_charges()
    mm_atom_coords = mm_mol.atom_coords()
    for j in range(mol.natm):
        q2, r2 = atom_charges[j], atom_coords[j]
        r = lib.norm(r2 - mm_atom_coords, axis=1)
        nuc_solv[j] = q2 * np.sum(mm_atom_charges / r)
    return mm_pot, nuc_solv


def _pcm(
    mol: gto.Mole, rdm1: np.ndarray, solvent_model: solvent.PCM
) -> tuple[np.ndarray, np.ndarray]:
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


def _trace(
    op: np.ndarray, rdm1: np.ndarray, scaling: float = 1.0
) -> Union[float, np.ndarray]:
    """
    this function returns the trace between an operator and an rdm1
    """
    if op.ndim == 2:
        return contract("ij,ij", op, rdm1) * scaling
    else:
        return contract("xij,ij->x", op, rdm1) * scaling
